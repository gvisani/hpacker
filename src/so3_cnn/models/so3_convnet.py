
import sys, os

import numpy as np
import torch
import e3nn
from e3nn import o3
from src.so3_cnn import nn

from torch import Tensor
from typing import *


class SO3_ConvNet(torch.nn.Module):
    '''
    CGNet-like model, but without the invariant skip connections
    '''

    def load_hparams(self, hparams: Dict):

        # this is just a global normalization factor applied to all data (e.g. the mean sqrt power of the training data)
        # useful to store it inside the model for doing inference on new test data
        # we usually don't *need* to do this for supervised models, but I noticed it at least speeds up convergence. so perhaps it is useful, but requires more testing
        self.input_normalizing_constant = torch.tensor(hparams['input_normalizing_constant'], requires_grad=False) if hparams['input_normalizing_constant'] is not None else None

        ## hyperparams of the CG blocks
        self.n_cg_blocks = hparams['n_cg_blocks']

        self.do_initial_linear_projection = hparams['do_initial_linear_projection'] # required to be true when the first value of self.ch_nonlin_rule_list is 'elementwise'
        self.ch_initial_linear_projection = hparams['ch_initial_linear_projection'] #   because the 'elementwise' Tensor Product requires an equal numver of channels per \ell

        # these two control the dimensionality of the CG blocks, in terms of maximum spherical degree \ell, and number of channels (equal for all \ell)
        self.lmax_list = hparams['lmax_list']
        self.ch_size_list = hparams['ch_size_list']

        # these two govern the Tensor Product rules in each block
        self.ls_nonlin_rule_list = hparams['ls_nonlin_rule_list']
        self.ch_nonlin_rule_list = hparams['ch_nonlin_rule_list']

        assert self.n_cg_blocks == len(self.lmax_list)
        assert self.n_cg_blocks == len(self.ch_size_list)
        assert self.n_cg_blocks == len(self.ls_nonlin_rule_list)
        assert self.n_cg_blocks == len(self.ch_nonlin_rule_list)

        self.use_additive_skip_connections = hparams['use_additive_skip_connections'] # zero-padded on one side if self.ch_size_list[i] > self.ch_size_list[i-1]

        self.weights_initializer = hparams['weights_initializer'] # do not bother with this

        # batch norm, if requested, is applied *at the start* of each CG block
        self.use_batch_norm = hparams['use_batch_norm']

        # hyperparams of the norm layer, outside of batch norm
        # I am listing the default values
        self.norm_type = hparams['norm_type'] # signal
        self.normalization = hparams['normalization'] # component
        self.norm_balanced = hparams['norm_balanced'] # False
        self.norm_affine = hparams['norm_affine'] # per_l
        self.norm_nonlinearity = hparams['norm_nonlinearity'] # None
        self.norm_location = hparams['norm_location'] # between

        self.linearity_first = hparams['linearity_first'] # whether to apply the linear transformation first (or the nonlinearity first), keep False
        self.filter_symmetric = hparams['filter_symmetric'] # keep True always, no reason to do otherwise. Does not change anything for 'efficient' ls_nonlin_rule, and reduces unnecessary computation for 'full' ls_nonlin_rule

        ## hyperparams of the fully-connected layers on the invariant (\ell=0) output of the CG blocks
        self.n_fc_blocks = hparams['n_fc_blocks']
        self.fc_h_dim = hparams['fc_h_dim']
        self.fc_nonlin = hparams['fc_nonlin']
        self.dropout_rate = hparams['dropout_rate']

        # self-evident
        self.output_dim = hparams['output_dim']



    def __init__(self,
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 hparams: Dict,
                 normalize_input_at_runtime: bool = False,
                 verbose: bool = False
                 ):
        super().__init__()

        self.irreps_in = irreps_in
        self.load_hparams(hparams)
        self.normalize_input_at_runtime = normalize_input_at_runtime

        assert self.n_cg_blocks == len(self.ch_size_list)
        assert self.lmax_list is None or self.n_cg_blocks == len(self.lmax_list)
        assert self.n_cg_blocks == len(self.ls_nonlin_rule_list)
        assert self.n_cg_blocks == len(self.ch_nonlin_rule_list)

        if self.do_initial_linear_projection:
            if verbose: print(self.irreps_in.dim, self.irreps_in)
            initial_irreps = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(self.irreps_in.ls), 1)).sort().irreps.simplify()
            self.initial_linear_projection = nn.SO3_linearity(self.irreps_in, initial_irreps)
            if verbose: print(initial_irreps.dim, initial_irreps)
        else:
            if verbose: print(self.irreps_in.dim, self.irreps_in)
            initial_irreps = self.irreps_in


        # equivariant, cg blocks
        prev_irreps = initial_irreps
        cg_blocks = []
        for i in range(self.n_cg_blocks):
            irreps_hidden = (self.ch_size_list[i]*o3.Irreps.spherical_harmonics(self.lmax_list[i], 1)).sort().irreps.simplify()
            cg_blocks.append(nn.CGBlock(prev_irreps,
                                                irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=self.linearity_first,
                                                filter_symmetric=self.filter_symmetric,
                                                use_batch_norm=self.use_batch_norm,
                                                ls_nonlin_rule=self.ls_nonlin_rule_list[i], # full, elementwise, efficient
                                                ch_nonlin_rule=self.ch_nonlin_rule_list[i], # full, elementwise
                                                norm_type=self.norm_type, # None, layer, signal
                                                normalization=self.normalization, # norm, component -> only if norm_type is not none
                                                norm_balanced=self.norm_balanced,
                                                norm_affine=self.norm_affine, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                                                norm_nonlinearity=self.norm_nonlinearity, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                                                norm_location=self.norm_location, # first, between, last
                                                weights_initializer=self.weights_initializer,
                                                init_scale=1.0))

            prev_irreps = cg_blocks[-1].irreps_out
            if verbose: print(prev_irreps.dim, prev_irreps)

        self.cg_blocks = torch.nn.ModuleList(cg_blocks)

        invariants_dim = [mul for (mul, _) in prev_irreps][0] # number of channels for l = 0


        # invariant, fully connected blocks
        prev_dim = invariants_dim
        fc_blocks = []
        for _ in range(self.n_fc_blocks):
            block = []
            block.append(torch.nn.Linear(prev_dim, self.fc_h_dim))
            # block.append(torch.nn.BatchNorm1d(self.fc_h_dim)) # remove this sometimes, with older models
            block.append(eval(nn.NONLIN_TO_ACTIVATION_MODULES[self.fc_nonlin]))
            if self.dropout_rate > 0.0:
                block.append(torch.nn.Dropout(self.dropout_rate))

            fc_blocks.append(torch.nn.Sequential(*block))
            prev_dim = self.fc_h_dim

        if len(fc_blocks) > 0:
            self.fc_blocks = torch.nn.ModuleList(fc_blocks)
        else:
            self.fc_blocks = None


        # output head
        self.output_head = torch.nn.Linear(prev_dim, self.output_dim)

    
    def forward(self, x: Dict[int, Tensor]) -> Tensor:

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(x)
        else:
            h = x
        
        # equivariant, cg blocks
        for i, block in enumerate(self.cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp
        
        invariants = h[0].squeeze(-1)


        # invariant, fully connected blocks
        h = invariants
        if self.fc_blocks is not None:
            for block in self.fc_blocks:
                h = block(h)
                # h += block(h) # skip connections

        # output head
        out = self.output_head(h)

        return out

    def predict(self,
                dataloader: torch.utils.data.DataLoader,
                emb_i: int = -1,
                device: str = 'cpu', verbose: bool = False, loading_bar: bool = False) -> Dict:
        from src.so3_cnn.so3.functional import put_dict_on_device
        from tqdm import tqdm

        if loading_bar:
            loading_bar = tqdm
        else:
            loading_bar = lambda x: x

        if verbose: print('Making predictions on %s.' % device)

        self.eval()
        
        # inference loop!
        embeddings_all = []
        y_hat_all_logits = []
        y_hat_all_index = []
        y_all = []
        res_ids_all = []
        for i, (X, X_vec, y, (rot, res_ids)) in loading_bar(enumerate(dataloader)):
            X = put_dict_on_device(X, device)
            y = y.to(device)
            self.eval()
            
            y_hat = self(X)
            embeddings = self.get_inv_embedding(X, emb_i=emb_i)

            assert np.allclose(self.output_head(embeddings).detach().cpu().numpy(), y_hat.detach().cpu().numpy())

            embeddings_all.append(embeddings.detach().cpu().numpy())
            y_hat_all_logits.append(y_hat.detach().cpu().numpy())
            y_hat_all_index.append(np.argmax(y_hat.detach().cpu().numpy(), axis=1))
            y_all.append(y.detach().cpu().numpy())
            res_ids_all.append(res_ids)

        embeddings_all = np.vstack(embeddings_all)
        y_hat_all_logits = np.vstack(y_hat_all_logits)
        y_hat_all_index = np.hstack(y_hat_all_index)
        y_all = np.hstack(y_all)
        res_ids_all = np.hstack(res_ids_all)
    
        return {
            'embeddings': embeddings_all,
            'logits': y_hat_all_logits,
            'best_indices': y_hat_all_index,
            'targets': y_all,
            'res_ids': res_ids_all
        }

    def get_inv_embedding(self, x: Dict[int, Tensor], emb_i: Union[int, str] = -1) -> Tensor:
        '''
        Gets invariant embedding from the FC blocks (backwards, must be negative), or from the input to the FC blocks
        '''
        assert emb_i in ['cg_output', 'all'] or emb_i in [-i for i in range(1, self.n_fc_blocks + 1)]
        self.eval()

        all_output = []

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(x)
        else:
            h = x
        
        # equivariant, cg blocks
        for i, block in enumerate(self.cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp
        
        invariants = h[0].squeeze(-1)


        if emb_i == 'cg_output':
            return invariants
        elif emb_i == 'all':
            all_output.append(invariants)
        
        h = invariants
        if self.fc_blocks is not None:
            for n, block in enumerate(self.fc_blocks):
                h = block(h)
                if emb_i == 'all':
                    all_output.append(h)
                elif n == len(self.fc_blocks) + emb_i:
                    return h
        
        # NB: only runs if emb_i == 'all'
        return all_output
