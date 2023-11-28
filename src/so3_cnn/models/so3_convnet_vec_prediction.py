
import sys, os

import numpy as np
import torch
import e3nn
from e3nn import o3
from src.so3_cnn import nn

from torch import Tensor
from typing import *

class SO3_ConvNet_VecPrediction(torch.nn.Module):
    '''
    CGNet-like model, but without the invariant skip connections
    Predicts vectors, not scalars
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
                 normalize_input_at_runtime: bool = False
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
            print(self.irreps_in.dim, self.irreps_in)
            initial_irreps = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(self.irreps_in.ls), 1)).sort().irreps.simplify()
            self.initial_linear_projection = nn.SO3_linearity(self.irreps_in, initial_irreps)
            print(initial_irreps.dim, initial_irreps)
        else:
            print(self.irreps_in.dim, self.irreps_in)
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
            print(prev_irreps.dim, prev_irreps)

        self.cg_blocks = torch.nn.ModuleList(cg_blocks)

        vectors_dim = [mul for (mul, _) in prev_irreps][1] # number of channels for l = 1
        

        # equivariant blocks to predict the next vector
        hidden_rep_irreps = o3.Irreps(f'{vectors_dim}x1e')
        irreps_out = o3.Irreps(f'{self.output_dim}x1e')

        self.vector_predictor = nn.SO3_linearity(hidden_rep_irreps, irreps_out, weights_initializer=self.weights_initializer)

    
    def forward(self, x: Dict[int, Tensor]) -> Tensor:

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant
        
        # print()
        # print('At the beginning')
        # for l in x:
        #     print(f'\t{l}: is inf? {torch.isinf(x[l]).any()} is nan? {torch.isnan(x[l]).any()}')
        # print()

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(x)
        else:
            h = x
        
        # print('After initial projection')
        # for l in h:
        #     print(f'\t{l}: is inf? {torch.isinf(h[l]).any()} is nan? {torch.isnan(h[l]).any()}')
        # print()
        
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

        # print('After CG blocks')
        # for l in h:
        #     print(f'\t{l}: is inf? {torch.isinf(h[l]).any()} is nan? {torch.isnan(h[l]).any()}')
        # print()

        # output head
        out = self.vector_predictor(h)[1]

        # print('After output head')
        # print(f'\tout: is inf? {torch.isinf(out).any()} is nan? {torch.isnan(out).any()}')
        # print()

        return out



class SO3_ConvNet_VecPrediction_WithInvariantConditioning(torch.nn.Module):
    '''
    CGNet-like model, but without the invariant skip connections
    Predicts vectors, not scalars
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

        # self-evident
        self.output_dim = hparams['output_dim']

        # conditioning
        self.conditioning_dim = hparams['conditioning_dim']
        self.conditioning_hidden_dim = hparams['conditioning_hidden_dim']



    def __init__(self,
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 hparams: Dict,
                 normalize_input_at_runtime: bool = False
                 ):
        super().__init__()

        self.irreps_in = irreps_in
        self.load_hparams(hparams)
        self.normalize_input_at_runtime = normalize_input_at_runtime

        assert self.n_cg_blocks == len(self.ch_size_list)
        assert self.lmax_list is None or self.n_cg_blocks == len(self.lmax_list)
        assert self.n_cg_blocks == len(self.ls_nonlin_rule_list)
        assert self.n_cg_blocks == len(self.ch_nonlin_rule_list)

        self.conditioning_projection = torch.nn.Linear(self.conditioning_dim, self.conditioning_hidden_dim)
        conditioned_irreps = (self.irreps_in + o3.Irreps(f'{self.conditioning_hidden_dim}x0e')).sort().irreps.simplify()

        if self.do_initial_linear_projection:
            print(conditioned_irreps.dim, conditioned_irreps)

            initial_irreps = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(conditioned_irreps.ls), 1)).sort().irreps.simplify()
            self.initial_linear_projection = nn.SO3_linearity(conditioned_irreps, initial_irreps)

            print(initial_irreps.dim, initial_irreps)
        else:
            print(conditioned_irreps.dim, conditioned_irreps)
            initial_irreps = conditioned_irreps


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
            print(prev_irreps.dim, prev_irreps)

        self.cg_blocks = torch.nn.ModuleList(cg_blocks)

        vectors_dim = [mul for (mul, _) in prev_irreps][1] # number of channels for l = 1

        # equivariant blocks to predict the next vector
        hidden_rep_irreps = o3.Irreps(f'{vectors_dim}x1e')
        irreps_out = o3.Irreps(f'{self.output_dim}x1e')

        self.vector_predictor = nn.SO3_linearity(hidden_rep_irreps, irreps_out, weights_initializer=self.weights_initializer)

    
    def forward(self, x: Dict[int, Tensor], c: Tensor) -> Tensor:

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant
        
        # conditioning!
        x[0] = torch.cat([x[0], self.conditioning_projection(c).unsqueeze(-1)], dim=1) # NOTE: h[0] has three dimensions, even though the third one is theoretically pointless: (batch, channels, 2*0 + 1)

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

        # only keep vectors
        h = {1: h[1]}

        # output head
        out = self.vector_predictor(h)[1]

        return out

