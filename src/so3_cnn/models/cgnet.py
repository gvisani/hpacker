
import sys, os

import numpy as np
import torch
import e3nn
from e3nn import o3
from src.so3_cnn import nn

from torch import Tensor
from typing import *


class CGNet(torch.nn.Module):
    '''
    H-CNN with Mike's implementation.
    It's a CGNet, except with the linearity *before* the nonlinearity
    '''

    def load_hparams(self, hparams: Dict):
        self.output_dim = hparams['output_dim']
        self.n_cg_blocks = hparams['n_cg_blocks']
        self.ch_dim = hparams['ch_dim']
        self.ls_nonlin_rule = hparams['ls_nonlin_rule']
        self.ch_nonlin_rule = hparams['ch_nonlin_rule']
        self.n_fc_blocks = hparams['n_fc_blocks']
        self.fc_h_dim = hparams['fc_h_dim']
        self.fc_nonlin = hparams['fc_nonlin']
        self.dropout_rate = hparams['dropout_rate']
        self.input_normalizing_constant = torch.tensor(hparams['input_normalizing_constant'], requires_grad=False) if hparams['input_normalizing_constant'] is not None else None

    def __init__(self,
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict,
                 hparams: Dict,
                 normalize_input_at_runtime: bool = False
                 ):
        super().__init__()

        self.irreps_in = irreps_in
        self.load_hparams(hparams)
        self.normalize_input_at_runtime = normalize_input_at_runtime
        self.lmax = max(self.irreps_in.ls)

        # equivariant, CG blocks
        prev_irreps = self.irreps_in
        cg_blocks = []
        invariants_dim = np.sum(np.array(prev_irreps.ls) == 0)
        for _ in range(self.n_cg_blocks):
            irreps_hidden = (self.ch_dim*o3.Irreps.spherical_harmonics(self.lmax, 1)).sort().irreps.simplify()
            cg_blocks.append(nn.CGBlock(prev_irreps,
                                                irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=True,
                                                filter_symmetric=True,
                                                use_batch_norm=True,
                                                ls_nonlin_rule=self.ls_nonlin_rule, # full, elementwise, efficient
                                                ch_nonlin_rule=self.ch_nonlin_rule, # full, elementwise
                                                norm_type=None # None, layer, signal
                                                ))
            prev_irreps = cg_blocks[-1].irreps_out
            print(prev_irreps)
            invariants_dim += np.sum(np.array(prev_irreps.ls) == 0)
        self.cg_blocks = torch.nn.ModuleList(cg_blocks)
        
        print('Invariants_dim: %d.' % (invariants_dim))
        sys.stdout.flush()

        # invariant, fully connected blocks
        prev_dim = invariants_dim
        fc_blocks = []
        for _ in range(self.n_fc_blocks):
            block = []
            block.append(torch.nn.Linear(prev_dim, self.fc_h_dim))
            block.append(eval(nn.NONLIN_TO_ACTIVATION_MODULES[self.fc_nonlin]))
            if self.dropout_rate > 0.0:
                block.append(torch.nn.Dropout(self.dropout_rate))

            fc_blocks.append(torch.nn.Sequential(*block))
            prev_dim = self.fc_h_dim

        self.fc_blocks = torch.nn.ModuleList(fc_blocks)


        # output head
        self.output_head = torch.nn.Linear(prev_dim, self.output_dim)

    
    def forward(self, x: Dict[int, Tensor]) -> Tensor:

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant

        # equivariant, CG blocks
        h = x
        invariants = [h[0].squeeze(-1)]
        for block in self.cg_blocks:
            h = block(h)
            invariants.append(h[0].squeeze(-1))
        
        invariants = torch.cat(invariants, axis=1)


        # invariant, fully connected blocks
        h = invariants
        if self.fc_blocks is not None:
            for block in self.fc_blocks:
                h = block(h)
        

        # output head
        out = self.output_head(h)

        return out

    def get_inv_embedding(self, x: Dict[int, Tensor], emb_i: Union[int, str]) -> Tensor:
        '''
        Gets invariant embedding from the FC blocks (backwards, muct be negative), or from the input to the FC blocks
        '''
        assert emb_i in ['cg_output', 'all'] or emb_i in [-i for i in range(1, self.n_fc_blocks + 1)]
        self.eval()

        all_output = []

        # equivariant, CG blocks
        h = x
        invariants = [h[0].squeeze(-1)]
        for block in self.cg_blocks:
            h = block(h)
            invariants.append(h[0].squeeze(-1))
        
        invariants = torch.cat(invariants, axis=1)

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
