from urllib.parse import non_hierarchical
import torch
from torch import Tensor
from e3nn import o3
import e3nn.nn as e3nn_nn
import functools
import sys, os

from .linearity import SO3_linearity
from .nonlinearity import TP_nonlinearity
from .normalization import layer_norm_nonlinearity, signal_norm, batch_norm, magnitudes_norm

from typing import *

NONLIN_TO_ACTIVATION_MODULES = {
    'swish': 'torch.nn.SiLU()',
    'sigmoid': 'torch.nn.Sigmoid()',
    'relu': 'torch.nn.ReLU()',
    'identity': 'torch.nn.Identity()'
}

class CGBlock(torch.nn.Module):
    def __init__(self,
                 irreps_in: o3.Irreps,
                 irreps_hidden: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 linearity_first: bool = False,
                 filter_symmetric: bool = True,
                 use_batch_norm: bool = True,
                 ls_nonlin_rule: str = 'full', # full, elementwise, efficient
                 ch_nonlin_rule: str = 'full',
                 norm_type: Optional[str] = None, # None, layer, signal, layer_and_signal
                 normalization: str = 'component', # norm, component -> only if norm_type is not none
                 norm_balanced: bool = False,
                 norm_affine: Optional[Union[str, bool, Tuple[bool, str]]] = None, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm, Tuple of both kinds --> layer_and_signal
                 norm_nonlinearity: Optional[str] = 'identity', # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                 norm_location: str = 'between', # first, between, last --> when norm_type = layer_and_signal, only specifies location of signal_norm (layer_norm goes first)
                 weights_initializer: Optional[Callable] = None,
                 init_scale: float = 1.0):
        '''
        "Note, it's common practice to avoid using batch normalization when training VAEs, 
        since the additional stochasticity due to using mini-batches may aggravate instability 
        on top of the stochasticity from sampling." --> https://www.tensorflow.org/tutorials/generative/cvae
        --> I think this is especially unnecessary in the decoder
        '''
        super().__init__()

        self.irreps_in = irreps_in
        self.linearity_first = linearity_first
        self.ls_nonlin_rule = ls_nonlin_rule
        self.ch_nonlin_rule = ch_nonlin_rule
        self.use_batch_norm = use_batch_norm

        filter_ir_out = [irr.ir for irr in irreps_hidden]

        layers = []
        if linearity_first:
            layers.append(SO3_linearity(irreps_in, irreps_hidden, weights_initializer=weights_initializer, scale_for_weights_init=init_scale))

            # with linearity first, the use of batch norm and other normalization techniques are mutually exclusive
            if self.use_batch_norm:
                layers.append(batch_norm(irreps_hidden, affine=True, momentum=0.1, normalization=normalization))
            elif norm_type == 'layer':
                layers.append(batch_norm(layers[-1].irreps_out.simplify(), layer=True, affine=norm_affine, normalization=normalization))
            elif norm_type == 'signal':
                layers.append(signal_norm(irreps_in, normalization=normalization, affine=norm_affine, balanced=norm_balanced))

            layers.append(TP_nonlinearity(irreps_hidden, w3j_matrices, filter_ir_out=filter_ir_out, ls_rule=ls_nonlin_rule, channel_rule=ch_nonlin_rule, filter_symmetric=filter_symmetric))
        
        else:
            if self.use_batch_norm:
                layers.append(batch_norm(irreps_in, affine=True, momentum=0.1, normalization=normalization))

            if norm_type == 'layer_and_signal':
                layers.append(batch_norm(irreps_in, instance=True, affine=True, normalization=normalization))
            
            if norm_location == 'first':
                if norm_type == 'layer_nonlin':
                    layers.append(layer_norm_nonlinearity(irreps_in, normalization=normalization, affine=norm_affine, nonlinearity=norm_nonlinearity))
                elif norm_type in ['signal', 'layer_and_signal']:
                    layers.append(signal_norm(irreps_in, normalization=normalization, affine=norm_affine, balanced=norm_balanced))
                elif norm_type == 'magnitudes':
                    layers.append(magnitudes_norm(irreps_in, return_magnitudes=False))
                elif norm_type == 'instance':
                    layers.append(batch_norm(irreps_in, instance=True, affine=norm_affine, normalization=normalization))
                elif norm_type == 'layer':
                    layers.append(batch_norm(irreps_in, layer=True, affine=norm_affine, normalization=normalization))

            layers.append(TP_nonlinearity(irreps_in, w3j_matrices, filter_ir_out=filter_ir_out, ls_rule=ls_nonlin_rule, channel_rule=ch_nonlin_rule, filter_symmetric=filter_symmetric))
            
            if norm_location == 'between':
                if norm_type == 'layer_nonlin':
                    layers.append(layer_norm_nonlinearity(layers[-1].irreps_out.simplify(), normalization=normalization, affine=norm_affine, nonlinearity=norm_nonlinearity))
                elif norm_type in ['signal', 'layer_and_signal']:
                    layers.append(signal_norm(layers[-1].irreps_out.simplify(), normalization=normalization, affine=norm_affine, balanced=norm_balanced))
                elif norm_type == 'magnitudes':
                    layers.append(magnitudes_norm(layers[-1].irreps_out.simplify(), return_magnitudes=False))
                elif norm_type == 'instance':
                    layers.append(batch_norm(layers[-1].irreps_out.simplify(), instance=True, affine=norm_affine, normalization=normalization))
                elif norm_type == 'layer':
                    layers.append(batch_norm(layers[-1].irreps_out.simplify(), layer=True, affine=norm_affine, normalization=normalization))

            layers.append(SO3_linearity(layers[-1].irreps_out.simplify(), irreps_hidden, weights_initializer=weights_initializer, scale_for_weights_init=init_scale))

            if norm_location == 'last':
                if norm_type == 'layer_nonlin':
                    layers.append(layer_norm_nonlinearity(irreps_hidden, normalization=normalization, affine=norm_affine, nonlinearity=norm_nonlinearity))
                elif norm_type in ['signal', 'layer_and_signal']:
                    layers.append(signal_norm(irreps_hidden, normalization=normalization, affine=norm_affine, balanced=norm_balanced))
                elif norm_type == 'magnitudes':
                    layers.append(magnitudes_norm(irreps_hidden, return_magnitudes=False))
                elif norm_type == 'instance':
                    layers.append(batch_norm(irreps_hidden, instance=True, affine=norm_affine, normalization=normalization))
                elif norm_type == 'layer':
                    layers.append(batch_norm(irreps_hidden, layer=True, affine=norm_affine, normalization=normalization))

        self.layers = torch.nn.ModuleList(layers)
        self.irreps_out = layers[-1].irreps_out.simplify()
    
    # @profile
    def forward(self, h: Tensor):
        for layer in self.layers:
            h = layer(h)
        
        # for layer in self.layers:
        #     if isinstance(layer, SO3_linearity):
        #         # is_inf_before = torch.any(torch.tensor([torch.any(torch.isinf(h[l])) for l in h])).item()
        #         # is_nan_before = torch.any(torch.tensor([torch.any(torch.isnan(h[l])) for l in h])).item()
        #         is_inf_before = 'N/A'
        #         is_nan_before = 'N/A'
        #         h = layer(h)
        #         is_inf_after = torch.any(torch.tensor([torch.any(torch.isinf(h[l])) for l in h])).item()
        #         is_nan_after = torch.any(torch.tensor([torch.any(torch.isnan(h[l])) for l in h])).item()
        #         print('Linearity:\tisinf [%s %s], isnan [%s %s]' % (is_inf_before, is_inf_after, is_nan_before, is_nan_after), file=sys.stderr)
        #     elif isinstance(layer, TP_nonlinearity):
        #         # is_inf_before = torch.any(torch.tensor([torch.any(torch.isinf(h[l])) for l in h])).item()
        #         # is_nan_before = torch.any(torch.tensor([torch.any(torch.isnan(h[l])) for l in h])).item()
        #         is_inf_before = 'N/A'
        #         is_nan_before = 'N/A'
        #         h = layer(h)
        #         is_inf_after = torch.any(torch.tensor([torch.any(torch.isinf(h[l])) for l in h])).item()
        #         is_nan_after = torch.any(torch.tensor([torch.any(torch.isnan(h[l])) for l in h])).item()
        #         print('Nonlinearity:\tisinf [%s %s], isnan [%s %s]' % (is_inf_before, is_inf_after, is_nan_before, is_nan_after), file=sys.stderr)
        #     elif isinstance(layer, batch_norm):
        #         # is_inf_before = torch.any(torch.tensor([torch.any(torch.isinf(h[l])) for l in h])).item()
        #         # is_nan_before = torch.any(torch.tensor([torch.any(torch.isnan(h[l])) for l in h])).item()
        #         is_inf_before = 'N/A'
        #         is_nan_before = 'N/A'
        #         h = layer(h)
        #         is_inf_after = torch.any(torch.tensor([torch.any(torch.isinf(h[l])) for l in h])).item()
        #         is_nan_after = torch.any(torch.tensor([torch.any(torch.isnan(h[l])) for l in h])).item()
        #         print('Batch norm:\tisinf [%s %s], isnan [%s %s]' % (is_inf_before, is_inf_after, is_nan_before, is_nan_after), file=sys.stderr)
        #     elif isinstance(layer, signal_norm):
        #         # is_inf_before = torch.any(torch.tensor([torch.any(torch.isinf(h[l])) for l in h])).item()
        #         # is_nan_before = torch.any(torch.tensor([torch.any(torch.isnan(h[l])) for l in h])).item()
        #         is_inf_before = 'N/A'
        #         is_nan_before = 'N/A'
        #         h = layer(h)
        #         is_inf_after = torch.any(torch.tensor([torch.any(torch.isinf(h[l])) for l in h])).item()
        #         is_nan_after = torch.any(torch.tensor([torch.any(torch.isnan(h[l])) for l in h])).item()
        #         print('Signal norm:\tisinf [%s %s], isnan [%s %s]' % (is_inf_before, is_inf_after, is_nan_before, is_nan_after), file=sys.stderr)
        #     else:
        #         h = layer(h)

        return h


class FFNN_block(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: Optional[int] = None,
                 nonlinearity: str = 'relu',
                 use_batch_norm: bool = False,
                 dropout_rate: float = 0.0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001))
            layers.append(eval(NONLIN_TO_ACTIVATION_MODULES[nonlinearity]))
            if dropout_rate > 0.0:
                layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        if output_dim:
            layers.append(torch.nn.Linear(prev_dim, output_dim))
            self.output_dim = output_dim
        else:
            self.output_dim = prev_dim
        
        if len(layers) > 0:
            self.ffnn = torch.nn.Sequential(*layers)
        else:
            self.ffnn = torch.nn.Identity()
    
    def forward(self, x: Tensor):
        return self.ffnn(x)
        
