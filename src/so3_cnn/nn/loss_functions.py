
'''
All functions are setup using the same input arguments
'''


import torch
import e3nn
from e3nn import o3
import functools

from typing import *

torch.set_default_dtype(torch.float32)

NAME_TO_LOSS_FN = {'mse': 'setup__mse_loss',
                   'mse_normalized': 'setup__mse_normalized_loss',
                   'cosine': 'setup__cosine_loss'}

def setup__mse_loss(irreps: o3.Irreps, device: str) -> Callable:
    return torch.nn.functional.mse_loss

def setup__mse_normalized_loss(irreps: o3.Irreps, device: str) -> Callable:
    signal_ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in irreps.ls])
    signal_norm = torch.tensor([2*l+1 for l in signal_ls_indices]).to(device)

    def mse_normalized_loss(input, target, norm=None):
        return torch.mean(torch.square(input - target) / norm)

    return functools.partial(mse_normalized_loss, norm=signal_norm)

def setup__cosine_loss(irreps: o3.Irreps, device: str) -> Callable:
    dot_product = o3.TensorProduct(irreps, irreps, "0e", [(i, i, 0, 'uuw', False)
                                                          for i, (mul, ir) in enumerate(irreps)],
                                                          irrep_normalization='norm').to(device)
    
    one = torch.tensor(1.0).to(device)
    eps = torch.tensor(1e-9).to(device)
    def cosine_loss(input, target, dot_product=None):
        return torch.mean(one - dot_product(input, target) / (torch.sqrt((dot_product(input, input) * dot_product(target, target))) + eps))
    
    return functools.partial(cosine_loss, dot_product=dot_product)