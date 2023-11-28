
import torch
import e3nn
from e3nn import o3

from torch import Tensor
from typing import *


def get_random_wigner_D(lmax: int):
    rot_matrix = o3.rand_matrix()
    alpha, beta, gamma = o3.matrix_to_angles(rot_matrix)
    wigner = {}
    for l in range(lmax + 1):
        wigner[l] = o3.wigner_D(l, alpha, beta, gamma)
    return wigner

def get_wigner_D_from_rot_matrix(lmax: int, rot_matrix: Tensor):
    alpha, beta, gamma = o3.matrix_to_angles(rot_matrix)
    wigner = {}
    for l in range(lmax + 1):
        wigner[l] = o3.wigner_D(l, alpha, beta, gamma)
    return wigner

def get_wigner_D_from_alpha_beta_gamma(lmax: int, alpha: Tensor, beta: Tensor, gamma: Tensor):
    wigner = {}
    for l in range(lmax + 1):
        wigner[l] = o3.wigner_D(l, alpha, beta, gamma)
    return wigner

def rotate_signal(signal: Tensor, irreps: o3.Irreps, wigner: Dict):
    '''
    wigner must contain wigner-D matrices for all l's in irreps, otherwise a KeyError will be thrown
    '''
    wigner_ls = [wigner[l] for l in irreps.ls]
    rot_mat = torch.block_diag(*wigner_ls)
    rotated_signal = torch.matmul(signal, torch.t(rot_mat)) # Compute R = S * W^T --> 
    return rotated_signal


def get_wigner_D_block_from_rot_matrices(irreps: o3.Irreps, rot_matrices: Tensor) -> Tuple[Dict[int, Tensor], Tensor]:
    '''
    rot_matrices: Tensor with shape (B, 3, 3)

    output: Dict indexed by \ell, with values batched Wigner-D matrices
    '''
    irreps_unique_ls = sorted(list(set(irreps.ls)))
    wigner = {}
    for l in irreps_unique_ls:
        wigner[l] = []

    block_diag_wigner = []
    for b in range(rot_matrices.shape[0]):
        rot_matrix = rot_matrices[b]
    
        alpha, beta, gamma = o3.matrix_to_angles(rot_matrix)
        temp_wigner = {}
        for l in irreps_unique_ls:
            temp_wigner[l] = o3.wigner_D(l, alpha, beta, gamma)
        
        for mul, ir in irreps:
            wigner[ir.l].append(torch.block_diag(*[temp_wigner[ir.l] for _ in range(mul)]))
    
    for l in irreps_unique_ls:
        wigner[l] = torch.stack(wigner[l], dim=0)
    
    block_diag_wigner = []
    for b in range(rot_matrices.shape[0]):
        block_diag_wigner.append(torch.block_diag(*[wigner[l][b] for l in wigner]))
    block_diag_wigner = torch.stack(block_diag_wigner, dim=0)                
    
    return wigner, block_diag_wigner

def get_wigner_D_fibers_from_rot_matrices(irreps: o3.Irreps, rot_matrices: Tensor):
    '''
    rot_matrices: Tensor with shape (B, 3, 3)

    output: Dict indexed by \ell, with values batched Wigner-D matrices
    '''
    batch_size = rot_matrices.shape[0]
    
    wigner = {}
    alpha, beta, gamma = o3.matrix_to_angles(rot_matrices)
    for mul, ir in irreps:
        wig_l = o3.wigner_D(ir.l, alpha, beta, gamma).unsqueeze(1)
        wigner[ir.l] = torch.stack([torch.block_diag(*torch.tile(wig_l[b], (mul, 1, 1))) for b in range(batch_size)], dim=0)           
    
    return wigner

def get_wigner_D_fibers_from_rot_matrices_v2(irreps: o3.Irreps, rot_matrices: Tensor):
    '''
    rot_matrices: Tensor with shape (B, 3, 3)

    output: Dict indexed by \ell, with values batched Wigner-D matrices
    '''
    batch_size = rot_matrices.shape[0]
    
    wigner = {}
    alpha, beta, gamma = o3.matrix_to_angles(rot_matrices)
    for mul, ir in irreps:
        wig_l = o3.wigner_D(ir.l, alpha, beta, gamma).unsqueeze(1)
        wigner[ir.l] = torch.stack([torch.tile(wig_l[b], (mul, 1, 1)) for b in range(batch_size)], dim=0)           
    
    return wigner


def rotate_signal_batch_block_diag(signals: Tensor, wigner_block_diag: Tensor):
    '''
    Batched matrix-vector multiplication
    '''
    rotated_signals = torch.einsum('bn,bnm->bm', signals, torch.einsum('bij->bji', wigner_block_diag))
    return rotated_signals

def rotate_signal_batch_fibers(signals: Dict[int, Tensor], wigner: Dict):
    batch_size = signals[0].shape[0]
    rotated_signals = {}
    for l in wigner:
        rotated_signals[l] = torch.einsum('bn,bnm->bm', signals[l].reshape(batch_size, -1), torch.einsum('bij->bji', wigner[l])).reshape(batch_size, -1, 2*l+1)
    return rotated_signals

def rotate_signal_batch_fibers_v2(signals: Dict[int, Tensor], wigner: Dict):
    batch_size = signals[0].shape[0]
    rotated_signals = {}
    for l in wigner:
        rotated_signals[l] = torch.einsum('bcn,bcnm->bcm', signals[l], torch.einsum('bcij->bcji', wigner[l]))
    return rotated_signals
