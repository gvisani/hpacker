import json
import numpy as np
import math
import torch

def get_atom_place__torch_batch(plane_norm, chi_angle, a2, a3, bond_length, bond_angle):
    """
    plane_1_normal -- normal vec of plane described by [a1, a2, a3]
    a2 -- coords of the a2 atom
    a3 -- coords of the a3 atom
    chi angle -- dihedral angle between [a1, a2, a3] and [a2, a3, a4]
    bond_angle -- angle of the a2-a3-a4 bond
    bond_length -- length of the a3-a4 bond
    
    return: coords of a4
    """

    chi_angle, bond_angle = to_radians(90 - chi_angle), to_radians(bond_angle)
    
    vec = rotate_chi__torch_batch(plane_norm, chi_angle, a2, a3, bond_length, bond_angle)
    vec, plane_norm_2 = rotate_bond__torch_batch(a2, a3, vec, bond_angle,bond_length)
    
    return vec, plane_norm_2


def rotate_chi__torch_batch(plane_norm, chi_angle, a2, a3, bond_length, bond_angle):
    
    # rotate plane normal around a2-a3 by chi 
    a2a3 = a2 - a3
    a2a3 = a2a3 / torch.norm(a2a3, dim=-1).unsqueeze(-1)
    vec = rotate_about__torch_batch(plane_norm, a2a3, chi_angle)
    
    vec = vec * bond_length.unsqueeze(-1) + a3
    
    return vec


def rotate_bond__torch_batch(a2, a3, vec, angle, length):
    
    plane_norm = get_normal_vector__torch_batch(a2, a3, vec)
    vec = a2 - a3
    vec = vec / torch.norm(vec, dim=-1).unsqueeze(-1)
    vec = rotate_about__torch_batch(vec, plane_norm, -angle)
    vec = vec * length.unsqueeze(-1) + a3

    return vec, plane_norm


def rotate_about__torch_batch(a, b, theta):
    """Rotate vector a about vector b by theta radians."""
    # Thanks user MNKY at http://math.stackexchange.com/a/1432182/81266
    par = (torch.sum(a * b, dim=-1) / torch.sum(b * b, dim=-1)).unsqueeze(-1) * b
    perp = a - par
    w = torch.cross(b, perp, dim=-1)
    w = w / torch.norm(w, dim=-1).unsqueeze(-1)
    return (par +
            perp * torch.cos(theta).unsqueeze(-1) +
            torch.norm(perp, dim=-1).unsqueeze(-1) * w * torch.sin(theta).unsqueeze(-1))


def get_normal_vector__torch_batch(p1, p2, p3):
    '''
    NOTE: the two versions of p2 give normal vectors with different directions.
    for the purpose of reconstruction, this empirically seems not to matter.
    perhaps it's because the only thing that is done to place the atom is a rotation about the norm,
    and thus the direction in which it points doesn't matter?
    That being said, we should be consistent with the way we compute norms in the preprocessing.
    '''
    v1 = p1 - p2
    v2 = p3 - p2
    # v2 = p1 - p3
    x = torch.cross(v1, v2, dim=-1)
    return x / torch.norm(x, dim=-1).unsqueeze(-1)


def torch_dot_batch(v1, v2):
    return torch.sum(v1 * v2, dim=-1)


def get_chi_angle__torch_batch(plane_norm_1, plane_norm_2, a2, a3):

    # normalize just to make sure
    plane_norm_1 = plane_norm_1 / torch.norm(plane_norm_1, dim=-1).unsqueeze(-1)
    plane_norm_2 = plane_norm_2 / torch.norm(plane_norm_2, dim=-1).unsqueeze(-1)
    
    sign_vec = a3 - a2
    sign_with_magnitude = torch_dot_batch(sign_vec, torch.cross(plane_norm_1, plane_norm_2, dim=-1))
    sign = sign_with_magnitude / torch.abs(sign_with_magnitude)
    
    chi_angle = sign * torch.acos(torch_dot_batch(plane_norm_1, plane_norm_2))
    
    return to_degrees(chi_angle)


def split_id(res_id: str):
    return res_id.split("_")


def decode_id(res_id: np.ndarray):
    return [x.decode('utf-8') for x in res_id]


def to_degrees(x):
    return x * 180 / torch.pi


def to_radians(x):
    return x * torch.pi / 180

