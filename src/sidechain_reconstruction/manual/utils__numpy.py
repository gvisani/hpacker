import json
import numpy as np
import math

# ==========vector arithmetic functions========================
def get_atom_place(plane_norm, chi_angle, a2, a3, bond_length, bond_angle):
    """
    plane_1_normal -- normal vec of plane described by [a1, a2, a3]
    a2 -- coords of the a2 atom
    a3 -- coords of the a3 atom
    chi angle -- dihedral angle between [a1, a2, a3] and [a2, a3, a4]
    bond_angle -- angle of the a2-a3-a4 bond
    bond_length -- length of the a3-a4 bond
    
    return: coords of a4
    """

    chi_angle, bond_angle = np.deg2rad(90 - chi_angle), np.deg2rad(bond_angle)
    
    vec = rotate_chi(plane_norm, chi_angle, a2, a3, bond_length, bond_angle)
    vec, plane_norm_2 = rotate_bond(a2,a3, vec,bond_angle,bond_length)
    
    return vec, plane_norm_2

def rotate_chi(plane_norm, chi_angle, a2, a3, bond_length, bond_angle):
    
    # rotate plane normal around a2-a3 by chi 
    a2a3 = a2 - a3
    a2a3 = a2a3/np.linalg.norm(a2a3)
    vec = rotate_about(plane_norm, a2a3, chi_angle)
    
    vec = vec * bond_length + a3
    
    return vec

def rotate_bond(a2, a3, vec, angle, length):
    
    plane_norm = get_normal_vector(a2, a3, vec)
    vec = a2 - a3
    vec = vec/np.linalg.norm(vec)
    vec = rotate_about(vec, plane_norm, -angle)
    vec = vec * length + a3

    return vec, plane_norm


def rmsd(p1, p2):
    _sum = 0
    for q, r in zip(p1, p2):
        _sum += math.pow(q - r, 2)
    _sum /= len(p1)
    return math.sqrt(_sum)

# stolen from https://gist.github.com/fasiha/6c331b158d4c40509bd180c5e64f7924#file-rotatevectors-py-L35-L42

def makeUnit(x):
    """Normalize entire input to norm 1. Not what you want for 2D arrays!"""
    return x / np.linalg.norm(x)


def xParV(x, v):
    """Project x onto v. Result will be parallel to v."""
    # (x' * v / norm(v)) * v / norm(v)
    # = (x' * v) * v / norm(v)^2
    # = (x' * v) * v / (v' * v)
    return np.dot(x, v) / np.dot(v, v) * v


def xPerpV(x, v):
    """Component of x orthogonal to v. Result is perpendicular to v."""
    return x - xParV(x, v)


def xProjectV(x, v):
    """Project x onto v, returning parallel and perpendicular components
    >> d = xProject(x, v)
    >> np.allclose(d['par'] + d['perp'], x)
    True
    """
    par = xParV(x, v)
    perp = x - par
    return {'par': par, 'perp': perp}


def rotate_about(a, b, theta):
    """Rotate vector a about vector b by theta radians."""
    # Thanks user MNKY at http://math.stackexchange.com/a/1432182/81266
    proj = xProjectV(a, b)
    w = np.cross(b, proj['perp'])
    return (proj['par'] +
            proj['perp'] * np.cos(theta) +
            np.linalg.norm(proj['perp']) * makeUnit(w) * np.sin(theta))


#=========other utils=================
def split_id(res_id: str):
    return res_id.split("_")

def decode_id(res_id: np.ndarray):
    return [x.decode('utf-8') for x in res_id]

def get_normal_vector(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    # v2 = p1 - p3
    x = np.cross(v1, v2)
    return x / np.linalg.norm(x)

def get_chi_angle(plane_norm_1, plane_norm_2, a2, a3):
    
    sign_vec = a3 - a2
    sign_with_magnitude = np.dot(sign_vec, np.cross(plane_norm_1, plane_norm_2))
    sign = sign_with_magnitude / np.abs(sign_with_magnitude)
    
    dot = np.dot(plane_norm_1, plane_norm_2) / (np.linalg.norm(plane_norm_1) * np.linalg.norm(plane_norm_2))
    chi_angle = sign * np.arccos(dot)
    
    return np.degrees(chi_angle)
