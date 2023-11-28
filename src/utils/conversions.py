

import numpy as np
import scipy
import torch

def spherical_to_cartesian__pytorch(x):
    '''
    standard notation for spherical angles (t == theta == elevation ; p == phi == azimuthal)
    '''
    # get cartesian coordinates
    r = x[:,0]
    t = x[:,1]
    p = x[:,2]

    # get spherical coords from cartesian
    x_ = torch.sin(t)*torch.cos(p)*r
    y_ = torch.sin(t)*torch.sin(p)*r
    z_ = torch.cos(t)*r

    # return x, y, z
    return torch.cat([x_.view(-1, 1), y_.view(-1, 1), z_.view(-1, 1)], dim=-1)

def cartesian_to_spherical__numpy(xyz):
    '''
    standard notation for spherical angles (t == theta == elevation ; p == phi == azimuthal)
    '''
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    t = np.arccos(z/r)
    p = np.arctan2(y,x)
    return np.hstack([r.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1)])

def spherical_to_cartesian__numpy(x):
    '''
    standard notation for spherical angles (t == theta == elevation ; p == phi == azimuthal)
    '''
    # get cartesian coordinates
    r = x[:,0]
    t = x[:,1]
    p = x[:,2]

    # get spherical coords from cartesian
    x_ = np.sin(t)*np.cos(p)*r
    y_ = np.sin(t)*np.sin(p)*r
    z_ = np.cos(t)*r

    # return x, y, z
    return np.hstack([x_.reshape(-1, 1), y_.reshape(-1, 1), z_.reshape(-1, 1)])

def cartesian_to_spherical__pytorch(xyz):
    '''
    standard notation for spherical angles (t == theta == elevation ; p == phi == azimuthal)
    '''
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = torch.sqrt(x*x + y*y + z*z)
    t = torch.acos(z/r)
    p = torch.atan2(y,x)
    return torch.cat([r.view(-1, 1), t.view(-1, 1), p.view(-1, 1)], dim=-1)
