'''

Part of this code was adapted from https://github.com/muhrin/mrs-tutorial

'''


import math
import torch
import numpy as np
import scipy as sp
import scipy.special

class ZernickeRadialFunctions:
    
    def __init__(self, rcut=10.0, number_of_basis=20, lmax=6, complex_sph=False, record_zeros=False):
        self.rcut = rcut
        self.number_of_basis = number_of_basis
        self.lmax = lmax
        self.complex_sph = complex_sph
        self.record_zeros = record_zeros
        self.radius_depends_on_l = True
        if record_zeros:
            self.multiplicities = [number_of_basis] * (lmax + 1)
        else:
            rv = torch.arange(number_of_basis)
            self.multiplicities = [rv[torch.logical_and(rv >= l, (rv - l) % 2 == 0)].shape[0] for l in range(lmax + 1)]

        # precompute the n-l combinations, in the order in which they should appear in the output
        ns = []
        ls = []
        nonzero_idxs = []
        i = 0
        for l in range(self.lmax+1):
            for n in range(self.number_of_basis):
                if not ((n-l) % 2 == 1 or (n < l)):
                    ls.append(l)
                    ns.append(n)
                    nonzero_idxs.append(i) 
                i += 1
        
        self.ls = np.array(ls)
        self.ns = np.array(ns)
        self.nonzero_idxs = np.array(nonzero_idxs)

    
    def __call__(self, r):
        try:
            r = r.numpy()
        except:
            r = r.detach().numpy()
        
        # cap radiuses at self.rcut
        r[r > self.rcut] = self.rcut

        r = np.tile(r, (self.nonzero_idxs.shape[0], 1))
        l = np.tile(self.ls, (r.shape[1], 1)).T
        n = np.tile(self.ns, (r.shape[1], 1)).T
        

        # dimension of the Zernike polynomial
        D = 3.
        # constituent terms in the polynomial
        A = np.power(-1,(n - l)/2.) 
        B = np.sqrt(2.*n + D)
        C = sp.special.binom((n + l + D) // 2 - 1,
                                (n - l) // 2)

        E = sp.special.hyp2f1(-(n - l)/2.,
                                (n + l + D)/2.,
                                l + D/2.,
                                r/self.rcut * r/self.rcut)
        F = np.power(r/self.rcut, l)
        
        temp_coeffs = A * B * C * E * F
        
        if self.record_zeros:
            coeffs = np.zeros(self.lmax*self.number_of_basis, r.shape[0])
            coeffs[self.nonzero_idxs, :] = temp_coeffs
        else:
            coeffs = temp_coeffs
        
        coeffs = coeffs.T
        
        return torch.tensor(coeffs).type(torch.float)
    

class ZernickeRadialFunctions_OLD:
    
    def __init__(self, rcut, number_of_basis, lmax, complex_sph=False, record_zeros=False):
        self.rcut = rcut
        self.number_of_basis = number_of_basis
        self.lmax = lmax
        self.complex_sph = complex_sph
        self.record_zeros = record_zeros
        self.radius_depends_on_l = True
        if record_zeros:
            self.multiplicities = [number_of_basis] * (lmax + 1)
        else:
            rv = torch.arange(number_of_basis)
            self.multiplicities = [rv[torch.logical_and(rv >= l, (rv - l) % 2 == 0)].shape[0] for l in range(lmax + 1)]

    
    def __call__(self, r):
        try:
            r = r.numpy()
        except:
            r = r.detach().numpy()
        
        # cap radiuses at self.rcut
        r[r > self.rcut] = self.rcut
        
        return_val = []
        for l in range(self.lmax+1):
            for n in range(self.number_of_basis):

                if (n-l) % 2 == 1 or (n < l):
                    if self.record_zeros:
                        return_val.append(np.full(r.shape[0], 0.0))
                    continue

                # dimension of the Zernike polynomial
                D = 3.
                # constituent terms in the polynomial
                A = np.power(-1,(n-l)/2.) 
                B = np.sqrt(2.*n + D)
                C = sp.special.binom(int((n+l+D)/2. - 1),
                                     int((n-l)/2.))

                E = sp.special.hyp2f1(-(n-l)/2.,
                                       (n+l+D)/2.,
                                       l+D/2.,
                                       np.array(r)/self.rcut*np.array(r)/self.rcut)
                F = np.power(np.array(r)/self.rcut,l)
                
                coeff = A*B*C*E*F
                
                return_val.append(coeff)
        
        return torch.tensor(np.transpose(np.vstack(return_val))).type(torch.float)
