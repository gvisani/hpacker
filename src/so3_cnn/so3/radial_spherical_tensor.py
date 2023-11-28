'''
Part of this code was adapted from https://github.com/muhrin/mrs-tutorial
'''


import math
import numpy as np
import torch

from e3nn import o3

# from projections import complex_spherical_harmonics

from typing import List, Union


class RadialSphericalTensor(o3.Irreps):
    r"""representation of a signal in 3-space or in a solid ball

    A 'RadialSphericalTensor' contains the coefficients of a function expansion in 3-space, potentially compactly supported on a solid ball.
    Each coefficient corresponds to a single basis function; each basis function is the product of a radial basis function and a single spherical harmonic.

    Arguments:

    nRadialBases: int>0, number of radial basis functions
    orthonormalRadialBases: a function or functional that accepts a vector of nR>0 radii,
        and returns an array of shape (nR,nRadialBases) containing the values of
        the orthonormal radial basis functions.
    lMax: int, the maximum degree of spherical harmonic functions
    p_val, p_arg: same as in SphericalTensor
    """

    def __new__(cls,
                num_radials, basis,  # Provide an orthonormal radial basis
                lmax, p_val=1, p_arg=-1,
                sph_normalization='component'):  # provide an angular basis, defaults to standard spherical harmonics

        cls.num_radials = num_radials
        cls.radialBases = basis
        cls.lmax = lmax
        cls.p_val = p_val
        cls.p_arg = p_arg
        cls.sph_normalization = sph_normalization
        
        multiplicities = basis.multiplicities
        cls.multiplicities = multiplicities
        
        if cls.radialBases.radius_depends_on_l:
            # Radial basis depends on l (e.g. Zernicke basis)
            radialSelector = []
            for l in range(lmax + 1):
                nm = 2 * l + 1
                for iRadial in range(multiplicities[l]):
                    for m in range(nm):
                        radialSelector.append(iRadial + sum(multiplicities[:l])) #(l * multiplicities[l]))
        else:
            # Radial basis does not depend on l
            radialSelector = []
            for l in range(lmax + 1):
                nm = 2 * l + 1
                for iRadial in range(num_radials):
                    for m in range(nm):
                        radialSelector.append(iRadial)
                    
        cls.radialSelector = torch.tensor(radialSelector)

        parities = {l: (p_val * p_arg ** l) for l in range(lmax + 1)}
        irreps = [(multiplicity, (l, parities[l])) for multiplicity, l in zip(multiplicities, range(lmax + 1))]
        ret = super().__new__(cls, irreps)

        # ls_indices is used to quickly get the coefficients for a specific l
        # just replicate values of l in ls each (2*l + 1) times
        ls_indices = []
        for l in ret.ls:
            ls_indices.append([l for _ in range(2*l + 1)])
        ret.ls_indices = torch.Tensor([l for ls_sublist in ls_indices for l in ls_sublist])

        return ret

    def _evaluateAngularBasis(self, vectors, radii=None):
        r"""Evaluate angular basis functions (spherical harmonics) at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """

        assert self[0][1].p == 1, "the spherical harmonics are only evaluable when p_val is 1, since the l=0 must have parity 1."  # pylint: disable=no-member

        if self.radialBases.complex_sph: # automatically normalized (for now? if otherwise is possible?)
            # no extra normalization for now, copying mike's code
            # Note: converting from tensors to numpy arrays and back. converting back for compatibility, but this is not a differentiable operation
            raise NotImplementedError()
            # print('evaluating complex spherical harmonics')
            # if radii is not None:
            #     angularCoeffs = torch.tensor(complex_spherical_harmonics(self, vectors.view(-1, 3).numpy() / radii.view(-1, 1).expand(-1, 3).numpy(), normalization='norm'))
            # else:
            #     angularCoeffs = torch.tensor(complex_spherical_harmonics(self, vectors.view(-1, 3).numpy(), normalization='norm'))
        else:
            if radii is not None:
                angularCoeffs = o3.spherical_harmonics(self, vectors.view(-1, 3) / radii.view(-1, 1).expand(-1, 3), normalize=False, normalization=self.sph_normalization)
            else:
                angularCoeffs = o3.spherical_harmonics(self, vectors.view(-1, 3), normalize=True, normalization=self.sph_normalization)

        finalShape = tuple(list(vectors.shape[:-1]) + [self.dim])
        basisValuesNotFlat = angularCoeffs.view(finalShape)

        return basisValuesNotFlat

    def _evaluateRadialBasis(self, vectors, radii=None):
        r"""Evaluate radial basis functions at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """

        if radii is not None:
            basesFlat = self.radialBases(radii.view(-1))
        else:
            basesFlat = self.radialBases(torch.norm(vectors, dim=-1).view(-1))
        
        basesFlat = basesFlat[:, self.radialSelector]
        finalShape = tuple(list(vectors.shape[:-1]) + [self.dim])
        basisValuesNotFlat = basesFlat.view(finalShape)

        return basisValuesNotFlat

    def _evaluateJointBasis(self, vectors, radii=None):
        r"""Evaluate joint (radial x angular) basis functions at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """

        radii = torch.norm(vectors, dim=-1)
        angularBasis = self._evaluateAngularBasis(vectors, radii)
        radialBasis = self._evaluateRadialBasis(vectors, radii)
        return (angularBasis * radialBasis)

    def _evaluateBasisOnGrid(self, cutoffRadius, res, cropBases, spherical_grid=True, cutoffRadiusInner=None, basis=None):
        if not spherical_grid:
            samplePointsLinear = torch.linspace(start=-cutoffRadius, end=cutoffRadius, steps=res)
            
            disjointPoints = (samplePointsLinear, samplePointsLinear, samplePointsLinear)
            triplePoints = torch.cartesian_prod(samplePointsLinear, samplePointsLinear, samplePointsLinear)
            samplePoints = triplePoints.view(res, res, res, -1)
        else:
            from utils import spherical_to_cartesian__pytorch, cartesian_to_spherical__numpy, spherical_to_cartesian__numpy, cartesian_to_spherical__numpy
            r_res = int(res / 1.0)
            t_res = res
            p_res = int(res * 1.0)
            r_samplePoints = torch.linspace(1e-6, cutoffRadius, steps=r_res)
            t_samplePoints = torch.linspace(0, np.pi, steps=t_res)
            p_samplePoints = torch.linspace(0, 2*np.pi, steps=p_res)
            samplePointsSphere = torch.cartesian_prod(r_samplePoints, t_samplePoints, p_samplePoints)
            
            disjointPoints = (r_samplePoints, t_samplePoints, p_samplePoints)
            triplePoints = torch.tensor(spherical_to_cartesian__numpy(samplePointsSphere.numpy()))
            samplePoints = triplePoints.view(r_res, t_res, p_res, -1)
        
        radii = torch.norm(samplePoints, dim=-1)

        if basis is not None:
            samples = basis(samplePoints, radii)
        else:
            samples = self._evaluateJointBasis(samplePoints, radii)

        if cropBases:
            samples[radii > cutoffRadius, :] = 0
            if cutoffRadiusInner is not None: samples[radii < cutoffRadiusInner, :] = 0

        return (disjointPoints, triplePoints, samples)

    def _getBasisOnGrid(self, cutoffRadius, res, cropBases, spherical_grid=True, cutoffRadiusInner=None):
        return self._evaluateBasisOnGrid(cutoffRadius, res, cropBases, spherical_grid=spherical_grid, cutoffRadiusInner=cutoffRadiusInner)

    def forward_projection_pointwise(self, coords):
        '''
        Computes spherical tensor for individual points (at coords). Does not sum them up.
        '''
        return self._evaluateJointBasis(coords) # --> Z = R * Y

    def with_peaks_at(self, coords, values=None, normalization=None):

        r"""Create a spherical tensor with peaks (forward fourier transform)
        The peaks are located in :math:`\vec r_i` and have amplitude :math:`\|\vec r_i \|`
        Parameters
        ----------
        vectors : `torch.Tensor` :math:`\vec r_i` tensor of shape ``(N, 3)``
        values : `torch.Tensor`, optional value on the peak, tensor of shape ``(N)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(self.dim,)``
        """

        bases = self._evaluateJointBasis(coords) # --> Z = R * Y

        if values is None: values = torch.ones(coords.shape[:-1], dtype=bases.dtype, device=coords.device)
        else: values = values.to(bases.dtype)
        
        if self.radialBases.complex_sph:
            raise NotImplementedError()
            # bases = torch.conj(bases)

        if normalization == 'sqrt_power':
            # This makes the total power of each point cloud equal to 1
            basesSelfDotsInv = 1.0 / torch.sqrt(torch.einsum('...a,...a,a->...', bases, bases, self.ls_indices.type(torch.double)))
            coeffs = torch.einsum('...b,...,...->b', bases, basesSelfDotsInv, values)
#             coeffs = torch.einsum('...b,...->b', bases, basesSelfDotsInv)
        elif normalization == 'square':
            # this (square normalization) is what the e3nn tutorial does. It makes the inversion contained between 0 and 1 (actually some value higher than 1), which is actually really nice, but does it make reconstructions worse?
            basesSelfDotsInv = 1.0 / torch.einsum('...a,...a->...', bases, bases)
            coeffs = torch.einsum('...b,...,...->b', bases, basesSelfDotsInv, values)
        else:
            # this is what Mike does --> no normalization
            coeffs = torch.einsum('...b,...->b', bases, values)

        return coeffs.float()

    def _evaluateSignal(self, signals, basis):
        r"""Expand signal into a weighted sum of bases
        Parameters
        ----------
        signals : `torch.Tensor` of shape ``({... signals}, self.dim)``
        basis : `torch.Tensor` of shape ``({... points}, self.dim)``
        Returns
        -------
        `torch.Tensor` of shape ``({... signals}, {... points})``
        """
        shapeFinal = tuple(list(signals.shape[:-1]) + list(basis.shape[:-1]))

        signalsFlat = signals.view(-1, self.dim)
        basisFlat = basis.view(-1, self.dim)

        retFlat = torch.einsum('sd,pd->sp', signalsFlat, basisFlat)
        ret = retFlat.view(shapeFinal)

        return ret

    def signal_xyz(self, signals, vectors):
        basisValues = self._evaluateJointBasis(vectors)
        return self._evaluateSignal(signals, basisValues)

    def signal_on_grid(self, signals, rcut, res, cropBases=True, spherical_grid=True, cutoffRadiusInner=None):
        disjointPoints, samplePoints, samplesBasis = self._getBasisOnGrid(rcut, res, cropBases, spherical_grid=spherical_grid, cutoffRadiusInner=cutoffRadiusInner)
        return disjointPoints, samplePoints, self._evaluateSignal(signals, samplesBasis)
    
    def get_vectors_at_l(self, vectors, l):
        return vectors[self.ls_indices == l]


class MultiChannelRadialSphericalTensor(o3.Irreps):
    def __new__(cls, 
                rst: RadialSphericalTensor, 
                num_channels: int):
        cls.rst = rst
        cls.num_channels = num_channels

        multiplicities = [rst.multiplicities[l] * num_channels for l in range(rst.lmax + 1)]
        
        parities = {l: (rst.p_val * rst.p_arg ** l) for l in range(rst.lmax + 1)}
        irreps = [(multiplicity, (l, parities[l])) for multiplicity, l in zip(multiplicities, range(rst.lmax + 1))]
        ret = super().__new__(cls, irreps)

        # ls_indices is used to quickly get the coefficients for a specific l
        # just replicate l in ls each (2*l + 1) times
        # Could also replicate l in rst.ls_indices by the number of channels, they would be equivalent
        ls_indices = []
        for l in ret.ls:
            ls_indices.append([l for _ in range(2*l + 1)])
        ret.ls_indices = torch.Tensor([l for ls_sublist in ls_indices for l in ls_sublist])

        # idea for combine: create the index_permuter in the same way as 'combine()' currently merges tensors, but do it with
        # indices, and just do it once
        indices = torch.tensor(np.vstack([np.arange(rst.dim)+(i*rst.dim) for i in range(num_channels)]))
        index_permuter = []
        lower_bound = 0
        for mul, ir in ret.rst:
            num_values = mul*(2*ir.l + 1)
            index_permuter.append(indices[:, lower_bound : lower_bound + num_values].reshape(-1, 1).squeeze())
            lower_bound += num_values
        ret.index_permuter = torch.cat(index_permuter, dim=-1)

        return ret
    
    def combine_inplace(self, tensors: torch.Tensor):
        if len(tensors.shape) == 1:
            tensors = torch.unsqueeze(tensors, 0)
        
        return tensors[:, self.index_permuter].squeeze()
        
    
    def combine(self, tensors: torch.Tensor, normalization=None):
        '''
        Combine tensors by concatenating all values indexed by l,m (num_radials [or n] such values) for all channels. e.g. 1C 2C 3C ... 1N 2N 3N
        Assumes tensors are stacked along the 0th dimension (vertically)
        tensors: dim = [batch_size, num_channels, rst.dim] or [num_channels, rst.dim]
        '''
        if len(tensors.shape) == 2:
            tensors = torch.unsqueeze(tensors, 0)

        assert tensors.shape[2] == self.rst.dim
        batch_size = tensors.shape[0]

        new_tensors = []
        lower_bound = 0
        for mul, ir in self.rst:
            num_values = mul*(2*ir.l + 1)
            new_tensors.append(tensors[:, :, lower_bound : lower_bound + num_values].reshape(batch_size, -1, 1).squeeze())
            lower_bound += num_values
        
        combined_tensors = torch.cat(new_tensors, dim=-1)
        
        if normalization == 'sqrt_power':
            basesSelfDotsInv = 1.0 / torch.einsum('...a,...a,a->...', combined_tensors, combined_tensors, self.ls_indices.type(torch.float))
            combined_tensors = torch.einsum('...b,...->b', combined_tensors, basesSelfDotsInv)
        elif normalization == 'square':
            basesSelfDotsInv = 1.0 / torch.einsum('...a,...a->...', combined_tensors, combined_tensors)
            combined_tensors = torch.einsum('...b,...->b', combined_tensors, basesSelfDotsInv)
        
        return combined_tensors
    
    def separate(self, tensors: torch.Tensor):
        '''
        Does the reverse of self.combine()
        tensors: dim = [batch_size, rst.dim * num_channels] or [rst.dim * num_channels]
        '''
        if len(tensors.shape) == 1:
            tensors = torch.unsqueeze(tensors, 0)
        
        batch_size = tensors.shape[0]

        new_tensors = []
        lower_bound = 0
        for mul, ir in self.rst:
            num_values = self.num_channels * mul*(2*ir.l+1)
            new_tensors.append(tensors[:, lower_bound : lower_bound + num_values].reshape(batch_size, self.num_channels, -1))
            lower_bound += num_values
        
        separated_tensors = torch.cat(new_tensors, dim=-1)

        if len(separated_tensors.shape) == 2:
            separated_tensors = separated_tensors.unsqueeze(0)
        
        return separated_tensors

    def get_vectors_at_l(self, vectors, l):
        return vectors[self.ls_indices == l]

