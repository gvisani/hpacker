

import numpy as np
import torch

from e3nn import o3

from src.so3_cnn.so3 import ZernickeRadialFunctions

from typing import *


class SphericalFourierEncoding(torch.nn.Module):

    def __init__(self, lmax,
                       elements,
                       biophysicals = [],
                       sph_normalization='component',
                       convert_output_to_dict=False,
                       radial_bases_obj=ZernickeRadialFunctions,
                       radial_bases_kwargs={},
                       rst_normalization=None):
        super().__init__()
        
        self.lmax = lmax
        self.elements = elements
        self.biophysicals = biophysicals
        self.num_channels = len(self.elements) + len(self.biophysicals)
        self.sph_normalization = sph_normalization
        self.convert_output_to_dict = convert_output_to_dict
        self.rst_normalization = rst_normalization
        assert self.rst_normalization in [None, 'square']
        
        self.radialBases = radial_bases_obj(**radial_bases_kwargs)
        
        radial_selector = []
        if self.radialBases.radius_depends_on_l:
            # Radial basis depends on l (e.g. Zernike basis)
            multiplicities = self.radialBases.multiplicities
            for l in range(lmax + 1):
                nm = 2 * l + 1
                for iRadial in range(multiplicities[l]):
                    for m in range(nm):
                        radial_selector.append(iRadial + sum(multiplicities[:l]))
        else:
            multiplicities = [self.radialBases.number_of_basis] * (lmax + 1)
            for l in range(lmax + 1):
                nm = 2 * l + 1
                for iRadial in range(multiplicities[l]):
                    for m in range(nm):
                        radial_selector.append(iRadial)
        self.radial_selector = torch.tensor(radial_selector)
        
        self.single_channel_irreps = o3.Irreps([(multiplicity, (l, 1)) for multiplicity, l in zip(multiplicities, range(lmax + 1))])
        self.irreps = (self.num_channels*self.single_channel_irreps).sort().irreps.simplify()
        self.ls_indices = torch.cat([torch.tensor([l]).repeat(2*l+1) for l in self.irreps.ls])
        
        # index selector for combining multiple channels
        indices = torch.tensor(np.vstack([np.arange(self.single_channel_irreps.dim)+(i*self.single_channel_irreps.dim) for i in range(self.num_channels)]))
        channel_combining_index_permuter = []
        lower_bound = 0
        for mul, ir in self.single_channel_irreps:
            num_values = mul*(2*ir.l + 1)
            channel_combining_index_permuter.append(indices[:, lower_bound : lower_bound + num_values].reshape(-1, 1).squeeze())
            lower_bound += num_values
        self.channel_combining_index_permuter = torch.cat(channel_combining_index_permuter, dim=-1)        


    def forward(self, x_coords_B_N3: List[Union[np.ndarray, torch.Tensor]], x_elements_B_N: List[List], x_biophysical_B_N: Optional[List[Dict[str, Union[np.ndarray, torch.Tensor]]]] = None):
        '''
        Currently will compute the forward FT for all atoms, even if they are not in the desired elements.
        Thus it may do more computation than necessary (beware of Hydrogens!)
        
        Assume that only coords for desired elements are present
        
        This performs the forward transform dynamically and in neighborhood batches
        
        NB: need to use a collate_fn for a dataloader

        With Bessel, it takes 1/15 seconds to process a batch of 100 neighborhoods with the 4 main elements
        With Zernike, it takes 1/4 seconds (~4 times slower)

        Batching makes things faster. With batch_size = 100, ~3.5 faster for Bessel, ~2 faster for Zernike

        "batch size" is the number of neighborhoods

        if batching across groups-of-neighborhoods/proteins, then 

        '''

        if not isinstance(x_coords_B_N3, list) and not isinstance(x_coords_B_N3, tuple):
            x_coords_B_N3 = [x_coords_B_N3]
            x_elements_B_N = [x_elements_B_N]
            if x_biophysical_B_N is not None:
                x_biophysical_B_N = [x_biophysical_B_N]
        
        batch_size = len(x_coords_B_N3)
        assert batch_size == len(x_elements_B_N)
        if x_biophysical_B_N is not None:
            assert batch_size == len(x_biophysical_B_N)

        if x_biophysical_B_N is None:
            x_biophysical_B_N = []
        
        # construct indicators of which atoms belong to which neighborhood
        nb_indices = torch.cat([torch.full((x_coords_B_N3[b].shape[0],), b) for b in range(batch_size)])
        
        # concatenate all the stuff together
        if isinstance(x_coords_B_N3[0], np.ndarray):
            x_coords = torch.tensor(np.concatenate(x_coords_B_N3, axis=0), dtype=torch.float32)
        else:
            x_coords = torch.cat(x_coords_B_N3, dim=0)
        
        # forward FT
        x_radii = torch.norm(x_coords, dim=-1)
        angular_coeffs = o3.spherical_harmonics(self.single_channel_irreps, x_coords.view(-1, 3) / x_radii.view(-1, 1).expand(-1, 3), normalize=False, normalization=self.sph_normalization)
        radial_coeffs = self.radialBases(x_radii)[:, self.radial_selector]
        pointwise_coeffs = (angular_coeffs * radial_coeffs)

        if self.rst_normalization == 'square':
            basesSelfDotsInv = 1.0 / torch.einsum('...f,...f->...', pointwise_coeffs, pointwise_coeffs)
        
        batched_disentangled_coeffs = []
        for b in range(batch_size):
            cuff_coeffs = pointwise_coeffs[nb_indices == b, :]
            curr_basesSelfDotsInv = basesSelfDotsInv[nb_indices == b] if self.rst_normalization == 'square' else None
            
            # merge coeffs from multiple element channels, and biophysical channels
            disentangled_coeffs = []
            for element in self.elements:
                if self.rst_normalization == 'square':
                    disentangled_coeffs.append(torch.einsum('...f,...->f', cuff_coeffs[np.array(x_elements_B_N[b]) == element], curr_basesSelfDotsInv[np.array(x_elements_B_N[b]) == element]))
                else:
                    disentangled_coeffs.append(torch.einsum('...f->f', cuff_coeffs[np.array(x_elements_B_N[b]) == element]))

            for biophysical in self.biophysicals: # biophysicals, for each requested element channel. note that these are pre-computed, and information leakage may come in from other elements
                if isinstance(x_biophysical_B_N[b][biophysical], np.ndarray):
                    x_biophysical_B_N[b][biophysical] = torch.tensor(x_biophysical_B_N[b][biophysical], dtype=torch.float32)
                
                if self.rst_normalization == 'square':
                    disentangled_coeffs.append(torch.einsum('...f,...,...->f', cuff_coeffs, curr_basesSelfDotsInv, x_biophysical_B_N[b][biophysical]))
                else:
                    disentangled_coeffs.append(torch.einsum('...f,...->f', cuff_coeffs, x_biophysical_B_N[b][biophysical]))
        
            # "zip" different channels together        
            batched_disentangled_coeffs.append(torch.cat(disentangled_coeffs, dim=0)[self.channel_combining_index_permuter])
        
        coeffs = torch.stack(batched_disentangled_coeffs, dim=0)
        
        dict_coeffs = {}
        if self.convert_output_to_dict:
            for _, ir in self.irreps:
                dict_coeffs[ir.l] = coeffs[:, self.ls_indices == ir.l].view(batch_size, -1, 2*ir.l+1)
            coeffs = dict_coeffs
        
        return coeffs
    