
import os
import json
import numpy as np
import math
import torch

try:
    from .utils__torch import *
except ImportError:
    from utils__torch import *

from typing import *

class Reconstructor():
    
    def __init__(self,
                 path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reconstruction_params__vectorized.json')):
        """
        Takes path to params.json
        """
        
        with open(path, 'r') as f:
            reconstruction = json.loads(f.read())
        
        # =====Beta carbon virtualization==========
        self.CA_CB_dict = torch.tensor(reconstruction['CA_CB_dict']) # averge length of CA-CB bond length, by AA type
        self.N_C_CA_dict = torch.tensor(reconstruction['N_C_CA_dict']) # average N-C-CA bond angle
        self.N_C_CA_CB_dict = torch.tensor(reconstruction['N_C_CA_CB_dict']) # avereg N-C-CA-CB dihedral angle
        
        # =====Chi Angle reconstruction===========
        self.ideal_bond_lengths = torch.tensor(reconstruction['ideal_bond_lengths']) # average bonds lengths by AA + chi
        self.ideal_bond_angles = torch.tensor(reconstruction['ideal_bond_angles']) # averge bond angle (NOT dihedral) by AA + chi
    
        self.aa_to_idx = {aa: i for i, aa in enumerate(reconstruction['AAs'])}
    
    def reconstruct_from_normal_vectors(self,
                                        atoms: List[torch.Tensor],
                                        AA: List[str],
                                        normal_vectors: torch.Tensor,
                                        true_CB: Optional[torch.Tensor] = None
                                        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        '''
        For backbone atoms, order is C, O, N, CA
        The order is important! In particular N-CA as they are used to compute the first and second chi angles
        '''
        
        # TODO: make this use the standard AA idxs that will be passed during training
        AA = torch.tensor([self.aa_to_idx[aa] for aa in AA])
    
        if true_CB is not None:
            # if we have the true CB location, use it instead of the virtualized one
            atoms.append(true_CB)
        else:
            #=====virtualize beta carbon============
            CB_norm = get_normal_vector__torch_batch(atoms[2], atoms[0], atoms[3])
            
            atoms.append(get_atom_place__torch_batch(
                CB_norm, self.N_C_CA_CB_dict[AA], 
                atoms[0], atoms[3], 
                self.CA_CB_dict[AA],
                self.N_C_CA_dict[AA]
            )[0])

        ordered_chi_angles = []
        
        #====place side chain atoms===========
        for chi_num in range(4):
            
            p1_norm = get_normal_vector__torch_batch(atoms[-3], atoms[-2], atoms[-1])
            p2_norm = normal_vectors[:, chi_num, :]
            chi_angle = get_chi_angle__torch_batch(p1_norm, p2_norm, atoms[-2], atoms[-1])

            bond_length, bond_angle = self.ideal_bond_lengths[AA, chi_num], self.ideal_bond_angles[AA, chi_num]

            predicted_place, _ = get_atom_place__torch_batch(p1_norm, chi_angle, atoms[-2], atoms[-1], bond_length, bond_angle)

            # Use predicted place downstream
            ordered_chi_angles.append(chi_angle)
            atoms.append(predicted_place)
        
        ordered_placed_atoms = atoms[4:]

        return ordered_placed_atoms, ordered_chi_angles

    
    def reconstruct_from_chi_angles(self,
                                    atoms: List[torch.Tensor],
                                    AA: List[str],
                                    chi_angles: torch.Tensor,
                                    true_CB: Optional[torch.Tensor] = None,
                                    )-> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        '''
        For backbone atoms, order is C, O, N, CA
        The order is important! In particular N-CA as they are used to compute the first and second chi angles
        
        --- Input ---
        atoms: list of length 4. each element is a torch tensor containing a batch (batch size = B) of atom coordinates for the atoms [C, O, N, CA], in this order
        AA: list of single-char amino-acid identifiers, of length B
        chi_angles: Tensor of shape (B x 4) containing desired chi angles, with NaN values for invalid angles
        
        --- Output ---
        ordered_placed_atoms: list of atom coordinates that have been placed, in the order of placement
        ordered_norms: list of plane norms of the sidechains, 
        '''
        
        # TODO: make this use the standard AA idxs that will be passed during training
        AA = torch.tensor([self.aa_to_idx[aa] for aa in AA])

    
        if true_CB is not None:
            # if we have the true CB location, use it instead of the virtualized one
            atoms.append(true_CB)
        else:
            #=====virtualize beta carbon============
            CB_norm = get_normal_vector__torch_batch(atoms[2], atoms[0], atoms[3])
            
            atoms.append(get_atom_place__torch_batch(
                CB_norm, self.N_C_CA_CB_dict[AA], 
                atoms[0], atoms[3], 
                self.CA_CB_dict[AA],
                self.N_C_CA_dict[AA]
            )[0])

        ordered_norms = []
        
        #====place side chain atoms===========
        for chi_num in range(4):
            
            if len(ordered_norms) == 0 and chi_num == 0:
                p1_norm = get_normal_vector__torch_batch(atoms[-3], atoms[-2], atoms[-1])
                # comopute and save the CB norm as well
                ordered_norms.append(p1_norm)
            else:
                p1_norm = ordered_norms[-1]

            chi_angle = chi_angles[:, chi_num]
            
            bond_length, bond_angle = self.ideal_bond_lengths[AA, chi_num], self.ideal_bond_angles[AA, chi_num]
            
            predicted_place, fake_p2_norm = get_atom_place__torch_batch(p1_norm, chi_angle, atoms[-2], atoms[-1], bond_length, bond_angle)

            # Use predicted place downstream
            atoms.append(predicted_place)
            # somehow p2_norm is in the opposite direction of what I thought it would be, but it reconstructs fine
            ordered_norms.append(get_normal_vector__torch_batch(atoms[-3], atoms[-2], atoms[-1]))
           
        ordered_placed_atoms = atoms[4:]

        return ordered_placed_atoms, ordered_norms
        