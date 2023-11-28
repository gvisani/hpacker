
import os
import json
import numpy as np
import math

try:
    from .utils__numpy import *
except ImportError:
    from utils__numpy import *

from typing import *

class Reconstructor():
    """
    Usage:
    >>> r = Reconstructor("/gscratch/scrubbed/wgalvin/python/reconstruction.json")
    >>> angles = [90, 75, -40, 30]
    >>> nb = ...
    >>> r.reconstruct(nb, angles)
    """
    def __init__(self,
                 path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reconstruction_params.json'),
                 vec_db=None):
        """
        Takes path to reconstruction.json

        If vec_db is not none, stores a map of {id -> normal vectors}
        """
        
        with open(path, 'r') as f:
            reconstruction = json.loads(f.read())
            
        self.vec_db = vec_db
        
        # =====Beta carbon virtualization==========
        self.CA_CB_dict = reconstruction['CA_CB_dict'] # averge length of CA-CB bond length, by AA type
        self.N_C_CA_dict = reconstruction['N_C_CA_dict'] # average N-C-CA bond angle
        self.N_C_CA_CB_dict = reconstruction['N_C_CA_CB_dict'] # avereg N-C-CA-CB dihedral angle
        
        # =====Chi Angle reconstruction===========
        self.ideal_bond_lengths = reconstruction['ideal_bond_lengths'] # average bonds lengths by AA + chi
        self.ideal_bond_angles = reconstruction['ideal_bond_angles'] # averge bond angle (NOT dihedral) by AA + chi
        
        #======misc lookup tables=================
        self.aa_symbols = reconstruction['aa_symbols']
        self.chi_atoms = reconstruction['chi_atoms']
        
    
    def reconstruct_from_normal_vectors(self, neighborhood, vecs) -> Tuple[dict, list]:
        
        res_id = decode_id(neighborhood['res_id'])
        AA = self.aa_symbols[res_id[0]]
        
        if AA in ['ALA', 'GLY']: return None # no Chi angles
    
        atoms = {'CA': np.zeros(3, dtype=float)}
        for atom, _res_id, coords in zip(neighborhood['atom_names'],neighborhood['res_ids'], neighborhood['coords']):
            if (decode_id(_res_id) != res_id): continue
            atom = atom.decode('utf-8').strip()
            atoms[atom] = coords
                
        #=====virtualize beta carbon============
        if 'N' not in atoms or 'C' not in atoms or 'CA' not in atoms: # check that all needed atoms are found in neighborhood
            return None
    
        CB_norm = get_normal_vector(atoms['N'], atoms['C'], atoms['CA'])
        
        atoms['CB'] = get_atom_place(
             CB_norm, self.N_C_CA_CB_dict[AA], 
             atoms['C'], atoms['CA'], 
             self.CA_CB_dict[AA],
             self.N_C_CA_dict[AA]
        )
        
        placed = {'CB': atoms['CB']}
        chi_angles = []
        
        #====place side chain atoms===========
        for chi_num in range(1, 5):
            
            if AA not in self.chi_atoms[f'chi{chi_num}']: break
            
            a1, a2, a3, a4 = self.chi_atoms[f'chi{chi_num}'][AA]
            if a4 not in atoms: continue
            
                
            p1_norm = get_normal_vector(atoms[a1], atoms[a2], atoms[a3])
            p2_norm = vecs[chi_num - 1]
            chi_angle = get_chi_angle(p1_norm, p2_norm, atoms[a2], atoms[a3])
            
            chi_angles.append(chi_angle)
            
            bond_length, bond_angle = self.ideal_bond_lengths[f'{AA}{chi_num - 1}'], self.ideal_bond_angles[f'{AA}{chi_num - 1}']

            predicted_place = get_atom_place(p1_norm, chi_angle, atoms[a2], atoms[a3], bond_length, bond_angle)

            # Use predicted place downstream
            atoms[a4] = predicted_place
            
            placed[a4] = atoms[a4]
        

        return placed, chi_angles
    

    def reconstruct_from_chi_angles(self,
                                    atoms: dict,
                                    AA: str,
                                    chi_angles: list[float],
                                    true_CB: Optional[np.ndarray] = None) -> tuple[dict, list]:
        """
        Takes a neighborhood with standard dt as used in get_neighborhood_pipeline, 
        and a list of chi angles
        
        returns a dict of {name -> coords} for sidechain atoms that were placed, 
        including the beta carbon
        """
        
        assert 'C' in atoms
        assert 'N' in atoms
        assert 'CA' in atoms

        AA = self.aa_symbols[AA]

        if true_CB is not None:
            # if we have the true CB location, use it instead of the virtualized one
            atoms['CB'] = true_CB
        else:
            #=====virtualize beta carbon============
            CB_norm = get_normal_vector(atoms['N'], atoms['C'], atoms['CA'])
            
            atoms['CB'] = get_atom_place(
                CB_norm, self.N_C_CA_CB_dict[AA], 
                atoms['C'], atoms['CA'], 
                self.CA_CB_dict[AA],
                self.N_C_CA_dict[AA]
            )[0]
        
        placed = {'CB': atoms['CB']}
        
        ordered_norms = []
        
        #====place side chain atoms===========
        for chi_num in range(1, 5):
            
            if AA not in self.chi_atoms[f'chi{chi_num}']: break
            
            a1, a2, a3, a4 = self.chi_atoms[f'chi{chi_num}'][AA]
            if a4 in atoms:
                print('WARNING: sidechain atom already present in neighborhood')

            p1_norm = get_normal_vector(atoms[a1], atoms[a2], atoms[a3])
            chi_angle = chi_angles[chi_num - 1]
            bond_length, bond_angle = self.ideal_bond_lengths[f'{AA}{chi_num - 1}'], self.ideal_bond_angles[f'{AA}{chi_num - 1}']

            predicted_place, fake_p2_norm = get_atom_place(p1_norm, chi_angle, atoms[a2], atoms[a3], bond_length, bond_angle)

            # Use predicted place downstream
            atoms[a4] = predicted_place
            placed[a4] = atoms[a4]
            ordered_norms.append(get_normal_vector(atoms[a2], atoms[a3], atoms[a4]))
        
        return placed, ordered_norms
        
        
    def reconstruct_from_chi_angles__neighborhood_version(self, neighborhood: np.ndarray, chi_angles: list[float]) -> dict:
        """
        Takes a neighborhood with standard dt as used in get_neighborhood_pipeline, 
        and a list of chi angles
        
        returns a dict of {name -> coords} for sidechain atoms that were placed, 
        including the beta carbon
        """
        
        res_id = decode_id(neighborhood['res_id'])
        AA = self.aa_symbols[res_id[0]]
        
        if AA in ['ALA', 'GLY']: return None # no Chi angles
    
        atoms = {'CA': np.zeros(3, dtype=float)}
        for atom, _res_id, coords in zip(neighborhood['atom_names'],neighborhood['res_ids'], neighborhood['coords']):
            if (decode_id(_res_id) != res_id): continue
            atom = atom.decode('utf-8').strip()
            atoms[atom] = coords
        
        #=====virtualize beta carbon============
        if 'N' not in atoms or'C' not in atoms or 'CA' not in atoms: # check that all needed atoms are found in neighborhood
            return None
    
        CB_norm = get_normal_vector(atoms['N'], atoms['C'], atoms['CA'])
        
        atoms['CB'] = get_atom_place(
             CB_norm, self.N_C_CA_CB_dict[AA], 
             atoms['C'], atoms['CA'], 
             self.CA_CB_dict[AA],
             self.N_C_CA_dict[AA]
        )
        
        placed = {'CB': atoms['CB']}
        
        
        #====place side chain atoms===========
        for chi_num in range(1, 5):
            
            if AA not in self.chi_atoms[f'chi{chi_num}']: break
            
            a1, a2, a3, a4 = self.chi_atoms[f'chi{chi_num}'][AA]
            if a4 not in atoms: continue
            
            if self.vec_db is not None:
                if chi_num == 1: self.vec_db["_".join(res_id)] = []
                p2_norm = get_normal_vector(atoms[a2], atoms[a3], atoms[a4])
                self.vec_db["_".join(res_id)].append(p2_norm)
                
            p1_norm = get_normal_vector(atoms[a1], atoms[a2], atoms[a3])
            chi_angle = chi_angles[chi_num - 1]
            bond_length, bond_angle = self.ideal_bond_lengths[f'{AA}{chi_num - 1}'], self.ideal_bond_angles[f'{AA}{chi_num - 1}']

            predicted_place, p2_norm = get_atom_place(p1_norm, chi_angle, atoms[a2], atoms[a3], bond_length, bond_angle)

            # Use predicted place downstream
            atoms[a4] = predicted_place
            
            placed[a4] = atoms[a4]
        

        return placed

if __name__ == '__main__':

    # test the reconstructions

    # plot error on reconstructed vs true chi angles for a batch of neighborhoods

    # just use my validation data

    from runtime.sidechain_prediction.src.data import load_data
    from torch.utils.data import DataLoader
    from protein_holography_pytorch.utils.protein_naming import ind_to_ol_size
    from sqlitedict import SqliteDict
    import h5py
    import hdf5plugin

    with open('/gscratch/spe/gvisan01/protein_holography-pytorch/runtime/sidechain_prediction/runs/so3_convnet-simple_task_100pdbs-zernike-ks-lmax=5/hparams.json', 'r') as f:
        hparams = json.load(f)
        pdb_list_filename = hparams['pdb_list_filename_template'].format(split='validation')

    with h5py.File('/gscratch/scrubbed/gvisan01/casp12/chi/neighborhoods/neighborhoods-easy_task_100pdbs_validation-r_max=10-central_residue_only.hdf5', 'r') as f:
        nbs = f['data'][:]

    valid_angles_dict = SqliteDict(hparams['angles_filepath'].format(pdb_list_filename=pdb_list_filename, **hparams))
    valid_vectors_dict = SqliteDict(hparams['vectors_filepath'].format(pdb_list_filename=pdb_list_filename, **hparams))

    reconstructor = Reconstructor()

    for nb in nbs:

        vecs = valid_vectors_dict["_".join(decode_id(nb['res_id']))]
        angles_valid = valid_angles_dict["_".join(decode_id(nb['res_id']))]

        try:
            _, chi_angles = reconstructor.reconstruct_from_normal_vectors(nb, vecs)
        except TypeError:
            continue

        chi_angles_reconstructed = np.full(4, np.nan)
        for i, chi in enumerate(chi_angles):
            chi_angles_reconstructed[i] = chi * (180/np.pi)
        angles_valid = np.array(angles_valid)

        # print(np.nanmin(chi_angles_reconstructed), np.nanmax(chi_angles_reconstructed))
        # print(np.nanmin(angles_valid), np.nanmax(angles_valid))

        # print(chi_angles_reconstructed[0])
        # print(angles_valid[0])

        # compute the error
        mae_4 = np.abs(chi_angles_reconstructed - angles_valid)

        print(f'Error:\t{mae_4[0]}\t{mae_4[1]}\t{mae_4[2]}\t{mae_4[3]}')