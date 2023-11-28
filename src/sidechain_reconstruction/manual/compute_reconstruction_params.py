

import os, sys

import numpy as np
import math

import h5py
import hdf5plugin

import json
import gzip, pickle

from tqdm import tqdm

from protein_holography_pytorch.utils.conversions import spherical_to_cartesian__numpy

from protein_holography_pytorch.utils.protein_naming import one_letter_to_aa, aa_to_one_letter, ol_to_ind_size

from typing import *

from .utils__numpy import get_normal_vector, split_id, decode_id

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--residues_file', type=str, default='/gscratch/scrubbed/gvisan01/dlpacker/neighborhoods/neighborhoods-dlpacker_training__0-r_max=10-central_residue_only.hdf5')
    parser.add_argument('--dataset_name', type=str, default='data')
    parser.add_argument('--num_residues', type=int, default=1000000)
    args = parser.parse_args()

    compute_reconstruction_parameters_from_residues_hdf5_file(args.residues_file, dataset_name=args.dataset_name, num_residues=args.num_residues)


def compute_reconstruction_parameters_from_residues_hdf5_file(residues_file: str, dataset_name: str = 'data', num_residues: Optional[int] = None):
    '''
    Expects as input an hdf5 file containing residues, of the kind that has to be collected from PDBs using the script `get_neighborhoods.py` with the flag `--central_residue_only` toggled.

    This is done in two steps:
        1) compute all the values and put them in a dictionary indexed by AA and chi number
        2) vectorize the whole thing, following an arbitrary AA order that gets saved alongside the parameters
    '''

    ## simple workaround to deal with having training data split into multiple files, but also allowing for the way of having one training file
    with h5py.File(residues_file, "r") as f:
        if num_residues is None:
            data = f[dataset_name][:]
        else:
            data = f[dataset_name][:num_residues]

    rec_params__numpy = _compute_unvectorized_ideal_reconstruction_params(data)

    rec_params__torch = _vectorize_reconstruction_params(rec_params__numpy)

    with open('reconstruction_params.json', 'w') as fp:
        json.dump(rec_params__numpy, fp, indent=4)
    
    with open('reconstruction_params__vecgtorized.json', 'w') as fp:
        json.dump(rec_params__torch, fp, indent=4)


def dict_array_to_list(adict):
    newdict = {}
    for key, value in adict.items():
        newdict[key] = value.tolist()
    return newdict

def _compute_unvectorized_ideal_reconstruction_params(data: np.ndarray):
    
    ## Some tables of constants

    chi_atoms = dict(
            chi1=dict(
                ARG=['N', 'CA', 'CB', 'CG'],
                ASN=['N', 'CA', 'CB', 'CG'],
                ASP=['N', 'CA', 'CB', 'CG'],
                CYS=['N', 'CA', 'CB', 'SG'],
                GLN=['N', 'CA', 'CB', 'CG'],
                GLU=['N', 'CA', 'CB', 'CG'],
                HIS=['N', 'CA', 'CB', 'CG'],
                ILE=['N', 'CA', 'CB', 'CG1'],
                LEU=['N', 'CA', 'CB', 'CG'],
                LYS=['N', 'CA', 'CB', 'CG'],
                MET=['N', 'CA', 'CB', 'CG'],
                PHE=['N', 'CA', 'CB', 'CG'],
                PRO=['N', 'CA', 'CB', 'CG'],
                SER=['N', 'CA', 'CB', 'OG'],
                THR=['N', 'CA', 'CB', 'OG1'],
                TRP=['N', 'CA', 'CB', 'CG'],
                TYR=['N', 'CA', 'CB', 'CG'],
                VAL=['N', 'CA', 'CB', 'CG1'],
            ),
            chi2=dict(
                ARG=['CA', 'CB', 'CG', 'CD'],
                ASN=['CA', 'CB', 'CG', 'OD1'],
                ASP=['CA', 'CB', 'CG', 'OD1'],
                GLN=['CA', 'CB', 'CG', 'CD'],
                GLU=['CA', 'CB', 'CG', 'CD'],
                HIS=['CA', 'CB', 'CG', 'ND1'],
                ILE=['CA', 'CB', 'CG1', 'CD1'],
                LEU=['CA', 'CB', 'CG', 'CD1'],
                LYS=['CA', 'CB', 'CG', 'CD'],
                MET=['CA', 'CB', 'CG', 'SD'],
                PHE=['CA', 'CB', 'CG', 'CD1'],
                PRO=['CA', 'CB', 'CG', 'CD'],
                TRP=['CA', 'CB', 'CG', 'CD1'],
                TYR=['CA', 'CB', 'CG', 'CD1'],
            ),
            chi3=dict(
                ARG=['CB', 'CG', 'CD', 'NE'],
                GLN=['CB', 'CG', 'CD', 'OE1'],
                GLU=['CB', 'CG', 'CD', 'OE1'],
                LYS=['CB', 'CG', 'CD', 'CE'],
                MET=['CB', 'CG', 'SD', 'CE'],
            ),
            chi4=dict(
                ARG=['CG', 'CD', 'NE', 'CZ'],
                LYS=['CG', 'CD', 'CE', 'NZ'],
            ),
        )

    aa_symbols = dict(
        ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', 
        GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F', 
        PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V',
    )
    keys = [a for a in aa_symbols.keys()]
    for key in keys:
        aa_symbols[aa_symbols[key]] = key 
        
        
    n_chi = dict( 
        ALA=0, ARG=4, ASN=2, ASP=2, CYS=1, 
        GLN=3, GLU=3, GLY=0, HIS=2, ILE=2, LEU=2, 
        LYS=4, MET=3, PHE=2, PRO=3, SER=1, THR=1, 
        TRP=2, TYR=2, VAL=1,
    )
    keys = [key for key in n_chi]
    for key in keys:
        n_chi[aa_symbols[key]] = n_chi[key]

    ## NOTE: converting from spherical coords to cartesian coords! assuming that the data is saved in spherical coords
    for nb in data:
        real_mask = nb['atom_names'] != b''
        nb['coords'][real_mask] = spherical_to_cartesian__numpy(nb['coords'][real_mask])

    ## Collect data on CB parameters

    CA = np.zeros(3, dtype=float)

    distances = []
    n_c_ca_angles = []
    dihedral_angles = []

    list_of_CA_CB_dict = {}
    list_of_N_C_CA_dict = {}
    list_of_N_C_CA_CB_dict = {}

    for neighborhood in tqdm(data):
        res_id = decode_id(neighborhood['res_id'])
        AA = one_letter_to_aa[res_id[0]]
        for atom, res_id, coords in zip(neighborhood['atom_names'],neighborhood['res_ids'], neighborhood['coords']):
            if (decode_id(res_id) != decode_id(neighborhood['res_id'])): continue
            atom = atom.decode('utf-8').strip()
            if atom == 'N': N = coords 
            if atom == 'C': C = coords
            if atom == 'CB': CB = coords
            
        # CA-CB distance
        dist = math.dist(CB, CA)
        distances.append(dist)
        if AA not in list_of_CA_CB_dict:
            list_of_CA_CB_dict[AA] = []
        list_of_CA_CB_dict[AA].append(float(dist))
            
        # N-C-CA
        v1, v2 = N - CA, C - CA
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        angle = (np.degrees(np.arccos(np.dot(v2, v1))))
        n_c_ca_angles.append(float(angle))

        if AA not in list_of_N_C_CA_dict:
            list_of_N_C_CA_dict[AA] = []
        list_of_N_C_CA_dict[AA].append(float(angle))
        
        # N-C-CA-CB
        v1 = get_normal_vector(N, C, CA) 
        v2 = get_normal_vector(C, CA, CB)
        angle = math.degrees((np.arccos(np.dot(v1, v2))))
        dihedral_angles.append(float(angle))
        
        if AA not in list_of_N_C_CA_CB_dict:
            list_of_N_C_CA_CB_dict[AA] = []
        list_of_N_C_CA_CB_dict[AA].append(float(angle))
    
    CA_CB_dict = {}
    N_C_CA_dict = {}
    N_C_CA_CB_dict = {}
    for aa in list_of_CA_CB_dict:
        CA_CB_dict[aa] = float(np.median(list_of_CA_CB_dict[aa]))
    for aa in list_of_N_C_CA_dict:
        N_C_CA_dict[aa] = float(np.median(list_of_N_C_CA_dict[aa]))
    for aa in list_of_N_C_CA_CB_dict:
        N_C_CA_CB_dict[aa] = float(np.median(list_of_N_C_CA_CB_dict[aa]))

    ## Collect data on side-chain parameters

    CA = np.zeros(3, dtype=float)

    bond_lengths = {}
    bond_angles = {}

    for neighborhood in tqdm(data):
        res_id = decode_id(neighborhood['res_id'])
        AA = one_letter_to_aa[res_id[0]]
        
        atoms = {'CA': CA}
        for atom, _res_id, coords in zip(neighborhood['atom_names'],neighborhood['res_ids'], neighborhood['coords']):
            if (decode_id(_res_id) != res_id): continue
            atom = atom.decode('utf-8').strip()
            atoms[atom] = coords
            
        for chi_num in range(1, 5):
            if AA not in chi_atoms[f'chi{chi_num}']: break
            a1, a2, a3, a4 = chi_atoms[f'chi{chi_num}'][AA]
            if a2 not in atoms or a3 not in atoms or a4 not in atoms: # check that all needed atoms are found in neighborhood
                continue
                
            # bond length
            if AA not in bond_lengths: bond_lengths[AA] = [[], [], [], []]
            dist = math.dist(atoms[a3], atoms[a4])
            bond_lengths[AA][chi_num - 1].append(float(dist))
            
            # a2-a3-a4 bond angles
            if AA not in bond_angles: bond_angles[AA] = [[], [], [], []]
            v1, v2 = atoms[a2] - atoms[a3], atoms[a4] - atoms[a3]
            v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
            angle = (np.degrees(np.arccos(np.dot(v2, v1))))
            bond_angles[AA][chi_num - 1].append(float(angle)) 
        
    ideal_bond_lengths, ideal_bond_angles = {}, {}
    for aa in bond_lengths:
        for chi_num in range(4):
            info = bond_lengths[aa][chi_num]
            if len(info) == 0: continue
            ideal_bond_lengths[f'{aa}{chi_num}'] = float(np.median(info))
            
    for aa in bond_angles:
        for chi_num in range(4):
            info = bond_angles[aa][chi_num]
            if len(info) == 0: continue
            ideal_bond_angles[f'{aa}{chi_num}'] = float(np.median(info))
    
    reconstruction_params = {
        'CA_CB_dict': CA_CB_dict,
        'N_C_CA_dict': N_C_CA_dict,
        'N_C_CA_CB_dict': N_C_CA_CB_dict,
        'ideal_bond_lengths': ideal_bond_lengths,
        'ideal_bond_angles': ideal_bond_angles,
        'aa_symbols': aa_symbols,
        'chi_atoms': chi_atoms,
        'n_chi': n_chi
    }

    return reconstruction_params


def _vectorize_reconstruction_params(reconstruction_params: Dict):

    def three_to_one_aa(rec_json, aa_to_one_letter):
        new_rec_json = {}
        for field in rec_json:
            if field == 'chi_atoms' or field == 'aa_symbols' or field == 'n_chi': continue
            new_rec_json[field] = {}
            for aa_key in rec_json[field]:
                new_aa_key = aa_to_one_letter[aa_key[:3]] + aa_key[3:]
                new_rec_json[field][new_aa_key] = rec_json[field][aa_key]
        
        return new_rec_json

    def aa_to_index(rec_json, VALID_AAS, null_value = 0.0):

        new_rec_json = {}
        for field in ['CA_CB_dict', 'N_C_CA_dict', 'N_C_CA_CB_dict']:
            new_rec_json[field] = [rec_json[field][aa] for aa in VALID_AAS]
        
        for field in ['ideal_bond_lengths', 'ideal_bond_angles']:
            values = []
            for aa in VALID_AAS:
                temp = []
                for i in range(4):
                    key = aa + str(i)
                    if key in rec_json[field]:
                        temp.append(rec_json[field][key])
                    else:
                        temp.append(null_value)
                values.append(temp)
            new_rec_json[field] = values
        
        return new_rec_json
    
    # first, turn the json into using single-char-AA instead of three-chars-AA
    vecotrized_reconstruction_params = three_to_one_aa(reconstruction_params, aa_to_one_letter)

    # second, order amino-acid-specific data according to the standard order in VALID_AAS
    # provide invalid bond lengths and angles as zero, or as nans
    AAs = ['W', 'N', 'I', 'G', 'H', 'V', 'M', 'T', 'S', 'Y', 'Q', 'F', 'E', 'K', 'P', 'C', 'L', 'A', 'D', 'R']
    vecotrized_reconstruction_params = aa_to_index(vecotrized_reconstruction_params, AAs, null_value=np.nan)
    vecotrized_reconstruction_params['AAs'] = AAs

    return vecotrized_reconstruction_params

if __name__ == '__main__':
    main()
