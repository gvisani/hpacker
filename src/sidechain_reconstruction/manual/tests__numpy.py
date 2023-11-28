


import os, sys

import gzip, pickle
import json

import h5py
import hdf5plugin

import numpy as np
import torch

from utils__numpy import *
from reconstruction__numpy import Reconstructor

from tqdm import tqdm

from runtime.sidechain_prediction.src.data import load_data
from torch.utils.data import DataLoader
from protein_holography_pytorch.utils.protein_naming import ind_to_ol_size, one_letter_to_aa

from tests__utils import *


def get_CB_coords(res_id, res_id_to_nb_dict):
    coords = res_id_to_nb_dict[res_id]['coords']
    atom_names = res_id_to_nb_dict[res_id]['atom_names']
    temp_CB_coords = coords[atom_names == 'CB']
    assert temp_CB_coords.shape[0] == 1
    CB_coords = temp_CB_coords[0]
    return CB_coords


def reconstruction_from_collected_chi_angles(dataloader, res_id_to_nb_dict, use_true_CB=False):
    reconstructor = Reconstructor()

    results_by_res_id = {}

    for batch_i, (X, X_vec, aa_label, angles, vectors, backbone_coords, (rot, data_ids)) in tqdm(enumerate(dataloader)):

        # same format as in the torch version... allows us to check for mistakes
        atoms = [backbone_coords[:, 0, :].cpu(), backbone_coords[:, 1, :].cpu(), backbone_coords[:, 2, :].cpu(), backbone_coords[:, 3, :].cpu()]
        AA = [ind_to_ol_size[aa_idx.item()] for aa_idx in aa_label]
        normal_vectors = vectors.cpu()

        if normal_vectors.shape[1] == 5:
            # print('Removing the first normal vector')
            normal_vectors = normal_vectors[:, 1:, :]
        
        for i in range(len(AA)):
            res_id = data_ids[i]
            ol_aa = AA[i]
        
            if use_true_CB:
                true_CB = get_CB_coords(res_id, res_id_to_nb_dict)
            else:
                true_CB = None

            ###### TODO: these don't match at all!!!! not even a permutation or something
            ## something is terribly off, and the worst part is that neither gives the correct result lol

            all_atoms = dict(zip(res_id_to_nb_dict[res_id]['atom_names'], res_id_to_nb_dict[res_id]['coords']))
            curr_atoms_from_nbs = {'C': all_atoms['C'], 'O': all_atoms['O'], 'N': all_atoms['N'], 'CA': np.zeros(3)}
            
            curr_atoms = {'C': atoms[0][i].numpy(), 'O': atoms[1][i].numpy(), 'N': atoms[2][i].numpy(), 'CA': atoms[3][i].numpy()}
            curr_angles = angles[i].numpy()[~np.isnan(angles[i].numpy())]
                        
            placed_atoms_dict, derived_norms = reconstructor.reconstruct_from_chi_angles(curr_atoms, ol_aa, curr_angles, true_CB=true_CB)

            pred_atoms_53 = np.full((5, 3), np.nan)
            for j, atom_name in enumerate(SIDECHAIN_ATOMS_AA_DICT[one_letter_to_aa[ol_aa]]):
                pred_atoms_53[j, :] = placed_atoms_dict[atom_name]


            true_atoms_53 = get_sidechain_atom_coords(ol_aa, res_id, res_id_to_nb_dict)

            # compute euclidean distance
            error_5 = np.sqrt(np.sum((pred_atoms_53 - true_atoms_53)**2, axis=1))

            # print(error_5)

            results_by_res_id[res_id] = error_5

    return results_by_res_id


def reconstruction_from_collected_chi_angles_from_nbs(angles_db, res_id_to_nb_dict, use_true_CB=False):
    '''
    We need this and the above function to match!
    
    TODO: figured out some studpid issues. made sure everything is in degrees and in cartesian coordinates. now error is small but still present
    # I think that with the true beta-carbon, the reconstruction error should be zero, but it seems higher (10^-1/10^-2 AA) than numerical error alone
    # good news is, numpy and torch versions match!
    '''
    reconstructor = Reconstructor()

    results_by_res_id = {}

    i = 0
    for res_id in TEST_RES_IDS:

        ol_aa = res_id.split('_')[0]

        if one_letter_to_aa[ol_aa] not in VEC_AA_ATOM_DICT:
            continue
            
        all_atoms = dict(zip(res_id_to_nb_dict[res_id]['atom_names'], res_id_to_nb_dict[res_id]['coords']))
        curr_atoms = {'C': all_atoms['C'], 'O': all_atoms['O'], 'N': all_atoms['N'], 'CA': np.zeros(3)}
        curr_angles = np.array(angles_db[res_id])

        if use_true_CB:
            true_CB = get_CB_coords(res_id, res_id_to_nb_dict)
        else:
            true_CB = None

        placed_atoms_dict, derived_norms = reconstructor.reconstruct_from_chi_angles(curr_atoms, ol_aa, curr_angles, true_CB=true_CB)

        pred_atoms_53 = np.full((5, 3), np.nan)
        for j, atom_name in enumerate(SIDECHAIN_ATOMS_AA_DICT[one_letter_to_aa[ol_aa]]):
            pred_atoms_53[j, :] = placed_atoms_dict[atom_name]


        true_atoms_53 = get_sidechain_atom_coords(ol_aa, res_id, res_id_to_nb_dict)

        # compute euclidean distance
        error_5 = np.sqrt(np.sum((pred_atoms_53 - true_atoms_53)**2, axis=1))

        # print(error_5)

        results_by_res_id[res_id] = error_5

        i += 1

        if i == 16:
            break

    return results_by_res_id


if __name__ == '__main__':

    batch_size = 16

    from sqlitedict import SqliteDict
    angles_db = SqliteDict('/gscratch/scrubbed/gvisan01/casp12/chi/chi_angles_and_vectors/angles-easy_task_100pdbs_validation.db')

    with open('/gscratch/spe/gvisan01/protein_holography-pytorch/runtime/sidechain_prediction/runs/so3_convnet-simple_task_100pdbs-zernike-ks-lmax=5/hparams.json', 'r') as f:
        hparams = json.load(f)

    datasets, data_irreps, norm_factor = load_data(hparams, splits=['valid'])
    dataset = datasets['valid']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with h5py.File('/gscratch/scrubbed/gvisan01/casp12/chi/neighborhoods/neighborhoods-easy_task_100pdbs_validation-r_max=10-central_residue_only.hdf5', 'r') as f:
        nbs = f['data'][:]
    
    if not os.path.exists('res_id_to_nb_dict.gz'):
        res_id_to_nb_dict = {}
        for nb in nbs:
            real_mask = nb['atom_names'] != b''
            atom_names = np.array([atom_name.decode('utf-8').strip() for atom_name in nb['atom_names'][real_mask]])
            from protein_holography_pytorch.utils.conversions import spherical_to_cartesian__numpy
            coords = spherical_to_cartesian__numpy(nb['coords'][real_mask])
            res_id_to_nb_dict["_".join(decode_id(nb['res_id']))] = {'atom_names': atom_names, 'coords': coords}
        with gzip.open('res_id_to_nb_dict.gz', 'wb') as f:
            pickle.dump(res_id_to_nb_dict, f)
    else:
        with gzip.open('res_id_to_nb_dict.gz', 'rb') as f:
            res_id_to_nb_dict = pickle.load(f)

    results_from_dataloader = reconstruction_from_collected_chi_angles(dataloader, res_id_to_nb_dict, use_true_CB=True)

    results_from_nbs = reconstruction_from_collected_chi_angles_from_nbs(angles_db, res_id_to_nb_dict, use_true_CB=True)

    print()
    for res_id in TEST_RES_IDS:
        print(res_id)
        print(results_from_dataloader[res_id])
        print(results_from_nbs[res_id])
        print()
    
    # reconstruction_from_collected_chi_angles(dataloader, res_id_to_nb_dict, use_true_CB=True)

    # reconstruction_from_collected_chi_angles_from_nbs(angles_db, res_id_to_nb_dict, use_true_CB=True)

    


