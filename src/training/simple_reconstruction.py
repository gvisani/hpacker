
import os, sys

import numpy as np
import torch
import math

import h5py
import hdf5plugin

import json
import gzip, pickle

from tqdm import tqdm

from torch.utils.data import DataLoader

from .data import load_data
from .losses import angle_loss
from src.utils.conversions import spherical_to_cartesian__numpy

from src.utils.protein_naming import one_letter_to_aa, aa_to_one_letter, ol_to_ind_size

from typing import *

from .data import stringify_array, stringify

from src.sidechain_reconstruction.manual import _compute_unvectorized_ideal_reconstruction_params, _vectorize_reconstruction_params
from src.sidechain_reconstruction.manual.reconstruction__torch import Reconstructor
from src.sidechain_reconstruction.manual.tests__torch import reconstruction_from_collected_chi_angles, reconstruction_from_collected_norms, get_sidechain_atom_coords
from src.sidechain_reconstruction.manual.tests__utils import atomic_reconstruction_barplot_by_aa, norms_error_barplot_by_aa, chi_angles_error_barplot_by_aa

def get_indices_mapping_x1_into_x2(x1, x2):
    # '''
    # Courtesy of ChatGPT
    # '''

    # x1_sorted_indices = np.argsort(x1)
    # x1_sorted = np.sort(x1)

    # # Use searchsorted to find the indices of A values in the sorted_B
    # indices_in_sorted_x1 = np.searchsorted(x1_sorted, x2)

    # # Use these indices to map the values in B to their corresponding positions in A
    # indices_in_x1 = np.argsort(x1_sorted_indices)[indices_in_sorted_x1]

    # return indices_in_x1

    from tqdm import tqdm

    list_x2 = list(x2)

    indices = []
    for x in tqdm(x1):
        indices.append(list_x2.index(x))

    return np.array(indices)



def plot_mae_per_chi_binned_by_proportion_of_sidechains_removed(model_dir):

    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
        
    with gzip.open(os.path.join(model_dir, 'per_example_results_dict.gz'), 'rb') as f:
        per_example_results_dict_by_split = pickle.load(f)
    
    num_residues_to_use = 10_000

    for split in per_example_results_dict_by_split:
        per_example_results_dict = per_example_results_dict_by_split[split]
        res_ids_N = per_example_results_dict['res_ids'][:num_residues_to_use]
        y_hat_N4_or_N8_or_N43 = torch.tensor(per_example_results_dict['y_hat'])
        true_angles_N4 = torch.tensor(per_example_results_dict['true_angles'])[:num_residues_to_use]
        predicted_angles_N4 = torch.tensor(per_example_results_dict['predicted_angles'])[:num_residues_to_use]
        backbone_coords_N43 = torch.tensor(per_example_results_dict['backbone_coords'])
        N = len(res_ids_N)
        aas_N = torch.tensor([ol_to_ind_size[res_id.split('_')[0]] for res_id in res_ids_N])
        error_on_angles_N4 = angle_loss(predicted_angles_N4, true_angles_N4, aas_N)

        if split == 'valid':
            split_for_filename = 'validation'
        elif split == 'test':
            split_for_filename = 'testing'
        pdb_list_filename = hparams['pdb_list_filename_template'].format(split=split_for_filename)
        downsampled_neighborhoods_filepath = hparams['downsampled_neighborhoods_filepath'].format(pdb_list_filename=pdb_list_filename, **hparams)
    
        with h5py.File(downsampled_neighborhoods_filepath, 'r') as f:
            res_ids_in_downsampled = stringify_array(f['data']['res_id'])
            prop_sidechains_removed = f['proportion_sidechain_removed'][:]
        
        print('Making indices...', flush=True)
        indices_mapping_prop_sidechain_removed_to_error_on_angles = get_indices_mapping_x1_into_x2(res_ids_N, res_ids_in_downsampled)
        print('Done making indices!', flush=True)

        rearranged_prop_sidechains_removed = prop_sidechains_removed[indices_mapping_prop_sidechain_removed_to_error_on_angles]

    
        sorted_idxs_of_prop_sidechains_removed = np.argsort(rearranged_prop_sidechains_removed)
        errors_sorted_by_increasing_prop_sidechains_removed = error_on_angles_N4[sorted_idxs_of_prop_sidechains_removed]
        prop_sidechains_removed_sorted = rearranged_prop_sidechains_removed[sorted_idxs_of_prop_sidechains_removed]

        num_bins = 10
        binsize = prop_sidechains_removed_sorted.shape[0] // num_bins
        avg_error_per_bin = []
        for i in range(num_bins):
            if i < num_bins - 1:
                avg_error_per_bin.append(np.nanmean(errors_sorted_by_increasing_prop_sidechains_removed[i * binsize : (i + 1) * binsize, :].numpy(), axis=0))
            else:
                avg_error_per_bin.append(np.nanmean(errors_sorted_by_increasing_prop_sidechains_removed[i * binsize : , :].numpy(), axis=0))
        
        avg_error_per_bin_B4 = np.vstack(avg_error_per_bin)

        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(4):
            plt.plot(np.arange(num_bins), avg_error_per_bin_B4[:, i], marker='o', label=f'chi_{i + 1}')
        plt.grid(ls='--', color='dimgrey', alpha=0.5)
        plt.ylabel('MAE')
        plt.xlabel('Proportion of Sidechains Removed')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'{split}__mae_per_chi_binned_by_proportion_of_sidechains_removed.png'))
        plt.close()
        




        
        

        




def reconstruction_from_predictions(model_dir):
    '''
    1) compute and save reconstruction parameters from training data, if they don't exist already
    2) compute, save and plot per-aa error on chi-angles 
    2) based on the type of model (i.e. predicts angles, sin/cos of angles, or norms) compute reconstructions and compute/save/plot per-aa errors
    '''

    # initialize reconstruction stuff --> compute reconstruction parameters if they don't exist already and get true atomic positions
    res_id_to_atoms_dict_by_split = _init_reconstruction_functions(model_dir)

    os.makedirs(os.path.join(model_dir, 'rec_errors'), exist_ok=True)

    with gzip.open(os.path.join(model_dir, 'per_example_results_dict.gz'), 'rb') as f:
        per_example_results_dict_by_split = pickle.load(f)
    
    # load hparams to know if model predicted norms, angles, or sin/cos of angles
    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    
    reconstructor = Reconstructor(os.path.join(model_dir, '../reconstruction_params__vectorized.json'))

    for split in per_example_results_dict_by_split:
        per_example_results_dict = per_example_results_dict_by_split[split]
        res_ids_N = per_example_results_dict['res_ids']
        y_hat_N4_or_N8_or_N43 = torch.tensor(per_example_results_dict['y_hat'])
        true_angles_N4 = torch.tensor(per_example_results_dict['true_angles'])
        predicted_angles_N4 = torch.tensor(per_example_results_dict['predicted_angles'])
        backbone_coords_N43 = torch.tensor(per_example_results_dict['backbone_coords'])
        N = len(res_ids_N)
        aas_N = torch.tensor([ol_to_ind_size[res_id.split('_')[0]] for res_id in res_ids_N])
        error_on_angles_N4 = angle_loss(predicted_angles_N4, true_angles_N4, aas_N)


        # chi-angles prediction plots
        results_dict_res_id_to_errors = dict(zip(res_ids_N, error_on_angles_N4))
        _, chi_angles_error_by_aa = chi_angles_error_barplot_by_aa(results_dict_res_id_to_errors, 'Chi Angles Error', os.path.join(model_dir, f'rec_errors/{split}__chi_angles_error_barplot_by_aa.png'))
        with open(os.path.join(model_dir, f'rec_errors/{split}__chi_angle_error_by_aa.json'), 'w+') as f:
            json.dump(dict_array_to_list(chi_angles_error_by_aa), f, indent=4)

        # atomic reconstruction! first compute them
        ol_AAs = [res_id.split('_')[0] for res_id in res_ids_N]
        atoms = [backbone_coords_N43[:, 0, :].cpu(), backbone_coords_N43[:, 1, :].cpu(), backbone_coords_N43[:, 2, :].cpu(), backbone_coords_N43[:, 3, :].cpu()]
        if '_vec' in hparams['model_type']:
            normal_vectors_N43 = y_hat_N4_or_N8_or_N43
            placed_atoms, _ = reconstructor.reconstruct_from_normal_vectors(atoms, ol_AAs, normal_vectors_N43)
            placed_atoms = torch.stack(placed_atoms, dim=1)
        else:
            # from .losses import AngleLoss, SinCosAngleLoss
            # loss_fn = SinCosAngleLoss() if '_sin_cos' in hparams['model_type'] else AngleLoss()
            # chi_angles_N4 = loss_fn.get_chi_angles_from_predictions(y_hat_N4_or_N8_or_N43)
            chi_angles_N4 = predicted_angles_N4
            
            placed_atoms, _ = reconstructor.reconstruct_from_chi_angles(atoms, ol_AAs, chi_angles_N4)
            placed_atoms = torch.stack(placed_atoms, dim=1)
        
        # then make plots against the ground truth
        atomic_reconstruction_error_by_res_id = {}
        for i in range(N):
            res_id = res_ids_N[i]
            ol_aa = ol_AAs[i]

            pred_atoms_53 = placed_atoms[i]
            true_atoms_53 = get_sidechain_atom_coords(ol_aa, res_id, res_id_to_atoms_dict_by_split[split])

            # compute euclidean distance on atoms
            euclidean_error_5 = torch.sqrt(torch.sum(torch.square(pred_atoms_53 - true_atoms_53), dim=1))
            atomic_reconstruction_error_by_res_id[res_id] = euclidean_error_5.cpu().numpy()
        
        _, atomic_reconstruction_error_by_aa = atomic_reconstruction_barplot_by_aa(atomic_reconstruction_error_by_res_id, 'Atomic Reconstruction Error', os.path.join(model_dir, f'rec_errors/{split}__atomic_reconstruction_error_barplot_by_aa.png'))
        with open(os.path.join(model_dir, f'rec_errors/{split}__atomic_reconstruction_error_by_aa.json'), 'w+') as f:
            json.dump(dict_array_to_list(atomic_reconstruction_error_by_aa), f, indent=4)




def reconstruction_null(model_dir: str,
                        batch_size = 64):
    '''
    Given a model_dir - with associated training, validation and testing data:
        1) compute and save reconstruction parameters from training data, if they don't exist already
        2) compute null per-aa reconstruction errors, save them in json files, and produce plots
    
    TODO: need to test this
    '''

    # load data

    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    datasets, data_irreps, norm_factor = load_data(hparams, splits=['train', 'valid', 'test'])
    dataloaders = {split: DataLoader(datasets[split], batch_size=batch_size, shuffle=False, drop_last=False) for split in ['train', 'valid', 'test']}

    # initialize reconstruction stuff --> compute reconstruction parameters if they don't exist already and get true atomic positions
    res_id_to_atoms_dict_by_split = _init_reconstruction_functions(model_dir)
    
    ## compute null reconstruction errors for validation and test sets

    os.makedirs(os.path.join(model_dir, 'null_rec_errors'), exist_ok=True)
    
    for split in ['valid', 'test']:
        print(f'Computing null reconstruction errors for {split} set...')
        dataloader = dataloaders[split]
        res_id_to_atoms_dict = res_id_to_atoms_dict_by_split[split]

        rec_error_from_chi_angles_virtual_CB, derived_norms_from_chi_angles_virtual_CB, norm_error_from_chi_angles_virtual_CB = reconstruction_from_collected_chi_angles(dataloader, res_id_to_atoms_dict, rec_params_filepath=os.path.join(model_dir, '../reconstruction_params__vectorized.json'), use_true_CB=False)
        rec_error_from_chi_angles_real_CB, derived_norms_from_chi_angles_real_CB, norm_error_from_chi_angles_real_CB = reconstruction_from_collected_chi_angles(dataloader, res_id_to_atoms_dict, rec_params_filepath=os.path.join(model_dir, '../reconstruction_params__vectorized.json'), use_true_CB=True)

        rec_error_from_norms_virtual_CB, derived_chi_angles_from_norms_virtual_CB, chi_angle_error_from_norms_virtual_CB = reconstruction_from_collected_norms(dataloader, res_id_to_atoms_dict, rec_params_filepath=os.path.join(model_dir, '../reconstruction_params__vectorized.json'), use_true_CB=False)
        rec_error_from_norms_real_CB, derived_chi_angles_from_norms_real_CB, chi_angle_error_from_norms_real_CB = reconstruction_from_collected_norms(dataloader, res_id_to_atoms_dict, rec_params_filepath=os.path.join(model_dir, '../reconstruction_params__vectorized.json'), use_true_CB=True)

        # reconstruction from chi-angles
        yaxismax, atomic_reconstruction_from_chi_angles_by_aa_with_virtual_CB__dict = atomic_reconstruction_barplot_by_aa(rec_error_from_chi_angles_virtual_CB, 'Atomic Reconstruction from Chi Angles with Virtual CB', os.path.join(model_dir, f'null_rec_errors/{split}__atomic_reconstruction_from_chi_angles_barplot_by_aa_with_virtual_CB.png'))
        _, atomic_reconstruction_from_chi_angles_by_aa_with_real_CB__dict = atomic_reconstruction_barplot_by_aa(rec_error_from_chi_angles_real_CB, 'Atomic Reconstruction from Chi Angles with Real CB', os.path.join(model_dir, f'null_rec_errors/{split}__atomic_reconstruction_from_chi_angles_barplot_by_aa_with_real_CB.png'), yaxismax=yaxismax)

        with open(os.path.join(model_dir, f'null_rec_errors/{split}__atomic_reconstruction_from_chi_angles_by_aa_with_virtual_CB.json'), 'w+') as f:
            json.dump(dict_array_to_list(atomic_reconstruction_from_chi_angles_by_aa_with_virtual_CB__dict), f, indent=4)
        with open(os.path.join(model_dir, f'null_rec_errors/{split}__atomic_reconstruction_from_chi_angles_by_aa_with_real_CB.json'), 'w+') as f:
            json.dump(dict_array_to_list(atomic_reconstruction_from_chi_angles_by_aa_with_real_CB__dict), f, indent=4)

        # recovering norms from chi-angles
        yaxismax, norm_error_from_chi_angles_by_aa_with_virtual_CB__dict = norms_error_barplot_by_aa(norm_error_from_chi_angles_virtual_CB, 'Norm Error from Chi Angles with Virtual CB', os.path.join(model_dir, f'null_rec_errors/{split}__norm_error_from_chi_angles_barplot_by_aa_with_virtual_CB.png'))
        _, norm_error_from_chi_angles_by_aa_with_real_CB__dict = norms_error_barplot_by_aa(norm_error_from_chi_angles_real_CB, 'Norm Error from Chi Angles with Real CB', os.path.join(model_dir, f'null_rec_errors/{split}__norm_error_from_chi_angles_barplot_by_aa_with_real_CB.png'), yaxismax=yaxismax)

        with open(os.path.join(model_dir, f'null_rec_errors/{split}__norm_error_from_chi_angles_by_aa_with_virtual_CB.json'), 'w+') as f:
            json.dump(dict_array_to_list(norm_error_from_chi_angles_by_aa_with_virtual_CB__dict), f, indent=4)
        with open(os.path.join(model_dir, f'null_rec_errors/{split}__norm_error_from_chi_angles_by_aa_with_real_CB.json'), 'w+') as f:
            json.dump(dict_array_to_list(norm_error_from_chi_angles_by_aa_with_real_CB__dict), f, indent=4)

        # reconstruction from norms
        yaxismax, atomic_reconstruction_from_norms_by_aa_with_virtual_CB__dict = atomic_reconstruction_barplot_by_aa(rec_error_from_norms_virtual_CB, 'Atomic Reconstruction from Norms with Virtual CB', os.path.join(model_dir, f'null_rec_errors/{split}__atomic_reconstruction_from_norms_barplot_by_aa_with_virtual_CB.png'))
        _, atomic_reconstruction_from_norms_by_aa_with_real_CB__dict = atomic_reconstruction_barplot_by_aa(rec_error_from_norms_real_CB, 'Atomic Reconstruction from Norms with Real CB', os.path.join(model_dir, f'null_rec_errors/{split}__atomic_reconstruction_from_norms_barplot_by_aa_with_real_CB.png'), yaxismax=yaxismax)

        with open(os.path.join(model_dir, f'null_rec_errors/{split}__atomic_reconstruction_from_norms_by_aa_with_virtual_CB.json'), 'w+') as f:
            json.dump(dict_array_to_list(atomic_reconstruction_from_norms_by_aa_with_virtual_CB__dict), f, indent=4)
        with open(os.path.join(model_dir, f'null_rec_errors/{split}__atomic_reconstruction_from_norms_by_aa_with_real_CB.json'), 'w+') as f:
            json.dump(dict_array_to_list(atomic_reconstruction_from_norms_by_aa_with_real_CB__dict), f, indent=4)

        # recovering chi-angles from norms
        yaxismax, chi_angle_error_from_norms_by_aa_with_virtual_CB__dict = chi_angles_error_barplot_by_aa(chi_angle_error_from_norms_virtual_CB, 'Chi Angle Error from Norms with Virtual CB', os.path.join(model_dir, f'null_rec_errors/{split}__chi_angle_error_from_norms_barplot_by_aa_with_virtual_CB.png'))
        _, chi_angle_error_from_norms_by_aa_with_real_CB__dict = chi_angles_error_barplot_by_aa(chi_angle_error_from_norms_real_CB, 'Chi Angle Error from Norms with Real CB', os.path.join(model_dir, f'null_rec_errors/{split}__chi_angle_error_from_norms_barplot_by_aa_with_real_CB.png'), yaxismax=yaxismax)

        with open(os.path.join(model_dir, f'null_rec_errors/{split}__chi_angle_error_from_norms_by_aa_with_virtual_CB.json'), 'w+') as f:
            json.dump(dict_array_to_list(chi_angle_error_from_norms_by_aa_with_virtual_CB__dict), f, indent=4)
        with open(os.path.join(model_dir, f'null_rec_errors/{split}__chi_angle_error_from_norms_by_aa_with_real_CB.json'), 'w+') as f:
            json.dump(dict_array_to_list(chi_angle_error_from_norms_by_aa_with_real_CB__dict), f, indent=4)
    


def _init_reconstruction_functions(model_dir):
    '''
    Utility function so we don't copy-paste code at the start of different functions
    '''
    
    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # load true residue atoms
    print('Loading true residue atoms...')
    train_true_residues_file = hparams['true_sidechains_filepath'].format(pdb_list_filename=hparams['pdb_list_filename_template'].format(split='training'), **hparams)
    valid_true_residues_file = hparams['true_sidechains_filepath'].format(pdb_list_filename=hparams['pdb_list_filename_template'].format(split='validation'), **hparams)
    test_true_residues_file = hparams['true_sidechains_filepath'].format(pdb_list_filename=hparams['pdb_list_filename_template'].format(split='testing'), **hparams)

    # construct simple dictionaries from res_id to atoms
    def make_res_id_to_atoms_dict(true_residues_file: str):
        with h5py.File(true_residues_file, "r") as f:
            data = f['data'][:]
        res_id_to_atoms_dict = {}
        for nb in tqdm(data):
            real_mask = nb['atom_names'] != b''
            atom_names = np.array([atom_name.decode('utf-8').strip() for atom_name in nb['atom_names'][real_mask]])
            coords = spherical_to_cartesian__numpy(nb['coords'][real_mask])
            res_id_to_atoms_dict["_".join(decode_id(nb['res_id']))] = {'atom_names': atom_names, 'coords': coords}
        return res_id_to_atoms_dict
    # train_res_id_to_atoms_dict = make_res_id_to_atoms_dict(train_true_residues_file)
    valid_res_id_to_atoms_dict = make_res_id_to_atoms_dict(valid_true_residues_file)
    test_res_id_to_atoms_dict = make_res_id_to_atoms_dict(test_true_residues_file)
    res_id_to_atoms_dict_by_split = {'valid': valid_res_id_to_atoms_dict, 'test': test_res_id_to_atoms_dict} # 'train': train_res_id_to_atoms_dict, 

    # compute reconstruction parameters if they don't exist already
    print('Computing reconstruction parameters...')
    if not os.path.exists(os.path.join(model_dir, '../reconstruction_params__vectorized.json')):
        rec_params, rec_params__vectorized = compute_ideal_reconstruction_parameters_from_data(train_true_residues_file)

        with open(os.path.join(model_dir, '../reconstruction_params.json'), 'w+') as f:
            json.dump(rec_params, f, indent=4)
        
        with open(os.path.join(model_dir, '../reconstruction_params__vectorized.json'), 'w+') as f:
            json.dump(rec_params__vectorized, f, indent=4)

    return res_id_to_atoms_dict_by_split


def compute_ideal_reconstruction_parameters_from_data_given_model_dir(model_dir: str):

    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    
    train_true_residues_file = hparams['true_sidechains_filepath'].format(pdb_list_filename=hparams['pdb_list_filename_template'].format(split='training'), **hparams)
    
    rec_params, rec_params__vectorized = compute_ideal_reconstruction_parameters_from_data(train_true_residues_file)

    return rec_params, rec_params__vectorized

def compute_ideal_reconstruction_parameters_from_data(training_file: str):
    '''
    This is done in two steps:
        1) compute all the values and put them in a dictionary indexed by AA and chi number
        2) vectorize the whole thing, following the AAs order in TEMPLATE_RECONSTRUCTION_PARAMS['AAs']
    '''

    ## simple workaround to deal with having training data split into multiple files, but also allowing for the way of having one training file
    try:
        training_file = training_file.replace('training', 'training__0')
        with h5py.File(training_file, "r") as f:
            data = f['data'][:100000] # only use the first hundred thousand conformations because that's plenty
    except OSError:
        training_file = training_file.replace('training__0', 'training')
        with h5py.File(training_file, "r") as f:
            data = f['data'][:100000] # only use the first hundred thousand conformations because that's plenty

    
    rec_params__numpy = _compute_unvectorized_ideal_reconstruction_params(data)

    rec_params__torch = _vectorize_reconstruction_params(rec_params__numpy)

    return rec_params__numpy, rec_params__torch




## Some utility functions

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

def dict_array_to_list(adict):
    newdict = {}
    for key, value in adict.items():
        newdict[key] = value.tolist()
    return newdict


