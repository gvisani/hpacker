


import os, sys

import gzip, pickle
import json

from tqdm import tqdm
import h5py
import hdf5plugin

import numpy as np
import torch

from .utils__torch import *
from .reconstruction__torch import Reconstructor

from runtime.sidechain_prediction.src.data import load_data
from torch.utils.data import DataLoader
from protein_holography_pytorch.utils.protein_naming import ind_to_ol_size, one_letter_to_aa

from .tests__utils import *

def get_sidechain_atom_coords(ol_aa, res_id, res_id_to_nb_dict):
    coords = res_id_to_nb_dict[res_id]['coords']
    atom_names = res_id_to_nb_dict[res_id]['atom_names']
    sidechain_coords = np.full((5, 3), np.nan)
    for i, atom in enumerate(SIDECHAIN_ATOMS_AA_DICT[one_letter_to_aa[ol_aa]]):
        temp_coords = coords[atom_names == atom]
        assert temp_coords.shape[0] == 1
        sidechain_coords[i, :] = temp_coords[0]
    return np.array(sidechain_coords)

def get_CB_coords_for_batch(res_id_N, res_id_to_nb_dict):
    CB_coords = np.full((len(res_id_N), 3), np.nan)
    for i, res_id in enumerate(res_id_N):
        coords = res_id_to_nb_dict[res_id]['coords']
        atom_names = res_id_to_nb_dict[res_id]['atom_names']
        temp_CB_coords = coords[atom_names == 'CB']
        assert temp_CB_coords.shape[0] == 1
        CB_coords[i, :] = temp_CB_coords[0]
    return torch.tensor(CB_coords).float()



def reconstruction_from_collected_chi_angles(dataloader, res_id_to_nb_dict, rec_params_filepath=None, use_true_CB=False):
    if rec_params_filepath is None:
        print('Rec params not provided, using default ones')
        reconstructor = Reconstructor()
    else:
        reconstructor = Reconstructor(rec_params_filepath)

    atomic_reconstruction_error_by_res_id = {}

    derived_norms_by_res_id = {}
    cosine_distance_on_norms_by_res_id = {}

    for batch_i, (X, X_vec, aa_label, angles, vectors, backbone_coords, (rot, data_ids)) in tqdm(enumerate(dataloader), total=len(dataloader)):

        # format the input in the way in which the reconstructor expects it
        atoms = [backbone_coords[:, 0, :].cpu(), backbone_coords[:, 1, :].cpu(), backbone_coords[:, 2, :].cpu(), backbone_coords[:, 3, :].cpu()]
        AA = [ind_to_ol_size[aa_idx.item()] for aa_idx in aa_label]
        normal_vectors = vectors.cpu()
        
        if use_true_CB:
            true_CB = get_CB_coords_for_batch(data_ids, res_id_to_nb_dict)
        else:
            true_CB = None
        
        placed_atoms, derived_norms = reconstructor.reconstruct_from_chi_angles(atoms, AA, angles, true_CB=true_CB)
        placed_atoms = torch.stack(placed_atoms, dim=1).cpu()
        derived_norms = torch.stack(derived_norms, dim=1).cpu()

        for i in range(len(AA)):
            res_id = data_ids[i]
            ol_aa = AA[i]

            pred_atoms_53 = placed_atoms[i]
            true_atoms_53 = get_sidechain_atom_coords(ol_aa, res_id, res_id_to_nb_dict)

            # compute euclidean distance on atoms
            euclidean_error_5 = torch.sqrt(torch.sum(torch.square(pred_atoms_53 - true_atoms_53), dim=1))
            atomic_reconstruction_error_by_res_id[res_id] = euclidean_error_5.cpu().numpy()

            # compute cosine distance error on norms
            derived_norms_53 = derived_norms[i].cpu().numpy()
            collected_norms_53 = normal_vectors[i].cpu().numpy()
            error_on_norms_53 = 1.0 - np.sum((derived_norms_53 / np.linalg.norm(derived_norms_53, axis=-1, keepdims=True)) * (collected_norms_53 / np.linalg.norm(collected_norms_53, axis=-1, keepdims=True)), axis=-1)

            derived_norms_by_res_id[res_id] = derived_norms_53
            cosine_distance_on_norms_by_res_id[res_id] = error_on_norms_53
    
    return atomic_reconstruction_error_by_res_id, derived_norms_by_res_id, cosine_distance_on_norms_by_res_id


def reconstruction_from_collected_norms(dataloader, res_id_to_nb_dict, rec_params_filepath=None, use_true_CB=False):
    if rec_params_filepath is None:
        print('Rec params not provided, using default ones')
        reconstructor = Reconstructor()
    else:
        reconstructor = Reconstructor(rec_params_filepath)

    atomic_reconstruction_error_by_res_id = {}

    derived_chi_angles_by_res_id = {}
    mae_on_chi_angles_by_res_id = {}

    for batch_i, (X, X_vec, aa_label, angles, vectors, backbone_coords, (rot, data_ids)) in tqdm(enumerate(dataloader), total=len(dataloader)):

        # format the input in the way in which the reconstructor expects it
        atoms = [backbone_coords[:, 0, :].cpu(), backbone_coords[:, 1, :].cpu(), backbone_coords[:, 2, :].cpu(), backbone_coords[:, 3, :].cpu()]
        AA = [ind_to_ol_size[aa_idx.item()] for aa_idx in aa_label]
        normal_vectors = vectors.cpu()

        # take out the first norm, since it's the CB norm
        normal_vectors = normal_vectors[:, 1:, :]

        if use_true_CB:
            true_CB = get_CB_coords_for_batch(data_ids, res_id_to_nb_dict)
        else:
            true_CB = None
        
        placed_atoms, derived_chi_angles = reconstructor.reconstruct_from_normal_vectors(atoms, AA, normal_vectors, true_CB=true_CB)
        placed_atoms = torch.stack(placed_atoms, dim=1).cpu()
        derived_chi_angles = torch.stack(derived_chi_angles, dim=1).cpu()

        for i in range(len(AA)):
            res_id = data_ids[i]
            ol_aa = AA[i]

            pred_atoms_53 = placed_atoms[i]
            true_atoms_53 = get_sidechain_atom_coords(ol_aa, res_id, res_id_to_nb_dict)

            # compute euclidean distance on atoms
            euclidean_error_5 = torch.sqrt(torch.sum(torch.square(pred_atoms_53 - true_atoms_53), dim=1))
            atomic_reconstruction_error_by_res_id[res_id] = euclidean_error_5.cpu().numpy()

            # compute mae on chi angles
            derived_chi_angles_4 = derived_chi_angles[i].cpu().numpy()
            collected_chi_angles_4 = angles[i].cpu().numpy()
            naive_error_on_chi_angles_4 = np.abs(derived_chi_angles_4 - collected_chi_angles_4)
            circular_erorr_on_chi_angles_4 = np.minimum(naive_error_on_chi_angles_4, 360 - naive_error_on_chi_angles_4)

            derived_chi_angles_by_res_id[res_id] = derived_chi_angles_4
            mae_on_chi_angles_by_res_id[res_id] = circular_erorr_on_chi_angles_4
            
    return atomic_reconstruction_error_by_res_id, derived_chi_angles_by_res_id, mae_on_chi_angles_by_res_id


def difference_between_our_angles_vs_pyrosettas():
    from sqlitedict import SqliteDict

    our_angles_db = SqliteDict('/gscratch/scrubbed/gvisan01/casp12/chi/chi_angles_and_vectors/angles-easy_task_100pdbs_validation.db')
    pyrosetta_angles_db = SqliteDict('/gscratch/scrubbed/gvisan01/casp12/chi/chi_angles_and_vectors/angles-easy_task_100pdbs_validation_pyrosetta.db')

    our_res_ids = set(our_angles_db.keys())
    pyrosetta_res_ids = set(pyrosetta_angles_db.keys())
    assert our_res_ids <= pyrosetta_res_ids and pyrosetta_res_ids <= our_res_ids

    # filter out res ids from amino-acids we don't care about
    res_ids = [res_id for res_id in our_res_ids if res_id.split('_')[0] in one_letter_to_aa and res_id.split('_')[0] not in {'G', 'A', 'P'}] # skip invalid residues, as well as glycine, alanine, and proline

    error_by_aa = {}
    for res_id in res_ids:
        aa = one_letter_to_aa[res_id.split('_')[0]]
        if aa not in error_by_aa:
            error_by_aa[aa] = []
        error_by_aa[aa].append(np.abs(np.array(our_angles_db[res_id]) - np.array(pyrosetta_angles_db[res_id])))
    
    mae_by_aa = {}
    for aa in error_by_aa:
        mae_by_aa[aa] = np.nanmean(error_by_aa[aa], axis=0)
    
    # sort the amino-acids by size (our standard sorting)
    aas = list(error_by_aa.keys())
    ol_aas = [aa_to_one_letter[aa] for aa in aas]
    size_aas = [ol_to_ind_size[ol_aa] for ol_aa in ol_aas]
    aas = np.array(aas)[np.argsort(size_aas)]

    ncols = 1
    nrows = 1
    fig, axs = plt.subplots(figsize=(10*ncols, 5*nrows), ncols=ncols, nrows=nrows, sharex=False, sharey=True)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # plot euclidean distance of individually-placed atoms
    ax = axs
    ind = np.arange(len(aas))
    width = 0.15
    for i in range(0, 4):
        ax.bar(ind + i*width, [mae_by_aa[aa][i] for aa in aas], width, color=colors[i], label=f'$\\chi_{i+1}$')
    ax.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    ax.set_xticks(ind + (4*width) / 2, aas)
    ax.set_ylabel('Average Absolute Error (Degrees)')
    ax.set_title('Absolute Error Between Our Chi Angles and Pyrosetta\'s Chi Angles')
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/absolute_error_between_our_chi_angles_and_pyrosettas_chi_angles.png')




if __name__ == '__main__':

    batch_size = 16

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
    

    # test difference between chi angles generated by us and by pyrosetta
    difference_between_our_angles_vs_pyrosettas()

    rec_error_from_chi_angles_virtual_CB, derived_norms_from_chi_angles_virtual_CB, norm_error_from_chi_angles_virtual_CB = reconstruction_from_collected_chi_angles(dataloader, res_id_to_nb_dict, use_true_CB=False)
    rec_error_from_chi_angles_real_CB, derived_norms_from_chi_angles_real_CB, norm_error_from_chi_angles_real_CB = reconstruction_from_collected_chi_angles(dataloader, res_id_to_nb_dict, use_true_CB=True)

    rec_error_from_norms_virtual_CB, derived_chi_angles_from_norms_virtual_CB, chi_angle_error_from_norms_virtual_CB = reconstruction_from_collected_norms(dataloader, res_id_to_nb_dict, use_true_CB=False)
    rec_error_from_norms_real_CB, derived_chi_angles_from_norms_real_CB, chi_angle_error_from_norms_real_CB = reconstruction_from_collected_norms(dataloader, res_id_to_nb_dict, use_true_CB=True)

    ## test reconstruction from chi-angles --> looks good!!!
    yaxismax, _ = atomic_reconstruction_barplot_by_aa(rec_error_from_chi_angles_virtual_CB, 'Atomic Reconstruction from Chi Angles  with Virtual CB', 'plots/atomic_reconstruction_from_chi_angles_barplot_by_aa_with_virtual_CB.png')
    atomic_reconstruction_barplot_by_aa(rec_error_from_chi_angles_real_CB, 'Atomic Reconstruction from Chi Angles with Real CB', 'plots/atomic_reconstruction_from_chi_angles_barplot_by_aa_with_real_CB.png', yaxismax=yaxismax)

    ## test recovering norms from chi-angles --> looks good!!!
    yaxismax, _ = norms_error_barplot_by_aa(norm_error_from_chi_angles_virtual_CB, 'Norm Error from Chi Angles with Virtual CB', 'plots/norm_error_from_chi_angles_barplot_by_aa_with_virtual_CB.png')
    norms_error_barplot_by_aa(norm_error_from_chi_angles_real_CB, 'Norm Error from Chi Angles with Real CB', 'plots/norm_error_from_chi_angles_barplot_by_aa_with_real_CB.png', yaxismax=yaxismax)


    ## test reconstruction from norms --> looks good!!!
    yaxismax, _ = atomic_reconstruction_barplot_by_aa(rec_error_from_norms_virtual_CB, 'Atomic Reconstruction from Norms with Virtual CB', 'plots/atomic_reconstruction_from_norms_barplot_by_aa_with_virtual_CB.png')
    atomic_reconstruction_barplot_by_aa(rec_error_from_norms_real_CB, 'Atomic Reconstruction from Norms with Real CB', 'plots/atomic_reconstruction_from_norms_barplot_by_aa_with_real_CB.png', yaxismax=yaxismax)

    ## test recovering chi-angles from norms --> looks good!!!
    yaxismax, _ = chi_angles_error_barplot_by_aa(chi_angle_error_from_norms_virtual_CB, 'Chi Angle Error from Norms with Virtual CB', 'plots/chi_angle_error_from_norms_barplot_by_aa_with_virtual_CB.png')
    chi_angles_error_barplot_by_aa(chi_angle_error_from_norms_real_CB, 'Chi Angle Error from Norms with Real CB', 'plots/chi_angle_error_from_norms_barplot_by_aa_with_real_CB.png', yaxismax=yaxismax)

    ## TODO test cyclically recovering the real chi-angles (chi-angles --> norms --> chi-angles)

    ## TODO test cyclically recovering the real norms (norms --> chi-angles --> norms)


