import os, sys
from glob import glob
import numpy as np
import h5py
import hdf5plugin
import torch
from e3nn import o3

from sqlitedict import SqliteDict

from torch.utils.data import Dataset

from typing import *

from src.utils.protein_naming import ol_to_ind_size
GLYCINE, ALANINE = ol_to_ind_size['G'], ol_to_ind_size['A']



def get_norm_factor(projections: np.ndarray, data_irreps: o3.Irreps):
    ls_indices = np.concatenate([[l]*(2*l+1) for l in data_irreps.ls])
    batch_size = 2000
    norm_factors = []
    num_batches = projections.shape[0] // batch_size
    for i in range(num_batches):
        signals = projections[i*batch_size : (i+1)*batch_size]
        batch_norm_factors = np.sqrt(np.einsum('bf,bf,f->b', signals, signals, 1.0 / (2*ls_indices + 1)))
        norm_factors.append(batch_norm_factors)
    
    # final batch for the remaining signals
    if (projections.shape[0] % batch_size) > 0:
        signals = projections[(i+1)*batch_size:]
        batch_norm_factors = np.sqrt(np.einsum('bf,bf,f->b', signals, signals, 1.0 / (2*ls_indices + 1)))
        norm_factors.append(batch_norm_factors)

    norm_factor = np.mean(np.concatenate(norm_factors, axis=-1))

    return norm_factor

def get_data_irreps(hparams):

    # get list of channels. currently, we only really need the length of this list to compute the data irreps
    if hparams['channels'] == 'dlpacker':
        channels = ['C', 'N', 'O', 'S', "all other elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all other AAs"]
    elif hparams['channels'] == 'AAs':
        channels = [b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                    b'S', b'T', b'W', b'Y', b'V', b'G', "all other AAs"]
    else:
        channels = hparams['channels'].split(',')
    
    # construct data irreps from hparams
    mul_by_l = []
    if hparams['radial_func_mode'] == 'ks':
        for l in range(hparams['lmax'] + 1):
            mul_by_l.append((hparams['radial_func_max']+1) * len(channels))
    
    elif hparams['radial_func_mode'] == 'ns':
        ns = np.arange(hparams['radial_func_max'] + 1)
        for l in range(hparams['lmax'] + 1):
            # assuming not keeping zeros... because why would we?
            mul_by_l.append(np.count_nonzero(np.logical_and(np.array(ns) >= l, (np.array(ns) - l) % 2 == 0)) * len(channels) * (2*l + 1))

    data_irreps = o3.Irreps('+'.join([f'{mul}x{l}e' for l, mul in enumerate(mul_by_l)]))
    # print('Data Irreps:', data_irreps)
    ls_indices = np.concatenate([[l]*(2*l+1) for l in data_irreps.ls])

    return data_irreps, ls_indices

def stringify(res_id):
    return '_'.join(list(map(lambda x: x.decode('utf-8'), list(res_id))))

def stringify_array(res_ids):
    return np.array([stringify(res_id) for res_id in res_ids])


class ZernikegramsSidechainDataset(Dataset):
    def __init__(self,
                 zgrams: np.ndarray,
                 irreps: o3.Irreps,
                 aa_labels: np.ndarray,
                 chi_angles: np.ndarray,
                 norm_vectors: np.ndarray,
                 backbone_coords: np.ndarray,
                 c: List):
        self.zgrams = zgrams # [N, dim]
        self.aa_labels = aa_labels # [N,]
        self.chi_angles = chi_angles # [N, 4]
        self.norm_vectors = norm_vectors # [N, 5, 3] # --> the first norm belongs to the backbone+CB. we're not learning it since we can virtualize CB almost exactly. but we could also learn it
        self.backbone_coords = backbone_coords # [N, 4, 3]
        self.c = c # [N, ANY]
        assert zgrams.shape[0] == aa_labels.shape[0]
        assert zgrams.shape[0] == chi_angles.shape[0]
        assert zgrams.shape[0] == norm_vectors.shape[0]
        assert zgrams.shape[0] == backbone_coords.shape[0]
        assert zgrams.shape[0] == len(c)
        assert zgrams.shape[1] == irreps.dim
        assert chi_angles.shape[1] == 4
        assert norm_vectors.shape[1] == 5
        assert norm_vectors.shape[2] == 3
        assert backbone_coords.shape[1] == 4
        assert backbone_coords.shape[2] == 3

        self.ls_indices = np.hstack([np.array([l]).repeat(2*l+1) for l in irreps.ls])
        self.unique_ls = sorted(list(set(irreps.ls)))
    
    def __len__(self):
        return self.zgrams.shape[0]
    
    def __getitem__(self, idx: int):
        zgrams_fiber = {}
        for l in self.unique_ls:
            zgrams_fiber[l] = self.zgrams[idx][self.ls_indices == l].reshape(-1, 2*l+1)
        
        return zgrams_fiber, \
               self.zgrams[idx], \
               self.aa_labels[idx], \
               self.chi_angles[idx], \
               self.norm_vectors[idx], \
               self.backbone_coords[idx], \
               self.c[idx]


def filter_out_bad_data(zgrams, labels, backbone_coords, chi_angles, norm_vectors, res_ids, frames):

    mask = np.logical_and(~np.logical_or.reduce(np.isnan(zgrams), axis=-1), ~np.logical_or.reduce(np.isinf(zgrams), axis=-1))
    mask = np.logical_and(mask, ~np.logical_and.reduce(zgrams == 0.0, axis=-1))
    print('There are %d Zernikegrams with NaNs or Infs or zeros. Filtering them out.' % np.sum(~mask))

    mask_yes_sidechain = ~np.isin(labels, [GLYCINE, ALANINE])
    print('There are %d Zernikegrams with no sidechain. Filtering them out.' % np.sum(~mask_yes_sidechain))
    mask = np.logical_and(mask, mask_yes_sidechain)

    print(mask.shape)
    zgrams = zgrams[mask]
    labels = labels[mask]
    res_ids = res_ids[mask]
    frames = frames[mask]
    backbone_coords = backbone_coords[mask]
    chi_angles = chi_angles[mask]
    norm_vectors = norm_vectors[mask]

    return zgrams, labels, backbone_coords, chi_angles, norm_vectors, res_ids, frames


def load_single_split_data(hparams, split, get_norm_factor_if_training=True, test_data_filepath=None):

    assert split in {'validation', 'testing'} or 'training' in split # accomodate for divided training data, usually of the kind 'training__N'

    data_irreps, ls_indices = get_data_irreps(hparams)

    if test_data_filepath is not None:
        raise NotImplementedError('The use of "test_data_filepath" is not implemented yet for the new standardized data loading procedure.')

    pdb_list_filename = hparams['pdb_list_filename_template'].format(split=split)

    with h5py.File(hparams['data_filepath'].format(pdb_list_filename=pdb_list_filename, **hparams), 'r') as f:
        data = f['data'][:]

        zgrams = data['zernikegram']
        labels = data['label']
        res_ids = np.array(list(map(stringify, data['res_id'])))
        pdbs = data['res_id'][:, 1]
        backbone_coords = data['backbone_coords']
        chi_angles = data['chi_angles']
        norm_vectors = data['norm_vecs']
        try:
            frames = f['data']['frame']
        except Exception as e:
            print('Warning: no frames.', file=sys.stderr)
            print(e)
            frames = np.zeros((labels.shape[0], 3, 3))
    
    # zgrams, labels, backbone_coords, chi_angles, norm_vectors, res_ids, frames = filter_out_bad_data(zgrams, labels, backbone_coords, chi_angles, norm_vectors, res_ids, frames)

    if 'training' in split and get_norm_factor_if_training and hparams['normalize_input']:
        ##### debugging #####
        power = np.mean(np.sqrt(np.einsum('bf,bf,f->b', zgrams[:1000], zgrams[:1000], 1.0 / (2*ls_indices + 1))))
        print('Power before norm: %.4f' % power)
        sys.stdout.flush()
        ##### debugging #####

        size = zgrams.shape[0]
        num_to_use = 100_000
        idxs = np.random.default_rng(1234567890).integers(size, size=num_to_use)
        print(f'Getting norm factor using a random sample with fixed seed of {num_to_use} zgrams.', flush=True)
        norm_factor = get_norm_factor(zgrams[idxs], data_irreps)
        # zgrams = zgrams / norm_factor # ---> we don't divide it now anymore!
        print('Done, norm_factor:', norm_factor)
        
        ##### debugging #####
        power = np.mean(np.sqrt(np.einsum('bf,bf,f->b', zgrams[:1000] / norm_factor, zgrams[:1000] / norm_factor, 1.0 / (2*ls_indices + 1))))
        print('Power after norm: %.4f' % power)
        sys.stdout.flush()
        ##### debugging #####
    else:
        norm_factor = None

    print('Running on %s set with %d examples.' % (split, zgrams.shape[0]))

    bad_pdbs = np.array([b'3zhi', b'5apz', b'4hry', b'5yqr', b'4xsj', b'3vw7', b'3laz', b'2dpq', b'4hp2'])
    print(pdbs.shape, bad_pdbs.shape)
    print(pdbs[:10], bad_pdbs)
    mask = ~np.isin(pdbs, bad_pdbs)
    print(f'Removing {np.sum(~mask)} examples from {split} set because they belong to structures above 50% sequence-similar to CASP13 and CASP14 structure.')

    zgrams = zgrams[mask]
    labels = labels[mask]
    res_ids = res_ids[mask]
    frames = frames[mask]
    backbone_coords = backbone_coords[mask]
    chi_angles = chi_angles[mask]
    norm_vectors = norm_vectors[mask]

    dataset = ZernikegramsSidechainDataset(zgrams,
                                            data_irreps,
                                            labels,
                                            chi_angles,
                                            norm_vectors,
                                            backbone_coords,
                                            list(zip(list(frames), list(res_ids))))
    return dataset, data_irreps, norm_factor


def load_data(hparams, splits=['train', 'valid'], get_norm_factor_if_training=True, test_data_filepath=None):
    
    for split in splits:
        assert split in {'train', 'valid', 'test'}
        
    norm_factor = None
    datasets = {}
    if 'train' in splits:
        train_dataset, data_irreps, norm_factor = load_single_split_data(hparams, 'training', get_norm_factor_if_training=get_norm_factor_if_training)
        datasets['train'] = train_dataset
    
    if 'valid' in splits:
        valid_dataset, data_irreps, _ = load_single_split_data(hparams, 'validation')
        datasets['valid'] = valid_dataset
    
    if 'test' in splits:
        test_dataset, data_irreps, _ = load_single_split_data(hparams, 'testing')
        datasets['test'] = test_dataset

    return datasets, data_irreps, norm_factor

