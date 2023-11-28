
import os, sys
import gzip, pickle
import numpy as np
import matplotlib.pyplot as plt

from src.utils.protein_naming import one_letter_to_aa
from src.sidechain_reconstruction.manual.tests__utils import sort_aas_by_size

from typing import *


def plot_chi_angle_predictions_distributions_vs_true(model_dir: str,
                                                     splits: List[str] = ['valid', 'test']):
    '''
    Inference must be run first, and results saved in model_dir with the standard naming.
    Makes a 20 x 4 plot. Each contains a scatterplot of true vs predicted chi angles, per amino-acid.
    Makes a 20 x 4 plot. Each contains a histogram of the error distribution, per amino-acid.
    '''

    for split in splits:
        assert split in ['train', 'valid', 'test']

    with gzip.open(os.path.join(model_dir, 'per_example_results_dict.gz'), 'rb') as f:
        per_example_results_dict_by_split = pickle.load(f)
    
    os.makedirs(os.path.join(model_dir, 'rec_errors'), exist_ok=True)
    
    for split in splits:
        per_example_results_dict = per_example_results_dict_by_split[split]
        res_ids_N = per_example_results_dict['res_ids']
        true_angles_N4 = per_example_results_dict['true_angles']
        predicted_angles_N4 = per_example_results_dict['predicted_angles']

        savepaths = [os.path.join(model_dir, 'rec_errors', f'{split}_chi_angle_predictions_distributions_vs_true.png'),
                     os.path.join(model_dir, 'rec_errors', f'{split}_chi_angle_predictions_errors_histograms.png')]

        _plot_chi_angle_predictions_distributions_vs_true(res_ids_N, true_angles_N4, predicted_angles_N4, savepaths)



def _plot_chi_angle_predictions_distributions_vs_true(res_ids_N: np.ndarray,
                                                      true_angles_N4: np.ndarray,
                                                      pred_angles_N4: np.ndarray,
                                                      savepaths: List[str]):
    '''
    Makes a 20 x 4 plot. Each contains a scatterplot of true vs predicted chi angles, per amino-acid.
    Makes a 20 x 4 plot. Each contains a histogram of the error distribution, per amino-acid.
    Utility function that can be used outside of our standard evaluation pipeline
    '''

    # first, group by amino acid and by chi angle
    # at this stage, NaNs are still present

    aa_to_chis = {}
    for res_id, true_chis, pred_chis in zip(res_ids_N, true_angles_N4, pred_angles_N4):
        aa = one_letter_to_aa[res_id[0]]
        if aa not in aa_to_chis:
            aa_to_chis[aa] = {'true': [], 'pred': []}
        aa_to_chis[aa]['true'].append(true_chis)
        aa_to_chis[aa]['pred'].append(pred_chis)
    
    for aa in aa_to_chis:
        aa_to_chis[aa]['true'] = np.vstack(aa_to_chis[aa]['true'])
        aa_to_chis[aa]['pred'] = np.vstack(aa_to_chis[aa]['pred'])

    # sort the amino-acids by size (our standard sorting)
    aas = sort_aas_by_size(list(aa_to_chis.keys()))

    ncols = 4
    nrows = len(aas)
    plotsize = 3
    fig, axs = plt.subplots(figsize=(plotsize*ncols, plotsize*nrows), ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    chi_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for row, aa in enumerate(aas):
        for col, chi in enumerate([1, 2, 3, 4]):
            true, pred = aa_to_chis[aa]['true'][:, chi-1], aa_to_chis[aa]['pred'][:, chi-1]

            # condition for invalid chi angle for the given amino acid
            # NOTE: this might be the wrong exact condition
            if np.isnan(true).all() or np.isnan(pred).all():
                continue

            ax = axs[row, col]
            ax.scatter(true, pred, s=8, c=chi_colors[chi-1], alpha=0.5)
            ax.grid(axis='both', ls='--', color='dimgrey', alpha=0.5)
            ax.set_title(f'{aa} $\\chi_{chi}$')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.plot([-180, 180], [-180, 180], c='k', ls='--')
            ax.set_aspect('equal', 'box')

            if row == nrows - 1:
                ax.set_xlabel('True')
            if col == 0:
                ax.set_ylabel('Predicted')

    plt.tight_layout()
    plt.savefig(savepaths[0])

    
    ncols = 4
    nrows = len(aas)
    plotsize = 3
    fig, axs = plt.subplots(figsize=(plotsize*ncols, plotsize*nrows), ncols=ncols, nrows=nrows, sharex=True, sharey=True)

    chi_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for row, aa in enumerate(aas):
        for col, chi in enumerate([1, 2, 3, 4]):
            true, pred = aa_to_chis[aa]['true'][:, chi-1], aa_to_chis[aa]['pred'][:, chi-1]

            # condition for invalid chi angle for the given amino acid
            # NOTE: this might be the wrong exact condition
            if np.isnan(true).all() or np.isnan(pred).all():
                continue
            
            error = np.abs(true - pred)
            error = np.minimum(error, 360 - error) # remember circular error!

            ax = axs[row, col]
            ax.hist(error, color=chi_colors[chi-1])
            ax.grid(axis='x', ls='--', color='dimgrey', alpha=0.5)
            ax.set_title(f'{aa} $\\chi_{chi}$')
            ax.set_xlim(-5, 185)

            if row == nrows - 1:
                ax.set_xlabel('MAE')
            if col == 0:
                ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(savepaths[1])

