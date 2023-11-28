
import numpy as np
import matplotlib.pyplot as plt

import torch

from protein_holography_pytorch.utils.protein_naming import ind_to_ol_size, ol_to_ind_size, one_letter_to_aa, aa_to_one_letter

TEST_RES_IDS = ['Q_1BGX_T_42_ _L', 'F_4XOT_A_244_ _E', 'Y_3EQX_A_297_ _L', 'V_2GVN_A_115_ _E', 'V_3CNU_A_106_ _H', 'T_1JM1_A_123_ _E', 'H_1TXO_A_59_ _H', 'Q_2COV_D_394_ _L', 'L_1JM1_A_53_ _L', 'T_3TG9_A_59_ _H', 'R_1G8M_A_4_ _L', 'Y_1BGX_T_729_ _L', 'L_3O5T_A_23_ _H', 'D_1JM1_A_167_ _E', 'E_3TG9_A_44_ _L', 'L_3SWO_D_382_ _H']

VEC_AA_ATOM_DICT = {
    'ARG' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','NE'], ['CD','NE','CZ']], #, ['NE','CZ','NH1']],
    'ASN' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','OD1']],
    'ASP' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','OD1']],
    'CYS' : [['N','CA','CB'], ['CA','CB','SG']],
    'GLN' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','OE1']],
    'GLU' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','OE1']],
    'HIS' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','ND1']],
    'ILE' : [['N','CA','CB'], ['CA','CB','CG1'], ['CB','CG1','CD1']],
    'LEU' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'LYS' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','CE'], ['CD','CE','NZ']],
    'MET' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','SD'], ['CG','SD','CE']],
    'PHE' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'PRO' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD']], # WARNING: proline is weird
    'SER' : [['N','CA','CB'], ['CA','CB','OG']],
    'THR' : [['N','CA','CB'], ['CA','CB','OG1']],
    'TRP' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'TYR' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'VAL' : [['N','CA','CB'], ['CA','CB','CG1']]
}

SIDECHAIN_ATOMS_AA_DICT = {}
for aa, atoms in VEC_AA_ATOM_DICT.items():
    SIDECHAIN_ATOMS_AA_DICT[aa] = [atom_list[-1] for atom_list in atoms]

def sort_aas_by_size(aas):
    # sort the amino-acids by size (our standard sorting)
    ol_aas = [aa_to_one_letter[aa] for aa in aas]
    size_aas = [ol_to_ind_size[ol_aa] for ol_aa in ol_aas]
    aas = np.array(aas)[np.argsort(size_aas)]
    return aas

def get_sidechain_atom_coords(ol_aa, res_id, res_id_to_nb_dict):
    coords = res_id_to_nb_dict[res_id]['coords']
    atom_names = res_id_to_nb_dict[res_id]['atom_names']
    sidechain_coords = np.full((5, 3), np.nan)
    for i, atom in enumerate(SIDECHAIN_ATOMS_AA_DICT[one_letter_to_aa[ol_aa]]):
        temp_coords = coords[atom_names == atom]
        assert temp_coords.shape[0] == 1
        sidechain_coords[i, :] = temp_coords[0]
    return np.array(sidechain_coords)


def atomic_reconstruction_barplot_by_aa(results_dict, title, savepath, yaxismax=None):
    '''
    results_dict: Dict[res_id --> np.array of euclidean errors of shape (5,), where the first one is the CB error, and the rest are sidechain atom errors]
    title: string title of the plot
    savepath: path to save the plot
    yaxismax: max value for the y-axis, optional
    '''

    results_by_aa = {}
    for res_id, errors in results_dict.items():
        aa = one_letter_to_aa[res_id.split('_')[0]]
        if aa not in results_by_aa:
            results_by_aa[aa] = []
        results_by_aa[aa].append(errors)
    
    for aa in results_by_aa:
        results_by_aa[aa] = np.vstack(results_by_aa[aa])
    
    mean_results_by_aa = {}
    for aa in results_by_aa:
        mean_results_by_aa[aa] = np.nanmean(results_by_aa[aa], axis=0)
    
    # sort the amino-acids by size (our standard sorting)
    aas = sort_aas_by_size(list(results_by_aa.keys()))

    ncols = 1
    nrows = 2
    fig, axs = plt.subplots(figsize=(10*ncols, 5*nrows), ncols=ncols, nrows=nrows, sharex=False, sharey=True)

    colors = ['tab:grey', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # plot euclidean distance of individually-placed atoms
    ax = axs[0]
    ind = np.arange(len(aas))
    width = 0.15
    ax.bar(ind, [mean_results_by_aa[aa][0] for aa in aas], width, color=colors[0], label='CB')
    for i in range(1, 5):
        ax.bar(ind + i*width, [mean_results_by_aa[aa][i] for aa in aas], width, color=colors[i], label=f'$\\chi_{i}$')
    ax.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    if yaxismax is not None: ax.set_ylim(0, yaxismax)
    ax.set_xticks(ind + (5*width) / 2, aas)
    ax.set_ylabel('Average Euclidean Distance (Angstroms)')
    ax.set_title(title)
    ax.legend()

    # plot RMSD (remember that this is just of the individual atoms defining the sidechain, not of the whole sidechain)
    ax = axs[1]
    ind = np.arange(len(aas))
    width = 0.35
    rmsd_no_CB = [np.nanmean(np.sqrt(np.nanmean(np.square(results_by_aa[aa][:, 1:]), axis=0))) for aa in aas]
    rmsd_yes_CB = [np.nanmean(np.sqrt(np.nanmean(np.square(results_by_aa[aa]), axis=0))) for aa in aas]
    ax.bar(ind, rmsd_no_CB, width, color='tab:cyan', label='No CB')
    ax.bar(ind + width, rmsd_yes_CB, width, color='tab:purple', label='Yes CB')
    if yaxismax is not None: ax.set_ylim(0, yaxismax)
    ax.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    ax.set_xticks(ind + width / 2, aas)
    ax.set_ylabel('Average RMSD (Angstroms)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(savepath)

    yaxismax = ax.get_ylim()[1]

    return yaxismax, mean_results_by_aa




def norms_error_barplot_by_aa(results_dict, title, savepath, yaxismax=None):
    '''
    results_dict: Dict[res_id --> np.array of cosine errors of shape (5,), where the first one is the CB error, and the rest are sidechain norms errors]
    title: string title of the plot
    savepath: path to save the plot
    yaxismax: max value for the y-axis, optional
    '''

    results_by_aa = {}
    for res_id, errors in results_dict.items():
        aa = one_letter_to_aa[res_id.split('_')[0]]
        if aa not in results_by_aa:
            results_by_aa[aa] = []
        results_by_aa[aa].append(errors)
    
    for aa in results_by_aa:
        results_by_aa[aa] = np.vstack(results_by_aa[aa])
    
    mean_results_by_aa = {}
    for aa in results_by_aa:
        mean_results_by_aa[aa] = np.nanmean(results_by_aa[aa], axis=0)
    
    # sort the amino-acids by size (our standard sorting)
    aas = sort_aas_by_size(list(results_by_aa.keys()))

    ncols = 1
    nrows = 1
    fig, axs = plt.subplots(figsize=(10*ncols, 5*nrows), ncols=ncols, nrows=nrows, sharex=False, sharey=True)

    colors = ['tab:grey', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # plot euclidean distance of individually-placed atoms
    ax = axs
    ind = np.arange(len(aas))
    width = 0.15
    ax.bar(ind, [mean_results_by_aa[aa][0] for aa in aas], width, color=colors[0], label='CB')
    for i in range(1, 5):
        ax.bar(ind + i*width, [mean_results_by_aa[aa][i] for aa in aas], width, color=colors[i], label=f'$\\chi_{i}$')
    ax.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    if yaxismax is not None: ax.set_ylim(0, yaxismax)
    ax.set_xticks(ind + (5*width) / 2, aas)
    ax.set_ylabel('Average Cosine Distance')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(savepath)

    yaxismax = ax.get_ylim()[1]

    return yaxismax, mean_results_by_aa


def chi_angles_error_barplot_by_aa(results_dict, title, savepath, yaxismax=None):
    '''
    results_dict: Dict[res_id --> np.array of absolute angle errors of shape (4,)]
    title: string title of the plot
    savepath: path to save the plot
    yaxismax: max value for the y-axis, optional
    '''

    results_by_aa = {}
    for res_id, errors in results_dict.items():
        aa = one_letter_to_aa[res_id.split('_')[0]]
        if aa not in results_by_aa:
            results_by_aa[aa] = []
        results_by_aa[aa].append(errors)
    
    for aa in results_by_aa:
        results_by_aa[aa] = np.vstack(results_by_aa[aa])
    
    mean_results_by_aa = {}
    for aa in results_by_aa:
        mean_results_by_aa[aa] = np.nanmean(results_by_aa[aa], axis=0)
    
    # sort the amino-acids by size (our standard sorting)
    aas = sort_aas_by_size(list(results_by_aa.keys()))

    ncols = 1
    nrows = 1
    fig, axs = plt.subplots(figsize=(10*ncols, 5*nrows), ncols=ncols, nrows=nrows, sharex=False, sharey=True)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # plot euclidean distance of individually-placed atoms
    ax = axs
    ind = np.arange(len(aas))
    width = 0.15
    for i in range(0, 4):
        ax.bar(ind + i*width, [mean_results_by_aa[aa][i] for aa in aas], width, color=colors[i], label=f'$\\chi_{i+1}$')
    ax.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    if yaxismax is not None: ax.set_ylim(0, yaxismax)
    ax.set_xticks(ind + (4*width) / 2, aas)
    ax.set_ylabel('Average Absolute Error (Degrees)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(savepath)

    yaxismax = ax.get_ylim()[1]

    return yaxismax, mean_results_by_aa