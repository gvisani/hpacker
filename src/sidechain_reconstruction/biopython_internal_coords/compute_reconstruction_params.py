
'''

This scripts computes reconstruction parameters from data.

By reconstruction parameters we mean the values of the internal coordinates: bond lengths, bond angles, and dihedral angles - which are either *somewhat constant* or can be *aproximately deterministically determined* from the backbone atoms and the chi angles.

There are multiple equivalent "parameters" that equally well describe the sidechain.
As our goal is to leverage Biopython's internal_coord module to do the reconstruction, we compute the paramaters that it expects in order to reconstruct the sidechain.

'''

import os
import json
import numpy as np

import itertools

from Bio.PDB import PDBParser, Selection

# the lines below hide neverending warnings from biopython
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio import BiopythonDeprecationWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
warnings.simplefilter('ignore', BiopythonDeprecationWarning)

from tqdm import tqdm

from typing import *

try:
    from .constants import *
except ImportError:
    from constants import *

INTERNAL_COORDS_NAMES_BY_RESNAME_FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'internal_coords_atomic_combos_by_resname.json')
REC_PARAMS_FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reconstruction_params.json')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_list', type=str, default='/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/dlpacker_training__0.txt',
                        help='Path to the list of pdbs to use for computing the reconstruction parameters. Must provide one PDB ID per line.')
    parser.add_argument('--pdbdir', type=str, default='/gscratch/scrubbed/gvisan01/dlpacker/pdbs')
    args = parser.parse_args()

    with open(args.pdb_list, 'r') as f:
        pdb_list = f.read().splitlines()
    
    compute_biopyton_reconstruction_parameters(pdb_list, args.pdbdir)


def compute_biopyton_reconstruction_parameters(pdb_list: List[str], pdbdir: str) -> Dict[str, Dict[str, Union[float, Tuple[str, float]]]]:
    '''
    Master function.

    Structure of output (also saved in REC_PARAMS_FILEPATH):
    {
        resname: { 
            atomic_combo: either a single float corresponding to the median value to use,
                          or a Tuple[str, float] corresponding respectively to the reference dihedral angle and the median offset value between the current atomic_combo (always a dihedral) and the reference dihedral angle (which is either a backbone dihedral, or a chi angle)
        }
    }
    '''
    
    ## compute internal_coords_atomic_combos_by_resname if it doesn't already exist
    if not os.path.exists(INTERNAL_COORDS_NAMES_BY_RESNAME_FILEPATH):
        print('Pre-computed names of atomic_combos not found. Computing them...')
        internal_coords_atomic_combos_by_resname = _precompute_internal_coords_atomic_combos_by_resname(pdbdir)
        with open(INTERNAL_COORDS_NAMES_BY_RESNAME_FILEPATH, 'w') as f:
            json.dump(internal_coords_atomic_combos_by_resname, f, indent=4)
        print('Done')
    else:
        print('Using pre-computed names of atomic_combos...')
        with open(INTERNAL_COORDS_NAMES_BY_RESNAME_FILEPATH, 'r') as f:
            internal_coords_atomic_combos_by_resname = json.load(f)
    
    ## compute internal_coords_trace --> statistics on all the internal coordinates
    print('Computing internal coordinates for all pdbs...')
    internal_coords_trace = _get_internal_coords_for_pdbs(pdb_list, pdbdir, internal_coords_atomic_combos_by_resname)
    print('Done')

    ## add the related dihedrals to reconstruction_params
    reconstruction_params = {}
    for resname in internal_coords_trace:

        if resname not in reconstruction_params:
            reconstruction_params[resname] = {}
        
        ## add the related dihedrals to reconstruction_params
        if resname in DESIRED_DIHEDRAL_TO_REFERENCE_DIHEDRAL:
            for desired_dihedral in DESIRED_DIHEDRAL_TO_REFERENCE_DIHEDRAL[resname]:
                reference_dihedral = DESIRED_DIHEDRAL_TO_REFERENCE_DIHEDRAL[resname][desired_dihedral]
                reconstruction_params[resname][desired_dihedral] = (reference_dihedral, _compute_dihedral_offset(internal_coords_trace[resname][reference_dihedral], internal_coords_trace[resname][desired_dihedral]))
        
        ## deal with the Arginine special case. it has two symmetric dihedral angles. ick 180 for one and 0 for the other
        ## it doens't matter which, because we will consider this symmetry in the computation of the loss
        if resname == 'ARG':
            reconstruction_params[resname]['CD:NE:CZ:NH1'] = 179.9 # some fuzziness to avoid collinearity
            reconstruction_params[resname]['CD:NE:CZ:NH2'] = 0.1 # some fuzziness to avoid collinearity
        
        ## add everything else as the median value. exclude backbone-only values, though! we don't need them anymore, we only needed N:CA:C:O in the related-dihedrals step. also exclude chi angles of course
        ## NOTE: doing this after the related-dihedrals is important, because we don't want to substitute them out with the median value, which would be wrong
        for atomic_combo in internal_coords_trace[resname]:

            if _is_only_backbone(atomic_combo) or 'chi' in atomic_combo or atomic_combo in reconstruction_params[resname]:
                continue
            
            if len(atomic_combo.split(':')) == 2: # bond length
                median_value = np.median(internal_coords_trace[resname][atomic_combo])
            else: # bond angle or dihedral angle
                median_value = _compute_median_for_angles(internal_coords_trace[resname][atomic_combo])
            
            reconstruction_params[resname][atomic_combo] = median_value
        
        ## add side-chain atoms, for convenience during the reconstruction
        reconstruction_params[resname]['sidechain atoms'] = SIDE_CHAINS[resname]
    
    print(f'Saving reconstruction parameters to {REC_PARAMS_FILEPATH}...')
    with open(REC_PARAMS_FILEPATH, 'w') as f:
        json.dump(reconstruction_params, f, indent=4)
    print('Done')
    
    return reconstruction_params



def _compute_median_for_angles(values):
    '''
    Computes the median value for angles, taking into account the case of +/- 180 degrees, which is a special case.
    '''

    if (np.max(values) - np.min(values)) > 355: # +/- 180 case! pick 180
        robust_median = 179.9 # some fuzziness to avoid collinearity
    else:
        robust_median = np.median(values)
    
    return robust_median


def _compute_dihedral_offset(reference_dihedral_values, desired_dihedral_values):
    '''
    reference_dihedral_values: values of the reference dihedral
    desired_dihedral_values: values of the dihedral to compare to the reference dihedral

    returns: median offset between the two dihedrals
    '''
    assert len(reference_dihedral_values) == len(desired_dihedral_values)

    diff = np.mod( (np.asarray(desired_dihedral_values) - np.asarray(reference_dihedral_values)) + 180, 360) - 180

    dihedral_offset = _compute_median_for_angles(diff)
    
    return dihedral_offset


def _precompute_internal_coords_atomic_combos_by_resname(pdbdir):
    '''
    NOTE: We also compute the backbone internal_coords here, both as a reminder of the ones Biopython uses, and also because we need N:CA:C:O values to compute the offset for O:C:CA:CB

    NOTE: this function is technically NOT GUARANTEED to find the accurate atomic combos. Technically seaking, using different PDBs might give different results, if we're particularly unlucky.
    In practice, I don't think it will ever be an issue. But worth noting here because this might be a spot to check if some weird bug happens downstream.
    '''

    pdbs = list(filter(lambda x: x.endswith('.pdb'), os.listdir(pdbdir)))[:10]
    parser = PDBParser()

    internal_coords_atomic_combos_by_resname = {aa: set() for aa in THE20}

    for pdb in tqdm(pdbs):

        structure = parser.get_structure(pdb.strip('.pdb'), os.path.join(pdbdir, pdb))
        structure.atom_to_internal_coordinates()

        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.resname in THE20:
                if not _only_has_backbone_and_CB_but_shouldnt(residue):
                    # sometimes, some residues miss the some internal coords... idk why, but, in order to avoid missing them here, I will re-compute them for multiple instances
                    # of the same residue, and keep the union of them. we do this for a few pdbs and that should be more than enough

                    try:
                        internal_coords_atomic_combos_by_resname[residue.resname] = internal_coords_atomic_combos_by_resname[residue.resname].union(set(_get_internal_coords_from_scratch(residue, get_backbone=True).keys()))
                    except AttributeError:
                        # sometimes the reisdue.internal_coord is None, for some reason --> NOTE: it seems to happen for the non-null insertion codes
                        print('Warning, residue.internal_coord is None for', residue.full_id, residue.resname)
                        continue
    
    for key in internal_coords_atomic_combos_by_resname:
        internal_coords_atomic_combos_by_resname[key] = list(internal_coords_atomic_combos_by_resname[key])
            
    return internal_coords_atomic_combos_by_resname


def _get_internal_coords_for_pdbs(pdb_list, pdbdir, internal_coords_atomic_combos):
    parser = PDBParser()

    internal_coords_trace = {resname: {} for resname in internal_coords_atomic_combos.keys()}

    for pdb in tqdm(pdb_list):
        try:
            structure = parser.get_structure(pdb, os.path.join(pdbdir, f'{pdb}.pdb'))
        except Exception as e:
            print(f'Error parsing {pdb}: {e}')
            continue
            
        structure.atom_to_internal_coordinates()

        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.resname in THE20:
                if not _only_has_backbone_and_CB_but_shouldnt(residue):
                    try:
                        internal_coords = _get_internal_coords_from_precomputed_keys(residue, internal_coords_atomic_combos[residue.resname], verbose=False)
                    except AttributeError:
                        # sometimes the reisdue.internal_coord is None, for some reason --> NOTE: it seems to happen for the non-null insertion codes
                        print('Warning, residue.internal_coord is None for', residue.full_id, residue.resname)
                        continue

                    if None in internal_coords.values(): # faulty residue that is missing some internal coords
                        continue

                    for atomic_combo in internal_coords:
                        if atomic_combo not in internal_coords_trace[residue.resname]:
                            internal_coords_trace[residue.resname][atomic_combo] = []
                        internal_coords_trace[residue.resname][atomic_combo].append(internal_coords[atomic_combo])
    
    return internal_coords_trace


def _get_internal_coords_from_precomputed_keys(residue, internal_coords_atomic_combos, verbose=False):

    internal_coords = {}
    for atomic_combo in internal_coords_atomic_combos:
        if len(atomic_combo.split(':')) == 2:
            value = residue.internal_coord.get_length(atomic_combo)
        elif len(atomic_combo.split(':')) in {3, 4} or 'chi' in atomic_combo:
            value = residue.internal_coord.get_angle(atomic_combo)
        else:
            raise ValueError(f'Atomic combo {atomic_combo} is not of length 2, 3, or 4')
        
        if value is None and verbose:
            print(f'Warning, None encountered for {atomic_combo} in {residue.full_id} - {residue.resname}')
        
        internal_coords[atomic_combo] = value
    
    return internal_coords


def _get_internal_coords_from_scratch(residue, get_backbone=True):

    internal_coords = {}
    for atomic_combo in _all_possible_atomic_combos(residue.resname, 2):
        if not _is_only_backbone(atomic_combo) or get_backbone:
            for perm in _all_permutations(atomic_combo):
                value = residue.internal_coord.get_length(perm)
                if value is not None:
                    internal_coords[perm] = value
                    break

    for atomic_combo in _all_possible_atomic_combos(residue.resname, 3):
        if not _is_only_backbone(atomic_combo) or get_backbone:
            for perm in _all_permutations(atomic_combo):
                value = residue.internal_coord.get_angle(perm)
                if value is not None:
                    internal_coords[perm] = value
                    break
    
    for atomic_combo in _all_possible_atomic_combos(residue.resname, 4):
        if not _is_only_backbone(atomic_combo) or get_backbone:
            for perm in _all_permutations(atomic_combo):
                value = residue.internal_coord.get_angle(perm)
                if value is not None:
                    internal_coords[perm] = value
                    break
    
    for i in range(1, 5):
        value = residue.internal_coord.get_angle(f'chi{i}')
        if value is not None:
            internal_coords[f'chi{i}'] = value
    
    internal_coords = _remove_chi_angle_atomic_combos(residue, internal_coords)
    
    return internal_coords


def _remove_chi_angle_atomic_combos(residue, internal_coords):
    '''
    If the atomic_combo is a chi angle, substitute the chi angle name
    '''
    if residue.resname in {'GLY', 'ALA'}: return internal_coords

    atomic_combos = list(internal_coords.keys())
    for i, chi_angle in enumerate(CHI_ANGLES[residue.resname]):
        for perm in _all_permutations(':'.join(chi_angle)):
            if perm in atomic_combos:
                del internal_coords[perm]
    return internal_coords


def _all_permutations(atomic_combo):
    return list(map(lambda x: ':'.join(x), list(itertools.permutations(atomic_combo.split(':')))))


def _all_possible_atomic_combos(resname, num_atoms):
    return list(map(lambda x: ':'.join(x), list(itertools.combinations(BB_ATOMS + SIDE_CHAINS[resname], num_atoms))))


def _is_only_backbone(atomic_combo):
    return all(map(lambda x: x in BB_ATOMS, atomic_combo.split(':')))


def _is_only_backbone_and_CB(atomic_combo):
    return all(map(lambda x: x in BB_ATOMS + ['CB'], atomic_combo.split(':')))


def _only_has_backbone_and_CB_but_shouldnt(residue):
    '''
    Catches the case in which the residue does not appear to have sidechain atoms for some reason
    '''
    if residue.resname in {'GLY', 'ALA'}: return False
    
    for atom in residue.get_atoms():
        if atom.id not in {'N', 'CA', 'C', 'O', 'CB'}:
            return False
    return True


if __name__ == '__main__':
    main()

    