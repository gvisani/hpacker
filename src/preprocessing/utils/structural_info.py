"""Module for extracting structural info from pyrosetta pose"""

from functools import partial
import logging
from pathlib import Path
import sys
from typing import List,Tuple

import h5py
import numpy as np


## un-comment the following three lines for faster, bulk processing with pyrosetta, and comment out the ones in the get_structural_info_from_protein__pyrosetta() function below
import pyrosetta
init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
pyrosetta.init(init_flags, silent=True)


##################### Copied from https://github.com/nekitmm/DLPacker/blob/main/utils.py

import os, re
from collections import defaultdict

# read in the charges from special file
CHARGES_AMBER99SB = defaultdict(lambda: 0) # output 0 if the key is absent
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charges.rtp'), 'r') as f:
    for line in f:
        if line[0] == '[' or line[0] == ' ':
            if re.match('\A\[ .{1,3} \]\Z', line[:-1]):
                key = re.match('\A\[ (.{1,3}) \]\Z', line[:-1])[1]
                CHARGES_AMBER99SB[key] = defaultdict(lambda: 0)
            else:
                l = re.split(r' +', line[:-1])
                CHARGES_AMBER99SB[key][l[1]] = float(l[3])

################################################################################


# NOTE: this does not include gly and ala because no chi angles
# should this include non-canonical amino acids? How?
# NOTE from GMV: we don't technically need to include the first vector - it is always given since we are given the backbone - but I will keep it bevause it's convenient to have it stored.
#                The last ARG vector is extra, idk why it is there.
#                ILE was missing the last vector, so I added it.
#                I am basine these upon the side-chain angles that William had collectd, which I believe matched DLPacker as well.
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
    'PRO' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD']],
    'SER' : [['N','CA','CB'], ['CA','CB','OG']],
    'THR' : [['N','CA','CB'], ['CA','CB','OG1']],
    'TRP' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'TYR' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'VAL' : [['N','CA','CB'], ['CA','CB','CG1']]
}

def get_norm_vecs(residue) -> np.ndarray:
    """
    Get normal vectors from a residue
    The vectors that can be used to calculate the chi angles in the side chain.
    Uses the tables available at http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html
    
    Parameters
    ----------
    residue : pyrosetta.rosetta.core.conformation.Residue
        The residue to extract vectors from
        
    Returns
    -------
    np.ndarray
        The normal vectors (5 of them as the CB one is included)
        Will be nan if there are no vectors for the residue there
    """
    vecs = np.full((5, 3), np.nan, dtype=float)
    atom_names = VEC_AA_ATOM_DICT.get(residue.name3())
    if atom_names is not None:
        for i in range(len(atom_names)):
            p1 = residue.xyz(atom_names[i][0])
            p2 = residue.xyz(atom_names[i][1])
            p3 = residue.xyz(atom_names[i][2])
            v1 = p1 - p2
            v2 = p3 - p2
            # v1 = p1 - p2
            # v2 = p1 - p3
            x = np.cross(v1, v2)
            vecs[i] = x / np.linalg.norm(x)
    return vecs

def get_chi_angle(plane_norm_1, plane_norm_2, a2, a3):
    
    sign_vec = a3 - a2
    sign_with_magnitude = np.dot(sign_vec, np.cross(plane_norm_1, plane_norm_2))
    sign = sign_with_magnitude / np.abs(sign_with_magnitude)
    
    dot = np.dot(plane_norm_1, plane_norm_2) / (np.linalg.norm(plane_norm_1) * np.linalg.norm(plane_norm_2))
    chi_angle = sign * np.arccos(dot)
    
    return np.degrees(chi_angle)

def get_chi_angles_and_norm_vecs(residue) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get chi angles and normal vectors (which are used to compute chi angles) from a residue.
    Uses the tables available at http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html

    Parameters
    ----------
    residue : pyrosetta.rosetta.core.conformation.Residue
        The residue to extract vectors from
        
    Returns
    -------
    np.ndarray
        The chi angles (4 of them)
        Will be nan if there are no vectors for the residue there

    np.ndarray
        The normal vectors (5 of them as the CB one is included)
        Will be nan if there are no vectors for the residue there
    '''
    vecs = np.full((5, 3), np.nan, dtype=float)
    chis = np.full(4, np.nan, dtype=float)
    atom_names = VEC_AA_ATOM_DICT.get(residue.name3())
    if atom_names is not None:

        for i in range(len(atom_names)):
            p1 = residue.xyz(atom_names[i][0])
            p2 = residue.xyz(atom_names[i][1])
            p3 = residue.xyz(atom_names[i][2])
            v1 = p1 - p2
            v2 = p3 - p2
            # v1 = p1 - p2
            # v2 = p1 - p3
            x = np.cross(v1, v2)
            vecs[i] = x / np.linalg.norm(x)
        
        for i in range(len(atom_names)-1):
            chis[i] = get_chi_angle(vecs[i], vecs[i+1], residue.xyz(atom_names[i][1]), residue.xyz(atom_names[i][2]))

    return chis, vecs

def get_structural_info_from_protein__pyrosetta(pdb_file: str) -> Tuple[
    str,
    Tuple[
        np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray
    ]
]:
    """
    Extract structural information from pyrosetta pose
    
    Parameters
    ----------
    pose : pyrosetta.rosetta.core.pose.Pose
        The pose created for the protein of interest
      
    Returns
    -------
    nested tuple of (bytes, (np.ndarray, np.ndarray, np.ndarray, np.ndarray,
      np.ndarray,np.ndarray)
        This nested tuple contains the pdb name followed by arrays containing
        the atom names, elements, residue ids, coordinates, SASAs, and charges 
        for each atom in the protein.
    """

    ## comment out these three lines for faster, bulk processing with pyrosetta, and uncomment the lines at the top of the script
    # import pyrosetta
    # init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
    # pyrosetta.init(init_flags, silent=True)

    from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.protocols.moves import DsspMover

    from protein_holography_pytorch.preprocessing_faster.utils.pyrosetta import calculate_sasa

    print(f"pdb name in protein routine {pdb_file.split('/')[-1].strip('.pdb')} - start", file=sys.stderr, flush=True)

    pose = pyrosetta.pose_from_pdb(pdb_file)
    
    # lists for each type of information to obtain
    atom_names = []
    elements = []
    sasas = []
    coords = []
    charges_pyrosetta = []
    charges_amber99sb = []
    res_ids = []

    angles_pyrosetta = []
    angles = []
    vecs = []
    res_ids_per_residue = []
    
    k = 0
    
    # extract secondary structure for use in res ids
    DSSP = DsspMover()
    DSSP.apply(pose)
      
    # extract physico-chemical information
    atom_sasa = calculate_sasa(pose)
    coords_rows = pose_coords_as_rows(pose)
    
    pi = pose.pdb_info()
    pdb = pi.name().split('/')[-1].strip('.pdb')
    L = len(pdb)

    logging.debug(f"pdb name in protein routine {pdb} - successfully loaded pdb into pyrosetta")

    # get structural info from each residue in the protein
    for i in range(1, pose.size()+1):
        
        # these data will form the residue id
        aa = pose.sequence()[i-1]
        chain = pi.chain(i)
        resnum = str(pi.number(i)).encode()
        icode = pi.icode(i).encode()
        ss = pose.secstruct(i)
        
        ## optional info to include in residue ids if analysis merits it
        ## - hbond info
        ## - chi1 angle
        #hbond_set = pose.get_hbonds()
        #chi1 = b''
        #print(aa)
        #if aa not in ['G','A','Z']:
        #    try:
        #        chi1 = str(pose.chi(1,i)).encode()
        #    except:
        #        print(pdb,aa,chain,resnum)
        #        #print(chi1)
        chis_pyrosetta = np.full(4, np.nan, dtype=float)
        # if aa not in ['G','A','Z']:
        num_angles = pose.residue_type(i).nchi()
        for chi_num in range(1, 5):
            if (num_angles >= chi_num): 
                chi = str(pose.chi(chi_num,i)).encode()
                chis_pyrosetta[chi_num - 1] = chi  
        angles_pyrosetta.append(chis_pyrosetta)

        chis, norms = get_chi_angles_and_norm_vecs(pose.residue(i))
        angles.append(chis)
        vecs.append(norms)

        res_id = np.array([
                aa,
                pdb,
                chain,
                resnum,
                icode,
                ss,
                #*hb_counts,
                #chi1
        ], dtype=f'S{L}')
        res_ids_per_residue.append(res_id)
        
        for j in range(1,len(pose.residue(i).atoms())+1):

            atom_name = pose.residue_type(i).atom_name(j)
            idx = pose.residue(i).atom_index(atom_name)
            atom_id = (AtomID(idx,i))
            element = pose.residue_type(i).element(j).name
            sasa = atom_sasa.get(atom_id)
            curr_coords = coords_rows[k]
            charge_pyrosetta = pose.residue_type(i).atom_charge(j)

            res_charges = CHARGES_AMBER99SB[pose.residue(i).name3()]
            if isinstance(res_charges, dict):
                charge_amber99sb = CHARGES_AMBER99SB[pose.residue(i).name3()][atom_name.strip().upper()]
            elif isinstance(res_charges, float) or isinstance(res_charges, int): # In which case it is zero because it is an unknown residue. I think it's a Z. pyrosetta ssigns all non-aminoacids a Z... so we lose the charges for those. But oh well, it's not that many probably
                charge_amber99sb = res_charges
            else:
                raise ValueError('Unknown charge type: {}. Something must be wrong.'.format(type(res_charges)))

            #hb_counts = get_hb_counts(hbond_set,i)
            
            res_id = np.array([
                aa,
                pdb,
                chain,
                resnum,
                icode,
                ss,
                #*hb_counts,
                #chi1
            ], dtype=f'S{L}')
            
            atom_names.append(atom_name.strip().upper().ljust(4)) # adding this to make sure all atom names are 4 characters long, because for some atims (the non-residue ones, and maybe others?) it is nsomehow ot the case
            elements.append(element)
            res_ids.append(res_id)
            coords.append(curr_coords)
            sasas.append(sasa)
            charges_pyrosetta.append(charge_pyrosetta)
            charges_amber99sb.append(charge_amber99sb)
            
            k += 1
            
    atom_names = np.array(atom_names,dtype='|S4')
    elements = np.array(elements, dtype='S2')
    sasas = np.array(sasas)
    coords = np.array(coords)
    charges_pyrosetta = np.array(charges_pyrosetta)
    charges_amber99sb = np.array(charges_amber99sb)
    res_ids = np.array(res_ids)
    
    res_ids_per_residue = np.array(res_ids_per_residue)
    angles_pyrosetta = np.array(angles_pyrosetta)
    angles = np.array(angles)
    vecs = np.array(vecs)

    return pdb,(atom_names,elements,res_ids,coords,sasas,charges_pyrosetta,charges_amber99sb,res_ids_per_residue,angles_pyrosetta,angles,vecs)


def get_structural_info_from_protein__biopython(
    pdb_file : str,
    remove_nonwater_hetero: bool = False,
    remove_waters: bool = True,
    ):
    
    '''
    TODO: IN PROGRESS!!!

    atom full id:
        - (PDB, model_num, chain, (hetero_flag, resnum, insertion_code), (atom_name, disorder_altloc))
    
    By default, biopyton selects only atoms with the highest occupancy, thus behaving like pyrosetta does with the flag "-ignore_zero_occupancy false"
    '''

    from Bio.PDB import PDBParser
    parser = PDBParser()

    structure = parser.get_structure(pdb_file[:-4], pdb_file)

    
    aa_to_one_letter = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER':'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

    # assume only one model is present in the structure
    models = list(structure.get_models())
    assert len(models) == 1

    # assume the pdb name was provided as id to create the structure
    pdb = structure.get_id()

    # lists for each type of information to obtain
    atom_names = []
    elements = []
    coords = []
    res_ids = []
    
    k = 0
    
    def pad_for_consistency(string):
        return (' ' + string).ljust(4, ' ')
    
    # get structural info from each residue in the protein
    for residue in structure.get_residues():

        chis_pyrosetta = np.full(4, np.nan, dtype=float)
        # if aa not in ['G','A','Z']:
        num_angles = pose.residue_type(i).nchi()
        for chi_num in range(1, 5):
            if (num_angles >= chi_num): 
                chi = str(pose.chi(chi_num,i)).encode()
                chis_pyrosetta[chi_num - 1] = chi  
        angles_pyrosetta.append(chis_pyrosetta)

        chis, norms = get_chi_angles_and_norm_vecs(pose.residue(i))
        angles.append(chis)
        vecs.append(norms)

        res_id = np.array([
                aa,
                pdb,
                chain,
                resnum,
                icode,
                ss,
                #*hb_counts,
                #chi1
        ], dtype=f'S{L}')
        res_ids_per_residue.append(res_id)

        for atom in residue.get_atoms():

            atom_full_id = atom.get_full_id()
            
            if remove_waters and atom_full_id[3][0] == 'W':
                continue
            
            if remove_nonwater_hetero and atom_full_id[3][0] not in {' ' 'W'}:
                continue

            chain = atom_full_id[2]
            resnum = atom_full_id[3][1]
            icode = atom_full_id[3][2]
            atom_name = pad_for_consistency(atom.get_name())
            element = atom.element
            coord = atom.get_coord()

            aa = atom.get_parent().resname
            if aa in aa_to_one_letter:
                aa = aa_to_one_letter[aa]

            res_id = np.array([aa,pdb,chain,resnum,icode,'null'],dtype='S5') # adding 'null' in place of secondary structure for compatibility
            
            atom_names.append(atom_name)
            elements.append(element)
            res_ids.append(res_id)
            coords.append(coord)
            
            k += 1
            
    atom_names = np.array(atom_names,dtype='|S4')
    elements = np.array(elements,dtype='S2')
    coords = np.array(coords)
    res_ids = np.array(res_ids)
    
    return pdb,(atom_names,elements,res_ids,coords)


# given a matrix, pad it with empty array
def pad(
    arr: np.ndarray,
    padded_length: int=100
) -> np.ndarray:
    """
    Pad an array long axis 0
    
    Parameters
    ----------
    arr : np.ndarray
    padded_length : int

    Returns
    -------
    np.ndarray
    """
    # get dtype of input array
    dt = arr.dtype

    # shape of sub arrays and first dimension (to be padded)
    shape = arr.shape[1:]
    orig_length = arr.shape[0]

    # check that the padding is large enough to accomdate the data
    if padded_length < orig_length:
        print('Error: Padded length of {}'.format(padded_length),
              'is smaller than original length of array {}'.format(orig_length))

    # create padded array
    padded_shape = (padded_length,*shape)
    mat_arr = np.zeros(padded_shape, dtype=dt)

    # add data to padded array
    mat_arr[:orig_length] = np.array(arr)
    
    return mat_arr

def pad_structural_info(
    ragged_structure: Tuple[np.ndarray, ...],
    padded_length: int=100
) -> List[np.ndarray]:
    """Pad structural into arrays"""
    pad_custom = partial(pad,padded_length=padded_length)
    mat_structure = list(map(pad_custom,ragged_structure))

    return mat_structure
