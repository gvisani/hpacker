"""Module for parallel processing of pdb files into structural info"""

from argparse import ArgumentParser
import logging
import os, sys
from time import time
from typing import Tuple

import h5py
from hdf5plugin import LZ4
import numpy as np
import pandas as pd
from progress.bar import Bar

from src.preprocessing.utils.structural_info import (
    get_structural_info_from_protein__pyrosetta, pad_structural_info
)
from src.preprocessing.preprocessors.preprocessor_pdbs import PDBPreprocessor
from src.utils.log_config import format
# from protein_holography_pytorch.utils.posterity import get_metadata,record_metadata

import json
from sqlitedict import SqliteDict # for storing angles data

from typing import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=format)



def get_structural_info(pdb_file: Union[str, List[str]],
                        padded_length: int=200000,
                        parser: str = 'pyrosetta',
                        which_charges: str = 'pyrosetta'):

    """ Get structural info from a single pdb file """

    assert parser in {'pyrosetta', 'biopython'}, f'Parser cannot be {parser}'
    assert which_charges in {'pyrosetta', 'amber99sb'}, f'Charges cannot be {which_charges}'

    if isinstance(pdb_file, str):
        L = len(pdb_file.split('/')[-1].split('.')[0])
    else:
        L = len(pdb_file[0].split('/')[-1].split('.')[0])
        for i in range(1, len(pdb_file)):
            L = max(L, len(pdb_file[i].split('/')[-1].split('.')[0]))

    dt = np.dtype([
        ('pdb',f'S{L}',()),
        ('atom_names', 'S4', (padded_length)),
        ('elements', 'S2', (padded_length)),
        ('res_ids', f'S{L}', (padded_length, 6)),
        ('coords', 'f4', (padded_length, 3)),
        ('SASAs', 'f4', (padded_length)),
        ('charges', 'f4', (padded_length)),
    ])

    if isinstance(pdb_file, str):
        pdb_file = [pdb_file]
    
    np_protein = np.zeros(shape=(len(pdb_file),), dtype=dt) 

    n = 0
    for i, pdb_file in enumerate(pdb_file):

        si = get_padded_structural_info(pdb_file, padded_length=padded_length, parser=parser)
        if si[0] is None:
            continue

        pdb,atom_names,elements,res_ids,coords,sasas,charges_pyrosetta,charges_amber99sb,res_ids_per_residue,angles_pyrosetta,angles,norm_vecs = si

        if which_charges == 'pyrosetta':
            np_protein[n] = (pdb,atom_names,elements,res_ids,coords,sasas,charges_pyrosetta,)
        else:
            np_protein[n] = (pdb,atom_names,elements,res_ids,coords,sasas,charges_amber99sb,)
        
        n += 1

    np_protein.resize((n,))

    return np_protein


def get_padded_structural_info(
    pdb_file: str, padded_length: int=200000, parser: str = 'pyrosetta') -> Tuple[
    bytes,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Extract structural info used for holographic projection from PyRosetta pose.
    
    Parameters
    ----------
    pose : pyrosetta.rosetta.core.pose.Pose
        Pose created by PyRosetta from pdb file
        
    Returns
    -------
    tuple of (bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
              np.ndarray)
        The entries in the tuple are
            bytes encoding the pdb name string
            bytes array encoding the atom names of shape [max_atoms]
            bytes array encoding the elements of shape [max_atoms]
            bytes array encoding the residue ids of shape [max_atoms,6]
            float array of shape [max_atoms,3] representing the 3D Cartesian 
              coordinates of each atom
            float array of shape [max_atoms] storing the SASA of each atom
            float array of shape [max_atoms] storing the partial charge of each atom
    """

    try:
        if parser == 'biopython':
            raise NotImplementedError("Use of Biopython parser not implemented yet")
        elif parser == 'pyrosetta':
            pdb, ragged_structural_info = get_structural_info_from_protein__pyrosetta(pdb_file)

        mat_structural_info = pad_structural_info(
            ragged_structural_info,padded_length=padded_length
            )
    except Exception as e:
        logger.error(f"Failed to process {pdb_file}")
        logger.error(e)
        return (None,)

    return (pdb, *mat_structural_info)


def get_structural_info_from_dataset(
    pdb_list_file: str,
    pdb_dir: str,
    max_atoms: int,
    which_charges: str,
    hdf5_out: str,
    output_dataset_name: str,
    parallelism: int,
    angle_db: str = None,
    vec_db: str = None,
    logging_level=logging.INFO
):
    """
    Parallel processing of pdbs into structural info
    
    Parameters
    ---------
    pdb_list : str
        path to csv file containing list of pdbs, under the column name 'pdb'
    pdb_dir : str
        Path where the pdb files are stored
    max_atoms : int
        Max number of atoms in a protein for padding purposes
    hdf5_out : str
        Path to hdf5 file to write
    parlellism : int
        Number of workers to use
    """

    logger.setLevel(logging_level)

    # metadata = get_metadata()
    
    with open(pdb_list_file, 'r') as f:
        pdb_list = [pdb.strip() for pdb in f.readlines()]

    pdb_list_from_dir = []
    for file in os.listdir(pdb_dir):
        if file.endswith(".pdb"):
            pdb = file.strip('.pdb')
            pdb_list_from_dir.append(pdb)
    
    # filter out pdbs that are not in the directory
    pdb_list = list(set(pdb_list) & set(pdb_list_from_dir))
    
    ds = PDBPreprocessor(pdb_list, pdb_dir)
    bad_neighborhoods = []
    L = np.max([ds.pdb_name_length, 5])
    logger.info(f"Maximum pdb name L = {L}")
    
    dt = np.dtype([
        ('pdb', f'S{L}',()),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S2', (max_atoms)),
        ('res_ids', f'S{L}', (max_atoms,6)),
        ('coords', 'f4', (max_atoms,3)),
        ('SASAs', 'f4', (max_atoms)),
        ('charges', 'f4', (max_atoms)),
    ])
    
    with h5py.File(hdf5_out,'w') as f:
        f.create_dataset(output_dataset_name,
                         shape=(ds.size,),
                         dtype=dt,
                         chunks=True,
                         compression=LZ4())
        # record_metadata(metadata, f[output_dataset_name])

    pyrosetta_angle_dict = {}
    angle_dict = {}
    vec_dict = {}

    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(hdf5_out,'r+') as f:
            n = 0
            for i,structural_info in enumerate(ds.execute(
                    get_padded_structural_info,
                    limit = None,
                    params = {'padded_length': max_atoms},
                    parallelism = parallelism)):
                if structural_info[0] is None:
                    bar.next()
                    continue
                try:
                    pdb,atom_names,elements,res_ids,coords,sasas,charges_pyrosetta,charges_amber99sb,res_ids_per_residue,angles_pyrosetta,angles,norm_vecs = (*structural_info,)
                    for res_id, curr_angles_pyrosetta, curr_angles, curr_norm_vecs in zip(res_ids_per_residue, angles_pyrosetta, angles, norm_vecs):
                        res_id = '_'.join([id_.decode("utf-8") for id_ in res_id])
                        pyrosetta_angle_dict[res_id] = curr_angles_pyrosetta.tolist()
                        angle_dict[res_id] = curr_angles.tolist()
                        vec_dict[res_id] = curr_norm_vecs.tolist()
                    
                    if which_charges == 'pyrosetta':
                        charges = charges_pyrosetta
                    else:
                        charges = charges_amber99sb
                    
                    f[output_dataset_name][n] = (
                        pdb, atom_names, elements,
                        res_ids, coords, sasas, charges
                    )
                    logger.info(f"Wrote to hdf5 for pdb = {pdb}")
                except Exception as e:
                    print(e, file=sys.stderr)
                    bar.next()
                    continue

                n+=1
                bar.next()
            
            print(f"----------> n = {n}")
            f[output_dataset_name].resize((n,)) # resize to account for errors

        if angle_db is not None:
            # # save angles computed by pyrosetta
            # pyrosetta_angle_db = angle_db.replace(".db", "_pyrosetta.db")
            # pyrosetta_angle_db = SqliteDict(pyrosetta_angle_db, autocommit=False)
            # for k, v in pyrosetta_angle_dict.items():
            #     pyrosetta_angle_db[k] = v
            # pyrosetta_angle_db.commit()
            # pyrosetta_angle_db.close()

            # save angles computed by us
            angle_db = SqliteDict(angle_db, autocommit=False)
            for k, v in angle_dict.items():
                angle_db[k] = v
            angle_db.commit()
            angle_db.close()
        
        if vec_db is not None:
            vec_db = SqliteDict(vec_db, autocommit=False)
            for k, v in vec_dict.items():
                vec_db[k] = v
            vec_db.commit()
            vec_db.close()

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--hdf5_out', type=str,
        help='Output hdf5 filename, where structural info will be stored.',
        required=True
    )
    parser.add_argument(
        '--output_dataset_name', type=str,
        help='Name of the dataset within output_hdf5 where the structural information will be saved. We recommend keeping this set to simply "data".',
        default='data'
    )
    parser.add_argument(
        '--pdb_list_file', type=str,
        help='Path to file containing list of PDB files of interest, one per row.',
        required=True
    )
    parser.add_argument(
        '--pdb_dir', type=str,
        help='directory of pdb files',
        required=True
    )
    parser.add_argument(
        '--parallelism', type=int,
        help='ouptput file name',
        default=4
    )
    parser.add_argument(
        '--max_atoms', type=int,
        help='max number of atoms per protein for padding purposes',
        default=200000
    )
    parser.add_argument(
        '--which_charges', type=str,
        help="Which charges to use. pyrosetta is used in our HCNN models. amber99sb is used by the sidechain predictors, only because it's easier to use them on-the-fly during reconstruction.",
        default='pyrosetta',
        choices=['pyrosetta', 'amber99sb']
    )
    parser.add_argument(
        '--logging', type=str,
        help='logging level',
        default="INFO"
    )
    parser.add_argument(
        '--angle_db', type=str, 
        help='path to chi angle database',
        default=None
    )
    parser.add_argument(
        '--vec_db', type=str, 
        help='path to normal vector database',
        default=None
    )
    
    args = parser.parse_args()

    # os.environ['NUMEXPR_MAX_THREADS'] = '4' #str(args.parallelism)

    get_structural_info_from_dataset(
        args.pdb_list_file,
        args.pdb_dir,
        args.max_atoms,
        args.which_charges,
        args.hdf5_out,
        args.output_dataset_name,
        args.parallelism,
        args.angle_db,
        args.vec_db,
        logging_level=eval(f'logging.{args.logging}')
    )

if __name__ == "__main__":
    start_time=time()
    main()
    print(f"Total time = {time() - start_time:.2f} seconds")
