import functools
import itertools
import logging
from multiprocessing import Pool, TimeoutError
import os
import signal
import sys
import time
from typing import Callable, List
import warnings

import h5py
import numpy as np

# def process_data(pdb: str, pdb_dir: str) -> np.ndarray:
#     """
#     Process data.

#     Parameters
#     ----------
#     pdb : str
#         Name of PDB file.
#     pdb_dir : str
#         Directory the PDB is in.

#     Returns
#     -------
#     numpy.ndarray
#         The processed pose.
#     """
#     assert(process_data.callback)

#     pdb = pdb.decode('utf-8')
#     # for ext in [".cif",   ".pdb"]:
#     ext = ".pdb"
#     pose = None
#     try:
#         logging.debug(f"Getting pose for {pdb}{ext}")
#         pdb_file = os.path.join(pdb_dir, pdb) + ext
#         pose = pyrosetta.pose_from_pdb(pdb_file)
#         pose.pdb_info().name(pdb)
#         # break
#     except:
#     #         logging.debug(f"Filetype {ext} could not be found for {pdb}")
#     # else:
#         logging.warning(f'Pose could not be created for protein {pdb_file}')
#         return process_data.callback(None,**process_data.params)
#     #logging.debug(f'pdb is {pdb}{pose.pdb_info().name()}')
#     return process_data.callback(pose, **process_data.params)

# def initializer(init, callback: Callable, params, init_params):
#     if init is not None:
#         init(**init_params)
#     process_data.callback = callback
#     process_data.params = params
#     signal.signal(signal.SIGINT, signal.SIG_IGN)

def process_data_dir(pdb,pdb_dir):
    assert(process_data_dir.callback)

    pdb = pdb if isinstance(pdb, str) else pdb.decode('utf-8')

    pdb_file = os.path.join(pdb_dir, pdb + '.pdb')

    #print('pdb is ',pdb,pose.pdb_info().name())
    return process_data_dir.callback(pdb_file, **process_data_dir.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data_dir.callback = callback
    process_data_dir.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class PDBPreprocessor:
    def __init__(self, pdb_list: List[str], pdb_dir: str):
        self.pdb_dir = pdb_dir
        self.__data = pdb_list
        self.size = len(pdb_list)
        self.pdb_name_length = np.max(list(map(len, self.__data)))
        
    def count(self) -> int:
        """
        Return the length of the data.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Length of the data.
        """
        return len(self.__data)

    def execute(self, callback: Callable, parallelism: int=8, limit: int=None, params = None, init = None, init_params = None) -> None:
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer, processes=parallelism, initargs = (init, callback, params, init_params)) as pool:
            all_loaded = True
            if all_loaded:
                # logging.info('All PDB files are loaded.')
                pass
            else:
                raise Exception("Some PDB files could not be loaded.")
            process_data_pdbs = functools.partial(
                process_data_dir,
                pdb_dir=self.pdb_dir
            )
            for res in pool.imap(process_data_pdbs, data):
                if res:
                    yield res

