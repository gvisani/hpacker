import functools
import logging
from multiprocessing import Pool
import os
import h5py
import numpy as np
import pandas as pd
import signal

from src.utils.log_config import format

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=format)

def process_data(ind,hdf5_file,protein_list):
    assert(process_data.callback)

    with h5py.File(hdf5_file,'r') as f:
        protein = f[protein_list][ind]
        #print('loaded protein',protein[0])
    return process_data.callback(protein, **process_data.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)

        
class HDF5Preprocessor:
    def __init__(self, hdf5_file, protein_list ):

        with h5py.File(hdf5_file,'r') as f:
            num_proteins = np.array(f[protein_list].shape[0])
            self.pdb_name_length = np.max(list(map(len, f[protein_list]["pdb"])))

        self.protein_list = protein_list
        self.hdf5_file = hdf5_file
        self.size = num_proteins
        self.__data = np.arange(num_proteins)
        
        logger.info(f"Preprocessed {self.size} proteins from {self.hdf5_file}")
        
    def count(self):
        return len(self.__data)

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(initializer = initializer,
                  processes=parallelism,
                  initargs = (init, callback, params, init_params)) as pool:

    
            all_loaded = True
            if all_loaded:
                # logger.info('All PDB files are loaded.')
                pass
            else:
                raise Exception("Some PDB files could not be loaded.")


            process_data_hdf5 = functools.partial(
                process_data,
                hdf5_file = self.hdf5_file,
                protein_list = self.protein_list
            )
            ntasks = self.size
            num_cpus = os.cpu_count()
            chunksize = ntasks // num_cpus + 1
            logger.debug(
                f"Data size = {ntasks}, " \
                f"cpus = {num_cpus}, " \
                f"chunksize = {chunksize}")

            if chunksize > 4:
                chunksize = 4
            logger.info(f"chunksize = {chunksize}")
            for res in pool.imap_unordered(
                    process_data_hdf5,
                    data,
                    chunksize=chunksize):
                #if res:
                yield res

