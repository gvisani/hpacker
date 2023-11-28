import signal
import numpy as np
import os
import logging
import functools
from multiprocessing import Pool
import h5py
import sys




def process_data(ind,hdf5_file,neighborhood_list):
    assert(process_data.callback)
    with h5py.File(hdf5_file,'r') as f:
        neighborhood = f[neighborhood_list][ind]
        if 'proportion_sidechain_removed' in f:
            proportion_sidechain_removed = f['proportion_sidechain_removed'][ind]
        else:
            proportion_sidechain_removed = None

    return process_data.callback(neighborhood, proportion_sidechain_removed=proportion_sidechain_removed, **process_data.params)

def initializer(init, callback, params, init_params):
    if init is not None:
        init(**init_params)
    process_data.callback = callback
    process_data.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class HDF5Preprocessor:
    def __init__(self, hdf5_file, neighborhood_list):

        with h5py.File(hdf5_file,'r') as f:
            num_neighborhoods = np.array(f[neighborhood_list].shape[0])
            self.pdb_name_length = np.max(
                list(map(len, f[neighborhood_list]["res_id"][:,1])))
            self.__max_atoms = f[neighborhood_list][0]['atom_names'].shape[0]
            self.__dtype = f[neighborhood_list].dtype

        self.neighborhood_list = neighborhood_list
        self.hdf5_file = hdf5_file
        self.size = num_neighborhoods
        self.__data = np.arange(num_neighborhoods)

        logging.info(f"Preprocessed {self.size} neighborhoods from {self.hdf5_file}")
        
    def count(self):
        return len(self.__data)
    
    def max_atoms(self):
        return self.__max_atoms

    def dtype(self):
        return self.__dtype

    def execute(self, callback, parallelism = 8, limit = None, params = None, init = None, init_params = None):
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
            process_data_hdf5 = functools.partial(
                process_data,
                hdf5_file=self.hdf5_file,
                neighborhood_list=self.neighborhood_list
            )
            ntasks = self.size
            num_cpus = os.cpu_count()
            chunksize = ntasks // num_cpus + 1

            print('Parallelism: ', parallelism, file=sys.stderr)
            print('Number of tasks: ', ntasks, file=sys.stderr)
            print('Number of cpus: ', num_cpus, file=sys.stderr)
            print('Chunksize: ', chunksize, file=sys.stderr)

            logging.debug(
                f"Data size = {ntasks}, " \
                f"cpus = {num_cpus}, " \
                f"chunksize = {chunksize}")

            if chunksize > 100:
                chunksize = 16
            
            print('Chunksize: ', chunksize, file=sys.stderr)
            
            for res in pool.imap(process_data_hdf5, data, chunksize=chunksize):
                if res:
                    yield res

