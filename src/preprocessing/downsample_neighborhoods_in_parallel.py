
'''

TODO: figure out why this doesn't actually speed up computation. Maybe multiprocessing doesn't like the use of a compiled c function? or maybe I'm just doing something worng?

'''

from argparse import ArgumentParser
import numpy as np
import h5py
from hdf5plugin import LZ4
import sys
import itertools
from progress.bar import Bar
from tqdm import tqdm
from src.preprocessing.preprocessors.preprocessor_hdf5_neighborhoods import HDF5Preprocessor
from src.preprocessing.utils.downsampling import downsample

import time

def downsample_callback(nb_in, args_p, rng, max_atoms, remove_central, remove_backbone_of_central):
     
    if args_p == 'random':
        p = rng.random()
    else:
        p = float(args_p)
    
    nb_out = downsample(nb_in, p, rng, max_atoms=max_atoms, 
                        remove_central=remove_central, remove_backbone_of_central=remove_backbone_of_central)
    
    return nb_out, p
     

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_in', type=str, required=True)
    parser.add_argument('--hdf5_out', type=str, required=True)
    parser.add_argument('--input_dataset_name', type=str, default='data')
    parser.add_argument('--output_dataset_name', type=str, default='data')
    parser.add_argument('--keep_central_residue', action='store_false', dest='remove_central_residue', help='keep central residue. by default, the sidechain only of the central residue is removed')
    parser.add_argument('--remove_backbone_of_central_residue', action='store_true', dest='remove_backbone_of_central_residue')
    parser.add_argument('--p', type=str, default='random') # 'random' or a number in [0, 1]
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--parallelism', type=int, default=1)

    args = parser.parse_args()
    
    # Better way to do this? Probably!
    if not args.remove_central_residue:
        print("keeping central")
        
    if args.remove_backbone_of_central_residue:
        print("removing backbone atoms of central residue")
    
    if args.seed is None:
        args.seed = int(time.time())
    rng = np.random.default_rng(args.seed)

    time.sleep(1)

    ds = HDF5Preprocessor(args.hdf5_in, args.input_dataset_name)
    n = ds.count()
    dt = ds.dtype()

    start = time.time()
    with h5py.File(args.hdf5_out, 'w') as f_out:
        # Initialize dataset
        f_out.create_dataset(args.output_dataset_name,
                        shape=(n,),
                        maxshape=(None,),
                        dtype=dt,
                        compression=LZ4())
        
        f_out.create_dataset('proportion_sidechain_removed',
                        shape=(n,),
                        maxshape=(None,),
                        dtype=np.float32,
                        compression=LZ4())
        
        f_out.create_dataset('seed',
                        shape=(1,),
                        maxshape=(None,),
                        dtype=np.float32,
                        compression=LZ4())

    print('Starting...')
    with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
        with h5py.File(args.hdf5_out, 'r+') as f_out:
                
            for i, (nb_out, p) in enumerate(ds.execute(downsample_callback,
                                                        limit = None,
                                                        params = {
                                                            'args_p': args.p,
                                                            'rng': rng,
                                                            'max_atoms': ds.max_atoms(),
                                                            'remove_central': args.remove_central_residue,
                                                            'remove_backbone_of_central': args.remove_backbone_of_central_residue
                                                        },
                                                        parallelism = args.parallelism)):
                
                f_out[args.output_dataset_name][i] = (*nb_out,)
                f_out['proportion_sidechain_removed'][i] = p
                bar.next()
            
            f_out['seed'][0] = args.seed
    
    print("Time elapsed: %.2fs" % (time.time() - start))
    print("Done")

    
    