

from argparse import ArgumentParser
import numpy as np
import h5py
from hdf5plugin import LZ4
import sys
import itertools
from progress.bar import Bar
from tqdm import tqdm
from src.preprocessing.utils.downsampling import downsample

import time

GLYCINE, ALANINE = 'GLY', 'ALA'

from src.preprocessing.utils.constants import CHI_ANGLES
from src.utils.protein_naming import one_letter_to_aa

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdf5_in', type=str, required=True)
    parser.add_argument('--hdf5_out', type=str, required=True)
    parser.add_argument('--input_dataset_name', type=str, default='data')
    parser.add_argument('--output_dataset_name', type=str, default='data')
    parser.add_argument('--keep_central_residue', action='store_false', dest='remove_central_residue', help='keep central residue. by default, the sidechain only of the central residue is removed')
    parser.add_argument('--remove_backbone_of_central_residue', action='store_true', dest='remove_backbone_of_central_residue')
    parser.add_argument('--CB_as_backbone', action='store_true', default=False)
    parser.add_argument('--chi', type=int, default=None, help='if None, all central res atoms are removed. if int, the atoms associated with the chi angles up until and excluding the specified "chi" are kept. Note that we provide the CB if "isinstance(chi, int)", but not if "chi is None".')
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--p', type=str, default='random') # 'random' or a number in [0, 1]
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--parallelism', type=int, default=1, help='Dummy argument for compatibility. Not used.')

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

    start = time.time()
    
    with h5py.File(args.hdf5_in, 'r') as f_in:
        # number of neighborhoods

        N = len(f_in[args.input_dataset_name])

        # N = 100
        # print('Warning: in testing mode on a reduced sample size!')

        dt = f_in[args.input_dataset_name].dtype
                
        with h5py.File(args.hdf5_out, 'w') as f_out:
            # Initialize dataset
            f_out.create_dataset(args.output_dataset_name,
                         shape=(N * args.num_repeats,),
                         maxshape=(None,),
                         dtype=dt,
                         compression=LZ4())
            
            f_out.create_dataset('proportion_sidechain_removed',
                         shape=(N * args.num_repeats,),
                         maxshape=(None,),
                         dtype=np.float32,
                         compression=LZ4())
            
            f_out.create_dataset('seed',
                         shape=(1,),
                         maxshape=(None,),
                         dtype=np.float32,
                         compression=LZ4())
        n = 0
        with h5py.File(args.hdf5_out, 'r+') as f_out:
            for _ in range(args.num_repeats):
                for i in tqdm(range(0, N)):
                    if args.p == 'random':
                        p = rng.random()
                    elif args.p == 'half_one_half_random':
                        if rng.random() < 0.5:
                            p = 1.0
                        else:
                            p = rng.random()
                    else:
                        p = float(args.p)
                    
                    nb_in = f_in[args.input_dataset_name][i]

                    # skip glycine and alanine, and neighborhoods that don't have cetrain chis!
                    aa = one_letter_to_aa[nb_in['res_id'][0].decode('utf-8')]
                    if aa in {'GLY', 'ALA'}:
                        continue
                    elif args.chi is None:
                        pass
                    elif len(CHI_ANGLES[aa]) < args.chi:
                        continue

                    nb_out = downsample(nb_in, p, rng, args.chi, max_atoms=nb_in['atom_names'].shape[0], 
                                        remove_central=args.remove_central_residue, remove_backbone_of_central=args.remove_backbone_of_central_residue, CB_as_backbone=args.CB_as_backbone)
                    
                    f_out[args.output_dataset_name][n] = (*nb_out,)
                    f_out['proportion_sidechain_removed'][n] = p

                    n += 1
            
            f_out['seed'][0] = args.seed
            f_out[args.output_dataset_name].resize((n,))
            f_out['proportion_sidechain_removed'].resize((n,))

    
    print("Done in %.3f seconds." % (time.time() - start))

    
    