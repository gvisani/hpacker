"""Module for parallel gathering of zernikegrams"""

import os, sys

from argparse import ArgumentParser
import logging
from progress.bar import Bar
from time import time
from typing import List

import h5py
from hdf5plugin import LZ4
import numpy as np
from sqlitedict import SqliteDict

from src.preprocessing.preprocessors.preprocessor_hdf5_neighborhoods import HDF5Preprocessor
from src.preprocessing.utils.spherical_bases import change_basis_complex_to_real
from src.preprocessing.utils.zernikegrams import get_hologram 
from src.utils.protein_naming import ol_to_ind_size
from src.utils.argparse import *

from src.preprocessing.utils.zernikegrams import get_hologram 
from src.utils.conversions import spherical_to_cartesian__numpy

from src.preprocessing.utils.constants import BACKBONE_ATOMS, N, CA, C, O, EMPTY_ATOM_NAME

GLYCINE, ALANINE = ol_to_ind_size['G'], ol_to_ind_size['A']

# get phdir for assigning paths
import os
from pathlib import Path
import src
phdir = Path(src.__file__).parents[1]
cob_mats = np.load(os.path.join(phdir, "src/preprocessing/utils/YZX_XYZ_cob.npy"), allow_pickle=True)[()]

def get_one_zernikegram(res_id: np.ndarray,
                         res_ids: np.ndarray,
                         spherical_coords: np.ndarray,
                         elements: np.ndarray,
                         atom_names: np.ndarray,
                         r_max: float = 10.0,
                         radial_func_max: int = 20,
                         Lmax: int = 6,
                         channels: List[str] = ['C', 'N', 'O', 'S'], 
                         backbone_only: bool = False,
                         request_frame: bool = False,
                         real_sph_harm: bool = True,
                         charges: Optional[np.ndarray] = None,
                         SASAs: Optional[np.ndarray] = None,
                         get_physicochemical_info_for_hydrogens: bool = True,
                         sph_harm_normalization: str = 'component',
                         rst_normalization: Optional[str] = None,
                         radial_func_mode = 'ns',
                         keep_zeros: bool = False
                        ) -> np.ndarray:

    if backbone_only: raise NotImplementedError("backbone_only not implemented yet")
    
    ks = np.arange(radial_func_max+1)

    if keep_zeros:
        num_combi_channels = [len(channels) * len(ks)] * Lmax
    else:
        num_combi_channels = [
            len(channels) * np.count_nonzero(
                np.logical_and(
                    (l%2) == np.array(ks)%2,
                    np.array(ks) >= l)
        ) for l in range(Lmax + 1)]
    
    nh = {
        "res_id": res_id,
        "res_ids": res_ids,
        "coords": spherical_coords,
        "elements": elements,
        "atom_names": atom_names
    }
    if charges is not None:
        nh["charges"] = charges
    if SASAs is not None:
        nh["SASAs"] = SASAs
    hgm, _ = get_hologram(nh, 
                            Lmax, 
                            ks, 
                            num_combi_channels, 
                            r_max,
                            mode=radial_func_mode, 
                            keep_zeros=keep_zeros, 
                            channels=channels,
                            get_physicochemical_info_for_hydrogens=get_physicochemical_info_for_hydrogens,
                            request_frame=request_frame,
                            rst_normalization=rst_normalization)
    
    for l in range(0, Lmax + 1):
        assert not np.any(np.isnan(hgm[str(l)]))
        assert not np.any(np.isinf(hgm[str(l)]))

    if real_sph_harm:
        for l in range(0, Lmax + 1):
            hgm[str(l)] = np.einsum(
                'nm,cm->cn', change_basis_complex_to_real(l), np.conj(hgm[str(l)]))
            if sph_harm_normalization == 'component': # code uses 'integral' normalization by default. Can just simply multiply by sqrt(4pi) to convert to 'component'
                if rst_normalization is None:
                    hgm[str(l)] *= np.sqrt(4*np.pi).astype(np.float32)
                elif rst_normalization == 'square':
                    hgm[str(l)] *= (1.0 / np.sqrt(4*np.pi)).astype(np.float32) # just by virtue of how the square normalization works... simple algebra

    hgm = make_flat_and_rotate_zernikegram(hgm, Lmax)

    return hgm


def get_zernikegrams(nbs: np.ndarray, # of custom dtype
                    r_max: float,
                    radial_func_max: int,
                    Lmax: int,
                    channels: List[str], 
                    backbone_only: bool = False,
                    request_frame: bool = False,
                    get_physicochemical_info_for_hydrogens: bool = True,
                    real_sph_harm: bool = True,
                    sph_harm_normalization: str = 'component',
                    rst_normalization: Optional[str] = None,
                    radial_func_mode = 'ns',
                    keep_zeros: bool = False) -> Dict:
    
    if backbone_only: raise NotImplementedError("backbone_only not implemented yet")
    
    ks = np.arange(radial_func_max+1)

    if keep_zeros:
        num_combi_channels = [len(channels) * len(ks)] * Lmax
    else:   
        num_combi_channels = [
            len(channels) * np.count_nonzero(
                np.logical_and(
                    (l%2) == np.array(ks)%2,
                    np.array(ks) >= l)
        ) for l in range(Lmax + 1)]
    
    num_components = get_num_components(Lmax, ks, keep_zeros, radial_func_mode, channels)
    L = np.max(list(map(len, nbs["res_id"][:,1])) + [5])
    dt = np.dtype(
        [('res_id', f'S{L}', (6,)),
        ('zernikegram', 'f4', (num_components,)),
        ('frame', 'f4', (3, 3)),
        ('label', '<i4')])
    
    zernikegrams, res_ids, frames, labels = [], [], [], []
    for np_nh in nbs:
        arr, res_id = get_single_zernikegram(np_nh, Lmax, ks, num_combi_channels, r_max, torch_dt=dt, mode=radial_func_mode, real_sph_harm=real_sph_harm, channels=channels, torch_format=True, request_frame=request_frame, sph_harm_normalization=sph_harm_normalization, rst_normalization=rst_normalization, get_physicochemical_info_for_hydrogens=get_physicochemical_info_for_hydrogens)
        zernikegrams.append(arr['zernikegram'])
        res_ids.append(res_id)
        if request_frame:
            frames.append(arr['frame'])
        labels.append(arr['label'])
    
    if request_frame:
        frames = np.stack(frames, axis=0)
    else:
        frames = None

    return {'zernikegram': np.vstack(zernikegrams),
            'res_id': np.vstack(res_ids),
            'frame': frames,
            'label': np.hstack(labels).reshape(-1)}


def make_flat_and_rotate_zernikegram(zgram, L_max):
    flattened_zgram = np.concatenate([
        np.einsum(
            'mn,Nn->Nm',
            cob_mats[i],
            zgram[str(i)],
        ).flatten().real for i in range(L_max + 1)])
    return flattened_zgram

def make_flat_zernikegram(zgram, L_max):
    flattened_zgram = np.concatenate([zgram[str(i)].flatten().real for i in range(L_max + 1)])
    return flattened_zgram

def get_num_components(Lmax, ks, keep_zeros, mode, channels):
    num_components = 0
    if mode == "ns":
        for l in range(Lmax + 1):
            if keep_zeros:
                num_components += np.count_nonzero(
                    np.array(ks) >= l) * len(channels) * (2*l + 1)
            else:
                num_components += np.count_nonzero(
                    np.logical_and(np.array(ks) >= l, (np.array(ks) - l) % 2 == 0)) * len(channels) * (2*l + 1)
                
    if mode == "ks":
        for l in range(Lmax + 1):
            num_components += len(ks) * len(channels) *  (2*l +1)
    return num_components

def stringify(res_id):
    return '_'.join(list(map(lambda x: x.decode('utf-8'), list(res_id))))

def get_single_zernikegram(
    np_nh,
    L_max,
    ks,
    num_combi_channels,
    r_max, 
    proportion_sidechain_removed: float=None,
    real_sph_harm: bool=True, 
    mode='ns',
    keep_zeros=False, 
    channels: List[str]=['C','N','O','S','H','SASA','charge'],
    get_physicochemical_info_for_hydrogens: bool=True,
    torch_format: bool=False,
    torch_dt: np.dtype=None, 
    request_frame: bool=False,
    sph_harm_normalization: str = 'component',
    rst_normalization: Optional[str] = None
):
    
    if np_nh["res_id"][0].decode("utf-8") in {'Z', 'X'}:
        logging.error(f"Skipping neighborhood with residue: {np_nh['res_id'][0].decode('-utf-8')}")
        return (None,)
    
    try:
        hgm, frame = get_hologram(np_nh, 
                                    L_max, 
                                    ks, 
                                    num_combi_channels, 
                                    r_max,
                                    mode=mode, 
                                    keep_zeros=keep_zeros, 
                                    channels=channels,
                                    get_physicochemical_info_for_hydrogens=get_physicochemical_info_for_hydrogens,
                                    request_frame=request_frame,
                                    rst_normalization=rst_normalization)
    except Exception as e:
       print(e)
       print('Error with',np_nh[0])
       #print(traceback.format_exc())
       return (None,)

    for l in range(0, L_max + 1):
        if np.any(np.isnan(hgm[str(l)])):
            logging.error(f"NaNs in hologram for {np_nh['res_id'][0].decode('-utf-8')}")
            return (None,)
        if np.any(np.isinf(hgm[str(l)])):
            logging.error(f"Infs in hologram for {np_nh['res_id'][0].decode('-utf-8')}")
            return (None,)

    if real_sph_harm:
        for l in range(0, L_max + 1):
            hgm[str(l)] = np.einsum(
                'nm,cm->cn', change_basis_complex_to_real(l), np.conj(hgm[str(l)]))
            if sph_harm_normalization == 'component': # code uses 'integral' normalization by default. Can just simply multiply by sqrt(4pi) to convert to 'component'
                if rst_normalization is None:
                    hgm[str(l)] *= np.sqrt(4*np.pi).astype(np.float32)
                elif rst_normalization == 'square':
                    hgm[str(l)] *= (1.0 / np.sqrt(4*np.pi)).astype(np.float32) # just by virtue of how the square normalization works... simple algebra

    if torch_format:

        # get backbone atom coords, in standardard [C, O, N, CA] order
        central_res_mask = np.logical_and.reduce(np_nh['res_ids'] == np_nh['res_id'], axis=-1)

        C_coords = np_nh['coords'][np.logical_and(central_res_mask, np_nh['atom_names'] == C)]
        assert C_coords.shape[0] == 1, f'C_coords.shape[0] is {C_coords.shape[0]} instead of 1'

        O_coords = np_nh['coords'][np.logical_and(central_res_mask, np_nh['atom_names'] == O)]
        assert O_coords.shape[0] == 1, f'O_coords.shape[0] is {O_coords.shape[0]} instead of 1'

        N_coords = np_nh['coords'][np.logical_and(central_res_mask, np_nh['atom_names'] == N)]
        assert N_coords.shape[0] == 1, f'N_coords.shape[0] is {N_coords.shape[0]} instead of 1'


        # convert coords to cartesian and add CA at [0, 0, 0]
        backbone_coords = np.vstack([spherical_to_cartesian__numpy(np.vstack([C_coords, O_coords, N_coords])), np.array([0.0, 0.0, 0.0])])

        # arr = np.zeros(dtype=torch_dt, shape=(1,))
        
        hgm = make_flat_and_rotate_zernikegram(hgm, L_max)

        # arr['res_id'] = np_nh['res_id']
        # arr['zernikegram'] = hgm
        # arr['frame'] = frame
        # arr['label'] = ol_to_ind_size[np_nh["res_id"][0].decode("-utf-8")]
        # arr['backbone_coords'] = backbone_coords
        
        arr = (np_nh['res_id'], hgm, frame, ol_to_ind_size[np_nh["res_id"][0].decode("-utf-8")], backbone_coords)
        
        return arr, np_nh['res_id'], proportion_sidechain_removed

    return hgm, np_nh['res_id']

def get_zernikegrams_from_dataset(
        hdf5_in,
        input_dataset_name,
        r_max,
        Lmax,
        ks,
        hdf5_out,
        output_dataset_name,
        parallelism,
        real_sph_harm: bool=False,
        keep_zeros: bool=True,
        mode: str='ns',
        channels: List[str]=['C','N','O','S','H','SASA','charge'],
        request_frame: bool=False,
        sph_harm_normalization: str='integral',
        rst_normalization: Optional[str] = None,
        torch_format: bool=False,
        exclude_residues_with_no_sidechain: bool=False,
        angles_db: Optional[str]=None,
        vectors_db: Optional[str]=None,
):
        
    # get metadata
    # metadata = get_metadata()

    start_time = time()

    logging.basicConfig(level=logging.DEBUG)
    ds = HDF5Preprocessor(hdf5_in, input_dataset_name)
    bad_neighborhoods = []
    n = 0
    ks = np.array(ks)
    # channels = ['C','N','O','S','H','SASA','charge']
    if keep_zeros:
        num_combi_channels = [len(channels) * len(ks)] * Lmax
    else:   
        num_combi_channels = [
            len(channels) * np.count_nonzero(
                np.logical_and(
                    (l%2) == np.array(ks)%2,
                    np.array(ks) >= l)
        ) for l in range(Lmax + 1)]

    L = np.max([5, ds.pdb_name_length])
    
    if torch_format:
        logging.info(f"Using torch format")
        num_components = get_num_components(Lmax, ks, keep_zeros, mode, channels)
        if angles_db is not None:
            assert vectors_db is not None
            dt = np.dtype(
                [('res_id', f'S{L}', (6,)),
                ('zernikegram', 'f4', (num_components,)),
                ('frame', 'f4', (3, 3)),
                ('label', '<i4'),
                ('backbone_coords', 'f4', (4, 3)),
                ('chi_angles', 'f4', (4,)),
                ('norm_vecs', 'f4', (5, 3))])
            angles_db = SqliteDict(angles_db, flag='r')
            vectors_db = SqliteDict(vectors_db, flag='r')
        else:
            dt = np.dtype(
                [('res_id', f'S{L}', (6,)),
                ('zernikegram', 'f4', (num_components,)),
                ('frame', 'f4', (3, 3)),
                ('label', '<i4'),
                ('backbone_coords', 'f4', (4, 3))])
            angles_db = None
            vectors_db = None
    
    if real_sph_harm and not torch_format:
        logging.info(f"Using real spherical harmonics")
        dt = np.dtype(
            [(str(l),'float32',(num_combi_channels[l],2*l+1)) 
            for l in range(Lmax + 1)])
    elif not torch_format:
        logging.info(f"Using complex spherical harmonics")
        dt = np.dtype(
            [(str(l),'complex64',(num_combi_channels[l],2*l+1)) 
            for l in range(Lmax + 1)])
    
    logging.info(f"Transforming {ds.size} in zernikegrams")
    logging.info("Writing hdf5 file")
    
    nhs = np.empty(shape=ds.size,dtype=(f'S{L}',(6)))
    with h5py.File(hdf5_out,'w') as f:
        f.create_dataset(output_dataset_name,
                         shape=(ds.size,),
                         dtype=dt,
                         compression=LZ4())
        f.create_dataset('nh_list',
                         dtype=(f'S{L}',(6)),
                         shape=(ds.size,),
                         compression=LZ4())
        f.create_dataset('proportion_sidechains_removed',
                            dtype='f4',
                            shape=(ds.size,),
                            compression=LZ4())
        # record_metadata(metadata, f[neighborhood_list])
        # record_metadata(metadata, f["nh_list"])

    if not torch_format:
        with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
            with h5py.File(hdf5_out,'r+') as f:
                n = 0
                for i,hgm in enumerate(ds.execute(
                        get_zernikegrams,
                        limit = None,
                        # NOTE these params are all wrong
                        params = {'L_max': Lmax,
                                    'ks':ks,
                                    'num_combi_channels': num_combi_channels,
                                    'r_max': r_max,
                                    "real_sph_harm": real_sph_harm,
                                    "keep_zeros": keep_zeros,
                                    "mode": mode,
                                    "channels": channels, 
                                    "sph_harm_normalization": sph_harm_normalization,
                                    "rst_normalization": rst_normalization},
                        parallelism = parallelism)):
                    if hgm is None or hgm[0] is None:
                        bar.next()
                        print('error')
                        continue
                    f['nh_list'][n] = hgm[1]
                    f[output_dataset_name][n] = hgm[0]
                    #print(hgm[0].shape)
                    bar.next()
                    n += 1

                    print(hgm['zernikegram'].shape)

                print(f'Resizing to {n}')
                f[output_dataset_name].resize((n,))
                f['nh_list'].resize((n,))
                    
    else:
        with Bar('Processing', max = ds.count(), suffix='%(percent).1f%%') as bar:
            with h5py.File(hdf5_out,'r+') as f:
                n = 0
                init_time = time()
                print('Time to start: %.5fs' % (init_time - start_time), file=sys.stderr)
                for i,hgm in enumerate(ds.execute(
                        get_single_zernikegram,
                        limit = None,
                        params = {'L_max': Lmax,
                                'ks':ks,
                                'num_combi_channels': num_combi_channels,
                                'r_max': r_max,
                                "real_sph_harm": real_sph_harm,
                                "keep_zeros": keep_zeros,
                                "mode": mode,
                                "channels": channels,
                                "torch_format": torch_format,
                                "torch_dt": dt,
                                "request_frame": request_frame,
                                "sph_harm_normalization": sph_harm_normalization,
                                "rst_normalization": rst_normalization},
                        parallelism = parallelism)):
                    
                    new_time = time()
                    # print('%d - %.5fs' % (i, new_time - init_time), end='\r', file=sys.stderr)

                    if hgm is None or hgm[0] is None:
                        bar.next()
                        print('error')
                        continue
                    
                    hgm_data, nh_info, proportion_sidechain_removed = hgm

                    res_id, zgram, frame, label, backbone_coords = hgm_data

                    
                    if exclude_residues_with_no_sidechain:
                        if label in {GLYCINE, ALANINE}:
                            bar.next()
                            continue
                    
                    if angles_db is not None:
                        stringified_res_id = stringify(res_id)
                        chi_angles = angles_db[stringified_res_id]
                        norm_vecs = vectors_db[stringified_res_id]
                        arr = (res_id, zgram, frame, label, backbone_coords, chi_angles, norm_vecs,)
                    else:
                        arr = (res_id, zgram, frame, label, backbone_coords)

                    f[output_dataset_name][n] = (*arr,)
                    f['nh_list'][n] = nh_info
                    if proportion_sidechain_removed is not None:
                        f['proportion_sidechains_removed'][n] = proportion_sidechain_removed
                    else:
                        f['proportion_sidechains_removed'][n] = -1.0
                    
                    bar.next()
                    n += 1
                
                print(f'Resizing to {n}')
                f[output_dataset_name].resize((n,))
                f['nh_list'].resize((n,))
                f['proportion_sidechains_removed'].resize((n,))

                


def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--hdf5_in', type=str,
        help='input hdf5 filename, containing protein neighborhoods',
        required=True
    )
    parser.add_argument(
        '--hdf5_out', dest='hdf5_out', type=str,
        help='ouptut hdf5 filename, which will contain zernikegrams.',
        required=True
    )
    parser.add_argument(
        '--input_dataset_name', type=str,
        help='Name of the dataset within hdf5_in where the neighborhoods are stored. We recommend keeping this set to simply "data".',
        default='data'
    )
    parser.add_argument(
        '--output_dataset_name', type=str,
        help='Name of the dataset within hdf5_out where the zernikegrams will be stored. We recommend keeping this set to simply "data".',
        default='data'
    )
    parser.add_argument(
        '--parallelism', type=int,
        help='Parallelism for multiprocessing.',
        default = 4
    )

    parser.add_argument(
        '--l_max', type=int,
        help='Maximum spherical frequency to use in projections',
        default = 6
    )
    parser.add_argument(
        '--radial_func_mode', type=str,
        help='Operation mode for radial functions: \
              ns (treating k input as literal n values to use), \
              ks (treating k values as wavelengths)',
        default = 'ns'
    )
    parser.add_argument(
        '--radial_func_max', type=int,
        help='Maximum radial frequency to use in projections',
        default = 20
    )
    parser.add_argument(
        '--keep_zeros', action='store_true',
        help='Keep zeros in zernikegrams. Only when radial_func_mode is "ns". When radial_func_mode is "ks", zeros are always removed.',
    )
    parser.add_argument(
        '--r_max', type=float,
        help='Radius of the neighborhoods.',
        default=10.0
    )
    parser.add_argument(
        '--channels', type=comma_sep_str_list,
        help='Channels to use in zernikegrams.',
        default=['C', 'N', 'O', 'S']
    )
    parser.add_argument(
        '--sph_harm_normalization', type=str,
        help='Normalization to use for spherical harmonics.'
             'Use "integral" for pre-trained tensorflow HCNN_AA, "component" for pre-trained pytorch H-(V)AE.',
        choices = ['integral', 'component'],
        default='component'
    )
    parser.add_argument(
        '--rst_normalization', type=optional_str,
        help="Normalization to use for the zernikegrams of individual Dirac-delta functions. We find that 'square' tends to work the best.",
        choices = [None, 'None', 'square'],
        default=None
    )

    parser.add_argument(
        '--use_complex_sph_harm',
        help='Use complex spherical harmonics, as opposed to real oness.',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--request_frame',
        help="Request frame from dataset.",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--sph_harm_convention', type=str, default='yzx',
        help="convention to use for L=1 spherical harmonics. "
        "Will influence all Y_l^m with l>0. However, this convention will "
        "not affect training. Only need to specify to compare values with a "
        "given convention "
    )
    parser.add_argument(
        '--tensorflow_format',
        dest='torch_format',
        help='Use tensorflow format for saving output',
        action='store_false',
        default=True
    )
    parser.add_argument(
        '--exclude_residues_with_no_sidechain',
        action='store_true',
        default=False,
        help='Effectively excludes neighborhoods whose central residue is a Glycine or an Alanine.'
    )
    parser.add_argument(
        '--angle_db',
        dest='angles_db',
        type=str,
        default=None
    )
    parser.add_argument(
        '--vec_db',
        dest='vectors_db',
        type=str,
        default=None
    )
                        
    args = parser.parse_args()

    print(args.channels)

    if args.channels[0] == 'dlpacker':
        print("using dlpacker")
        # NOTE my code is different than williams, I switched amino acid 'O' with amino acid
        # 'G' because glycine is canonical, whereas pyrrolysine is not.
        # My example follows DLPackers code, which lists their amino acids in utils.py "THE20"
        # TODO: check if dl packer puts hydrogen in the all other elements channel, if not remove it
        args.channels = ['C', 'N', 'O', 'S', "all_other_elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all_other_AAs"]
    
    elif args.channels[0] == 'AAs':
        args.channels = [b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', b'S', b'T', b'W', b'Y', b'V', b'G', "all_other_AAs"]
    

    print(args.channels)

    '''
    NOTE: we assume spherical coordinates
    '''
    
    get_zernikegrams_from_dataset(
        args.hdf5_in,
        args.input_dataset_name,
        args.r_max,
        args.l_max,
        np.arange(args.radial_func_max + 1),
        args.hdf5_out,
        args.output_dataset_name,
        args.parallelism,
        real_sph_harm=not args.use_complex_sph_harm,
        keep_zeros=args.keep_zeros,
        mode=args.radial_func_mode,
        channels=args.channels,
        sph_harm_normalization=args.sph_harm_normalization,
        rst_normalization=args.rst_normalization,
        torch_format=args.torch_format,
        exclude_residues_with_no_sidechain=args.exclude_residues_with_no_sidechain,
        angles_db=args.angles_db,
        vectors_db=args.vectors_db,
    )
    
if __name__ == "__main__":
    s = time()
    main()
    print(f"Time of computation: {time() - s} secs")
