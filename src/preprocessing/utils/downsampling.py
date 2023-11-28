
import os
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
from tqdm import tqdm


so_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils.so') # compile utils.c with: gcc -shared -o utils.so -fPIC utils.c
c_functions = CDLL(so_file)

max_atoms = 1000


from src.preprocessing.utils.constants import BACKBONE_ATOMS, EMPTY_ATOM_NAME, CHI_ATOMS, CHI_ANGLES, CB

MAX_SITES = 100_00
MAX_CHAINS = 20

get_mask = c_functions.get_mask
get_mask.restype = c_int
get_mask.argtypes = [
    c_int, 
    c_int, 
    c_int,
    c_int,
    c_int,
    c_char_p,
    ndpointer(c_long, flags="C_CONTIGUOUS"),
    c_void_p, 
    ndpointer(c_bool, flags="C_CONTIGUOUS"),
    ndpointer(c_bool, flags="C_CONTIGUOUS"), 
    ndpointer(c_double, flags="C_CONTIGUOUS"),
    ndpointer(c_bool, flags="C_CONTIGUOUS"), 
    c_float, 
    c_void_p, 
    c_int,
    c_char,
    c_long
]


# @profile
def get_mask_c(n, p, rng, chi=None, CB_as_backbone=False):
    # unpadding only necessary because converting sites -> int
    unpad_mask = n['atom_names'] != EMPTY_ATOM_NAME
    res_ids = n['res_ids'][unpad_mask]
    chains = res_ids[:, 2].tobytes()
    sites = np.array(res_ids[:, 3], dtype=int)

    # super simple trick to make sure there are no negative sites, and generally to internally make sites just start at 0
    # so we also don't overflow in the positive direction
    # some PDBs have negative site numbers (god knows why) and sometimes very high values that might make us go over the allotted maximum
    min_sites = int(np.min(sites))
    sites = sites - min_sites
    central_site = int(n['res_id'][3]) - min_sites

    SIZE = len(res_ids)

    names = n['atom_names'][unpad_mask].tobytes()

    rand = rng.random((max_atoms))

    seen = np.zeros(MAX_CHAINS * MAX_SITES, dtype=bool)
    removed = np.zeros(MAX_CHAINS * MAX_SITES, dtype=bool)

    mask = np.zeros(max_atoms, dtype=bool)

    central_chain = n['res_id'][2].tobytes()

    size_of_string_in_res_id = int(str(n.dtype['res_id']).split("'")[1][1:])
    min_chain_ascii_value = int(np.min(list(map(ord, res_ids[:, 2]))))

    try:
        aa_ol = n['res_id'][0].decode('utf-8')
    except:
        aa_ol = n['res_id'][0]
    
    if CB_as_backbone:
        BACKBONE_ATOMS_PLUS_MAYBE_CB = np.hstack([BACKBONE_ATOMS, np.array([CB])])
    else:
        BACKBONE_ATOMS_PLUS_MAYBE_CB = BACKBONE_ATOMS

    if chi is not None and not (aa_ol not in CHI_ATOMS):
        assert isinstance(chi, int) and chi < 5 and chi > 0, "chi must be an integer between 1 and 4"
        ATOMS_TO_KEEP = np.array(list(set(list(BACKBONE_ATOMS_PLUS_MAYBE_CB) + CHI_ATOMS[aa_ol][:chi]))) # throw into set to potentially reove duplicate CB
        len_atoms_to_keep = len(ATOMS_TO_KEEP)
        ATOMS_TO_KEEP = ATOMS_TO_KEEP.tobytes()

    else:
        len_atoms_to_keep = len(BACKBONE_ATOMS_PLUS_MAYBE_CB)
        ATOMS_TO_KEEP = BACKBONE_ATOMS_PLUS_MAYBE_CB.tobytes()
        from protein_holography_pytorch.utils.protein_naming import one_letter_to_aa
        aa = one_letter_to_aa[aa_ol]
        assert chi is None or aa not in CHI_ANGLES


    success = get_mask(
        SIZE, MAX_SITES, MAX_CHAINS, size_of_string_in_res_id, min_chain_ascii_value,
        chains, sites, names, 
        seen, removed, rand, mask, p, ATOMS_TO_KEEP, len_atoms_to_keep,
        central_chain, central_site
    )

    if not success:
        print("Error: get_mask failed")
        print(n['res_id'])
        print(chains)
        print(size_of_string_in_res_id)
        print(np.min(sites), np.max(sites))
        print(np.unique(res_ids[:, 2]))
        print()
        raise Exception("get_mask failed")

    return mask


def pad(arrays, max_atoms):
    """
    Returns LIST of ndarrays padded to max_atoms
    """
    return [pad_arr(arr, max_atoms) for arr in arrays]

def pad_arr(arr, padded_length):
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

# @profile
def downsample(neighborhood, p, rng, chi=None, max_atoms=1000, remove_central=True, remove_backbone_of_central=False, CB_as_backbone=False):
    """
    Takes a neighborhood, removes p proportion
    of sidechains.
    
    Leaves backbone atoms in place. 
    
    Returns a copy of the neighborhood modified

    chi: None or int. if None, all side-chain atoms are removed. If int, the atoms associated with the chi angles up until that index are kept.
    
    ```
    # USAGE: 
    for neighborhood in data:
        neighborhood = downsample(neighborhood, .5)
        ...
    ```
    """
    assert remove_central or not remove_backbone_of_central, "remove_backbone_of_central only makes sense if remove_central is True"
    dt = neighborhood.dtype
    decode_id = lambda res_id: "_".join([x.decode('utf-8') for x in res_id])
    encode_id = lambda res_id: np.array(res_id.split('_'), dtype='S')
    decode_ids = lambda res_ids: np.array([decode_id(res_id) for res_id in res_ids])

    if CB_as_backbone:
        BACKBONE_ATOMS_PLUS_MAYBE_CB = np.hstack([BACKBONE_ATOMS, np.array([CB])])
    else:
        BACKBONE_ATOMS_PLUS_MAYBE_CB = BACKBONE_ATOMS

    backbone_in = np.isin(neighborhood['atom_names'], BACKBONE_ATOMS_PLUS_MAYBE_CB)

    mask = get_mask_c(neighborhood, p, rng, chi, CB_as_backbone)

    if not remove_central:
        print('keeping central residue')
        m = np.array(list(map(decode_id, neighborhood['res_ids'])))
        is_central_res_atom = m == decode_id(neighborhood['res_id'])
        mask = np.logical_or(mask, is_central_res_atom)
    
    if remove_backbone_of_central:
        print('removing backbone of central residue')
        m = np.array(list(map(decode_id, neighborhood['res_ids'])))
        is_central_res_atom = m == decode_id(neighborhood['res_id'])
        mask = np.logical_and(mask, ~is_central_res_atom)

    info = [neighborhood['res_id'], 
    neighborhood['atom_names'][mask],
    neighborhood['elements'][mask],
    neighborhood['res_ids'][mask],
    neighborhood['coords'][mask],
    neighborhood['SASAs'][mask],
    neighborhood['charges'][mask]]

    backbone_out = np.isin(info[1], BACKBONE_ATOMS_PLUS_MAYBE_CB)

    if np.sum(backbone_in) != np.sum(backbone_out):
        print("Error: backbone atoms changed")
        print(neighborhood['res_id'])
        print(np.sum(backbone_in))
        print(np.sum(backbone_out))
        print()

    # if np.any(backbone_in != backbone_out):
    #     print("Error: backbone atoms changed")
    #     print(neighborhood['res_id'])
    #     print(backbone_in)
    #     print(backbone_out)
    #     print()
    #     raise Exception("backbone atoms changed")


    # ## tests to check everything works as intended

    temp_neighborhood = {}
    temp_neighborhood['res_id'] = neighborhood['res_id']
    temp_neighborhood['atom_names'] = neighborhood['atom_names'][mask]
    temp_neighborhood['elements'] = neighborhood['elements'][mask]
    temp_neighborhood['res_ids'] = neighborhood['res_ids'][mask]
    temp_neighborhood['coords'] = neighborhood['coords'][mask]
    temp_neighborhood['SASAs'] = neighborhood['SASAs'][mask]
    temp_neighborhood['charges'] = neighborhood['charges'][mask]


    def get_atoms_of_res_id(res_id):
        m = np.array(list(map(decode_id, temp_neighborhood['res_ids'])))
        is_res_id = m == decode_id(res_id)
        return np.sum(is_res_id), temp_neighborhood['atom_names'][is_res_id]

    def get_num_atoms_for_all_res_ids(temp_neighborhood):
        # get unique res_ids
        unique_res_ids = np.unique(list(map(decode_id, temp_neighborhood['res_ids'])))[:-1] # last one is empty one

        # get number of atoms for each res_id
        num_atoms = []
        for res_id in unique_res_ids:
            n_atoms, _ = get_atoms_of_res_id(encode_id(res_id))
            num_atoms.append(n_atoms)
        return num_atoms

    num_atoms, atoms = get_atoms_of_res_id(temp_neighborhood['res_id'])
    num_expected_atoms = 3
    if CB_as_backbone:
        num_expected_atoms += 1
    from protein_holography_pytorch.utils.protein_naming import one_letter_to_aa
    if chi is not None and neighborhood['res_id'][0].decode('utf-8') in CHI_ATOMS:
        num_expected_atoms += chi
    
    if num_atoms != num_expected_atoms:
        print(f"Error: central residue doesn't have {num_expected_atoms} atoms")
        print(neighborhood['res_id'])
        print(num_atoms)
        print(atoms)
        print(get_num_atoms_for_all_res_ids(temp_neighborhood))
        print()

    # # print(num_atoms)
    # # print(atoms)
    # # print(get_num_atoms_for_all_res_ids(temp_neighborhood))
    # # print()

    
    info[1:] = pad(info[1:], max_atoms)
    
    x = np.zeros(shape=(1), dtype=dt)
    x[0] = (*info, )
    return x[0]