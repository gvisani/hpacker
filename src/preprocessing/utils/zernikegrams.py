
"""Zenrikegram projection"""
import logging
from typing import *

import numpy as np
import scipy as sp
import scipy.special

from .spherical_bases import change_basis_complex_to_real
from src.utils.log_config import format

from src.preprocessing.utils.constants import BACKBONE_ATOMS, N, CA, C, O, EMPTY_ATOM_NAME

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format=format)


def ks_to_ns_zernike(l: int, ks: Union[List[int], np.ndarray]) -> np.ndarray:
    """Converts a list of frequencies to a list of Zernike n indices."""
    return np.array(ks, dtype=int) * 2 + l

def remove_zero_ns(l: int, ns: Union[List[int], np.ndarray]) -> np.ndarray:
    """Removes Zernike n indices that are zero."""
    ns_arr = np.array(ns, dtype=int)
    return ns_arr[
        np.logical_and(
            ns_arr >= l,
            ns_arr % 2 == (l % 2)
        )
    ]

def get_3D_zernike_function_indices(
    L_max: int, radial_nums: Union[List[int], np.ndarray], mode: str,
    keep_zeros: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get combined indices n, l, m for Zernike polynomials."""
    ns = []
    ls = []
    ms = []
    # logger.debug(f"Value of keep_zeros is {keep_zeros}.")
    # logger.debug(f"Value of ~keep_zeros is {not keep_zeros}.")
    if mode not in ["ks", "ns"]:
        logger.error(f"Unspecified mode {mode} supplied. Expected 'ks' or 'ns'")
    if mode == "ks":
        # logger.debug(f"keep_zeros set to false and mode is {mode}.")
        if keep_zeros:
            logger.warning(
                f"keep_zeros set to true but modes is {mode}. This combination "
                "is unexpected since no zeros index combinations are outputted "
                "in ks mode")
        for l in range(L_max + 1):
            n_vals_per_l = ks_to_ns_zernike(l, radial_nums)
            for n in n_vals_per_l:
                m_to_append = np.arange(-l, l + 1)
                ns.append(np.zeros(shape=(2*l + 1), dtype=int) + n)
                ls.append(np.zeros(shape=(2*l + 1), dtype=int) + l)
                ms.append(m_to_append)
    elif mode == "ns" and not keep_zeros:
        # logger.debug(f"keep_zeros set to false and mode is {mode}.")
        for l in range(L_max + 1):
            n_vals_per_l = remove_zero_ns(l, radial_nums)
            for n in n_vals_per_l:
                m_to_append = np.arange(-l, l + 1)
                ns.append(np.zeros(shape=(2*l + 1), dtype=int) + n)
                ls.append(np.zeros(shape=(2*l + 1), dtype=int) + l)
                ms.append(m_to_append)
    elif mode == "ns" and keep_zeros:
        # logger.debug(f"keep_zeros set to true and mode is {mode}.")
        n_vals_per_l = radial_nums
        for l in range(L_max + 1):
            for n in n_vals_per_l:
                m_to_append = np.arange(-l, l + 1)
                ns.append(np.zeros(shape=(2*l + 1), dtype=int) + n)
                ls.append(np.zeros(shape=(2*l + 1), dtype=int) + l)
                ms.append(m_to_append)

    ns = np.concatenate(ns)
    ls = np.concatenate(ls)
    ms = np.concatenate(ms)
    return ns, ls, ms


def zernike_coeff_lm_new(
        r: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        n: np.ndarray,
        r_max: np.float32,
        l: np.ndarray,
        m: np.ndarray,
        weights: np.ndarray,
        rst_normalization: Optional[str] = None,
) -> np.ndarray:
    """
    Compute Zernike coefficients.

    This implementation uses vectorized operations and avoids unnecessary 
    computation when possible.

    Parameters
    ----------
    r : np.ndarray
        Radii magnitudes.
    t : np.ndarray
        Theta values.
    p : np.ndarray
        Phi values.
    n : np.ndarray

    r_max : np.float64

    l : np.ndarray

    m :  np.ndarray

    weights : np.ndarray


    Returns
    -------
    coeffs : np.ndarray
        Zerkine coefficients.
    """    
    
    # logger.debug(f"r: {r}")
    # Dimension of the Zernike polynomial.
    D = 3.
    
    # Constituent terms in the polynomial.
    A = np.power(-1.0 + 0j, (n - l) / 2.)

    B = np.sqrt(2. * n + D)
    C = sp.special.binom((n + l + D) // 2 - 1, (n - l) // 2)

    nl_unique_combs, nl_inv_map = np.unique(np.vstack([n, l]).T, axis=0,
                                            return_inverse=True)

    num_nl_combs = nl_unique_combs.shape[0]
    n_hyp2f1_tile = np.tile(nl_unique_combs[:, 0], (r.shape[1], 1)).T
    l_hyp2f1_tile = np.tile(nl_unique_combs[:, 1], (r.shape[1], 1)).T

    E_unique = sp.special.hyp2f1(-(n_hyp2f1_tile - l_hyp2f1_tile) / 2.,
                                 (n_hyp2f1_tile + l_hyp2f1_tile + D) /2.,
                                 l_hyp2f1_tile + D / 2.,
                                 r[:num_nl_combs, :]**2 / r_max**2)
    E = E_unique[nl_inv_map]

    l_unique, l_inv_map = np.unique(l, return_inverse=True)
    l_power_tile = np.tile(l_unique, (r.shape[1], 1)).T
    F_unique = np.power(r[:l_unique.shape[0]] / r_max, l_power_tile)
    F = F_unique[l_inv_map]

    # Spherical harmonic component.
    lm_unique_combs, lm_inv_map = np.unique(np.vstack([l, m]).T, axis=0,
                                            return_inverse=True)
    num_lm_combs = lm_unique_combs.shape[0]
    l_sph_harm_tile = np.tile(lm_unique_combs[:, 0], (p.shape[1], 1)).T
    m_sph_harm_tile = np.tile(lm_unique_combs[:, 1], (p.shape[1], 1)).T

    y_unique = np.conj(sp.special.sph_harm(m_sph_harm_tile, l_sph_harm_tile,
                                           p[:num_lm_combs], t[:num_lm_combs]))
    y = y_unique[lm_inv_map]

    if True in np.isinf(E):
        print('Error: E is inf')
        print(f'E={E}, n={n}, l={l}, D={D}, r={np.array(r)}, rmax={r_max}')
    # logger.debug(f"y: {y}")
    # logger.debug(f"radial: {A * B * C * np.einsum('cN,nN,nN->Ncn', weights, E, F)[1]}")
    # logger.debug(f"shape of radial: {(A * B * C * np.einsum('cN,nN,nN->Ncn', weights, E, F)).shape}")
    # logger.debug(f"zipped n,l,m: {list(zip(n, l, m))}")
    # n indexes the combinations of n, l, m and N indexes the points in the point cloud
    if rst_normalization is None:
        coeffs = A * B * C * np.einsum('cN,nN,nN,nN->cn', weights, E, F, y)
    elif rst_normalization == 'square':
        # all_points_coeffs = A * B * C * np.einsum('cN,nN,nN,nN->cnN', weights, E, F, y)
        # square_norm = 1.0 / np.einsum('cnN->N' , np.real( all_points_coeffs * np.conj(all_points_coeffs) ))
        # coeffs = np.einsum('cnN,N->cn', all_points_coeffs, square_norm)

        all_points_coeffs = (A * B * C)[:, None] * E * F * y  # shape: nN
        square_norm = 1.0 / \
            np.einsum('nN->N', all_points_coeffs * np.conj(all_points_coeffs))
        coeffs = np.einsum('nN,cN,N->cn', all_points_coeffs,
                           weights, square_norm)

    return coeffs

def zernike_radial_coeff_lm_new(
        r: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        n: np.ndarray,
        r_max: np.float32,
        l: np.ndarray,
        m: np.ndarray,
        weights: np.ndarray,
) -> np.ndarray:
    """
    Compute Zerkinke coefficients.

    This implementation uses vectorized operations and avoids unnecessary 
    computation when possible.

    Parameters
    ----------
    r : np.ndarray
        Radii magnitudes.
    t : np.ndarray
        Theta values.
    p : np.ndarray
        Phi values.
    n : np.ndarray

    r_max : np.float64

    l : np.ndarray

    m :  np.ndarray

    weights : np.ndarray


    Returns
    -------
    coeffs : np.ndarray
        Zernike coefficients.
    """
    # log the inputs
    # logging.debug(f"r={r}")
    # logging.debug(f"t={t}")
    # logging.debug(f"p={p}")
    # logging.debug(f"n={n}")
    # logging.debug(f"r_max={r_max}")
    # logging.debug(f"l={l}")
    # logging.debug(f"m={m}")
    # logging.debug(f"weights={weights}")
    # Dimension of the Zernike polynomial.
    D = 3.

    # Constituent terms in the polynomial.
    A = np.power(-1.0 + 0j, (n - l) / 2.)

    B = np.sqrt(2. * n + D)
    print("div")
    C = sp.special.binom((n + l + D) // 2 - 1, (n - l) // 2)

    nl_unique_combs, nl_inv_map = np.unique(np.vstack([n, l]).T, axis=0,
                                            return_inverse=True)

    num_nl_combs = nl_unique_combs.shape[0]
    n_hyp2f1_tile = np.tile(nl_unique_combs[:, 0], (r.shape[1], 1)).T
    l_hyp2f1_tile = np.tile(nl_unique_combs[:, 1], (r.shape[1], 1)).T

    E_unique = sp.special.hyp2f1(-(n_hyp2f1_tile - l_hyp2f1_tile) / 2.,
                                 (n_hyp2f1_tile + l_hyp2f1_tile + D) /2.,
                                 l_hyp2f1_tile + D / 2.,
                                 r[:num_nl_combs, :]**2 / r_max**2)
    print(f"E_unique: {E_unique}")
    E = E_unique[nl_inv_map]

    l_unique, l_inv_map = np.unique(l, return_inverse=True)
    l_power_tile = np.tile(l_unique, (r.shape[1], 1)).T
    F_unique = np.power(r[:l_unique.shape[0]] / r_max, l_power_tile)
    F = F_unique[l_inv_map]

    # n indexes the combinations of n, l, m and N indexes the points in the point cloud
    coeffs = A * B * C * np.einsum('cN,nN,nN->cn', weights, E, F)

    return coeffs

def get_channel_weights(channel, nh, elements, real_locs, backbone_mask=None):
        # element weights are one-hot, channels with reasonable numerical values are not
        if channel == "all_other_elements":
            # account for the fact that elements may be represented as bytes or as strings
            # all other elements includes all elements except for C, N, O, S, H
            # elements only includes atoms that actually exist
            if isinstance(elements[0], str):
                return np.array((elements != 'C') & (elements != 'N') & (elements != 'O') & (elements != 'S') & (elements != 'H'), dtype=float)
            elif isinstance(elements[0], bytes):
                return np.array((elements != b'C') & (elements != b'N') & (elements != b'O') & (elements != b'S') & (elements != b'H'), dtype=float)
            else:
                raise ValueError("element type not recognized")
        elif channel in {'C','N','O','S','H'}:
            # account for the fact that elements may be represented as bytes or as strings
            if isinstance(elements[0], str):
                return np.array(elements == channel, dtype=float)
            elif isinstance(elements[0], bytes):
                return np.array(elements == channel.encode(), dtype=float)
            else:
                raise ValueError("element type not recognized")
        elif channel == 'SASA':
            return nh['SASAs'][real_locs]
        elif channel == 'charge':
            return nh['charges'][real_locs]
        # AAs should be all atoms on the BACKBONE of the AA that matches
        elif type(channel) == bytes:
            if backbone_mask is None:
                raise ValueError("backbone mask must be provided for AA channels")
            
            if isinstance(nh['res_ids'][:,0][real_locs][0], str):
                # convert channel to string
                channel = channel.decode()

            return np.array(np.logical_and(nh['res_ids'][:,0][real_locs] == channel, backbone_mask), dtype=float)
        
        elif channel == "all_other_AAs":
            if backbone_mask is None:
                raise ValueError("backbone mask must be provided for AA channels")
            
            AAS = [b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', b'S', b'T', b'W', b'Y', b'V', b'O']

            if isinstance(nh['res_ids'][:,0][real_locs][0], str):
                # convert channel to string
                AAS = [aa.decode() for aa in AAS]

            return np.array(np.logical_and(
                np.logical_and.reduce([nh['res_ids'][:,0][real_locs] != aa for aa in AAS]),
                backbone_mask), dtype=float)
        else:
            raise ValueError("channel %s not recognized"%channel)

def get_hologram(
    nh: np.ndarray,
    L_max: int,
    radial_nums: Union[List, np.ndarray], 
    num_combi_channels: int, 
    r_max: np.float32,
    mode: str="ks",
    keep_zeros: bool=False,
    real_sph_harm: bool=False,
    channels: List[str]=['C','N','O','S','H','SASA','charge'],
    get_physicochemical_info_for_hydrogens: bool=True,
    request_frame: bool=False,
    rst_normalization: Optional[str] = None
):
    
    # print("getting hologram")

    # get info from nh (note this gets all the info that matters, the location of only the atoms we care about)
    num_channels = len(channels)
    atom_names = nh['atom_names']
    real_locs = np.logical_and(atom_names != EMPTY_ATOM_NAME, nh['coords'][:,0] <= r_max)
    backbone_mask = np.logical_or.reduce(
                [nh['atom_names'][real_locs] == b for b in BACKBONE_ATOMS]
            )
    # real_locs = atom_names != b''
    # NOTE William treats atoms with no name as not existing, so those should not
    # be included in "all other elements" channel.
    elements = nh['elements'][real_locs]
    padded_coords = nh['coords']
    # curr_SASA = nh['SASAs'][real_locs]
    # curr_charge = nh['charges'][real_locs]
    atom_coords = padded_coords[real_locs]
    r,t,p = np.einsum('ij->ji',atom_coords)


    # indices are independent of what will be in those indices
    # TODO: compare william and my ns, ls, ms, r, t, p, need to know which neighborhood we getting these for
    ns, ls, ms = get_3D_zernike_function_indices(
        L_max, radial_nums, mode=mode, keep_zeros=keep_zeros)

    if keep_zeros:
        l_greater_n = ns < ls
        odds = ((ns - ls) % 2 == 1)
        nonzero_idxs = ~(l_greater_n | odds)
        nonzero_len = np.count_nonzero(nonzero_idxs)
        nmax = len(radial_nums) #
    else:
        nonzero_len = len(ns)
        nmax_per_l = np.array(
            [len(np.unique(ns[ls == l])) for l in range(L_max + 1)])
    if real_sph_harm:
        value_dtype = "float32"
    else:
        value_dtype = "complex64"
    if keep_zeros:
        dt = np.dtype([(str(l),'complex64',(num_combi_channels,2*l+1)) for l in range(L_max + 1)])
        arr = np.zeros(shape=(1,),dtype=dt)
    else:
        dt = np.dtype([(str(l),'complex64',(nmax_per_l[l] * len(channels),2*l+1)) for l in range(L_max + 1)])
    if real_sph_harm:
        dt_real = np.dtype([(str(l),'float32',(nmax_per_l[l] * len(channels),2*l+1)) for l in range(L_max + 1)])
        arr_real = np.zeros(shape=(1,),dtype=dt_real)

    arr = np.zeros(shape=(1,),dtype=dt)
    arr_weights = np.empty(shape=(num_channels, r.shape[-1],))
    
    # the weights for each channel are dependent upon what is there
    # the computation for the zernike coefficients happens later in this function
    for i,ch in enumerate(channels):
        arr_weights[i] = get_channel_weights(ch, nh, elements, real_locs, backbone_mask)
    # arr_weights[0] = np.array(elements == b'C', dtype=float)
    # arr_weights[1] = np.array(elements == b'N', dtype=float)
    # arr_weights[2] = np.array(elements == b'O', dtype=float)
    # arr_weights[3] = np.array(elements == b'S', dtype=float)
    # arr_weights[4] = np.array(elements == b'H', dtype=float)
    # arr_weights[5] = curr_SASA
    # arr_weights[6] = curr_charge
    # logger.debug(f"arr_weights: {arr_weights}")
    # logger.debug(f"elements: {elements}")
    ch_num = len(channels)
    out_z = np.zeros(shape=(ch_num, ns.shape[0]), dtype=np.complex64)

    rs = np.tile(r, (nonzero_len, 1))
    ts = np.tile(t, (nonzero_len, 1))
    ps = np.tile(p, (nonzero_len, 1))

    if keep_zeros:
        out_z[:,nonzero_idxs] = zernike_coeff_lm_new(
            rs, ts, ps, ns[nonzero_idxs], r_max, ls[nonzero_idxs],
            ms[nonzero_idxs], arr_weights, rst_normalization)
    else:
        out_z[:] = zernike_coeff_lm_new(
            rs, ts, ps, ns, r_max, ls, ms, arr_weights, rst_normalization)
    
    #return out_z
    low_idx = 0
    if keep_zeros:
        for l in range(L_max + 1):
            num_m = (2 * l + 1)
            idxs = ls ==  l
            arr[0][l][:,:] = out_z[:,idxs].reshape(nmax*ch_num, num_m, )
    else:
        for l in range(L_max + 1):
            num_m = (2 * l + 1)
            idxs = ls ==  l
            arr[0][l][:,:] = out_z[:,idxs].reshape(nmax_per_l[l]*ch_num, num_m, )
    if real_sph_harm:
        print("undoing conjugation")
        for l in range(L_max + 1):
            arr_real[0][l] = np.einsum(
                "nm,cm->cn", change_basis_complex_to_real(l), np.conj(arr[0][l])).real
        return arr_real[0], np.array(list(zip(ns, ls, ms))) 
    if request_frame:
       frame = get_frame(nh)
    else:
        frame = None
    return arr[0], frame #, np.array(list(zip(ns, ls, ms))) 


def get_frame(nh):
    try:
        cartesian_coords = nh["coords"]
        central_res = np.logical_and.reduce(nh['res_ids'] == nh['res_id'], axis=-1)
        central_CA_coords = np.array([0.0, 0.0, 0.0]) # since we centered the neighborhood on the alpha carbon
        central_N_coords = np.squeeze(cartesian_coords[central_res][nh['atom_names'][central_res] == N])
        central_C_coords = np.squeeze(cartesian_coords[central_res][nh['atom_names'][central_res] == C])
        print(central_N_coords.shape, central_C_coords.shape, central_CA_coords.shape)
        print(central_N_coords)
        # if central_N_coords.shape[0] == 3:
        #     print('-----'*16)
        #     print(nh['res_id'])
        #     print(nh['res_ids'])
        #     print(nh['atom_names'])
        #     print(central_N_coords)
        #     print(central_C_coords)
        #     print(nh['atom_names'].shape)
        #     print(nh['coords'].shape)
        #     print('-----'*16)

        # assert that there is only one atom with three coordinates
        assert (central_CA_coords.shape[0] == 3), 'first assert'
        assert (len(central_CA_coords.shape) == 1), 'second assert'
        assert (central_N_coords.shape[0] == 3), 'third assert'
        assert (len(central_N_coords.shape) == 1), 'fourth assert'
        assert (central_C_coords.shape[0] == 3), 'fifth assert'
        assert (len(central_C_coords.shape) == 1), 'sixth assert'

        # y is unit vector perpendicular to x and lying on the plane between CA_N (x) and CA_C
        # z is unit vector perpendicular to x and the plane between CA_N (x) and CA_C
        x = central_N_coords - central_CA_coords
        x = x / np.linalg.norm(x)

        CA_C_vec = central_C_coords - central_CA_coords

        z = np.cross(x, CA_C_vec)
        z = z / np.linalg.norm(z)

        y = np.cross(z, x)

        frame = (x, y, z)
    except Exception as e:
        print(e)
        print('No central residue (or other unwanted error).')
        frame = None
    return frame
