
import numpy as np

from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.id import (
    AtomID_Map_double_t,AtomID_Map_bool_t)
from pyrosetta.rosetta.core.scoring import calc_per_atom_sasa
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
from pyrosetta.rosetta.utility import vector1_double

from typing import *

def get_pose_residue_number(
    pose: Pose, 
    chain: str, 
    resnum: int, 
    icode: str=' '
) -> int:
    """Translate pdb residue id to pyrosetta index"""
    return pose.pdb_info().pdb2pose(chain, resnum, icode)

def get_pdb_residue_info(
    pose : Pose, 
    resnum : int
) -> Tuple[str, int, str]:
    """Translate pyrosetta index to pdb residue id"""
    pi = pose.pdb_info()
    return (pi.chain(resnum), pi.number(resnum), pi.icode(resnum))

def calculate_sasa(
    pose : Pose,
    probe_radius : float=1.4
) -> AtomID_Map_double_t:
    """Calculate SASA for a pose"""
    # pyrosetta structures for returning of sasa information
    all_atoms = AtomID_Map_bool_t()
    atom_sasa = AtomID_Map_double_t()
    rsd_sasa = vector1_double()
    
    # use pyrosetta to calculate SASA per atom
    calc_per_atom_sasa(
        pose,
        atom_sasa,
        rsd_sasa,
        probe_radius
    )
    
    return atom_sasa

def get_hb_counts(
    hbond_set: HBondSet,
    i: int
):
    """
    Classifies a pose's h-bonds by main- and side-chain linkages
    
    Parameters
    ----------
    hbond_set : 
        The h-bond object from pyrosetta
    i : int
       
    Returns
    -------
    np.ndarray
        Float array of shape [8] where each entry is the number of
        h-bonds where the central residue and the partner are categorized 
        according to the donor/accceptor role and the backbone (bb) vs. 
        side-chain (sc) location of the bond. Specifically, the array is 
        organized as
            central|partner
            ---------------
            acc-bb  don-bb
            don-bb  acc-bb
            acc-bb  don-sc
            don-bb  acc-sc
            acc-sc  don-bb
            don-sc  acc-bb
            acc-sc  don-sc
            don-sc  acc-sc
    """
    counts = np.zeros(8,dtype=int)
    for hb in hbond_set.residue_hbonds(i):
        ctrl_don = hb.don_res() == i
        if ctrl_don:
            ctrl_side = not hb.don_hatm_is_backbone()
            nb_side = not hb.acc_atm_is_backbone()
        else:
            ctrl_side = not hb.acc_atm_is_backbone()
            nb_side = not hb.don_hatm_is_backbone()
        counts[4*ctrl_side + 2*nb_side + 1*ctrl_don] += 1
    return counts