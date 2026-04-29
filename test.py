from hpacker import HPacker
import numpy as np


def parse_pdb_atoms(path):
    """Return dict mapping (chain, resnum, atom_name) -> (x, y, z)."""
    atoms = {}
    with open(path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            atom_name = line[12:16].strip()
            chain = line[21].strip() or ' '
            resnum = line[22:26].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            atoms[(chain, resnum, atom_name)] = np.array([x, y, z])
    return atoms


def compute_rmsd(coords1, coords2):
    diff = coords1 - coords2
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def rmsd_between_pdbs(path1, path2, atom_filter=None):
    atoms1 = parse_pdb_atoms(path1)
    atoms2 = parse_pdb_atoms(path2)
    common_keys = sorted(set(atoms1) & set(atoms2))
    if atom_filter is not None:
        common_keys = [k for k in common_keys if k[2] in atom_filter]
    if not common_keys:
        return float('nan'), 0
    coords1 = np.stack([atoms1[k] for k in common_keys])
    coords2 = np.stack([atoms2[k] for k in common_keys])
    return compute_rmsd(coords1, coords2), len(common_keys)


BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}
SIDECHAIN_EXCLUDE = BACKBONE_ATOMS | {'OXT'}

ref_pdb = 'T0950.pdb'
bb_only_pdb = 'T0950_bb_only.pdb'
pred_pdb = 'reconstructed_from_bb_only_T0950.pdb'

hpacker = HPacker(bb_only_pdb)
chi_trace = hpacker.reconstruct_sidechains(num_refinement_iterations=5, return_trace_of_predicted_angles=True)
hpacker.write_pdb(pred_pdb)

all_atom_rmsd, n_all = rmsd_between_pdbs(ref_pdb, pred_pdb)
bb_rmsd, n_bb = rmsd_between_pdbs(ref_pdb, pred_pdb, atom_filter=BACKBONE_ATOMS)

# sidechain atoms: anything not in backbone set and not OXT
atoms_ref = parse_pdb_atoms(ref_pdb)
sc_atom_names = {k[2] for k in atoms_ref if k[2] not in SIDECHAIN_EXCLUDE}
sc_rmsd, n_sc = rmsd_between_pdbs(ref_pdb, pred_pdb, atom_filter=sc_atom_names)

print(f"All-atom RMSD:   {all_atom_rmsd:.4f} Å  ({n_all} atoms)")
print(f"Backbone RMSD:   {bb_rmsd:.4f} Å  ({n_bb} atoms)")
print(f"Sidechain RMSD:  {sc_rmsd:.4f} Å  ({n_sc} atoms)")

