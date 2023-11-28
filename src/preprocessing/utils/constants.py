
import numpy as np

N  = b'N   '
CA = b'CA  '
C  = b'C   '
O  = b'O   '
CB = b'CB  '
EMPTY_ATOM_NAME = b''

BACKBONE_ATOMS = np.array([N, CA, C, O])

CHI_ANGLES = {
    'ARG' : [['N','CA','CB','CG'], ['CA','CB','CG','CD'], ['CB','CG','CD','NE'], ['CG','CD','NE','CZ']], #, ['CD','NE','CZ','NH1']],
    'ASN' : [['N','CA','CB','CG'], ['CA','CB','CG','OD1']],
    'ASP' : [['N','CA','CB','CG'], ['CA','CB','CG','OD1']],
    'CYS' : [['N','CA','CB','SG']],
    'GLN' : [['N','CA','CB','CG'], ['CA','CB','CG','CD'], ['CB','CG','CD','OE1']],
    'GLU' : [['N','CA','CB','CG'], ['CA','CB','CG','CD'], ['CB','CG','CD','OE1']],
    'HIS' : [['N','CA','CB','CG'], ['CA', 'CB','CG','ND1']],
    'ILE' : [['N','CA','CB','CG1'], ['CA','CB','CG1','CD1']],
    'LEU' : [['N','CA','CB','CG'], ['CA','CB','CG','CD1']],
    'LYS' : [['N','CA','CB','CG'], ['CA','CB','CG','CD'], ['CB','CG','CD','CE'], ['CG','CD','CE','NZ']],
    'MET' : [['N','CA','CB','CG'], ['CA','CB','CG','SD'], ['CB','CG','SD','CE']],
    'PHE' : [['N','CA','CB','CG'], ['CA','CB','CG','CD1']],
    'PRO' : [['N','CA','CB','CG'], ['CA','CB','CG','CD']],
    'SER' : [['N','CA','CB','OG']],
    'THR' : [['N','CA','CB','OG1']],
    'TRP' : [['N','CA','CB','CG'], ['CA','CB','CG','CD1']],
    'TYR' : [['N','CA','CB','CG'], ['CA','CB','CG','CD1']],
    'VAL' : [['N','CA','CB','CG1']]
}

# transform chi angle atoms by left-justifying anf converting to bytes
from src.utils.protein_naming import aa_to_one_letter
CHI_ATOMS = {aa_to_one_letter[aa]: [CB] + [atoms[-1].ljust(4).encode('utf-8') for atoms in CHI_ANGLES[aa]] for aa in CHI_ANGLES}

