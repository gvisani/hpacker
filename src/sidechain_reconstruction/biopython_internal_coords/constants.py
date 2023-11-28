

THE20 = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
         'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
         'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
         'TYR', 'VAL'}

BB_ATOMS = ['C', 'CA', 'N', 'O']

SIDE_CHAINS = {'MET': ['CB', 'CE', 'CG', 'SD'],
               'ILE': ['CB', 'CD1', 'CG1', 'CG2'],
               'LEU': ['CB', 'CD1', 'CD2', 'CG'],
               'VAL': ['CB', 'CG1', 'CG2'],
               'THR': ['CB', 'CG2', 'OG1'],
               'GLY': [],
               'ALA': ['CB'],
               'ARG': ['CB', 'CD', 'CG', 'CZ', 'NE', 'NH1', 'NH2'],
               'SER': ['CB', 'OG'],
               'LYS': ['CB', 'CD', 'CE', 'CG', 'NZ'],
               'HIS': ['CB', 'CD2', 'CE1', 'CG', 'ND1', 'NE2'],
               'GLU': ['CB', 'CD', 'CG', 'OE1', 'OE2'],
               'ASP': ['CB', 'CG', 'OD1', 'OD2'],
               'PRO': ['CB', 'CD', 'CG'],
               'GLN': ['CB', 'CD', 'CG', 'NE2', 'OE1'],
               'TYR': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'OH'],
               'TRP': ['CB', 'CD1', 'CD2', 'CE2', 'CE3', 'CG', 'CH2', 'CZ2', 'CZ3', 'NE1'],
               'CYS': ['CB', 'SG'],
               'ASN': ['CB', 'CG', 'ND2', 'OD1'],
               'PHE': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ']}

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

RELATED_DIHEDRALS = {
    'ALA' : [('N:CA:C:O', 'O:C:CA:CB')],
    'ARG' : [('N:CA:C:O', 'O:C:CA:CB')],
    'ASN' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi2', 'CA:CB:CG:ND2')],
    'ASP' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi2', 'CA:CB:CG:OD2')],
    'CYS' : [('N:CA:C:O', 'O:C:CA:CB')],
    'GLN' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi3', 'CB:CG:CD:NE2')],
    'GLU' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi3', 'CB:CG:CD:OE2')],
    'HIS' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi2', 'CA:CB:CG:CD2')],
    'ILE' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi1', 'N:CA:CB:CG2')],
    'LEU' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi2', 'CA:CB:CG:CD2')],
    'LYS' : [('N:CA:C:O', 'O:C:CA:CB')],
    'MET' : [('N:CA:C:O', 'O:C:CA:CB')],
    'PHE' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi2', 'CA:CB:CG:CD2')],
    'PRO' : [('N:CA:C:O', 'O:C:CA:CB')],
    'SER' : [('N:CA:C:O', 'O:C:CA:CB')],
    'THR' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi1', 'N:CA:CB:CG2')],
    'TRP' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi2', 'CA:CB:CG:CD2')],
    'TYR' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi2', 'CA:CB:CG:CD2')],
    'VAL' : [('N:CA:C:O', 'O:C:CA:CB'), ('chi1', 'N:CA:CB:CG2')],
}
DESIRED_DIHEDRAL_TO_REFERENCE_DIHEDRAL = {resname: {pair[1]: pair[0] for pair in RELATED_DIHEDRALS[resname]} for resname in RELATED_DIHEDRALS}