#
# Python module protein.py for holographic machine learning
#
# This module contains all structures relating to protein conventions including...
#  - dictionaries for translating amino acids to indices and vice-versa
#  - dictionaries for translating elements to indices and vice-versa
#  - dictionaries for amino acid information such as the number of atoms typically
#     associated with that amino acid in pdb structures
#


# dictionaries for converting amino acid to index and vice-versa
ind_to_aa = {0: 'GLY', 1: 'ALA', 2: 'CYS', 3: 'SER', 4: 'PRO',
             5: 'THR', 6: 'VAL', 7: 'ASP', 8: 'ILE', 9: 'LEU',
             10: 'ASN', 11: 'MET', 12: 'GLN', 13: 'LYS', 14: 'GLU',
             15: 'HIS', 16: 'PHE', 17: 'ARG', 18: 'TYR', 19: 'TRP'}
aa_to_ind = {v: k for k, v in ind_to_aa.items()}

# dictionaries to convert element to index
el_to_ind = {'C':0 , 'N':1, 'O':2, 'S':3}
ind_to_el = {0:'C', 1:'N', 2:'O', 3:'S'}
# dictionaries for amino acid statistics
atoms_per_aa = {'CYS': 6, 'ASP': 8, 'SER': 6, 'GLN': 9, 'LYS': 9,
                'ASN': 8, 'PRO': 7, 'GLY': 4, 'THR': 7, 'PHE': 11,
                'ALA': 5, 'MET': 8, 'HIS': 10, 'ILE': 8, 'LEU': 8,
                'ARG': 11, 'TRP': 14, 'VAL': 7, 'GLU': 9, 'TYR': 12}

aas = aa_to_ind.keys()
els = el_to_ind.keys()

aa_to_one_letter = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER':'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
one_letter_to_aa = {v: k for v, k in aa_to_one_letter.items()}

#Mike uses this one
aa_to_ind_size = {'CYS': 2, 'ILE': 8, 'GLN': 12, 'VAL': 6, 'LYS': 13,
       'PRO': 4, 'GLY': 0, 'THR': 5, 'PHE': 16, 'GLU': 14,
       'HIS': 15, 'MET': 11, 'ASP': 7, 'LEU': 9, 'ARG': 17,
       'TRP': 19, 'ALA': 1, 'ASN': 10, 'TYR': 18, 'SER': 3}
ind_to_aa_size = {0: 'GLY', 1: 'ALA', 2: 'CYS', 3: 'SER', 4: 'PRO',
       5: 'THR', 6: 'VAL', 7: 'ASP', 8: 'ILE', 9: 'LEU',
       10: 'ASN', 11: 'MET', 12: 'GLN', 13: 'LYS', 14: 'GLU',
       15: 'HIS', 16: 'PHE', 17: 'ARG', 18: 'TYR', 19: 'TRP'}
aa_to_ind_one_letter_order = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3,
                        'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7,
                        'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11,
                        'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER':15,
                        'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19}
background_freqs = {'ALA': 7.4, 'CYS': 3.3, 'ASP': 5.9, 'GLU': 3.7,
                        'PHE': 4., 'GLY': 7.4, 'HIS': 2.9, 'ILE': 3.8,
                        'LYS': 7.2, 'LEU': 7.6, 'MET': 1.8, 'ASN': 4.4,
                        'PRO': 5., 'GLN': 5.8, 'ARG': 4.2, 'SER': 8.1,
                        'THR': 6.2, 'VAL': 6.8, 'TRP': 1.3, 'TYR': 3.3}
ind_to_aa_one_letter_order = {0: 'ALA', 1: 'CYS', 2: 'ASP', 3: 'GLU',
                        4: 'PHE', 5: 'GLY', 6: 'HIS', 7: 'ILE',
                        8: 'LYS', 9: 'LEU', 10: 'MET', 11: 'ASN',
                        12: 'PRO', 13: 'GLN', 14: 'ARG', 15: 'SER', 
                        16: 'THR', 17: 'VAL', 18: 'TRP', 19: 'TYR'}
aa_to_ind_hydro = {'ALA': 8, 'ARG': 15, 'ASN': 17, 'ASP': 14,
                   'CYS': 6, 'GLN': 13, 'GLU': 10, 'GLY': 11,
                   'HIS': 18, 'ILE': 1, 'LEU': 0, 'LYS': 16,
                   'MET': 5, 'PHE': 2, 'PRO': 19, 'SER': 12,
                   'THR': 9, 'TRP': 3, 'TYR': 7, 'VAL': 4}
ind_to_aa_hydro = {8: 'ALA', 15: 'ARG', 17: 'ASN', 14: 'ASP',
                   6: 'CYS', 13: 'GLN', 10: 'GLU', 11: 'GLY',
                   18: 'HIS', 1: 'ILE', 0: 'LEU', 16: 'LYS', 
                   5: 'MET', 2: 'PHE', 19: 'PRO', 12: 'SER',
                   9: 'THR', 3: 'TRP', 7: 'TYR', 4: 'VAL'}

aa_to_one_letter = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER':'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
one_letter_to_aa = {'A': 'ALA',
 'C': 'CYS',
 'D': 'ASP',
 'E': 'GLU',
 'F': 'PHE',
 'G': 'GLY',
 'H': 'HIS',
 'I': 'ILE',
 'K': 'LYS',
 'L': 'LEU',
 'M': 'MET',
 'N': 'ASN',
 'P': 'PRO',
 'Q': 'GLN',
 'R': 'ARG',
 'S': 'SER',
 'T': 'THR',
 'V': 'VAL',
 'W': 'TRP',
 'Y': 'TYR'}
aa_to_ind = {'CYS': 2, 'ILE': 8, 'GLN': 12, 'VAL': 6, 'LYS': 13,
       'PRO': 4, 'GLY': 0, 'THR': 5, 'PHE': 16, 'GLU': 14,
       'HIS': 15, 'MET': 11, 'ASP': 7, 'LEU': 9, 'ARG': 17,
       'TRP': 19, 'ALA': 1, 'ASN': 10, 'TYR': 18, 'SER': 3}
ind_to_aa = {0: 'GLY', 1: 'ALA', 2: 'CYS', 3: 'SER', 4: 'PRO',
       5: 'THR', 6: 'VAL', 7: 'ASP', 8: 'ILE', 9: 'LEU',
       10: 'ASN', 11: 'MET', 12: 'GLN', 13: 'LYS', 14: 'GLU',
       15: 'HIS', 16: 'PHE', 17: 'ARG', 18: 'TYR', 19: 'TRP'}

ind_to_ol_nisthal = {0: 'D', 1: 'E', 2: 'H', 3: 'K',
                        4: 'R', 5: 'A', 6: 'F', 7: 'I',
                        8: 'L', 9: 'M', 10: 'V', 11: 'Y',
                        12: 'N', 13: 'Q', 14: 'S', 15: 'T', 
                        16: 'G', 17: 'P',18:'C',19:'W'}
ol_to_ind_nisthal = {'D': 0,
 'E': 1,
 'H': 2,
 'K': 3,
 'R': 4,
 'A': 5,
 'F': 6,
 'I': 7,
 'L': 8,
 'M': 9,
 'V': 10,
 'Y': 11,
 'N': 12,
 'Q': 13,
 'S': 14,
 'T': 15,
 'G': 16,
 'P': 17,
 'C': 18,
 'W': 19}
ind_to_aa_ward = {0: 'GLY', 1: 'PRO', 2: 'TRP', 3: 'PHE', 4: 'TYR',

       5: 'ALA', 6: 'SER', 7: 'LYS', 8: 'ARG', 9: 'GLN',

       10: 'GLU', 11: 'MET', 12: 'HIS', 13: 'ASP', 14: 'ASN',

       15: 'CYS', 16: 'THR', 17: 'LEU', 18: 'VAL', 19: 'ILE'}
aa_to_ind_ward = {'GLY': 0,
 'PRO': 1,
 'TRP': 2,
 'PHE': 3,
 'TYR': 4,
 'ALA': 5,
 'SER': 6,
 'LYS': 7,
 'ARG': 8,
 'GLN': 9,
 'GLU': 10,
 'MET': 11,
 'HIS': 12,
 'ASP': 13,
 'ASN': 14,
 'CYS': 15,
 'THR': 16,
 'LEU': 17,
 'VAL': 18,
 'ILE': 19}
aa_to_ind_cosine = {
    'TRP' : 0 ,
    'TYR' : 1 ,
    'PHE' : 2 ,
    'LEU' : 3 ,
    'MET' : 4 ,
    'ARG' : 5 ,
    'LYS' : 6 ,
    'HIS' : 7 ,
    'GLN' : 8 ,
    'GLU' : 9 ,
    'ASP' : 10 ,
    'ASN' : 11 ,
    'ALA' : 12 ,
    'SER' : 13 ,
    'THR' : 14 ,
    'CYS' : 15 ,
    'ILE' : 16 ,
    'VAL' : 17 ,
    'PRO' : 18 ,
    'GLY' : 19 ,
}
ind_to_aa_cosine = {
    0 : 'TRP' ,
    1 : 'TYR' ,
    2 : 'PHE' ,
    3 : 'LEU' ,
    4 : 'MET' ,
    5 : 'ARG' ,
    6 : 'LYS' ,
    7 : 'HIS' ,
    8 : 'GLN' ,
    9 : 'GLU' ,
    10 : 'ASP' ,
    11 : 'ASN' ,
    12 : 'ALA' ,
    13 : 'SER' ,
    14 : 'THR' ,
    15 : 'CYS' ,
    16 : 'ILE' ,
    17 : 'VAL' ,
    18 : 'PRO' ,
    19 : 'GLY' ,
}

background_freqs = {'ALA': 7.4, 'CYS': 3.3, 'ASP': 5.9, 'GLU': 3.7,
                    'PHE': 4., 'GLY': 7.4, 'HIS': 2.9, 'ILE': 3.8,
                    'LYS': 7.2, 'LEU': 7.6, 'MET': 1.8, 'ASN': 4.4,
                    'PRO': 5., 'GLN': 5.8, 'ARG': 4.2, 'SER': 8.1,
                    'THR': 6.2, 'VAL': 6.8, 'TRP': 1.3, 'TYR': 3.3}



#for real one
ind_to_aa_for_real_one_letter = dict()
for i,aa in enumerate(ind_to_aa.values()):
    ind_to_aa_for_real_one_letter[i] = aa_to_one_letter[aa]
ind_to_ol_size = {x:aa_to_one_letter[ind_to_aa_size[x]] for x in range(20)}
ol_to_ind_size = {val:key for key,val in ind_to_ol_size.items()}
