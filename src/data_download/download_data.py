
import os
from tqdm import tqdm
import urllib.request
import numpy as np

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.PDB import PDBParser

import warnings
warnings.filterwarnings("ignore")

from src.utils.protein_naming import aa_to_one_letter

BASE_DIR = '/gscratch/scrubbed/gvisan01/dlpacker'

# assume the following three sets of files exist already
DLPACKER_TRAIN = os.path.join(BASE_DIR, 'dlpacker_pdb_lists/train.txt')
DLPACKER_VAL = os.path.join(BASE_DIR, 'dlpacker_pdb_lists/val.txt')
DLPACKER_VAL_NEWEST = os.path.join(BASE_DIR, 'dlpacker_pdb_lists/val_newest.txt')

DLPACKER_PDBS_DIR = os.path.join(BASE_DIR, 'pdbs')
CASP_PDBS_DIR = os.path.join(BASE_DIR, 'casp_targets')
PDB_LISTS_DIR = os.path.join(BASE_DIR, 'pdb_lists')

# fasta files for filtering training data based on similarity to testing data
TRAINING_FASTA_OUTPUT_PATH = os.path.join(BASE_DIR, 'fasta_files/dlpacker.fasta')
TESTING_FASTA_OUTPUT_PATH = os.path.join(BASE_DIR, 'fasta_files/casp13_casp14.fasta')
FILTERED_TRAINING_FASTA_OUTPUT_PATH = os.path.join(BASE_DIR, 'fasta_files/dlpacker_50percent_to_casp13_casp14.fasta')

# make dirs in case they don't exist
os.makedirs(DLPACKER_PDBS_DIR, exist_ok=True)
os.makedirs(CASP_PDBS_DIR, exist_ok=True)
os.makedirs(PDB_LISTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'fasta_files'), exist_ok=True)



def main():

    # print('Downloading dlpacker training and validation PDBs...', flush=True)
    # download_dlpacker_pdbs()
    # make_training_pdb_lists()

    print('Downloading CASP13 and CASP14 targets (testing data)...', flush=True)
    download_casp13_and_14_targets()

    # print('Filter training data based on similarity to testing data...', flush=True)
    # print('\tMaking fasta files...')
    # make_fasta_files(TRAINING_FASTA_OUTPUT_PATH, TESTING_FASTA_OUTPUT_PATH)

    # print('\tRunning cd-hit and output filtered fasta file...', flush=True)
    # run_cd_hit_for_filtering(TRAINING_FASTA_OUTPUT_PATH, TESTING_FASTA_OUTPUT_PATH, FILTERED_TRAINING_FASTA_OUTPUT_PATH)

    # print('\tRemaking PDB lists from filtered fasta file...', flush=True)
    # remake_pdb_lists_from_filtered_fasta_file(FILTERED_TRAINING_FASTA_OUTPUT_PATH)


    # print('Done!', flush=True)




def download_dlpacker_pdbs():

    locations_of_existing_pdbs = [
        '/gscratch/scrubbed/gvisan01/casp12/validation',
        '/gscratch/scrubbed/gvisan01/casp12/training_30'
    ]

    pdb_lists = [
        DLPACKER_TRAIN,
        DLPACKER_VAL,
        DLPACKER_VAL_NEWEST
    ]

    os.makedirs(DLPACKER_PDBS_DIR, exist_ok=True)

    for pdb_list in pdb_lists:
        with open(pdb_list, 'r') as f:
            for line in tqdm(f):
                pdb = line.strip().lower()
                if not os.path.exists(os.path.join(DLPACKER_PDBS_DIR, pdb + '.pdb')):
                    # check if it already exists somewhere!
                    for location in locations_of_existing_pdbs:
                        if os.path.exists(os.path.join(location, f'{pdb.upper()}.pdb')) or os.path.exists(os.path.join(location, f'{pdb}.pdb')):
                            os.system(f"cp {os.path.join(location, f'{pdb.upper()}.pdb')} {DLPACKER_PDBS_DIR}") # it's gonna be uppercase
                            os.system(f"mv {os.path.join(DLPACKER_PDBS_DIR, f'{pdb.upper()}.pdb')} {os.path.join(DLPACKER_PDBS_DIR, f'{pdb}.pdb')}")
                            break
                    else: # download it
                        try:
                            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb}.pdb', os.path.join(DLPACKER_PDBS_DIR, f'{pdb}.pdb'))
                        except Exception as e:
                            print(f'Error downloading {pdb}: {e}')
                            continue

def make_training_pdb_lists():
    '''
    PDB Lists are the lists of PDBs that we use for training, validation, and testing of our models
    '''

    ## seed for splitting dlpacker's "validation" data into validationg and testing
    seed = 1234567890
    np.random.seed(seed)

    ## load the pdbs
    with open(DLPACKER_TRAIN, 'r') as f:
        dlpacker_train_pdbs = [line.strip().lower() for line in f]
    
    with open(DLPACKER_VAL, 'r') as f:
        dlpacker_val_pdbs = [line.strip().lower() for line in f]
    
    with open(DLPACKER_VAL_NEWEST, 'r') as f:
        dlpacker_val_newest_pdbs = [line.strip().lower() for line in f]
    

    ## save them into a new folder with the splits and the names we want
    os.makedirs(PDB_LISTS_DIR, exist_ok=True)

    ## train, split it 10 ways
    num_splits = 10
    num_pdbs_in_split = len(dlpacker_train_pdbs) // num_splits
    for i in range(num_splits):

        if i < num_splits - 1:
            curr_train_pdbs = dlpacker_train_pdbs[i*num_pdbs_in_split:(i+1)*num_pdbs_in_split]
        else:
            curr_train_pdbs = dlpacker_train_pdbs[i*num_pdbs_in_split:]
        
        with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_training__{i}.txt'), 'w') as f:
            for pdb in curr_train_pdbs:
                f.write(pdb + '\n')

    ## val and test, split dlpacker's "validation" pdbs in half at random. I think this is what they did
    np.random.shuffle(dlpacker_val_pdbs)
    val_pdbs = dlpacker_val_pdbs[:len(dlpacker_val_pdbs)//2]
    test_pdbs = dlpacker_val_pdbs[len(dlpacker_val_pdbs)//2:]

    with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_validation.txt'), 'w') as f:
        for pdb in val_pdbs:
            f.write(pdb + '\n')
    
    with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_testing.txt'), 'w') as f:
        for pdb in test_pdbs:
            f.write(pdb + '\n')
    

    ## val_newest, which we save as "test_newest" because that's what it is
    with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_testing_newest.txt'), 'w') as f:
        for pdb in dlpacker_val_newest_pdbs:
            f.write(pdb + '\n')


def download_casp13_and_14_targets():

    # download pdbs
    with open('download_casp13_and_14_targets.sh', 'w+') as f_out:
        with open('download_casp13_and_14_targets__TEMPLATE.sh', 'r') as f_in:
            f_out.write(f_in.read().format(BASE_DIR=BASE_DIR))
    os.system('bash download_casp13_and_14_targets.sh')
    os.remove('download_casp13_and_14_targets.sh')


    pdb_lists_from_attnpacker_paper = {
        13: 'T0949, T0950, T0951, T0953s1, T0953s2, T0954, T0955, T0957s1, T0957s2, T0958, T0959, T0960, T0961, T0962, T0963,T0964, T0965, T0966, T0967, T0968s1, T0968s2, T0969, T0970, T0971, T0973, T0974s1, T0974s2, T0975, T0976, T0977,T0978, T0979, T0980s1, T0980s2, T0981, T0982, T0983, T0984, T0985, T0986s1, T0986s2, T0987, T0988, T0989, T0990,T0991, T0992, T0993s1, T0993s2, T0994, T0995, T0996, T0997, T0998, T1000, T1001, T1002, T1003, T1004, T1005, T1006,T1008, T1009, T1010, T1011, T1013, T1014, T1015s1, T1015s2, T1016, T1016_A, T1017s1, T1017s2, T1018, T1019s1,T1019s2, T1020, T1021s1, T1021s2, T1021s3, T1022s1, T1022s2'.replace(' ', '').split(','),
        14: 'T1024, T1025, T1026, T1027, T1028, T1030, T1031, T1032, T1033, T1034, T1035, T1036s1, T1037, T1038, T1039, T1040,T1042, T1043, T1045s1, T1045s2, T1046s1, T1046s2, T1047s1, T1047s2, T1048, T1049, T1052, T1053, T1054, T1055, T1056,T1057, T1058, T1060s2, T1060s3, T1062, T1064, T1065s1, T1065s2, T1067, T1068, T1070, T1072s1, T1073, T1074, T1078,T1079, T1080, T1082, T1083, T1084, T1087, T1088, T1089, T1090, T1091, T1092, T1093, T1094, T1095, T1096, T1098,T1099, T1100'.replace(' ', '').split(',')
    }

    # get pdb_lists
    for casp in [13, 14]:
        pdb_list_in_dir = set([pdbname.strip('.pdb') for pdbname in os.listdir(CASP_PDBS_DIR)])
        pdb_list_in_paper = set(pdb_lists_from_attnpacker_paper[casp])
        
        missing_targets = sorted(list(pdb_list_in_paper - pdb_list_in_dir))
        if len(missing_targets) > 0:
            print('There are %d/%d missing targets for CASP%d, relative to the targets in the AttnPacker paper. Check the download links!!!' % (len(missing_targets), len(pdb_list_in_paper), casp))
            print('The missiing targets are: ', missing_targets)
        else:
            print('There are no missing targets for CASP%d, relative to the targets in the AttnPacker paper.' % casp)
        
        pdb_list = sorted(list(pdb_list_in_dir.intersection(pdb_list_in_paper)))

        # write pdb_list to file
        with open(os.path.join(PDB_LISTS_DIR, 'casp%d_targets_testing.txt' % casp), 'w+') as f:
            for pdb in pdb_list:
                f.write(pdb + '\n')


def make_fasta_files(training_fasta_output_path, testing_fasta_output_path):
    '''
    Makes fasta files for training data and testing data.
    '''
    
    # get training PDB paths
    DLPACKER_PDBS_DIR
    dlpacker_pdbs = []
    for i in tqdm(range(10)):
        with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_training__{i}.txt'), 'r') as f:
            for line in f:
                dlpacker_pdbs.append(line.strip())
    with open(os.path.join(PDB_LISTS_DIR, f'pdb_lists/dlpacker_validation.txt'), 'r') as f:
        for line in f:
            dlpacker_pdbs.append(line.strip())

    training_pdbs = [os.path.join(DLPACKER_PDBS_DIR, pdb+'.pdb') for pdb in dlpacker_pdbs]

    # get testing PDB paths
    casp13_pdblist_file = os.path.join(PDB_LISTS_DIR, 'casp13_targets_testing.txt')
    with open(casp13_pdblist_file, 'r') as f:
        casp13_pdbs = [line.strip() for line in f]
    casp13_pdbs = [os.path.join(CASP_PDBS_DIR, pdb + '.pdb') for pdb in casp13_pdbs]
    casp14_pdblist_file = os.path.join(PDB_LISTS_DIR, 'casp14_targets_testing.txt')
    with open(casp14_pdblist_file, 'r') as f:
        casp14_pdbs = [line.strip() for line in f]
    casp14_pdbs = [os.path.join(CASP_PDBS_DIR, pdb + '.pdb') for pdb in casp14_pdbs]
    testing_pdbs = casp13_pdbs + casp14_pdbs

    def get_sequence_from_pdb(pdbpath):
        '''
        Gets the sequences from a PDB file.
        '''
        parser = PDBParser()
        structure = parser.get_structure('X', pdbpath)
        chains = list(structure.get_chains())
        sequence = []
        for chain in chains:
            for residue in chain.get_residues():
                resname = residue.get_resname()
                if resname in aa_to_one_letter:
                    sequence.append(aa_to_one_letter[resname])
        sequence = ''.join(sequence)

        return sequence
    
    def get_sequences_from_pdbs_and_save_to_fasta_file(pdbpath_list, fasta_output_path):
        sequences = []
        for pdbpath in tqdm(pdbpath_list):
            pdb = pdbpath.split('/')[-1].split('.')[0]
            try:
                sequence = get_sequence_from_pdb(pdbpath)
            except Exception as e:
                print(e)
                print(f'Could not get sequence from {pdbpath}')
                continue
            sequences.append(SeqRecord(seq=Seq(sequence), id=pdb, description=''))
        with open(fasta_output_path, 'w+') as f:
            SeqIO.write(sequences, fasta_output_path, 'fasta')
    
    # make fasta files
    get_sequences_from_pdbs_and_save_to_fasta_file(training_pdbs, training_fasta_output_path)
    get_sequences_from_pdbs_and_save_to_fasta_file(testing_pdbs, testing_fasta_output_path)


def run_cd_hit_for_filtering(training_fasta_output_path, testing_fasta_output_path, filtered_training_fasta_output_path):
    '''
    Runs cd-hit to filter the training data at 50% sequence similarity with the test data.
    Use -n 2
    Make it so that sequence length is just not considered?

    What is below is from the docs:
    
        Basic command:

        cd-hit-2d -i db1 -i2 db2 -o db2novel -c 0.9 -n 5, where

        db1 & db2 are inputs,
        db2novel is output,
        0.9, means 90% identity, is the comparing threshold
        5 is the size of word

        Please note that by default, I only list matches where sequences in db2 are not longer than sequences in db1. You may use options -S2 or -s2 to overwrite this default. You can also run command:
        cd-hit-2d -i db2 -i2 db1 -o db1novel -c 0.9 -n 5

        Choose of word size (same as cd-hit):
        -n 5 for thresholds 0.7 ~ 1.0
        -n 4 for thresholds 0.6 ~ 0.7
        -n 3 for thresholds 0.5 ~ 0.6
        -n 2 for thresholds 0.4 ~ 0.5 --> use this because we want 40%

        

        More options:

        Options, -b, -M, -l, -d, -t, -s, -S, -B, -p, -aL, -AL, -aS, -AS are same to CD-HIT, here are few more cd-hit-2d specific options:

        -i2 input filename for db2 in fasta format, required
        -s2 length difference cutoff for db1, default 1.0
            by default, seqs in db1 >= seqs in db2 in a same cluster
            if set to 0.9, seqs in db1 may just >= 90% seqs in db2
        -S2 length difference cutoff, default 0
            by default, seqs in db1 >= seqs in db2 in a same cluster
            if set to 60, seqs in db2 may 60aa longer than seqs in db1
    
    '''

    command = f'cd-hit-2d -i {testing_fasta_output_path} -i2 {training_fasta_output_path} -o {filtered_training_fasta_output_path} -c 0.5 -n 2 -T 0 -d 0 -M 16000 -s2 0.0'

    os.system(command)


def remake_pdb_lists_from_filtered_fasta_file(filtered_training_fasta_output_path):
    '''
    Remakes the PDB lists from the filtered fasta file.
    '''

    filtered_pdbs = []
    for record in SeqIO.parse(filtered_training_fasta_output_path, 'fasta'):
        filtered_pdbs.append(record.id)
    filtered_pdbs = set(filtered_pdbs)
    print(f'There are {len(filtered_pdbs)} PDBs in the filtered training + validation set.')


    for i in tqdm(range(10)):
        curr_pdbs = []
        with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_training__{i}.txt'), 'r') as f:
            for line in f:
                pdb = line.strip()
                if pdb in filtered_pdbs:
                    curr_pdbs.append(pdb)
        with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_training__{i}.txt'), 'w+') as f:
            for pdb in curr_pdbs:
                f.write(f'{pdb}\n')
        
        curr_pdbs = []
        with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_validation.txt'), 'r') as f:
            for line in f:
                pdb = line.strip()
                if pdb in filtered_pdbs:
                    curr_pdbs.append(pdb)
        with open(os.path.join(PDB_LISTS_DIR, f'dlpacker_validation.txt'), 'w+') as f:
            for pdb in curr_pdbs:
                f.write(f'{pdb}\n')




if __name__ == '__main__':
    main()








