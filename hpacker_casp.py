
'''
Script used to evaluate the reconstruction performance of HPacker on CASP targets.
'''

import os
from hpacker import HPacker

pdb_lists = {
    'CASP13': '/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/casp13_targets_testing.txt',
    'CASP14': '/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/casp14_targets_testing.txt',
}
pdbdir = '/gscratch/scrubbed/gvisan01/dlpacker/casp_targets/'
outputdir = '/gscratch/spe/gvisan01/hpacker/reconstructions/'

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_refinement_iterations', type=int, default=5)
    args = parser.parse_args()

    for dataset in ['CASP13', 'CASP14']:
        pdb_list = pdb_lists[dataset]
        with open(pdb_list, 'r') as f:
            pdbs = f.read().splitlines()
        
        curr_outputdir = os.path.join(outputdir, dataset)
        os.makedirs(curr_outputdir, exist_ok=True)
        
        for i, pdb in enumerate(pdbs):
            print(f'{i+1}/{len(pdbs)} - {pdb}')
            pdbpath = os.path.join(pdbdir, pdb + '.pdb')
        
            hpacker = HPacker(pdbpath, remove_sidechains=True)
            metrics = hpacker.reconstruct_sidechains_and_evaluate(num_refinement_iterations = args.num_refinement_iterations)
            mae_per_angle_4, accuracy_per_angle_4, real_chis, predicted_chis, aas, res_ids_dict, rmsds = metrics
            hpacker.write_pdb(os.path.join(curr_outputdir, f'rec_{pdb}_refinement={args.num_refinement_iterations}.pdb'))    
    


