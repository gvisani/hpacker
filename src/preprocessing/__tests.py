
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import pickle, gzip

from tqdm import tqdm

def stringify(res_id):
    return b'_'.join(res_id).decode('utf-8')

def stringify_array(res_ids):
    return np.array([stringify(res_id) for res_id in res_ids])


if __name__ == '__main__':


    
    # chosen_pdb = b'2qa1'
    # chosen_res_id = stringify([b'I', b'2qa1', b'A', b'438', b' ', b'E']) # --> this one is NOT cut off!

    # with h5py.File('test_struc_info.hdf5', 'r') as f:
    #     structs = f['data']

    #     for struct in structs:
    #         if struct['pdb'] == chosen_pdb:
    #             ALL_CHOSEN_RES_IDS = np.unique(stringify_array(struct['res_ids']))
    #             # print(list(struct['atom_names'][struct['atom_names'] != b'']))
    #             print(list(struct['elements'][struct['atom_names'] != b'']))
    #             print(np.sum(struct['elements'] == b'Zn'))
    #             print(np.sum(struct['atom_names'] == b''))
    #             print()
    #             print(list(struct['atom_names'][stringify_array(struct['res_ids']) == chosen_res_id]))

    #             break
    
    # # check the original neighborhoods, if the res_id has backbone atoms in it
    # # if yes, then the downsampling scheme makes mistakes
    # # if no, then there is something wrong with either the PDB, or with get_structural_info, or with get_neighborhoods
    # # ANSWER: yes, but the atom_names are cut off!! 
    # print()
    # print()
    
    # # chosen_res_id = stringify([b'I', b'2qa1', b'A', b'438', b' ', b'E']) # --> this one is cut off!
    # chosen_res_id = stringify([b'A', b'2qa1', b'A', b'412', b' ', b'L']) # --> this one is also cut off!
    # # chosen_res_id = ALL_CHOSEN_RES_IDS[1]

    # with h5py.File('test_neighborhoods.hdf5', 'r') as f:
    #     nbs = f['data']

    #     for nb in nbs:
    #         if stringify(nb['res_id']) == chosen_res_id:
    #             mask_central_res = stringify_array(nb['res_ids']) == stringify(nb['res_id'])
    #             print(nb['res_id'])
    #             print(nb['atom_names'][mask_central_res])
    #             print(nb['atom_names'][nb['atom_names'] != b''])
    #             break

    from protein_holography_pytorch.preprocessing_faster.utils.constants import BACKBONE_ATOMS, EMPTY_ATOM_NAME
    num_backbone_atoms_before = {}
    with h5py.File('test_neighborhoods.hdf5', 'r') as f:
        nbs = f['data']
        for nb in tqdm(nbs):
            num_backbone_atoms_before[stringify(nb['res_id'])] = np.sum(np.isin(nb['atom_names'][nb['atom_names'] != EMPTY_ATOM_NAME], BACKBONE_ATOMS))
    
    
    res_ids_no_backbone_atoms = []
    num_atoms_in_central_res = []
    with h5py.File('test_downsampled_neighborhoods.hdf5', 'r') as f:
        nbs = f['data']
        ps = f['proportion_sidechain_removed']

        prop_sidechain_removed = []
        num_atoms_in_nbs = []
        for i_nb in tqdm(range(len(nbs))):

            nb = nbs[i_nb]

            real_locs = nb['atom_names'] != b''

            # check that central res sidechain is always removed
            mask_central_res = stringify_array(nb['res_ids']) == stringify(nb['res_id'])
            num_atoms_in_central_res.append(np.sum(mask_central_res))
            if np.sum(mask_central_res) != 3:
                print(nb['res_id'])
                print(np.sum(mask_central_res) )
                res_ids_no_backbone_atoms.append(nb['res_id'])
            # print(nb['atom_names'][mask_central_res])

            prop_sidechain_removed.append(ps[i_nb])
            num_atoms_in_nbs.append(np.sum(real_locs))

            num_backbone_atoms = np.sum(np.isin(nb['atom_names'][nb['atom_names'] != EMPTY_ATOM_NAME], BACKBONE_ATOMS))

            if num_backbone_atoms_before[stringify(nb['res_id'])] != num_backbone_atoms:
                print('Different number of backbone atoms!')
                print(nb['res_id'])

        # there should be a roughly linear inverse relationship between p and num atoms
        plt.scatter(prop_sidechain_removed, num_atoms_in_nbs, alpha=0.2)
        plt.xlabel('p')
        plt.ylabel('num atoms')
        plt.savefig('num_atoms_vs_p.png')

        from collections import Counter
        print(Counter(num_atoms_in_central_res))

            

    exit(0)

    res_ids_no_backbone_atoms = []
    num_atoms_in_central_res = []

    with h5py.File(os.path.join(basedir, f'neighborhoods/downsampled_neighborhoods-{split}-r_max=10-p=random.hdf5'), 'r') as f: # downsampled_neighborhoods-dlpacker_training__0-r_max=10-p=random - neighborhoods_no_central_res-dlpacker_training__0-r_max=10
        
        # num_nbs = len(f['data'])
        num_nbs = 10000

        nbs = f['data']
        ps = f['proportion_sidechain_removed']
    
        prop_sidechain_removed = []
        num_atoms_in_nbs = []
        for i_nb in tqdm(range(num_nbs)):

            nb = nbs[i_nb]

            real_locs = nb['atom_names'] != b''

            # check that central res sidechain is always removed
            mask_central_res = stringify_array(nb['res_ids']) == stringify(nb['res_id'])
            num_atoms_in_central_res.append(np.sum(mask_central_res))
            if np.sum(mask_central_res) != 3:
                print(nb['res_id'])
                print(np.sum(mask_central_res) )
                res_ids_no_backbone_atoms.append(nb['res_id'])
            # print(nb['atom_names'][mask_central_res])

            prop_sidechain_removed.append(ps[i_nb])
            num_atoms_in_nbs.append(np.sum(real_locs))

        # there should be a roughly linear inverse relationship between p and num atoms
        plt.scatter(prop_sidechain_removed, num_atoms_in_nbs, alpha=0.2)
        plt.xlabel('p')
        plt.ylabel('num atoms')
        plt.savefig('num_atoms_vs_p.png')

        from collections import Counter
        print(Counter(num_atoms_in_central_res))

        with gzip.open('res_ids_no_backbone_atoms.pkl.gz', 'wb') as f:
            pickle.dump(res_ids_no_backbone_atoms, f)

        
    # check the original neighborhoods, if the res_id has backbone atoms in it
    # if yes, then the downsampling scheme makes mistakes
    # if no, then there is something wrong with either the PDB, or with get_structural_info, or with get_neighborhoods
    # ANSWER: yes, but the atom_names are cut off!! 

    with gzip.open('res_ids_no_backbone_atoms.pkl.gz', 'rb') as f:
        res_ids_no_backbone_atoms = list(stringify_array(pickle.load(f)))
    
    chosen_res_id = res_ids_no_backbone_atoms[np.random.randint(len(res_ids_no_backbone_atoms))]

    with h5py.File(os.path.join(basedir, f'neighborhoods/neighborhoods-{split}-r_max=10.hdf5'), 'r') as f:
        nbs = f['data']

        nb = nbs[np.random.randint(len(nbs))]
        mask_central_res = stringify_array(nb['res_ids']) == stringify(nb['res_id'])
        print(nb['res_id'])
        print(nb['atom_names'][mask_central_res])
        print()

        for nb in nbs:
            if stringify(nb['res_id']) == chosen_res_id:
                mask_central_res = stringify_array(nb['res_ids']) == stringify(nb['res_id'])
                print(nb['res_id'])
                print(nb['atom_names'][mask_central_res])
                break
    
