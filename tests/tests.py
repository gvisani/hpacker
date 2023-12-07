
import os, sys
import numpy as np

sys.path.append('..')
from hpacker import HPacker

# model_dirs = [
#     '../pretrained_models/initial_guess',
#     '../pretrained_models/refinement',
#     '../pretrained_models/initial_guess_conditioned'
# ]

def test__refine_sidechains_with_specific_sites():
    '''
    PASS - Checked on pymol
    '''

    pdbpath = 'T0950.pdb'

    hpacker = HPacker(pdbpath)

    res_ids_to_refine = hpacker.get_res_ids()[:10]

    hpacker.refine_sidechains(res_ids=res_ids_to_refine)

    hpacker.write_pdb('T0950_partially_refined.pdb')


def test__detect_res_ids_with_missing_sidechains():
    '''
    PASS
    '''

    pdbpath = 'T0950.pdb'

    # no sidechains are missing
    hpacker = HPacker(pdbpath)
    res_ids_with_missing_sidechains = hpacker.detect_res_ids_with_missing_sidechains()
    assert len(res_ids_with_missing_sidechains) == 0

    # 10 sidechains are missing (no GLY!)
    hpacker = HPacker(pdbpath)
    all_res_ids = hpacker.get_res_ids()
    all_resnames = [hpacker.get_resname(res_id) for res_id in all_res_ids]
    res_id_indices_no_gly = list(filter(lambda x: x is not None, [i if resname != 'GLY' else None for i, resname in enumerate(all_resnames)]))
    indices = np.random.choice(res_id_indices_no_gly, size=min(10, len(res_id_indices_no_gly)), replace=False)
    res_ids_no_sidechains = [all_res_ids[i] for i in indices]
    hpacker.remove_sidechains_for_res_ids(res_ids_no_sidechains)
    res_ids_with_missing_sidechains = hpacker.detect_res_ids_with_missing_sidechains()
    assert set(res_ids_no_sidechains) == set(res_ids_with_missing_sidechains)

    # 100 sidechains are missing (no GLY!)
    hpacker = HPacker(pdbpath)
    all_res_ids = hpacker.get_res_ids()
    all_resnames = [hpacker.get_resname(res_id) for res_id in all_res_ids]
    res_id_indices_no_gly = list(filter(lambda x: x is not None, [i if resname != 'GLY' else None for i, resname in enumerate(all_resnames)]))
    indices = np.random.choice(res_id_indices_no_gly, size=min(10, len(res_id_indices_no_gly)), replace=False)
    res_ids_no_sidechains = [all_res_ids[i] for i in indices]
    hpacker.remove_sidechains_for_res_ids(res_ids_no_sidechains)
    res_ids_with_missing_sidechains = hpacker.detect_res_ids_with_missing_sidechains()
    assert set(res_ids_no_sidechains) == set(res_ids_with_missing_sidechains)

    # all sidechains are missing (no GLY!)
    hpacker = HPacker(pdbpath)
    all_res_ids = hpacker.get_res_ids()
    all_resnames = [hpacker.get_resname(res_id) for res_id in all_res_ids]
    res_id_indices_no_gly = list(filter(lambda x: x is not None, [i if resname != 'GLY' else None for i, resname in enumerate(all_resnames)]))
    res_ids_no_sidechains = [all_res_ids[i] for i in res_id_indices_no_gly]
    hpacker.remove_sidechains_for_res_ids(res_ids_no_sidechains)
    res_ids_with_missing_sidechains = hpacker.detect_res_ids_with_missing_sidechains()
    assert set(res_ids_no_sidechains) == set(res_ids_with_missing_sidechains)

    # all sidechains are missing, different function to remove sidechains
    hpacker = HPacker(pdbpath)
    all_res_ids = hpacker.get_res_ids()
    all_resnames = [hpacker.get_resname(res_id) for res_id in all_res_ids]
    res_id_indices_no_gly = list(filter(lambda x: x is not None, [i if resname != 'GLY' else None for i, resname in enumerate(all_resnames)]))
    res_ids_no_sidechains = [all_res_ids[i] for i in res_id_indices_no_gly]
    hpacker.remove_all_sidechains()
    res_ids_with_missing_sidechains = hpacker.detect_res_ids_with_missing_sidechains()
    assert set(res_ids_no_sidechains) == set(res_ids_with_missing_sidechains)


def test__find_residues_in_surrounding():
    '''
    PASS - checked with pymol
    '''

    pdbpath = 'T0950.pdb'
    hpacker = HPacker(pdbpath)

    res_ids_anchor = hpacker.get_res_ids()[10:12]

    print(res_ids_anchor)
    res_ids_in_surrounding = hpacker.find_residues_in_surrounding(res_ids_anchor, 5.0)
    print(res_ids_in_surrounding)


def test__reconstruction():
    '''
    PASS - checked with pymol

    All the v2 should be the same as their counterparts without v2
    All the P should be the same as their counterparts without P
    '''

    ## specify which residues to reconstruct
    pdbpath = 'T0950.pdb'
    for num_refinement_iterations in [0, 5]:

        print(f'num_refinement_iterations: {num_refinement_iterations}')

        print(0)
        # this should not change anything since no residues with no side-chains are detected
        hpacker = HPacker(pdbpath)
        hpacker.reconstruct_sidechains(num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_null_{num_refinement_iterations}_all_rec.pdb')

        print(1)
        hpacker = HPacker(pdbpath)
        hpacker.reconstruct_sidechains(reconstruct_all_sidechains=True, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_{num_refinement_iterations}_all_rec.pdb')

        print(2)
        hpacker = HPacker(pdbpath, remove_sidechains=True)
        hpacker.reconstruct_sidechains(num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_v2_{num_refinement_iterations}_all_rec.pdb')

        print(3)
        hpacker = HPacker(pdbpath)
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        hpacker.reconstruct_sidechains(res_ids_to_reconstruct=res_ids_to_reconstruct, proximity_cutoff_for_refinement=10.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_10_{num_refinement_iterations}_partially_reconstructed__proximity_radius=10.pdb')

        print(4)
        hpacker = HPacker(pdbpath)
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        hpacker.reconstruct_sidechains(res_ids_to_reconstruct=res_ids_to_reconstruct, proximity_cutoff_for_refinement=5.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_5_{num_refinement_iterations}_partially_reconstructed__proximity_radius=5.pdb')

        print(5)
        hpacker = HPacker(pdbpath)
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        hpacker.reconstruct_sidechains(res_ids_to_reconstruct=res_ids_to_reconstruct, proximity_cutoff_for_refinement=0.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_0_{num_refinement_iterations}_partially_reconstructed__proximity_radius=0.pdb')

        print(6)
        hpacker = HPacker(pdbpath) # should be the same as the one above
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        hpacker.reconstruct_sidechains(res_ids_to_reconstruct=res_ids_to_reconstruct, proximity_cutoff_for_refinement=10.0, res_ids_to_refine=[], num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_0v2_{num_refinement_iterations}_partially_reconstructed__proximity_radius=10_but_override_with_no_residues.pdb')

        print(7)
        hpacker = HPacker(pdbpath) # testing mutations
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        res_id_to_resname = {res_id: 'TRP' for res_id in res_ids_to_reconstruct}
        hpacker.reconstruct_sidechains(res_ids_to_reconstruct=res_ids_to_reconstruct, res_id_to_resname=res_id_to_resname, proximity_cutoff_for_refinement=0.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_TRP_0_{num_refinement_iterations}_partially_reconstructed__proximity_radius=0.pdb')

        print(8)
        hpacker = HPacker(pdbpath) # testing mutations, without passing res_ids_to_reconstruct explicitly
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        res_id_to_resname = {res_id: 'TRP' for res_id in res_ids_to_reconstruct}
        hpacker.reconstruct_sidechains(res_id_to_resname=res_id_to_resname, proximity_cutoff_for_refinement=0.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_TRPv2_0_{num_refinement_iterations}_partially_reconstructed__proximity_radius=0.pdb')


    ## provide partial structure
    ## these should give the **same** structures as above
    hpacker = HPacker('T0950.pdb')
    hpacker.remove_sidechains_for_res_ids(hpacker.get_res_ids()[:10])
    hpacker.write_pdb('T0950_partial.pdb')

    pdbpath = 'T0950_partial.pdb'

    for num_refinement_iterations in [0, 5]:

        print(f'num_refinement_iterations: {num_refinement_iterations}')

        print(0)
        hpacker = HPacker(pdbpath)
        hpacker.reconstruct_sidechains(reconstruct_all_sidechains=True, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_P_{num_refinement_iterations}_from_partial_all_rec.pdb')

        print(1)
        hpacker = HPacker(pdbpath)
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        hpacker.reconstruct_sidechains(proximity_cutoff_for_refinement=10.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_P_10_{num_refinement_iterations}_partially_reconstructed__proximity_radius=10.pdb')

        print(2)
        hpacker = HPacker(pdbpath)
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        hpacker.reconstruct_sidechains(proximity_cutoff_for_refinement=5.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_P_5_{num_refinement_iterations}_partially_reconstructed__proximity_radius=5.pdb')

        print(3)
        hpacker = HPacker(pdbpath)
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        hpacker.reconstruct_sidechains(proximity_cutoff_for_refinement=0.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_P_0_{num_refinement_iterations}_partially_reconstructed__proximity_radius=0.pdb')

        print(4)
        hpacker = HPacker(pdbpath) # should be the same as the one above
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        hpacker.reconstruct_sidechains(proximity_cutoff_for_refinement=10.0, res_ids_to_refine=[], num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_P_0v2_{num_refinement_iterations}_partially_reconstructed__proximity_radius=10_but_override_with_no_residues.pdb')

        print(5)
        hpacker = HPacker(pdbpath) # testing mutations
        res_ids_to_reconstruct = hpacker.get_res_ids()[:10]
        res_id_to_resname = {res_id: 'TRP' for res_id in res_ids_to_reconstruct}
        hpacker.reconstruct_sidechains(res_id_to_resname=res_id_to_resname, proximity_cutoff_for_refinement=0.0, num_refinement_iterations=num_refinement_iterations)
        hpacker.write_pdb(f'T0950_P_TRP_0_{num_refinement_iterations}_partially_reconstructed__proximity_radius=0.pdb')



if __name__ == '__main__':
    # test__refine_sidechains_with_specific_sites()
    # test__detect_res_ids_with_missing_sidechains()
    # test__find_residues_in_surrounding()
    test__reconstruction()