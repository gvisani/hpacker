
import os
import json
from tqdm import tqdm

from Bio.PDB import PDBParser, PDBIO, Superimposer, Structure, Chain, Model, Residue, Atom, Selection, NeighborSearch
from Bio.PDB.ic_rebuild import structure_rebuild_test

# the lines below hide neverending warnings from biopython
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio import BiopythonDeprecationWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
warnings.simplefilter('ignore', BiopythonDeprecationWarning)
warnings.simplefilter('ignore', RuntimeWarning)

import math
import numpy as np

from copy import deepcopy

try:
    from .constants import *
    from .compute_reconstruction_params import compute_biopyton_reconstruction_parameters, REC_PARAMS_FILEPATH
except ImportError:
    from constants import *
    from compute_reconstruction_params import compute_biopyton_reconstruction_parameters, REC_PARAMS_FILEPATH


from typing import *

class FullStructureReconstructor(object):

    def __init__(self,
                 pdbpath: str,
                 filter: bool = True, # remove hydrogens, water, and altloc
                 remove_sidechains: bool = False,
                 keep_CB: bool = False,
                 virtual_CB: bool = False,
                 recompute_rec_params: bool = False,
                 pdb_list_for_rec_params: Optional[List[str]] = None,
                 pdbdir_for_rec_params: Optional[str] = None,
                 use_extended_symmetries: bool = False): # whether to use more "imperfect" symmetries (the ones in AttnPacker) than the ones we know to be strictly true, when computing residue-level RMSD
        
        self.parser = PDBParser(PERMISSIVE = 1) # PDB files reader
        self.sup = Superimposer() # Superimposer for aligning structuees and residues
        self.io = PDBIO() # PDB files writer
        self.altloc = ['A', 'B']  # initial altloc selection order preference
        self.use_extended_symmetries = use_extended_symmetries

        self.pdbpath = pdbpath
        self._read_pdb(pdbpath, filter=filter)

        ## save original structure for reference
        # when evaluating a model's reconstruction, we will need to compare the original structure with the reconstructed one
        #   in that case, our original structure will be 
        self.original_structure = self.structure.copy()

        assert not (keep_CB and virtual_CB), 'Cannot specify both keep_CB and virtual_CB.'
        self.keep_CB = keep_CB
        self.virtual_CB = virtual_CB

        if remove_sidechains:
            self._remove_all_sidechains(self.structure, keep_CB=keep_CB)
        
        ## make copy of structure
        ## workaround for biopython's weird behavior when it comes to internal_coords,
        ##  whereby I can't updae internal_coord if I compute them once before adding the atoms
        ## the idea of the workaround is the following:
        ##      - use one structure to compute the locations of sidechains, a structure where I will have to add all the atoms beforehand, which is why I can't use it to collect negihborhoods
        ##      - use another structure to collect neighborhoods, a structure where I will have to add all the atoms afterwards from the other structure, aligning the residues by the backbone
        self.structure_copy = self.structure.copy()

        ## make dictionaries that allow us to access residue objects and resnames directly from the res_id
        self.res_id_to_residue = self._make_res_id_to_residue_dict(self.structure)
        self.res_id_to_resname = self._make_res_id_to_resname_dict(self.structure)
        
        self.original_structure__res_id_to_residue = self._make_res_id_to_residue_dict(self.original_structure)
        self.original_structure__res_id_to_resname = self._make_res_id_to_resname_dict(self.original_structure)

        self.copy__res_id_to_residue = self._make_res_id_to_residue_dict(self.structure_copy)
        self.copy__res_id_to_resname = self._make_res_id_to_resname_dict(self.structure_copy)
        
        self._load_reconstruction_params(pdb_list_for_rec_params, pdbdir_for_rec_params, recompute_rec_params)

        if virtual_CB:
            self._add_all_dummy_atoms(self.structure_copy)
            self.add_all_virtual_CB()


    def _read_pdb(self, pdbpath, filter: bool = True):
        
        self.structure = self.parser.get_structure(os.path.basename(pdbpath).strip('.pdb'), pdbpath)

        if filter:
            self._remove_hydrogens(self.structure) # we never use hydrogens
            # self._convert_mse(self.structure)      # converts MSE to MET NOTE: We currently don't use this! as we don't apply this change in the training process of our models.
            self._remove_water(self.structure)     # waters are not used anyway
            try:
                self._remove_altloc(self.structure)    # only leave one altloc
            except:
                # sometimes there's an error here... but I think I can just ignore it in good faith
                pass


    def _remove_hydrogens(self, structure: Structure):
        ''' Taken from DLPacker: https://github.com/nekitmm/DLPacker/blob/main/dlpacker.py '''
        # Removes all hydrogens.
        # This code is not suited to work with hydrogens
        for residue in Selection.unfold_entities(structure, 'R'):
            remove = []
            for atom in residue:
                if atom.element == 'H': remove.append(atom.get_id())
            for i in remove: residue.detach_child(i)
    
    def _convert_mse(self, structure: Structure):
        ''' Taken from DLPacker: https://github.com/nekitmm/DLPacker/blob/main/dlpacker.py '''
        ''' NOTE: We currently don't use this! as we don't apply this change in the training process of our models. '''
        # Changes MSE residues to MET
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() == 'MSE':
                residue.resname = 'MET'
                for atom in residue:
                    if atom.element == 'SE':
                        new_atom = Atom.Atom('SD',\
                                             atom.coord,\
                                             atom.bfactor,\
                                             atom.occupancy,\
                                             atom.altloc,\
                                             'SD  ',\
                                             atom.serial_number,\
                                             element='S')
                        residue.add(new_atom)
                        atom_to_remove = atom.get_id()
                residue.detach_child(atom_to_remove)
    
    def _remove_water(self, structure: Structure):
        ''' Taken from DLPacker: https://github.com/nekitmm/DLPacker/blob/main/dlpacker.py '''
        # Removes all water molecules
        residues_to_remove = []
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() == 'HOH':
                residues_to_remove.append(residue)
        for r in residues_to_remove:
            r.get_parent().detach_child(r.get_id())
    
    def _remove_altloc(self, structure: Structure):
        ''' Taken from DLPacker: https://github.com/nekitmm/DLPacker/blob/main/dlpacker.py '''
        # Only leaves one altloc with the largest sum of occupancies
        total_occupancy = {}
        for atom in Selection.unfold_entities(structure, 'A'):
            if atom.is_disordered():
                for alt_atom in atom:
                    occupancy = alt_atom.get_occupancy()
                    if alt_atom.get_altloc() in total_occupancy:
                        total_occupancy[alt_atom.get_altloc()] += occupancy
                    else:
                        total_occupancy[alt_atom.get_altloc()] = occupancy

        # optionally select B altloc if it has larger occupancy
        # rare occasion, but it happens
        if 'A' in total_occupancy and 'B' in total_occupancy:
            if total_occupancy['B'] > total_occupancy['A']:
                self.altloc = ['B', 'A']
        
        # only leave one altloc
        disordered_list, selected_list = [], []
        for residue in Selection.unfold_entities(structure, 'R'):
            for atom in residue:
                if atom.is_disordered():
                    disordered_list.append(atom)
                    # sometimes one of the altlocs just does not exist!
                    try:
                        selected_list.append(atom.disordered_get(self.altloc[0]))
                    except:
                        selected_list.append(atom.disordered_get(self.altloc[1]))
                    selected_list[-1].set_altloc(' ')
                    selected_list[-1].disordered_flag = 0
        
        for d, a in zip(disordered_list, selected_list):
            p = d.get_parent()
            p.detach_child(d.get_id())
            p.add(a)
    
    def _is_hetero(self, residue: Residue):
        return residue.id[0] != ' '
    
    def _make_res_id_to_residue_dict(self, structure: Structure):
        '''
        Returns a dictionary mapping each residue's res_id to the residue object itself.
        '''
        res_id_to_residue = {}
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.resname in THE20 and not self._is_hetero(residue):
                res_id_to_residue[self._get_residue_id(residue)] = residue
        return res_id_to_residue


    def _make_res_id_to_resname_dict(self, structure: Structure):
        '''
        Returns a dictionary mapping each residue's res_id to its amino acid type.
        '''
        res_id_to_resname = {}
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.resname in THE20 and not self._is_hetero(residue):
                res_id_to_resname[self._get_residue_id(residue)] = residue.resname
        return res_id_to_resname
    
    def _update_resnames(self, res_id_to_resname_to_update: Dict[Tuple, str], original_structure: bool = False, copy_structure: bool = False):
        '''
        Updates the res_id_to_resname dictionary.
        Used whenever we want to change the amino acid type of a residue, for mutations and whatnot.
        '''
        assert not (original_structure and copy_structure), 'Cannot specify both original_structure and copy_structure.'

        def _check_keys_exist(previous_dict, update_dict):
            for key in update_dict.keys():
                assert key in previous_dict, f'Key {key} not found in previous dictionary.'

        if original_structure:
            _check_keys_exist(self.original_structure__res_id_to_resname, res_id_to_resname_to_update)
            self.original_structure__res_id_to_resname.update(res_id_to_resname_to_update)
        elif copy_structure:
            _check_keys_exist(self.original_structure__res_id_to_resname, res_id_to_resname_to_update)
            self.copy__res_id_to_resname.update(res_id_to_resname_to_update)
        else:
            _check_keys_exist(self.original_structure__res_id_to_resname, res_id_to_resname_to_update)
            self.res_id_to_resname.update(res_id_to_resname_to_update)
        
        for res_id, resname in res_id_to_resname_to_update.items():
            # update residue object with new resname
            residue = self._get_residue_from_res_id(res_id, original_structure=original_structure, copy_structure=copy_structure)
            residue.resname = resname

    def get_res_ids(self):
        '''
        Returns a list of all res_ids in the structure.
        '''
        return list(self.res_id_to_residue.keys())

    def get_resname(self, res_id: Tuple, original_structure: bool = False, copy_structure: bool = False):
        '''
        Returns the amino acid type of the specified res_id.
        '''
        return self._get_resname_from_res_id(res_id, original_structure=original_structure, copy_structure=copy_structure)

    def _get_residue_from_res_id(self, res_id: Tuple, original_structure: bool = False, copy_structure: bool = False):
        '''
        Returns the residue object corresponding to the specified res_id.
        '''
        assert not (original_structure and copy_structure), 'Cannot specify both original_structure and copy_structure.'

        if original_structure:
            return self.original_structure__res_id_to_residue[res_id]
        elif copy_structure:
            return self.copy__res_id_to_residue[res_id]
        else:
            return self.res_id_to_residue[res_id]


    def _get_resname_from_res_id(self, res_id: Tuple, original_structure: bool = False, copy_structure: bool = False):
        '''
        Returns the amino acid type corresponding to the specified res_id.
        '''
        assert not (original_structure and copy_structure), 'Cannot specify both original_structure and copy_structure.'

        if original_structure:
            return self.original_structure__res_id_to_resname[res_id]
        elif copy_structure:
            return self.copy__res_id_to_resname[res_id]
        else:
            return self.res_id_to_resname[res_id]
    

    def _get_residue_id(self, residue: Residue):
        # return a tuple of the form (chain, resnum, icode)
        # we don't return the resname because the res_id just needs to point to a particular SITE in the structure, for which we may want to change the resname
        # and sometimes, we don't even have the resname
        chain = residue.get_full_id()[2]
        resnum = residue.get_id()[1]
        icode = residue.get_id()[2]
        return (chain, resnum, icode)
    

    def _load_reconstruction_params(self, pdb_list: Optional[List[str]] = None, pdbdir: Optional[str] = None, recompute_params: bool = False):
        '''
        Loads reconstruction parameters from the file specified in REC_PARAMS_FILEPATH.
        If the reconstruction params aren't found, computes them and saves them to the file, using the optionally-provided list of PDBs.
        '''
        if os.path.exists(REC_PARAMS_FILEPATH) and not recompute_params:
            with open(REC_PARAMS_FILEPATH, 'r') as f:
                self.RECONSTRUCTION_PARAMS = json.load(f)
        else:
            print('Reconstruction parameters not found. Computing them now.')
            assert pdb_list is not None, 'Must provide a list of PDBs to compute reconstruction parameters.'
            assert pdbdir is not None, 'Must provide a directory containing the PDBs to compute reconstruction parameters.'

            self.RECONSTRUCTION_PARAMS = compute_biopyton_reconstruction_parameters(pdb_list, pdbdir)

            with open(REC_PARAMS_FILEPATH, 'w') as f:
                json.dump(self.RECONSTRUCTION_PARAMS, f, indent=4)

    def _remove_sidechain(self,
                          residue: Residue,
                          keep_CB: bool = False):
        '''
        Removes sidechain at desired site from internal representation of structure.
        Helpful for the purposes of inducing mutations.
        '''
        bb_atoms = {'N', 'CA', 'C', 'O'}
        if keep_CB:
            bb_atoms.add('CB')
        
        atoms_to_detach = []
        for atom in residue:
            if atom.id not in bb_atoms:
                atoms_to_detach.append(atom.id)
        for atom_id in atoms_to_detach:
            residue.detach_child(atom_id)


    def _remove_all_sidechains(self, structure: Structure, keep_CB: bool = False):
        '''
        Removes all sidechains from internal representation of structure.
        NOTE: DOES NOT remove hetero residues. This is not ideal, BUT the hetero residues do not play nice with the internal_coords module, so we have to keep them.
        '''
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.resname in THE20 and not self._is_hetero(residue):
                self._remove_sidechain(residue, keep_CB=keep_CB)
    
    def add_all_virtual_CB(self):
        res_ids = self.get_res_ids()
        for res_id in res_ids:
            self.add_atoms_to_sidechain(res_id, ['CB'], [])

    def _add_all_dummy_atoms(self, structure: Structure):
        '''
        Adds all dummy atoms to internal representation of structure. NOTE: it depends on the amino acid type (resname).
        '''
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.resname in THE20:
                DUMMY_COORD = residue.child_dict['CA'].coord + np.array([1, 1, 1])
                for atom_name in self.RECONSTRUCTION_PARAMS[residue.resname]['sidechain atoms']:
                    if atom_name not in residue.child_dict:
                        self._add_atom(residue, atom_name, DUMMY_COORD)
    
    def _add_dummy_atoms_for_res_ids(self, res_ids: List[Tuple], copy_structure: bool = True):
        '''
        Adds dummy atoms to internal representation of structure for requested residues. NOTE: it depends on the amino acid type (resname).
        Choise of "real" structure or copy structure. Defaults to copy structure.
        '''
        for res_id in res_ids:
            residue = self._get_residue_from_res_id(res_id, copy_structure=copy_structure)
            if residue.resname in THE20:
                DUMMY_COORD = residue.child_dict['CA'].coord + np.array([1, 1, 1])
                for atom_name in self.RECONSTRUCTION_PARAMS[residue.resname]['sidechain atoms']:
                    self._add_atom(residue, atom_name, DUMMY_COORD)

    def _add_atom(self, residue: Residue, atom_name: str, coord: np.ndarray):
        # dummy values follow DLPacker
        # NOTE: we might have to artificially remove altloc? though idk I think pyrosetta handles it automatically
        # pyrosetta does automatically handle occupancy (it just keeps the highest occupancy)

        atom = Atom.Atom(atom_name,
                            coord,
                            0, # dummy bfactor
                            1, # dummy occupancy
                            ' ', # dummy altloc
                            atom_name, # we put fullname in without spaces because who cares anyway
                            2, # dummy serial number
                            element = atom_name[:1]) # for protein atoms, this is always true!
        residue.add(atom)
    
    def _get_parent_structure(self, residue: Residue):
        ''' Taken from DLPacker: https://github.com/nekitmm/DLPacker/blob/main/dlpacker.py '''
        # Returns the parent structure of the given residue
        return residue.get_parent().get_parent().get_parent()
    
    def _align_structures(self, structure_a, structure_b):
        ''' Taken from DLPacker: https://github.com/nekitmm/DLPacker/blob/main/dlpacker.py '''
        # Aligns the second structure to the first one using backbone atoms
        bb_a, bb_b = [], []
        residues_a = Selection.unfold_entities(structure_a, 'R')
        residues_b = Selection.unfold_entities(structure_b, 'R')
        for a, b in zip(residues_a, residues_b):
            for n in BB_ATOMS:
                if a.has_id(n):
                    bb_a.append(a[n])
                    bb_b.append(b[n])
                    
        self.sup.set_atoms(bb_a, bb_b)
        self.sup.apply(structure_b)
    
    def _align_residues(self, residue1: Residue, residue2: Residue, return_copy: bool = False):
        ''' Inspired from DLPacker: https://github.com/nekitmm/DLPacker/blob/main/dlpacker.py '''
        # Aligns the second residue to the first one using backbone atoms
        bb_a, bb_b = [], []
        for n in BB_ATOMS:
            if residue1.has_id(n) and residue2.has_id(n):
                bb_a.append(residue1[n])
                bb_b.append(residue2[n])
                
        self.sup.set_atoms(bb_a, bb_b)
        if return_copy:
            residue2_copy = residue2.copy()
            self.sup.apply(residue2_copy)
            return residue2_copy
        else:
            self.sup.apply(residue2)


    def _is_length(self, atomic_combo: str):
        return len(atomic_combo.split(':')) == 2

    def _is_angle(self, atomic_combo: str):
        return len(atomic_combo.split(':')) in {3, 4}
    
    def _is_dihedral_with_dependence(self, atomic_combo_value: Union[float, Iterable]):
        return not isinstance(atomic_combo_value, float)
    
    def _add_offset(self, angle: float, offset: float):
        return (((angle + offset) + 180) % 360) - 180

    # @profile
    def _update_internal_coords_with_chi_angles(self,
                                                residue: Residue,
                                                chi_angles: List[float],
                                                update_atomic_coords=True):
        '''
        Updates the sidechain with desired chi angles to the internal_coords of the residue object.
        Also updates the cartesian coordinates of the whole structure, which might apply a rigid-body transformation to the whole structure.
        Assumes that resnames have already been updated as desired already.

        '''


        ## add chi angle values
        ## NOTE: this has to be done before adding the other internal coordinates, because some of the dihedrals depend on the chi angles
        for i, chi_angle in enumerate(chi_angles):
            residue.internal_coord.set_angle(f'chi{i+1}', chi_angle)
        
        ## add all the other internal coordinates
        for atomic_combo in self.RECONSTRUCTION_PARAMS[residue.resname]:
            if atomic_combo != 'sidechain atoms':
                atomic_combo_value = self.RECONSTRUCTION_PARAMS[residue.resname][atomic_combo]

                if self._is_dihedral_with_dependence(atomic_combo_value):
                    try:
                        residue.internal_coord.set_angle(atomic_combo, self._add_offset(residue.internal_coord.get_angle(atomic_combo_value[0]), atomic_combo_value[1]))
                    except Exception as e:
                        # print('Error in _update_internal_coords_with_chi_angles')
                        # print(e)
                        # print(residue.full_id)
                        # print(residue.resname)
                        # print(list(residue.get_atoms()))
                        continue
                elif self._is_length(atomic_combo):
                    residue.internal_coord.set_length(atomic_combo, atomic_combo_value)
                elif self._is_angle(atomic_combo):
                    residue.internal_coord.set_angle(atomic_combo, atomic_combo_value)
                else:
                    raise ValueError(f'Unrecognized atomic combo: {atomic_combo}.')
                
        if update_atomic_coords:
            ## update cartesian coordinates. NOTE: this might apply a rigid-body transformation to the whole structure
            residue.get_parent().internal_to_atom_coordinates()
    
    def _update_internal_coords_with_specified_chi_angles_and_atom_names(self, residue: Residue, atom_names: List[str], chi_angles: List[float] = []):
        '''
        updates the internal_coords of the residue object with the internal coords pertinent to the specified atom_names.
        '''
        ## add chi angle values
        ## NOTE: this has to be done before adding the other internal coordinates, because some of the dihedrals depend on the chi angles
        for i, chi_angle in enumerate(chi_angles):
            residue.internal_coord.set_angle(f'chi{i+1}', chi_angle)
        for j in range(len(chi_angles), 4):
            residue.internal_coord.set_angle(f'chi{j+1}', 45) # some dummy chi angle so that errors are not thrown?

        for atomic_combo in self.RECONSTRUCTION_PARAMS[residue.resname]:
            if atomic_combo != 'sidechain atoms':
                # atoms_in_combo = atomic_combo.split(':')
                # for atom in atoms_in_combo:
                #     if atom in atom_names:
                atomic_combo_value = self.RECONSTRUCTION_PARAMS[residue.resname][atomic_combo]

                if self._is_dihedral_with_dependence(atomic_combo_value):
                    try:
                        residue.internal_coord.set_angle(atomic_combo, self._add_offset(residue.internal_coord.get_angle(atomic_combo_value[0]), atomic_combo_value[1]))
                    except:
                        print('Error')
                        continue
                elif self._is_length(atomic_combo):
                    residue.internal_coord.set_length(atomic_combo, atomic_combo_value)
                elif self._is_angle(atomic_combo):
                    residue.internal_coord.set_angle(atomic_combo, atomic_combo_value)
                else:
                    raise ValueError(f'Unrecognized atomic combo: {atomic_combo}.')
        
        ## update cartesian coordinates. NOTE: this might apply a rigid-body transformation to the whole structure
        residue.get_parent().internal_to_atom_coordinates()
    
    
    def add_atoms_to_sidechain(self, res_id: Tuple[str], atom_names: List[str], chi_angles: List[float] = []):
        '''
        NOTE: the relevant sidechains must also be added. TODO: throw Warnings or errors if they aren't.
        '''
        residue_in_final_struc = self._get_residue_from_res_id(res_id)
        residue_in_copy = self._get_residue_from_res_id(res_id, copy_structure=True)

        # make dummy chain with only one residue!
        residue_in_copy_copy = residue_in_copy.copy()
        dummy_chain = Chain.Chain('A')
        dummy_chain.add(residue_in_copy_copy)
        dummy_chain.atom_to_internal_coordinates()

        # update CB internal coords
        self._update_internal_coords_with_specified_chi_angles_and_atom_names(residue_in_copy_copy, atom_names, chi_angles)

        self._align_residues(residue_in_final_struc, residue_in_copy_copy)

        self._copy_over_residue_requested_atoms(residue_in_final_struc, residue_in_copy_copy, atom_names)

    
    # def add_all_CBs(self):
    #     '''
    #     Assumes dummy atoms have been added to the structure.
    #     '''
    #     for res_id in tqdm(self.get_res_ids()):
    #         self._init_CB(res_id)
    
    def _copy_over_residue_requested_atoms(self,
                                            residue_to_copy_to: Residue,
                                            residue_to_copy_from: Residue,
                                            atom_names: List[str]):
        '''
        Copies over the requested atoms from the residue_to_copy_from to the residue_to_copy_to.
        '''
        for atom in residue_to_copy_from:
            if atom.id in atom_names:
                residue_to_copy_to.add(atom.copy())


    def _copy_over_residue_sidechain(self,
                                     residue_to_copy_to: Residue,
                                     residue_to_copy_from: Residue):
        '''
        Copies over the sidechain from the residue_to_copy_from to the residue_to_copy_to.
        '''
        atoms_already_present = set([atom.id for atom in residue_to_copy_to.get_atoms()])
        for atom in residue_to_copy_from:
            if atom.id not in BB_ATOMS:
                if atom.id not in atoms_already_present:
                    residue_to_copy_to.add(atom.copy())

    # @profile
    def _add_sidechain_with_chi_angles(self,
                                       res_id: Tuple,
                                       chi_angles: List[float]):
        '''
        Updates both self.structure_copy and self.structure with the desired sidechain at the desired site.
        Assumes that resnames have already been updated as desired already.
        Assumes that dummy atoms have already been added to the copy structure.
        '''
        residue_in_final_struc = self._get_residue_from_res_id(res_id)

        residue_in_copy = self._get_residue_from_res_id(res_id, copy_structure=True)

        # make dummy chain with only one residue!
        residue_in_copy_copy = residue_in_copy.copy()
        dummy_chain = Chain.Chain('A')
        dummy_chain.add(residue_in_copy_copy)
        dummy_chain.atom_to_internal_coordinates()

        self._update_internal_coords_with_chi_angles(residue_in_copy_copy, chi_angles)

        self._align_residues(residue_in_final_struc, residue_in_copy_copy)

        self._copy_over_residue_sidechain(residue_in_final_struc, residue_in_copy_copy)
    

    def _add_multiple_sidechains_with_chi_angles(self, res_id_to_chi_angles: Dict[Tuple, List[float]]):
        '''
        Adds multiple sidechains to the self.structure_copy, and carries them over to self.structure as well
        Assumes that resnames have already been updated as desired already.
        '''

        for res_id, chi_angles in res_id_to_chi_angles.items():
            residue_in_copy = self._get_residue_from_res_id(res_id, copy_structure=True)
            self._update_internal_coords_with_chi_angles(residue_in_copy, chi_angles, update_atomic_coords=False)
        
        ## update cartesian coordinates of copy structure
        ## NOTE: this is kind of an expensive operation, so it's better to use this function only for reconstructing the whole structure at once, or at least a large proportion if it
        self.structure_copy.internal_to_atom_coordinates()

        ## carry over sidechains to the final structure
        for res_id in res_id_to_chi_angles.keys():
            residue_in_final_struc = self._get_residue_from_res_id(res_id)
            residue_in_copy = self._get_residue_from_res_id(res_id, copy_structure=True)
            copy_of_residue_in_copy = self._align_residues(residue_in_final_struc, residue_in_copy, return_copy=True)
            self._copy_over_residue_sidechain(residue_in_final_struc, copy_of_residue_in_copy)

    def add_sidechain(self,
                       res_id: Tuple):
        '''
        Adds a sidechain of the specified amino acid type to the internal representation of the structure.
        Calls _add_sidechain_with_chi_angles with the appropriate chi angles.
        '''
        raise NotImplementedError('Child class must implement this method.')


    def populate_sidechains(self,
                            res_id_to_resname: Optional[Dict[Tuple, str]] = None,
                            **kwargs):
        '''
        Populates the internal representation of the structure with the desired amino acid compositions.
        If provided, use the specified amino acid types for the specified residues.
        Otherwise, use the amino acid types specified in the PDB file.
        '''
        if res_id_to_resname is not None:
            self._update_resnames(res_id_to_resname)
        
        raise NotImplementedError('Child class must implement this method. But remember to update resnames!')
    

    def write_pdb(self,
                  output_pdbpath: str,
                  original_structure = False,
                  copy_structure = False):
        '''
        Writes the current structure to a PDB file.
        '''
        assert not (original_structure and copy_structure), 'Cannot specify both original_structure and copy_structure.'

        if original_structure:
            structure_to_write = self.original_structure
        elif copy_structure:
            structure_to_write = self.structure_copy
        else:
            structure_to_write = self.structure
        
        self.io.set_structure(structure_to_write)
        self.io.save(output_pdbpath)
    
    def _get_chi_angles(self, residue: Residue):
        '''
        Returns the chi angles of the specified residue.
        '''
        chi_angles = []
        for i in range(1, 5):
            chi_angle = residue.internal_coord.get_angle(f'chi{i}')
            if chi_angle is not None:
                chi_angles.append(chi_angle)
        return chi_angles
    

    def _finalize_structure(self):
        '''
        Simply copies over the copy-structure to the "real" structure that we want to save.
        The benefit of keeping a "real" structure is that it's aligned with the original structure, which is useful for computing RMSD.
        '''
        for res_id in tqdm(self.res_id_to_residue.keys()):
            residue_in_final_struc = self._get_residue_from_res_id(res_id)
            residue_in_copy = self._get_residue_from_res_id(res_id, copy_structure=True)
            copy_of_residue_in_copy = self._align_residues(residue_in_final_struc, residue_in_copy, return_copy=True)
            self._copy_over_residue_sidechain(residue_in_final_struc, copy_of_residue_in_copy)

    # @profile
    def reconstruct_from_self_one_by_one(self):
        '''
        Reconstructs the structure from itself. Designed to test the null error of the reconstruction process..
        '''
        
        ## add all dummy atoms to the copy structure
        self._add_all_dummy_atoms(self.structure_copy)

        ## compute internal coords for copy structure and the original structure
        self.original_structure.atom_to_internal_coordinates()

        ## iterate over res_ids, comupting chi angles and adding sidechains
        for res_id in self.res_id_to_residue.keys():
            try:
                chi_angles = self._get_chi_angles(self._get_residue_from_res_id(res_id, original_structure=True))
            except:
                print('Error')
                continue
            self._add_sidechain_with_chi_angles(res_id, chi_angles)
    
    def reconstruct_from_self_all_at_once(self):
        ## add all dummy atoms to the copy structure
        self._add_all_dummy_atoms(self.structure_copy)

        ## compute internal coords for copy structure and the original structure
        self.structure_copy.atom_to_internal_coordinates()
        self.original_structure.atom_to_internal_coordinates()

        res_id_to_chi_angles = {}
        ## iterate over res_ids, comupting chi angles and adding sidechains
        for res_id in self.res_id_to_residue.keys():
            try:
                chi_angles = self._get_chi_angles(self._get_residue_from_res_id(res_id, original_structure=True))
            except:
                print('Error')
                continue
            res_id_to_chi_angles[res_id] = chi_angles

        self._add_multiple_sidechains_with_chi_angles(res_id_to_chi_angles)
    
    def standardize_original_structure(self):
        '''
        Just reconstructs from sels, but sets the original structure to the reconstructed version
        '''
        copy_of_structure = self.structure.copy()
        copy_of_copy_structure = self.structure_copy.copy()
        
        self.reconstruct_from_self_all_at_once() # now self.structure is the standardized version of the original structure

        self.original_structure = self.structure.copy()
        self.structure_copy = copy_of_copy_structure
        self.structure = copy_of_structure
    

    def update_internal_coords_with_average_values(self):
        '''
        The idea here is just to modify the internal coords of each residue, except for, of course, the chi-angles.
        TODO: DOES NOT WORK
        '''
        print('Warning: this function ("update_internal_coords_with_average_values") does not work!')
        self.structure.atom_to_internal_coordinates()

        all_res_ids = self.get_res_ids()
        for res_id in all_res_ids:
            residue = self._get_residue_from_res_id(res_id)
            self._update_internal_coords_with_chi_angles(residue, [], update_atomic_coords=False) # no need to compute chi angles, they are already there!

        self.structure.internal_to_atom_coordinates()

        self._align_structures(self.original_structure, self.structure)


    def plot_average_rmsd_per_resname(self, rmsd_per_res_id):
        '''
        Plots the average RMSD per amino acid type.
        '''
        resname_to_rmsds = {}
        for res_id, rmsd in rmsd_per_res_id.items():
            resname = self._get_resname_from_res_id(res_id)
            if resname not in resname_to_rmsds:
                resname_to_rmsds[resname] = []
            resname_to_rmsds[resname].append(rmsd)
        
        resnames = []
        avg_rmsds = []
        for resname, rmsds in resname_to_rmsds.items():
            resnames.append(resname)
            avg_rmsds.append(np.mean(np.array(rmsds).flatten())) # --> this makes it work with both a list of rmsds per res_id (coming from multiple structures, which may share res_ids but we don't care for the urposes of this plot), or a single rmsd per res_id
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 4))
        plt.bar(resnames, avg_rmsds)
        plt.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
        plt.ylabel('Average RMSD', fontsize=14)
        plt.savefig('average_rmsd_per_resname.png')
        plt.close()
    
    def structure_rmsd(self):
        '''
        Returns the RMSD between the original structure and the reconstructed structure. Computed over the non-hetero sidechain atoms only.
        '''
        res_id_to_sd = self._compute_sidechain_rmsd_per_res_id(return_sd_instead=True)
        distances = np.hstack(list(res_id_to_sd.values()))
        return np.sqrt(np.mean(distances))

    def structure_square_distances(self):
        res_id_to_sd = self._compute_sidechain_rmsd_per_res_id(return_sd_instead=True)
        distances = np.hstack(list(res_id_to_sd.values()))
        return distances

    def _compute_sidechain_rmsd_per_res_id(self, return_sd_instead: bool = False):
        '''
        Returns a dictionary mapping each residue's res_id to the RMSD between the sidechains of the original structure and the reconstructed structure.
        '''
        rmsd_per_res_id = {}
        for res_id in self.get_res_ids():
            if self._get_resname_from_res_id(res_id) in {'GLY', 'ALA'}: continue

            residue_in_final_struc = self._get_residue_from_res_id(res_id)
            residue_in_original_struc = self._get_residue_from_res_id(res_id, original_structure=True)
            copy_of_residue_in_original_struc = self._align_residues(residue_in_final_struc, residue_in_original_struc, return_copy=True)

            try:
                rmsd_per_res_id[res_id] = self._rmsd_of_sidechain(residue_in_final_struc, copy_of_residue_in_original_struc, return_sd_instead=return_sd_instead)
            except Exception as e:
                # print('Error in computing RMSD for res_id', res_id)
                # print(e)
                continue
        
        return rmsd_per_res_id
    
    def _compute_sidechain_rmsd_per_res_id_with_standardized_ground_truth(self, return_sd_instead: bool = False):
        '''
        Returns a dictionary mapping each residue's res_id to the RMSD between the sidechains of the original structure and the reconstructed structure.
        '''
        rmsd_per_res_id = {}
        for res_id in self.get_res_ids():
            if self._get_resname_from_res_id(res_id) in {'GLY', 'ALA'}: continue

            residue_in_final_struc = self._get_residue_from_res_id(res_id)
            residue_in_original_struc = self._get_residue_from_res_id(res_id, original_structure=True)

            # make dummy chain with only one residue!
            residue_in_original_struc_copy = residue_in_original_struc.copy()
            residue_in_original_struc_copy.internal_coord = None
            dummy_chain = Chain.Chain('A')
            dummy_chain.add(residue_in_original_struc_copy)
            dummy_chain.atom_to_internal_coordinates()

            # update CB internal coords
            self._update_internal_coords_with_chi_angles(residue_in_original_struc_copy, [], update_atomic_coords=True)

            copy_of_residue_in_original_struc = self._align_residues(residue_in_final_struc, residue_in_original_struc_copy, return_copy=True)

            try:
                rmsd_per_res_id[res_id] = self._rmsd_of_sidechain(residue_in_final_struc, copy_of_residue_in_original_struc, return_sd_instead=return_sd_instead)
            except Exception as e:
                # print('Error in computing RMSD for res_id', res_id)
                # print(e)
                continue
        
        return rmsd_per_res_id
    
    def _swap_values(self, adict, key1, key2):
        temp_value = adict[key1]
        adict[key1] = adict[key2]
        adict[key2] = temp_value
    
    def _rmsd_of_sidechain(self, residue1: Residue, residue2: Residue, return_sd_instead: bool=False):
        '''
        Returns the RMSD between the sidechains of the two residues.
        '''

        SYMMETRIC_RESIDUES = {
            'ARG': ('NH1', 'NH2'),
            'TYR': ('CD1-CE1', 'CD2-CE2'),
            'PHE': ('CD1-CE1', 'CD2-CE2'),
            'ASP': ('OD1', 'OD2'),
            'GLU': ('OE1', 'OE2'),
        }

        if self.use_extended_symmetries:
            SYMMETRIC_RESIDUES.update({
                ## the symmetries below are used in AttnPacker. I believe the first two symmetries also equate to symmetries in chi angles
                'VAL': ('CG1', 'CG2'),
                'LEU': ('CD1', 'CD2'),
                'HIS': ('ND1-NE2', 'CD2-CE1'),
                'GLN': ('OE1', 'NE2'),
                'ASN': ('OD1', 'ND2'),
            })

        atoms1, atoms2 = {}, {}
        for atom in residue1:
            if atom.id not in BB_ATOMS:
                atoms1[atom.id] = atom
        
        for atom in residue2:
            if atom.id not in BB_ATOMS:
                atoms2[atom.id] = atom
        
        def _are_sets_equal(set1, set2):
            return len(set1) == len(set2) and set1 <= set2 and set2 <= set1
                
        assert _are_sets_equal(set(atoms1.keys()), set(atoms2.keys())), 'Residues have different atoms.'

        def _compute_distance(atoms1: Dict, atom2: Dict):
            atom_names = list(atoms1.keys())
            distances = {}
            for atom_name in atom_names:
                atom1, atom2 = atoms1[atom_name], atoms2[atom_name]
                distances[atom_name] = math.dist(atom1.coord, atom2.coord)
            return distances
        
        distances = _compute_distance(atoms1, atoms2)
        
        if residue1.resname in SYMMETRIC_RESIDUES:
            atom_names1, atom_names2 = SYMMETRIC_RESIDUES[residue1.resname]
            atom_names1 = atom_names1.split('-')
            atom_names2 = atom_names2.split('-')
            for atom_name1, atom_name2 in zip(atom_names1, atom_names2):
                self._swap_values(atoms2, atom_name1, atom_name2) # swap the symmetric atoms in one of the dictionaries

            new_distances = _compute_distance(atoms1, atoms2)
            if np.mean(list(new_distances.values())) < np.mean(list(distances.values())): # if the distances are smaller, then keep the new distances
                distances = new_distances
        
        square_distances = np.array([distance**2 for distance in list(distances.values())])

        if return_sd_instead:
            return square_distances
        else:
            rmsd = np.sqrt(np.mean(square_distances))
            return rmsd
    
    def get_coords_and_atom_names(self, structure: Structure):
        atoms = structure.get_atoms()
        coords = []
        atom_names = []
        for atom in atoms:
            coords.append(atom.coord)
            atom_names.append(atom.id)
        return np.vstack(coords), atom_names
    
    def compute_residue_centrality(self, central_residue: Residue, radius: float = 10.0, exclude_self: bool = False, coords_and_atom_names: Optional[Tuple[np.ndarray, List[str]]] = None):

        central_CB = central_residue.child_dict['CB']

        structure_of_interest = self._get_parent_structure(central_residue)
        NBSearch = NeighborSearch(list(structure_of_interest.get_atoms()))
        nb_atom_objects = NBSearch.search(central_CB.coord, radius, level='A')

        centrality = 0
        for atom in nb_atom_objects:
            if exclude_self and atom.get_parent().id == central_residue.id:
                continue
            if atom.id == 'CB':
                centrality += 1
        
        # # use scipy kd tree to compute the same quantity
        # coords, atom_names = coords_and_atom_names
        # from scipy.spatial import KDTree
        # tree = KDTree(coords)
        # nb_indices = tree.query_ball_point(central_CB.coord, radius)
        # nb_indices = np.array(nb_indices)
        # nb_atom_names = np.array(atom_names)[nb_indices]
        # nb_CBs = nb_atom_names == 'CB'

        # assert np.sum(nb_CBs) == centrality # --> this is always true...

        return centrality




def check_chi_angles_match(structure1, structure2):

    def get_res_id_to_chi_angles(structure):
        structure.atom_to_internal_coordinates()
        res_id_to_chi_angles = {}
        res_id_to_resname = {}
        for residue in structure.get_residues():
            res_id = residue.id

            if residue.resname in {'GLY', 'ALA'}:
                continue

            if residue.internal_coord is None:
                continue

            res_id_to_resname[res_id] = residue.resname

            for i in range(1, 5):
                chi_angle = residue.internal_coord.get_angle(f'chi{i}')
                if chi_angle is not None:
                    if res_id not in res_id_to_chi_angles:
                        res_id_to_chi_angles[res_id] = []
                    res_id_to_chi_angles[res_id].append(chi_angle)
        
        return res_id_to_chi_angles, res_id_to_resname
    
    res_id_to_chi_angles1, res_id_to_resname1 = get_res_id_to_chi_angles(structure1)
    res_id_to_chi_angles2, res_id_to_resname2 = get_res_id_to_chi_angles(structure2)

    if set(res_id_to_chi_angles1.keys()) != set(res_id_to_chi_angles2.keys()):
        print('Residue IDs do not match! 1: ', len(set(res_id_to_chi_angles1.keys())), '2: ', len(set(res_id_to_chi_angles2.keys())))

    for res_id in res_id_to_chi_angles1:
        chi_angles1 = res_id_to_chi_angles1[res_id]
        chi_angles2 = res_id_to_chi_angles2[res_id]

        if len(chi_angles1) != len(chi_angles2):
            print('Different number of chi angles! %d and %d for resnames %s and %s' % (len(chi_angles1), len(chi_angles2), res_id_to_resname1[res_id], res_id_to_resname2[res_id]))

        for chi_angle1, chi_angle2 in zip(chi_angles1, chi_angles2):
            if not np.isclose(chi_angle1, chi_angle2):
                print('Chi angles do not match! %s , %.3f , %.3f' % (res_id, chi_angle1, chi_angle2))


if __name__ == '__main__':

    pdbdir = '/gscratch/scrubbed/gvisan01/dlpacker/casp_all/'
    pdb_lists = {
        'CASP13': '/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/casp13_targets_testing.txt',
        'CASP14': '/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/casp14_targets_testing.txt',
    }
    resname_to_rmsd_list = []
    for dataset in ['CASP13', 'CASP14']:
        pdb_list = pdb_lists[dataset]
        with open(pdb_list, 'r') as f:
            pdbs = f.read().splitlines()

        for i, pdb in enumerate(pdbs):
            pdbpath = os.path.join(pdbdir, pdb + '.pdb')

            from time import time
            from copy import deepcopy

            fsr = FullStructureReconstructor(pdbpath, remove_sidechains=True)
            original_original_structure = deepcopy(fsr.original_structure)
            fsr.standardize_original_structure()
            check_chi_angles_match(original_original_structure, fsr.original_structure)

            print('--')

            # start = time()
            # fsr = FullStructureReconstructor(pdbpath, remove_sidechains=True)
            # fsr.reconstruct_from_self_all_at_once()
            # # print(time() - start)
            # print('All at once')
            # check_chi_angles_match(fsr.original_structure, fsr.structure)
            # print()
            # # fsr.write_pdb(f'{pdb}__self_at_once.pdb')

            # # print('\n----------\n')

            # start = time()
            # fsr = FullStructureReconstructor(pdbpath, remove_sidechains=True)
            # fsr.reconstruct_from_self_one_by_one()
            # # print(time() - start)
            # print('One by one')
            # check_chi_angles_match(fsr.original_structure, fsr.structure)
            # print()
            # # fsr.write_pdb(f'{pdb}__self_one_by_one.pdb')

            # print()


    # ## null reconstruction on the CASP13 and 14 PDBs together
    # pdbdir = '/gscratch/scrubbed/gvisan01/dlpacker/casp_all/'
    # pdb_lists = {
    #     'CASP13': '/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/casp13_targets_testing.txt',
    #     'CASP14': '/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/casp14_targets_testing.txt',
    # }
    # resname_to_rmsd_list = []
    # for dataset in ['CASP13', 'CASP14']:
    #     pdb_list = pdb_lists[dataset]
    #     with open(pdb_list, 'r') as f:
    #         pdbs = f.read().splitlines()

    #     for i, pdb in tqdm(enumerate(pdbs), total=len(pdbs)):
    #         pdbpath = os.path.join(pdbdir, pdb + '.pdb')

    #         fsr = FullStructureReconstructor(pdbpath, remove_sidechains=True)
    #         fsr.reconstruct_from_self_all_at_once()
    #         res_id_to_rmsd = fsr._compute_sidechain_rmsd_per_res_id()
    #         resname_to_rmsd = {}
    #         for res_id in res_id_to_rmsd:
    #             resname = fsr._get_resname_from_res_id(res_id)
    #             if resname not in resname_to_rmsd:
    #                 resname_to_rmsd[resname] = []
    #             resname_to_rmsd[resname].append(res_id_to_rmsd[res_id])
    #         resname_to_rmsd_list.append(resname_to_rmsd)

    # import gzip, pickle
    # with gzip.open('null_reconstruction_resname_to_rmsd_list__100PDBs__symmetric_LEU_and_VAL.pkl.gz', 'wb') as f:
    #     pickle.dump(resname_to_rmsd_list, f)




    # ## boxplot of error between null rec with 1700 PDBs and 100 PDBs

    # import gzip, pickle
    # with gzip.open('null_reconstruction_resname_to_rmsd_list__symmetric_LEU_and_VAL.pkl.gz', 'rb') as f:
    #     resname_to_rmsd_list = pickle.load(f)
    # with gzip.open('null_reconstruction_resname_to_rmsd_list__100PDBs__symmetric_LEU_and_VAL.pkl.gz', 'rb') as f:
    #     resname_to_rmsd_list__smaller = pickle.load(f)
    
    # from constants import THE20

    # for resname_to_rmsd in resname_to_rmsd_list:
    #     for resname in THE20:
    #         if resname not in {'GLY', 'ALA'}:
    #             if resname not in resname_to_rmsd:
    #                 resname_to_rmsd[resname] = np.array([])
    # resname_to_rmsd_distribution = {}
    # for resname in THE20:
    #     if resname not in {'GLY', 'ALA'}:
    #         resname_to_rmsd_distribution[resname] = np.hstack([resname_to_rmsd[resname] for resname_to_rmsd in resname_to_rmsd_list])
    #         resname_to_rmsd_distribution[resname] = resname_to_rmsd_distribution[resname][~np.isnan(resname_to_rmsd_distribution[resname])]
    
    # full_structure_rmsd_distribution = np.hstack([resname_to_rmsd[resname] for resname_to_rmsd in resname_to_rmsd_list for resname in resname_to_rmsd])
    # print(np.mean(full_structure_rmsd_distribution), np.std(full_structure_rmsd_distribution))

    # for resname_to_rmsd__smaller in resname_to_rmsd_list__smaller:
    #     for resname in THE20:
    #         if resname not in {'GLY', 'ALA'}:
    #             if resname not in resname_to_rmsd__smaller:
    #                 resname_to_rmsd__smaller[resname] = np.array([])
    # resname_to_rmsd_distribution__smaller = {}
    # for resname in THE20:
    #     if resname not in {'GLY', 'ALA'}:
    #         resname_to_rmsd_distribution__smaller[resname] = np.hstack([resname_to_rmsd__smaller[resname] for resname_to_rmsd__smaller in resname_to_rmsd_list__smaller])
    #         resname_to_rmsd_distribution__smaller[resname] = resname_to_rmsd_distribution__smaller[resname][~np.isnan(resname_to_rmsd_distribution__smaller[resname])]
    
    # full_structure_rmsd_distribution__smaller = np.hstack([resname_to_rmsd__smaller[resname] for resname_to_rmsd__smaller in resname_to_rmsd_list__smaller for resname in resname_to_rmsd__smaller])
    # print(np.mean(full_structure_rmsd_distribution__smaller), np.std(full_structure_rmsd_distribution__smaller))

    # # order resnames by size
    # from protein_holography_pytorch.utils.protein_naming import ind_to_ol_size, ol_to_ind_size, aa_to_one_letter, one_letter_to_aa
    # ordered_resnames = [] 
    # for i in range(20):
    #     resname = one_letter_to_aa[ind_to_ol_size[i]]
    #     if resname not in {'GLY', 'ALA'}:
    #         ordered_resnames.append(resname)
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 4))
    # plt.boxplot([resname_to_rmsd_distribution[resname] - resname_to_rmsd_distribution__smaller[resname] for resname in ordered_resnames] + [full_structure_rmsd_distribution - full_structure_rmsd_distribution__smaller], positions=list(range(18)) + [20])
    # plt.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    # plt.ylabel('$\Delta$ RMSD', fontsize=16)

    # plt.xticks(list(range(18)) + [20], ordered_resnames + ['Full\nStructure'])
    # tick_positions = plt.xticks()[0]
    # tick_labels = plt.xticks()[1]
    # rotate_indices = list(range(18))
    # for i in rotate_indices:
    #     tick_labels[i].set_rotation(90)
    # plt.xticks(tick_positions, tick_labels, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title('Difference in Null Reconstruction on CASP13/14 Targets\nbetween using Statistics from 1,700 vs 100 PDBs', fontsize=16)
    # plt.tight_layout()
    # plt.savefig('../../plots/null_reconstruction_rmsd_per_resname__rmsd_avg__symmetric_LEU_and_VAL__comparison_1700_vs_100_pdbs.png')
    # plt.savefig('../../plots/null_reconstruction_rmsd_per_resname__rmsd_avg__symmetric_LEU_and_VAL__comparison_1700_vs_100_pdbs.pdf')

    # ## boxplot of null reconstruction RMSDs!

    # # order resnames by size
    # from protein_holography_pytorch.utils.protein_naming import ind_to_ol_size, ol_to_ind_size, aa_to_one_letter, one_letter_to_aa
    # ordered_resnames = [] 
    # for i in range(20):
    #     resname = one_letter_to_aa[ind_to_ol_size[i]]
    #     if resname not in {'GLY', 'ALA'}:
    #         ordered_resnames.append(resname)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 4))

    # plt.boxplot([resname_to_rmsd_distribution[resname] for resname in ordered_resnames] + [full_structure_rmsd_distribution], positions=list(range(18)) + [20])
    # plt.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    # plt.ylabel('RMSD', fontsize=16)

    # plt.xticks(list(range(18)) + [20], ordered_resnames + ['Full\nStructure'])
    # tick_positions = plt.xticks()[0]
    # tick_labels = plt.xticks()[1]
    # rotate_indices = list(range(18))
    # for i in rotate_indices:
    #     tick_labels[i].set_rotation(90)
    # plt.xticks(tick_positions, tick_labels, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title('Null Reconstruction on CASP13/14 Targets\nusing Statistics from 1,700 PDBs', fontsize=16)
    # plt.tight_layout()
    # plt.savefig('../../plots/null_reconstruction_rmsd_per_resname__rmsd_avg__symmetric_LEU_and_VAL.png')
    # plt.savefig('../../plots/null_reconstruction_rmsd_per_resname__rmsd_avg__symmetric_LEU_and_VAL.pdf')




    # ## boxplot of error between null rec with 1700 PDBs and 100 PDBs

    # import gzip, pickle
    # with gzip.open('null_reconstruction_resname_to_square_errors_list.pkl.gz', 'rb') as f:
    #     resname_to_square_errors_list = pickle.load(f)
    # with gzip.open('null_reconstruction_resname_to_square_errors_list__100pdbs.pkl.gz', 'rb') as f:
    #     resname_to_square_errors_list__smaller = pickle.load(f)
    
    # from constants import THE20

    # for resname_to_square_errors in resname_to_square_errors_list:
    #     for resname in THE20:
    #         if resname != 'GLY':
    #             if resname not in resname_to_square_errors:
    #                 resname_to_square_errors[resname] = np.array([])
    # resname_to_rmsd_distribution = {}
    # for resname in THE20:
    #     if resname != 'GLY':
    #         resname_to_rmsd_distribution[resname] = np.hstack([np.sqrt(np.nanmean(resname_to_square_errors[resname])) for resname_to_square_errors in resname_to_square_errors_list])
    #         resname_to_rmsd_distribution[resname] = resname_to_rmsd_distribution[resname][~np.isnan(resname_to_rmsd_distribution[resname])]
    # full_structure_rmsd_distribution = np.hstack([np.sqrt(np.nanmean(np.hstack([resname_to_square_errors[resname] for resname in resname_to_square_errors]))) for resname_to_square_errors in resname_to_square_errors_list])
    # print(np.mean(full_structure_rmsd_distribution), np.std(full_structure_rmsd_distribution))

    # for resname_to_square_errors__smaller in resname_to_square_errors_list__smaller:
    #     for resname in THE20:
    #         if resname != 'GLY':
    #             if resname not in resname_to_square_errors__smaller:
    #                 resname_to_square_errors__smaller[resname] = np.array([])
    # resname_to_rmsd_distribution_smaller = {}
    # for resname in THE20:
    #     if resname != 'GLY':
    #         resname_to_rmsd_distribution_smaller[resname] = np.hstack([np.sqrt(np.nanmean(resname_to_square_errors__smaller[resname])) for resname_to_square_errors__smaller in resname_to_square_errors_list__smaller])
    #         resname_to_rmsd_distribution_smaller[resname] = resname_to_rmsd_distribution_smaller[resname][~np.isnan(resname_to_rmsd_distribution_smaller[resname])]
    # full_structure_rmsd_distribution_smaller = np.hstack([np.sqrt(np.nanmean(np.hstack([resname_to_square_errors__smaller[resname] for resname in resname_to_square_errors__smaller]))) for resname_to_square_errors__smaller in resname_to_square_errors_list__smaller])
    # print(np.mean(full_structure_rmsd_distribution_smaller), np.std(full_structure_rmsd_distribution_smaller))

    # # order resnames by size
    # from protein_holography_pytorch.utils.protein_naming import ind_to_ol_size, ol_to_ind_size, aa_to_one_letter, one_letter_to_aa
    # ordered_resnames = [] 
    # for i in range(20):
    #     resname = one_letter_to_aa[ind_to_ol_size[i]]
    #     if resname != 'GLY':
    #         ordered_resnames.append(resname)
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 4))
    # plt.boxplot([resname_to_rmsd_distribution[resname] - resname_to_rmsd_distribution_smaller[resname] for resname in ordered_resnames] + [full_structure_rmsd_distribution - full_structure_rmsd_distribution_smaller], positions=list(range(19)) + [21])
    # plt.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    # plt.ylabel('$\Delta$ RMSD', fontsize=16)

    # plt.xticks(list(range(19)) + [21], ordered_resnames + ['Full\nStructure'])
    # tick_positions = plt.xticks()[0]
    # tick_labels = plt.xticks()[1]
    # rotate_indices = list(range(19))
    # for i in rotate_indices:
    #     tick_labels[i].set_rotation(90)
    # plt.xticks(tick_positions, tick_labels, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title('Difference in Null Reconstruction on CASP13/14 Targets\nbetween using Statistics from 1,700 vs 100 PDBs', fontsize=16)
    # plt.tight_layout()
    # plt.savefig('../../plots/null_reconstruction_rmsd_per_resname__comparison_1700_vs_100_pdbs.png')
    # plt.savefig('../../plots/null_reconstruction_rmsd_per_resname__comparison_1700_vs_100_pdbs.pdf')

    # ## boxplot of null reconstruction RMSDs!

    # import gzip, pickle
    # with gzip.open('null_reconstruction_resname_to_square_errors_list.pkl.gz', 'rb') as f:
    #     resname_to_square_errors_list = pickle.load(f)
    
    # from constants import THE20

    # for resname_to_square_errors in resname_to_square_errors_list:
    #     for resname in THE20:
    #         if resname != 'GLY':
    #             if resname not in resname_to_square_errors:
    #                 resname_to_square_errors[resname] = np.array([])
    
    # resname_to_rmsd_distribution = {}
    # for resname in THE20:
    #     if resname != 'GLY':
    #         resname_to_rmsd_distribution[resname] = np.hstack([np.sqrt(np.nanmean(resname_to_square_errors[resname])) for resname_to_square_errors in resname_to_square_errors_list])
    #         resname_to_rmsd_distribution[resname] = resname_to_rmsd_distribution[resname][~np.isnan(resname_to_rmsd_distribution[resname])]

    # full_structure_rmsd_distribution = np.hstack([np.sqrt(np.nanmean(np.hstack([resname_to_square_errors[resname] for resname in resname_to_square_errors]))) for resname_to_square_errors in resname_to_square_errors_list])
    # print(np.mean(full_structure_rmsd_distribution), np.std(full_structure_rmsd_distribution))

    # # order resnames by size
    # from protein_holography_pytorch.utils.protein_naming import ind_to_ol_size, ol_to_ind_size, aa_to_one_letter, one_letter_to_aa
    # ordered_resnames = [] 
    # for i in range(20):
    #     resname = one_letter_to_aa[ind_to_ol_size[i]]
    #     if resname != 'GLY':
    #         ordered_resnames.append(resname)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 4))

    # plt.boxplot([resname_to_rmsd_distribution[resname] for resname in ordered_resnames] + [full_structure_rmsd_distribution], positions=list(range(19)) + [21])
    # plt.grid(axis='y', ls='--', color='dimgrey', alpha=0.5)
    # plt.ylabel('RMSD', fontsize=16)

    # plt.xticks(list(range(19)) + [21], ordered_resnames + ['Full\nStructure'])
    # tick_positions = plt.xticks()[0]
    # tick_labels = plt.xticks()[1]
    # rotate_indices = list(range(19))
    # for i in rotate_indices:
    #     tick_labels[i].set_rotation(90)
    # plt.xticks(tick_positions, tick_labels, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title('Null Reconstruction on CASP13/14 Targets\nusing Statistics from 1,700 PDBs', fontsize=16)
    # plt.tight_layout()
    # plt.savefig('../../plots/null_reconstruction_rmsd_per_resname.png')
    # plt.savefig('../../plots/null_reconstruction_rmsd_per_resname.pdf')



    # pdb_lists = {
    #     'CASP13': '/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/casp13_targets_testing.txt',
    #     'CASP14': '/gscratch/scrubbed/gvisan01/dlpacker/pdb_lists/casp14_targets_testing.txt',
    # }
    # pdbdir = '/gscratch/scrubbed/gvisan01/dlpacker/casp_all/'

    # res_id_to_centrality = {}

    # for dataset in ['CASP13', 'CASP14']:
    #     pdb_list = pdb_lists[dataset]
    #     with open(pdb_list, 'r') as f:
    #         pdbs = f.read().splitlines()

    #     num_all_all = 0
    #     total_num_residues_including_hetero = 0
    #     centralities = []
    #     for i, pdb in tqdm(enumerate(pdbs), total=len(pdbs)):
    #         # print(f'{i+1}/{len(pdbs)} - {pdb}')
    #         pdbpath = os.path.join(pdbdir, pdb + '.pdb')

    #         fsr = FullStructureReconstructor(pdbpath, remove_sidechains=False, filter=True)
    #         coords, atom_names = fsr.get_coords_and_atom_names(fsr.original_structure)
                                                                    
    #         total_num_residues_including_hetero += len(list(fsr.original_structure.get_residues()))

    #         for res_id in fsr.get_res_ids():
    #             num_all_all += 1
    #             residue = fsr._get_residue_from_res_id(res_id, original_structure=True)
    #             if residue.resname in {'GLY'}: continue
                
    #             try:
    #                 centrality = fsr.compute_residue_centrality(residue, 10.5, exclude_self=False, coords_and_atom_names=(coords, atom_names))
    #             except:
    #                 print('error')
    #                 continue
    #             res_id_to_centrality[res_id] = centrality
    #             centralities.append(centrality)
            
    #     centralities = np.array(centralities)
        
    #     num_all = centralities.shape[0]
    #     num_surface = np.sum(centralities <= 15)
    #     num_core = np.sum(centralities >= 20)

    #     print()
    #     print(f'total_num_residues_including_hetero: {total_num_residues_including_hetero}')
    #     print(f'num_all_all: {num_all_all}')
    #     print(f'num_all: {num_all}')
    #     print(f'num_core: {num_core}')
    #     print(f'num_surface: {num_surface}')





    # ## test reconstructing a single residue from self, print out the coordinates
    # def print_coords(residue: Residue):
    #     for atom in residue:
    #         print(atom.name, atom.coord)

    # res_id = list(fsr.res_id_to_residue.keys())[10]
    # resname = fsr._get_resname_from_res_id(res_id)
    # print(resname)
    # residue_in_struc = fsr._get_residue_from_res_id(res_id)
    # residue_in_original_struc = fsr._get_residue_from_res_id(res_id, original_structure=True)

    # # reconstruct
    # chi_angles = fsr._get_chi_angles(residue_in_original_struc)
    # print(chi_angles)
    # fsr._add_sidechain_with_chi_angles(residue_in_struc, resname, chi_angles)

    # # print coords
    # print()
    # print('original:')
    # print_coords(residue_in_original_struc)
    # print()
    # print('reconstructed:')
    # print_coords(residue_in_struc)







