
import os, sys
import time
import json

from Bio.PDB import PDBParser, PDBIO, Superimposer, Structure, Chain, Model, Residue, Atom, Selection, NeighborSearch
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

from src.preprocessing import get_one_zernikegram
from src.utils.conversions import cartesian_to_spherical__numpy
from src.so3_cnn.so3.functional import make_dict, put_dict_on_device
from src.utils.protein_naming import ind_to_ol_size, ol_to_ind_size, aa_to_one_letter, one_letter_to_aa
from src.training.utils import one_hot_encode, NUM_AAS

from src.training.utils import general_model_init
from src.training.data import get_data_irreps
from src.training.losses import loss_per_chi_angle

from src.sidechain_reconstruction.biopython_internal_coords.full_structure_reconstructor import FullStructureReconstructor
from src.sidechain_reconstruction.biopython_internal_coords.constants import CHI_ANGLES

from typing import *

DLPACKER_CHANNELS = ['C', 'N', 'O', 'S', "all_other_elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all_other_AAs"]

AA_CHANNELS = [b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
               b'S', b'T', b'W', b'Y', b'V', b'G', "all_other_AAs"]

INITIAL_GUESS_MODEL_IDX = 0
REFINEMENT_MODEL_IDX = 1
INITIAL_GUESS_CONDITIONED_MODEL_IDX = 2

class HPacker(FullStructureReconstructor):

    def __init__(self,
                 *args, 
                 model_dirs: List[str] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models/initial_guess'),
                                          os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models/refinement'),
                                          os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models/initial_guess_conditioned')],
                 charges_filepath: Optional[str] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/preprocessing/utils/charges.rtp'),
                 verbose: str = False,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.verbose = verbose

        self._init_models(model_dirs)
        self._init_charges_dict(charges_filepath)
        self._add_charges_to_structure(self.structure)

        self.bb_atoms = {'CA', 'C', 'O', 'N'}
        if self.virtual_CB:
            self.bb_atoms.add('CB')

    def _init_charges_dict(self,
                           charges_filepath):
        '''
        Copied from https://github.com/nekitmm/DLPacker/blob/main/utils.py
        '''

        import re
        from collections import defaultdict

        self.charge_filepath = charges_filepath

        # read in the charges from special file
        self.CHARGES_AMBER99SB = defaultdict(lambda: 0) # output 0 if the key is absent
        if charges_filepath is not None:
            with open(charges_filepath, 'r') as f:
                for line in f:
                    if line[0] == '[' or line[0] == ' ':
                        if re.match('\A\[ .{1,3} \]\Z', line[:-1]):
                            key = re.match('\A\[ (.{1,3}) \]\Z', line[:-1])[1]
                            self.CHARGES_AMBER99SB[key] = defaultdict(lambda: 0)
                        else:
                            l = re.split(r' +', line[:-1])
                            self.CHARGES_AMBER99SB[key][l[1]] = float(l[3])
        else:
            print('Warning: charges are not provided. All atoms will be considered neutral.', file=sys.stderr)
    

    def _init_models(self, model_dirs: List[str]):

        assert len(model_dirs) == 3 # first model is backbone only, second model is all sidechains (minus central one of course), third model is some side-chains only

        self.model_dirs = model_dirs
        self.hparams = []
        self.data_irreps = []
        self.ls_indices = []
        self.model = []
        self.loss_fn = []
        for i, model_dir in enumerate(model_dirs):
            with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
                hparams = json.load(f)

            data_irreps, ls_indices = get_data_irreps(hparams)
            model, loss_fn, device = general_model_init(model_dir, hparams, data_irreps, verbose=self.verbose)
            model.load_state_dict(torch.load(os.path.join(model_dir, 'lowest_valid_loss_model.pt'), map_location=torch.device(device)))
            model.eval()

            self.hparams.append(hparams)
            self.data_irreps.append(data_irreps)
            self.ls_indices.append(ls_indices)
            self.model.append(model)
            self.loss_fn.append(loss_fn)

        self.device = device # assume it's always the same device, which just can't be otherwise

    def _get_neighborhood(self,
                          center_residue_res_id: Tuple,
                          model_idx: int) -> Union[np.ndarray, Dict]:
        ''' This is good, keep as is '''

        # get center residue from res_id
        center_residue = self._get_residue_from_res_id(center_residue_res_id)
        # print(aa_to_one_letter[center_residue.resname])

        # get CA coords of center residue
        center_residue_ca_coords = center_residue['CA'].coord

        # print(center_residue_res_id)
        # print(center_residue_ca_coords)

        # get neighborhood atoms
        NBSearch = NeighborSearch(list(self.structure.get_atoms()))
        nb_atom_objects = NBSearch.search(center_residue_ca_coords, self.hparams[model_idx]['rcut'], level='A')

        # get neighborhood coords, atom names, elements, and charges
        # center all of them around the center residue CA (just remove center_residue_ca_coords)
        # put aside the toms from the center residue

        pdb = self.structure.get_id()
        ss = '-'# just a dummy one here

        def aa_to_one_letter_func(resname):
            if resname not in aa_to_one_letter:
                return 'Z'
            else:
                return aa_to_one_letter[resname]

        preprocessing_center_residue_res_id = np.array([aa_to_one_letter_func(center_residue.resname), pdb, ss, center_residue_res_id[0], center_residue_res_id[1], center_residue_res_id[2], ss], dtype='S5')

        coords = []
        atom_names = []
        elements = []
        charges = []
        preprocessing_res_ids = []
        backbone_atoms = {}
        num_atoms_in_center_residue = 0
        for atom in nb_atom_objects:

            residue = atom.get_parent()
            res_id = self._get_residue_id(residue)

            if res_id == center_residue_res_id:
                backbone_atoms[atom.get_name()] = atom.coord - center_residue_ca_coords
                if atom.get_name() == 'CA': continue # skip center residue's CA because we don't want it for computing the zernikegram
                if atom.get_name() not in self.bb_atoms: continue # we don't want sidechain atoms of the central residue!
            
            if res_id == center_residue_res_id:
                num_atoms_in_center_residue += 1

            preprocessing_res_id = np.array([aa_to_one_letter_func(residue.resname), pdb, ss, res_id[0], res_id[1], res_id[2], ss], dtype='S5')

            preprocessing_res_ids.append(preprocessing_res_id)
            coords.append(atom.coord - center_residue_ca_coords)
            atom_names.append(atom.get_name().strip().upper().ljust(4))
            elements.append(atom.element)
            charges.append(atom.get_charge())

        assert num_atoms_in_center_residue == 3, f'There are {num_atoms_in_center_residue} atoms in the central residue!'

        preprocessing_res_ids = np.vstack(preprocessing_res_ids)
        coords = np.vstack(coords)
        atom_names = np.array(atom_names, dtype='|S4')
        elements = np.array(elements, dtype='S2')
        charges = np.array(charges)



        # put the backbone atoms in the standard [C, O, N, CA] order
        backbone_atoms = np.vstack([backbone_atoms['C'], backbone_atoms['O'], backbone_atoms['N'], backbone_atoms['CA']])

        nb = {
            'res_id': preprocessing_center_residue_res_id,
            'res_ids': preprocessing_res_ids,
            'atom_names': atom_names,
            'elements': elements,
            'coords': coords,
            'charges': charges,
        }

        return nb, backbone_atoms


    def _get_zernikegram(self,
                         nb: Union[np.ndarray, Dict],
                         model_idx: int) -> torch.Tensor:
        '''
        This is good, keep as is
        '''
        
        # convert coordinates to spherical
        nb['coords'] = cartesian_to_spherical__numpy(nb['coords'])

        if self.hparams[model_idx]['channels'] == 'dlpacker':
            channels = DLPACKER_CHANNELS
        elif self.hparams[model_idx]['channels'] == 'AAs':
            channels = AA_CHANNELS
        else:
            channels = self.hparams[model_idx]['channels'].split(',')

        zernikegram = get_one_zernikegram(nb['res_id'],
                                            nb['res_ids'],
                                            nb['coords'],
                                            nb['elements'],
                                            nb['atom_names'],
                                            r_max = self.hparams[model_idx]['rcut'],
                                            radial_func_max = self.hparams[model_idx]['radial_func_max'],
                                            Lmax = self.hparams[model_idx]['lmax'],
                                            channels = channels, 
                                            charges = nb['charges'],
                                            rst_normalization = self.hparams[model_idx]['rst_normalization'],
                                            radial_func_mode = self.hparams[model_idx]['radial_func_mode'])
        
        zernikegram = torch.tensor(zernikegram, dtype=torch.float32)

        return zernikegram


    def _get_predicted_chi_angles(self,
                                  res_id: Tuple,
                                  model_idx: int):
        
        resname = self._get_resname_from_res_id(res_id)
        num_chi = len(CHI_ANGLES[resname])
        
        nb, backbone_coords = self._get_neighborhood(res_id, model_idx)

        zernikegram = self._get_zernikegram(nb, model_idx)

        zernikegram_dict = put_dict_on_device(make_dict(zernikegram.unsqueeze(0), self.data_irreps[model_idx]), self.device)

        aa_label = torch.tensor([ol_to_ind_size[aa_to_one_letter[resname]]]).to(self.device)

        self.model[model_idx].eval()
        if '_cond_and_vec_cond' in self.hparams[model_idx]['model_type']:
            y_hat = self.model[model_idx](zernikegram_dict, one_hot_encode(aa_label, NUM_AAS).to(self.device).float(), torch.tensor(backbone_coords).unsqueeze(0).to(self.device).float())
        elif '_cond' in self.hparams[model_idx]['model_type']:
            y_hat = self.model[model_idx](zernikegram_dict, one_hot_encode(aa_label, NUM_AAS).to(self.device).float())
        else:
            y_hat = self.model[model_idx](zernikegram_dict)
        
        predicted_chi_angles = self.loss_fn[model_idx].get_chi_angles_from_predictions(y_hat, aa_label, torch.tensor(backbone_coords).unsqueeze(0))[0, :num_chi].detach().cpu().numpy().tolist()

        return predicted_chi_angles

    # @profile
    def _get_predicted_chi_angles_batch(self,
                                        res_ids: List[Tuple],
                                        model_idx: int):
        zernikegrams = []
        aa_labels = []
        all_backbone_coords = []
        num_chis = []
        for res_id in res_ids:
            resname = self._get_resname_from_res_id(res_id)
            num_chi = len(CHI_ANGLES[resname])
            
            nb, backbone_coords = self._get_neighborhood(res_id, model_idx)

            zernikegram = self._get_zernikegram(nb, model_idx)

            zernikegrams.append(zernikegram)
            aa_labels.append(ol_to_ind_size[aa_to_one_letter[resname]])
            all_backbone_coords.append(torch.tensor(backbone_coords))
            num_chis.append(num_chi)

        zernikegram = torch.stack(zernikegrams, dim=0)
        aa_label = torch.tensor(aa_labels).to(self.device)
        backbone_coords = torch.stack(all_backbone_coords, dim=0).to(self.device).float()

        zernikegram_dict = put_dict_on_device(make_dict(zernikegram, self.data_irreps[model_idx]), self.device)

        self.model[model_idx].eval()
        if '_cond_and_vec_cond' in self.hparams[model_idx]['model_type']:
            y_hat = self.model[model_idx](zernikegram_dict, one_hot_encode(aa_label, NUM_AAS).to(self.device).float(), backbone_coords)
        elif '_cond' in self.hparams[model_idx]['model_type']:
            y_hat = self.model[model_idx](zernikegram_dict, one_hot_encode(aa_label, NUM_AAS).to(self.device).float())
        else:
            y_hat = self.model[model_idx](zernikegram_dict)
        
        predicted_chi_angles_padded = self.loss_fn[model_idx].get_chi_angles_from_predictions(y_hat, aa_label, backbone_coords).detach().cpu().numpy()

        predicted_chi_angles = []
        for i, res_id in enumerate(res_ids):
            predicted_chi_angles.append(predicted_chi_angles_padded[i, :num_chis[i]].tolist())

        return predicted_chi_angles
    
    def _add_charges_to_structure(self,
                                  structure: Structure):
        
        for residue in Selection.unfold_entities(structure, 'R'):
            self._add_charges_to_residue(residue)

    def _add_charges_to_residue(self,
                                residue: Residue):
        
        for atom in residue.get_atoms():
            atom_name = atom.get_name()

            if not isinstance(self.CHARGES_AMBER99SB[residue.resname], int):
                atom.set_charge(self.CHARGES_AMBER99SB[residue.resname][atom_name])
            else:
                atom.set_charge(self.CHARGES_AMBER99SB[residue.resname])
    
    def add_sidechain(self,
                      res_id: Tuple):
        '''
        Adds a sidechain to the specified residue.
        TODO: Update this. Currently uses the initial guess model only, i.e. only uses backbone information. Give the option to use the refinement model or a conditiuoned/downsampled model.
        '''
        resname = self._get_resname_from_res_id(res_id)
        if resname == 'GLY': # nothing to do!
            return None
        elif resname == 'ALA': # add CB but return None
            self._add_sidechain_with_chi_angles(res_id, [])
            self._add_charges_to_residue(self._get_residue_from_res_id(res_id))
            return None
        else:
            chi_angles = self._get_predicted_chi_angles(res_id, INITIAL_GUESS_MODEL_IDX)
            self._add_sidechain_with_chi_angles(res_id, chi_angles)
            self._add_charges_to_residue(self._get_residue_from_res_id(res_id))
            return chi_angles

    def initial_guess(self, res_ids=None, batch_size=1, **kwargs):
        self.structure_copy.atom_to_internal_coordinates()
        if batch_size == 1:
            if res_ids is None: # use model trained only on backbone neighborhoods
                return self.predict_and_add_sidechains(INITIAL_GUESS_MODEL_IDX, **kwargs)
            else: # use model trained on partial neighborhoods
                return self.predict_and_add_sidechains(INITIAL_GUESS_CONDITIONED_MODEL_IDX, res_ids_to_predict=res_ids, **kwargs)
        else:
            if res_ids is None:
                return self.predict_and_add_sidechains__batch(INITIAL_GUESS_MODEL_IDX, batch_size=batch_size, **kwargs)
            else:
                return self.predict_and_add_sidechains__batch(INITIAL_GUESS_CONDITIONED_MODEL_IDX, res_ids_to_predict=res_ids, batch_size=batch_size, **kwargs)
    
    def refinement(self, res_ids=None, batch_size=1, **kwargs):
        self.structure_copy.atom_to_internal_coordinates()
        if batch_size == 1:
            return self.predict_and_add_sidechains(REFINEMENT_MODEL_IDX, res_ids_to_predict=res_ids, **kwargs)
        else:
            return self.predict_and_add_sidechains__batch(REFINEMENT_MODEL_IDX, res_ids_to_predict=res_ids, batch_size=batch_size, **kwargs)
    
    def predict_and_add_sidechains(self,
                                   model_idx: int,
                                   res_ids_to_predict: Optional[List[Tuple]] = None):
        assert model_idx in {INITIAL_GUESS_MODEL_IDX, REFINEMENT_MODEL_IDX, INITIAL_GUESS_CONDITIONED_MODEL_IDX}
        if res_ids_to_predict is None:
            res_ids_to_predict = self.get_res_ids()
        res_id_to_chi_angles = {}
        for res_id in res_ids_to_predict:
            resname = self._get_resname_from_res_id(res_id)
            if resname == 'GLY':
                continue
            elif resname == 'ALA':
                res_id_to_chi_angles[res_id] = []
            else:
                res_id_to_chi_angles[res_id] = self._get_predicted_chi_angles(res_id, model_idx)
        
        if model_idx == REFINEMENT_MODEL_IDX:
            # trick: just strip sidechains of the structure and then add them back in with the new chi angles
            self.remove_sidechains_for_res_ids(res_ids_to_predict, keep_CB=self.virtual_CB) # the present CBs now are the virtual ones!

        self._add_multiple_sidechains_with_chi_angles(res_id_to_chi_angles)
        self._add_charges_to_structure(self.structure)

        return res_id_to_chi_angles
    
    # @profile
    def predict_and_add_sidechains__batch(self,
                                          model_idx: int,
                                          res_ids_to_predict: Optional[List[Tuple]] = None,
                                          batch_size: int = 128):
        '''
        If res_ids is None, then predict and add sidechains to all residues in the structure.
        '''
        assert model_idx in {INITIAL_GUESS_MODEL_IDX, REFINEMENT_MODEL_IDX, INITIAL_GUESS_CONDITIONED_MODEL_IDX}
        res_id_to_chi_angles = {}
        if res_ids_to_predict is None:
            res_ids_to_predict = self.get_res_ids()
        res_ids_ala = list(filter(lambda res_id: self._get_resname_from_res_id(res_id) == 'ALA', res_ids_to_predict))
        res_ids = list(filter(lambda res_id: self._get_resname_from_res_id(res_id) not in {'GLY', 'ALA'}, res_ids_to_predict))
        num_batches = len(res_ids) // batch_size
        if len(res_ids) % batch_size != 0:
            num_batches += 1

        res_id_to_chi_angles = {}
        for batch_i in range(num_batches):
            if batch_i == num_batches - 1:
                batch_res_ids = res_ids[batch_i*batch_size:]
            else:
                batch_res_ids = res_ids[batch_i*batch_size : (batch_i+1)*batch_size]
            
            batch_predicted_chi_angles = self._get_predicted_chi_angles_batch(batch_res_ids, model_idx)

            res_id_to_chi_angles.update(dict(zip(batch_res_ids, batch_predicted_chi_angles)))
        
        res_id_to_chi_angles.update(dict(zip(res_ids_ala, [[] for _ in range(len(res_ids_ala))])))
        
        if model_idx == REFINEMENT_MODEL_IDX:
            # trick: just strip sidechains of the structure and then add them back in with the new chi angles
            self.remove_sidechains_for_res_ids(res_ids_to_predict, keep_CB=self.virtual_CB) # the present CBs now are the virtual ones!

        self._add_multiple_sidechains_with_chi_angles(res_id_to_chi_angles)
        self._add_charges_to_structure(self.structure)

        return res_id_to_chi_angles
                   
    def compute_error_per_chi_angle(self,
                                    real_res_id_to_chi_angles: Dict[Tuple, List[float]],
                                    predicted_res_id_to_chi_angles: Dict[Tuple, List[float]],
                                    res_id_to_centrality: Dict[Tuple, float]):
        real = {
            'all': [],
            'core': [],
            'surface': []
        }
        predicted = {
            'all': [],
            'core': [],
            'surface': []
        }
        aas = {
            'all': [],
            'core': [],
            'surface': []
        }
        res_ids = {
            'all': [],
            'core': [],
            'surface': []
        }
        for res_id in real_res_id_to_chi_angles:
            resname = self._get_resname_from_res_id(res_id)
            if resname in {'GLY', 'ALA'}: continue

            real_chi_angles = real_res_id_to_chi_angles[res_id]
            predicted_chi_angles = predicted_res_id_to_chi_angles[res_id]

            real['all'].append(np.pad(np.array(real_chi_angles), (0, 4 - len(real_chi_angles)), mode='constant', constant_values=np.nan))
            predicted['all'].append(np.pad(np.array(predicted_chi_angles), (0, 4 - len(predicted_chi_angles)), mode='constant', constant_values=np.nan))
            aas['all'].append(ol_to_ind_size[aa_to_one_letter[self._get_resname_from_res_id(res_id)]])
            res_ids['all'].append(res_id)

            if res_id_to_centrality[res_id] >= 20:
                real['core'].append(np.pad(np.array(real_chi_angles), (0, 4 - len(real_chi_angles)), mode='constant', constant_values=np.nan))
                predicted['core'].append(np.pad(np.array(predicted_chi_angles), (0, 4 - len(predicted_chi_angles)), mode='constant', constant_values=np.nan))
                aas['core'].append(ol_to_ind_size[aa_to_one_letter[self._get_resname_from_res_id(res_id)]])
                res_ids['core'].append(res_id)
            
            elif res_id_to_centrality[res_id] <= 15:
                real['surface'].append(np.pad(np.array(real_chi_angles), (0, 4 - len(real_chi_angles)), mode='constant', constant_values=np.nan))
                predicted['surface'].append(np.pad(np.array(predicted_chi_angles), (0, 4 - len(predicted_chi_angles)), mode='constant', constant_values=np.nan))
                aas['surface'].append(ol_to_ind_size[aa_to_one_letter[self._get_resname_from_res_id(res_id)]])
                res_ids['surface'].append(res_id)

        for key in real:
            if len(real[key]) == 0:
                real[key] = np.array([])
            else:
                real[key] = np.vstack(real[key])
        for key in predicted:
            if len(predicted[key]) == 0:
                predicted[key] = np.array([])
            else:
                predicted[key] = np.vstack(predicted[key])
        for key in aas:
            if len(aas[key]) == 0:
                aas[key] = np.array([])
            else:
                aas[key] = np.array(aas[key])
        for key in aas:
            if len(aas[key]) == 0:
                aas[key] = np.array([])
            else:
                aas[key] = np.array(aas[key])
        
        print('All residues:', len(real['all']))
        print('Core residues:', len(real['core']))
        print('Surface residues:', len(real['surface']))

        mae_per_angle_4, accuracy_per_angle_4 = loss_per_chi_angle(torch.tensor(predicted['all']), torch.tensor(real['all']), torch.tensor(aas['all']))

        return mae_per_angle_4, accuracy_per_angle_4, real, predicted, aas, res_ids
    
    def pretty_print_error_per_chi_angle(self, mae_per_angle_4, accuracy_per_angle_4):

        print('Accuracy:', end='\t')
        for chi_angle_idx in range(4):
            print('%.0f' % (accuracy_per_angle_4[chi_angle_idx].item()*100), end='\t')
        print()
        print('MAE:', end='\t')
        for chi_angle_idx in range(4):
            print('%.0f' % (mae_per_angle_4[chi_angle_idx].item()), end='\t')
        print()
        print()
    
    def _get_chi_angles_our_way(self, residue: Residue):
        '''
        Returns the chi angles of the specified residue.
        '''
        from src.preprocessing.utils.structural_info import VEC_AA_ATOM_DICT
        vecs = np.full((5, 3), np.nan, dtype=float)
        chis = []
        atom_names = VEC_AA_ATOM_DICT.get(residue.resname)
        if atom_names is not None:

            for i in range(len(atom_names)):
                p1 = residue.child_dict[atom_names[i][0]].coord
                p2 = residue.child_dict[atom_names[i][1]].coord
                p3 = residue.child_dict[atom_names[i][2]].coord
                v1 = p1 - p2
                v2 = p3 - p2
                # v1 = p1 - p2
                # v2 = p1 - p3
                x = np.cross(v1, v2)
                vecs[i] = x / np.linalg.norm(x)
            
            for i in range(len(atom_names)-1):
                chis.append(self._get_chi_angle_our_way_helper(vecs[i], vecs[i+1], residue.child_dict[atom_names[i][1]].coord, residue.child_dict[atom_names[i][2]].coord))

        return chis
    
    def _get_chi_angle_our_way_helper(self, plane_norm_1, plane_norm_2, a2, a3):
        
        sign_vec = a3 - a2
        sign_with_magnitude = np.dot(sign_vec, np.cross(plane_norm_1, plane_norm_2))
        sign = sign_with_magnitude / np.abs(sign_with_magnitude)
        
        dot = np.dot(plane_norm_1, plane_norm_2) / (np.linalg.norm(plane_norm_1) * np.linalg.norm(plane_norm_2))
        chi_angle = sign * np.arccos(dot)
        
        return np.degrees(chi_angle)
    
    def get_rmsds(self, res_id_to_centrality, standardize_original_structure=False):
        if standardize_original_structure:
            res_id_to_rmsd = self._compute_sidechain_rmsd_per_res_id_with_standardized_ground_truth()
        else:
            res_id_to_rmsd = self._compute_sidechain_rmsd_per_res_id()

        rmsds = {
            'all':[],
            'core':[],
            'surface':[]
        }
        for res_id in res_id_to_rmsd:
            if res_id not in res_id_to_centrality:
                continue

            rmsds['all'].append(res_id_to_rmsd[res_id]) # all!

            if res_id_to_centrality[res_id] >= 20: # core!
                rmsds['core'].append(res_id_to_rmsd[res_id])
            elif res_id_to_centrality[res_id] <= 15: # surface!
                rmsds['surface'].append(res_id_to_rmsd[res_id])
        
        return rmsds

    def compute_refinement_model_predictions_on_true_structure(self,
                                                               standardize_original_structure: bool = False):
        '''
        The results of this model are a lower bound on what we can hope to achieve.
        '''
        from copy import deepcopy

        original_no_sidechain_structure = deepcopy(self.structure)

        self.structure = deepcopy(self.original_structure)
        self._add_charges_to_structure(self.structure)

        ## compute internal coords for the original structure
        self.original_structure.atom_to_internal_coordinates()

        # get real chi angle values
        res_ids = self.get_res_ids()
        real_res_id_to_angles = {}
        res_id_to_centrality = {}
        for res_id in res_ids:
            resname = self._get_resname_from_res_id(res_id)
            if resname in {'GLY', 'ALA'}: continue
            original_struc_residue = self._get_residue_from_res_id(res_id, original_structure=True)
            try:
                chi_angles = self._get_chi_angles(original_struc_residue)
            except Exception as e:
                print('Error in computing real chi angles for residue', res_id, 'in original structure')
                print(e)
                continue
            try:
                centrality = self.compute_residue_centrality(original_struc_residue)
            except Exception as e:
                print('Error in computing centrality for residue', res_id, 'in original structure')
                print(e)
                continue
            real_res_id_to_angles[res_id] = chi_angles
            res_id_to_centrality[res_id] = centrality
        
        batch_size = 512
        
        res_id_to_chi_angles = {}
        res_ids = self.get_res_ids()
        res_ids_ala = list(filter(lambda res_id: self._get_resname_from_res_id(res_id) == 'ALA', res_ids))
        res_ids = list(filter(lambda res_id: self._get_resname_from_res_id(res_id) not in {'GLY', 'ALA'}, res_ids))
        num_batches = len(res_ids) // batch_size
        if len(res_ids) % batch_size != 0:
            num_batches += 1

        res_id_to_chi_angles = {}
        for batch_i in range(num_batches):
            if batch_i == num_batches - 1:
                batch_res_ids = res_ids[batch_i*batch_size:]
            else:
                batch_res_ids = res_ids[batch_i*batch_size : (batch_i+1)*batch_size]
            
            batch_predicted_chi_angles = self._get_predicted_chi_angles_batch(batch_res_ids, REFINEMENT_MODEL_IDX)

            res_id_to_chi_angles.update(dict(zip(batch_res_ids, batch_predicted_chi_angles)))
        
        res_id_to_chi_angles.update(dict(zip(res_ids_ala, [[] for _ in range(len(res_ids_ala))])))

        mae_per_angle_4, accuracy_per_angle_4, real, predicted, aas = self.compute_error_per_chi_angle(real_res_id_to_angles, res_id_to_chi_angles, res_id_to_centrality)
        print('Refinement model predictions on true structure:')
        self.pretty_print_error_per_chi_angle(mae_per_angle_4, accuracy_per_angle_4)

        ## add all dummy atoms to the copy structure
        self._add_all_dummy_atoms(self.structure_copy)
        self.structure_copy.atom_to_internal_coordinates()

        self.structure = original_no_sidechain_structure

        self._add_multiple_sidechains_with_chi_angles(res_id_to_chi_angles)
        self._add_charges_to_structure(self.structure)
        rmsds = self.get_rmsds(res_id_to_centrality, standardize_original_structure=standardize_original_structure)

        return mae_per_angle_4, accuracy_per_angle_4, real, predicted, aas, rmsds


    def reconstruct_sidechains_and_evaluate(self,
                                            num_refinement_iterations: int = 5,
                                            batch_size: int = 128,
                                            standardize_original_structure: bool = False):
        '''
        Use this function when reconstructing a structure that already has sidechains (e.g. a ground truth crystal structure), and wish to compare against it.
        Populates the internal representation of the structure with the rotamers for the original amino acid compositions.
        '''

        ## add all dummy atoms to the copy structure
        self._add_all_dummy_atoms(self.structure_copy)

        ## compute internal coords for the original structure
        self.original_structure.atom_to_internal_coordinates()

        ## preliminarily remove all sidechains, though I am not sure it matters (still, better to be safe than sorry)
        self.remove_all_sidechains()

        # get real chi angle values
        res_ids = self.get_res_ids()
        real_res_id_to_angles = {}
        res_id_to_centrality = {}
        for res_id in res_ids:
            resname = self._get_resname_from_res_id(res_id)
            if resname in {'GLY', 'ALA'}: continue
            original_struc_residue = self._get_residue_from_res_id(res_id, original_structure=True)
            try:
                chi_angles = self._get_chi_angles(original_struc_residue)
            except Exception as e:
                print('Error in computing real chi angles for residue', res_id, 'in original structure')
                print(e)
                continue
            try:
                centrality = self.compute_residue_centrality(original_struc_residue)
            except Exception as e:
                print('Error in computing centrality for residue', res_id, 'in original structure')
                print(e)
                continue
            real_res_id_to_angles[res_id] = chi_angles
            res_id_to_centrality[res_id] = centrality

        # make initial guess with backboone atoms only
        start = time.time()
        initial_guess_res_id_to_angles = self.initial_guess(batch_size=batch_size)
        if self.verbose: print('Initial guess took %.2f seconds' % (time.time() - start))

        initial_guess_mae_per_angle_4, initial_guess_accuracy_per_angle_4, initial_guess_real, initial_guess_predicted, initial_guess_aas, initial_guess_res_ids_dict = self.compute_error_per_chi_angle(real_res_id_to_angles, initial_guess_res_id_to_angles, res_id_to_centrality)
        self.pretty_print_error_per_chi_angle(initial_guess_mae_per_angle_4, initial_guess_accuracy_per_angle_4)
        initial_guess_rmsds = self.get_rmsds(res_id_to_centrality, standardize_original_structure=standardize_original_structure)

        if num_refinement_iterations == 0:
            return initial_guess_mae_per_angle_4, initial_guess_accuracy_per_angle_4, initial_guess_real, initial_guess_predicted, initial_guess_aas, initial_guess_res_ids_dict, initial_guess_rmsds
        
        # refinement!
        time_refinement = 0
        for i in range(num_refinement_iterations):
            if self.verbose: print(f'Iteration {i+1}/{num_refinement_iterations}')

            start = time.time()
            refined_res_id_to_angles = self.refinement(batch_size=batch_size)
            elapsed = time.time() - start
            time_refinement += elapsed
            if self.verbose: print('Refinement for %d times took %.2f seconds' % (i+1, time_refinement))

            mae_per_angle_4, accuracy_per_angle_4, real, predicted, aas, res_ids_dict = self.compute_error_per_chi_angle(real_res_id_to_angles, refined_res_id_to_angles, res_id_to_centrality)
            self.pretty_print_error_per_chi_angle(mae_per_angle_4, accuracy_per_angle_4)

        refined_rmsds = self.get_rmsds(res_id_to_centrality, standardize_original_structure=standardize_original_structure)

        return mae_per_angle_4, accuracy_per_angle_4, real, predicted, aas, res_ids_dict, refined_rmsds


    def reconstruct_sidechains(self,
                                num_refinement_iterations: int = 5,
                                res_id_to_resname: Optional[Dict[Tuple, str]] = None,

                                reconstruct_all_sidechains: bool = False,
                                res_ids_to_reconstruct: List[Tuple] = None,
                                proximity_cutoff_for_refinement: float = 10.0,
                                res_ids_to_refine: List[Tuple] = None,

                                batch_size: int = 128,
                                return_trace_of_predicted_angles: bool = False):
        '''
        Use this when not interested in evaluating the reconstruction.
        Populates the internal representation of the structure with the desired amino acid compositions.
        If provided, use the specified amino acid types for the specified residues.
        Otherwise, use the amino acid types specified in the PDB file.

        reconstruct_all_sidechains :: Overrides everything else
        res_id_to_resname :: Dictionary of mutations, 
        res_ids_to_reconstruct :: Residues that are to be reconstructed from scratch (initial_guess plus refinement). Overrides default choice of reconstructing res_ids without sidechains.
        proximity_cutoff_for_refinement :: Distance between CBs that defines which residues are considered close enough to be refined, even if they weren't reconstructed from scratch.
        res_ids_to_refine :: Overrides default choice of res_ids close to the ones reconstructed from scratch. Note that residues reconstructed from scratch also get refined by default.

        # NOTE: update resnames and add dummy atomas to copy structure AFTER computing residues that have no sidechains, otherwise the computation would give a bunch of errors
        # NOTE: behavior when res_ids_to_reconstruct does not contain all residues that have missing side-chains is **undefined**. It might return something but the predictions might not be good since the refinement model will be run on partial neighborhoods.
        '''

        assert proximity_cutoff_for_refinement >= 0, 'proximity_cutoff_for_refinement must be non-negative'

        # add the residues in 
        if res_ids_to_reconstruct is None and res_id_to_resname is not None:
            res_ids_to_reconstruct = list(res_id_to_resname.keys())
        elif res_ids_to_reconstruct is not None and res_id_to_resname is not None:
            res_ids_to_reconstruct = list(set(res_ids_to_reconstruct + list(res_id_to_resname.keys())))
        
        angles_trace = []

        # make initial guess with backbone atoms only
        start = time.time()

        if reconstruct_all_sidechains:
            self.remove_all_sidechains()
            if res_id_to_resname is not None:
                self.update_resnames(res_id_to_resname)
            
            self._add_all_dummy_atoms(self.structure_copy)
            initial_guess_res_id_to_angles = self.initial_guess(batch_size=batch_size)
        else:
            # by default, only reconstruct sidechains for all residues that don't have them
            if res_ids_to_reconstruct is None:
                res_ids_to_reconstruct = self.detect_res_ids_with_missing_sidechains()

            else:
                self.remove_sidechains_for_res_ids(res_ids_to_reconstruct) # remove side-chains for the specified residues, just to be safe
                self.remove_sidechains_for_res_ids(res_ids_to_reconstruct, copy_structure=True)
            
            # if all sidechains are missing, then use the initial_guess model, so set res_ids to None
            if self.all_sidechains_are_missing(res_ids_to_reconstruct):
                res_ids_to_reconstruct = None

            if res_id_to_resname is not None:
                self.update_resnames(res_id_to_resname)
            
            self._add_all_dummy_atoms(self.structure_copy)
            initial_guess_res_id_to_angles = self.initial_guess(batch_size=batch_size, res_ids=res_ids_to_reconstruct)

        if self.verbose: print('Initial guess took %.2f seconds' % (time.time() - start))

        angles_trace.append(initial_guess_res_id_to_angles)

        # refinement!
        time_refinement = 0
        for i in range(num_refinement_iterations):
            if self.verbose: print(f'Iteration {i+1}/{num_refinement_iterations}')

            start = time.time()

            if reconstruct_all_sidechains or res_ids_to_reconstruct is None:
                refined_res_id_to_angles = self.refinement(batch_size=batch_size)
            else:
                # by default, only refine sidechains for all residues that don't have them
                if res_ids_to_refine is None:
                    res_ids_to_refine = list(set(res_ids_to_reconstruct + self.find_residues_in_surrounding(res_ids_to_reconstruct, radius=proximity_cutoff_for_refinement)))
                else:
                    res_ids_to_refine = list(set(res_ids_to_reconstruct + res_ids_to_refine)) # add the residues that were reconstructed from scratch

                refined_res_id_to_angles = self.refinement(batch_size=batch_size, res_ids=res_ids_to_refine)
            
            elapsed = time.time() - start
            time_refinement += elapsed
            if self.verbose: print('Refinement for %d times took %.2f seconds' % (i+1, time_refinement))

            angles_trace.append(deepcopy(refined_res_id_to_angles))
        
        if return_trace_of_predicted_angles: return angles_trace
    
    def refine_sidechains(self,
                            num_refinement_iterations: int = 5,
                            res_ids: Optional[List[Tuple]] = None,
                            batch_size: int = 128,
                            return_trace_of_predicted_angles: bool = False):
        '''
        Use this when you already have resonably good sidechains in the structure and only want to refine them.
        If res_ids is None, then refine all residues in the structure.
        '''

        ## add all dummy atoms to the copy structure
        self._add_all_dummy_atoms(self.structure_copy)
        
        angles_trace = []

        # refinement!
        time_refinement = 0
        for i in range(num_refinement_iterations):
            if self.verbose: print(f'Iteration {i+1}/{num_refinement_iterations}')

            start = time.time()
            refined_res_id_to_angles = self.refinement(res_ids=res_ids, batch_size=batch_size)
            elapsed = time.time() - start
            time_refinement += elapsed
            if self.verbose: print('Refinement for %d times took %.2f seconds' % (i+1, time_refinement))

            angles_trace.append(deepcopy(refined_res_id_to_angles))
        
        if return_trace_of_predicted_angles: return angles_trace

            
def stack_dicts(dict_list):
    newdict = {
        'all': [d['all'] for d in dict_list if len(d['all']) > 0],
        'core': [d['core'] for d in dict_list if len(d['core']) > 0],
        'surface': [d['surface'] for d in dict_list if len(d['surface']) > 0]
    }
    return newdict

def pretty_print_scores(dataset, accuracy_per_angle_all, mae_per_angle_all, global_accuracy_all, global_accuracy_core, global_accuracy_surface, aas_trace, accuracy_per_angle_trace, mae_per_angle_trace, rmsds):

    print(f'{dataset} Accuracy and MAE per angle:')

    print('Accuracy:', end='\t')
    for chi_angle_idx in range(4):
        print('%.0f' % (accuracy_per_angle_all[chi_angle_idx].item()*100), end='\t')
    print()
    print('MAE:', end='\t')
    for chi_angle_idx in range(4):
        print('%.2f' % (mae_per_angle_all[chi_angle_idx].item()), end='\t')
    print()
    print('Global Accuracies:', end='\t')
    print('%.1f' % (global_accuracy_all.item()*100), end='\t')
    print('%.1f' % (global_accuracy_core.item()*100), end='\t')
    print('%.1f' % (global_accuracy_surface.item()*100), end='\t')
    print('RMSDs:', end='\t')
    print('%.3f' % (np.mean(np.hstack(rmsds['all']))), end='\t')
    print('%.3f' % (np.mean(np.hstack(rmsds['core']))), end='\t')
    print('%.3f' % (np.mean(np.hstack(rmsds['surface']))), end='\t')
    print()
    print('Number of residues (all, core, surface): %d\t%d\t%d' % (np.hstack(stack_dicts(aas_trace)['all']).shape[0], np.hstack(stack_dicts(aas_trace)['core']).shape[0], np.hstack(stack_dicts(aas_trace)['surface']).shape[0]))
    print()
    print()

    print(f'(UN-WEIGHTED AVERAGED) {dataset} Accuracy and MAE per angle:')

    print('Accuracy:', end='\t')
    for chi_angle_idx in range(4):
        print('%.0f' % (np.mean(np.vstack(accuracy_per_angle_trace), axis=0)[chi_angle_idx].item()*100), end='\t')
    print()
    print('MAE:', end='\t')
    for chi_angle_idx in range(4):
        print('%.0f' % (np.mean(np.vstack(mae_per_angle_trace), axis=0)[chi_angle_idx].item()), end='\t')
    print()
    print()
