
import os, sys
import json
import numpy as np
import torch

from src.utils.protein_naming import ol_to_ind_size, ind_to_ol_size, aa_to_one_letter

GLU = ol_to_ind_size[aa_to_one_letter['GLU']]
ASP = ol_to_ind_size[aa_to_one_letter['ASP']]
PHE = ol_to_ind_size[aa_to_one_letter['PHE']]
TYR = ol_to_ind_size[aa_to_one_letter['TYR']]
VAL = ol_to_ind_size[aa_to_one_letter['VAL']]
LEU = ol_to_ind_size[aa_to_one_letter['LEU']]


## NOTE: VAL and LEU symmetries are used by AttnPacker
SYMMETRIC_AAS = torch.tensor([GLU, ASP, PHE, TYR]) #, VAL, LEU])
SYMMETRIC_AA_TO_CHI = {
    GLU: 3,
    ASP: 2,
    PHE: 2,
    TYR: 2,
    # VAL: 1,
    # LEU: 2
}

def make_single_mask(mask_aa, aas_N, second_dim):
    N = aas_N.shape[0]
    assert second_dim in {4, 8}

    return torch.logical_and( \
                            torch.transpose(torch.tile(torch.isin(aas_N, mask_aa), [second_dim, 1]), 0, 1), \
                            torch.tile(torch.eye(4, dtype=torch.bool, device=aas_N.device)[SYMMETRIC_AA_TO_CHI[mask_aa] - 1], [N, 1]) if second_dim == 4 \
                                else torch.tile(torch.eye(8, dtype=torch.bool, device=aas_N.device)[SYMMETRIC_AA_TO_CHI[mask_aa] - 1] +
                                                torch.eye(8, dtype=torch.bool, device=aas_N.device)[SYMMETRIC_AA_TO_CHI[mask_aa] - 1 + 4],
                                                [N, 1])
    )

def make_flipped_mask(aas_N, second_dim):
    N = aas_N.shape[0]
    assert second_dim in {4, 8}

    mask = torch.zeros((N, second_dim), dtype=torch.bool, device=aas_N.device)
    for aa, chi in SYMMETRIC_AA_TO_CHI.items():
        mask[aas_N == aa, chi-1] = True
        if second_dim == 8:
            mask[aas_N == aa, chi-1+4] = True

    ## these two ways of computing masks are equivalent
    
    # mask = make_single_mask(GLU, aas_N, second_dim)
    # mask = torch.logical_or(mask, make_single_mask(ASP, aas_N, second_dim))
    # mask = torch.logical_or(mask, make_single_mask(PHE, aas_N, second_dim))
    # mask = torch.logical_or(mask, make_single_mask(TYR, aas_N, second_dim))     

    return mask

class AngleLoss():
    '''
    Somple MSE loss function on chi-angles
    We compute the loss circularly, meaning that angles -179 and 179 are considered to be 2 degrees apart instead of 358 degrees apart.

    Internally, we make the predictions between -1 and 1 (essentially, in units of \pi), to use lower numbers in the scale of the other models, so at least learning rate is more easily comparable
    '''
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, y_hat_N4, y_N4, aas_N, chi=None, weight_chis=False, epoch=None):
        if chi is not None:
            raise NotImplementedError('chi is not implemented for this loss function')

        y_N4 = y_N4 * 1/180
        flipped_y_N4 = torch.remainder(y_N4 + 1 + 1, 2) - 1
        
        isnan_mask_N4 = torch.isnan(y_N4) # it's the same for all three components
        n_N = torch.sum((~isnan_mask_N4).float(), dim=1) # it's the same for all three components
        VALUE = 1.0
        n_N[n_N == 0] = VALUE # this value will always be at the denominator of a zero so it doesn't matter what it is, it just can't be zero

        y_N4[isnan_mask_N4] = y_hat_N4[isnan_mask_N4]
        flipped_y_N4[isnan_mask_N4] = y_hat_N4[isnan_mask_N4]

        if epoch == 0:
            # in the first epoch, don't do circular loss nor modding the predictions,
            # so as to stabilize the networks' predictions in the range of chi-angle values

            # normal loss
            base_diff_squared_N4 = torch.square(y_hat_N4 - y_N4)

            # loss with flipped angles
            flipped_diff_squared_N4 = torch.square(y_hat_N4 - flipped_y_N4)

            # min of the two losses
            min_diff_squared_N4 = torch.min(base_diff_squared_N4, flipped_diff_squared_N4)

            # use the flipped loss for the symmetric chi angles
            # need to temporarily flatten the tensors to use torch.where
            flipped_mask_N4 = make_flipped_mask(aas_N, 4)
            # flipped_mask_N4 = torch.zeros((aas_N.shape[0], 4), dtype=torch.bool, device=aas_N.device)

            diff_squared_N4 = torch.where(flipped_mask_N4, min_diff_squared_N4, base_diff_squared_N4)

            # diff_squared_N4 = base_diff_squared_N4

        else:
            # compute the circular loss, modding the predictions
            y_hat_N4 = torch.remainder(y_hat_N4 + 1, 2) - 1

            # normal loss
            base_abs_diff_N4 = torch.abs(y_hat_N4 - y_N4)
            base_circular_abs_diff_N4 = torch.min(base_abs_diff_N4, 2 - base_abs_diff_N4)
            base_diff_squared_N4 = torch.square(base_circular_abs_diff_N4)

            # loss with flipped angles
            flipped_abs_diff_N4 = torch.abs(y_hat_N4 - flipped_y_N4)
            flipped_circular_abs_diff_N4 = torch.min(flipped_abs_diff_N4, 2 - flipped_abs_diff_N4)
            flipped_diff_squared_N4 = torch.square(flipped_circular_abs_diff_N4)

            # min of the two losses
            min_diff_squared_N4 = torch.min(base_diff_squared_N4, flipped_diff_squared_N4)

            # use the flipped loss for the symmetric chi angles
            # need to temporarily flatten the tensors to use torch.where
            flipped_mask_N4 = make_flipped_mask(aas_N, 4)
            # flipped_mask_N4 = torch.zeros((aas_N.shape[0], 4), dtype=torch.bool, device=aas_N.device)

            diff_squared_N4 = torch.where(flipped_mask_N4, min_diff_squared_N4, base_diff_squared_N4)

            # diff_squared_N4 = base_diff_squared_N4

        if weight_chis:
            # up-weight later chi angles based on the frequency within which they occur
            # NOTE: lower chi angles for residues with a lot of chi angles will be downweighted I think
            weights = torch.tensor([1.0, 18.0/14.0, 18.0/5.0, 18.0/2.0], device=y_hat_N4.device).unsqueeze(0)
            weights = weights / torch.mean(weights)
            diff_squared_N4 = diff_squared_N4 * weights

        mse_per_residue_N = torch.sum(diff_squared_N4, dim=-1) / n_N

        return torch.mean(mse_per_residue_N)


        # # average across the 4 angles, ignoring nans and thus keeping the scale for each angle the same
        # mse_per_residue_N = torch.nanmean(diff_squared_N4, dim=-1)

        # # average across the residues
        # # we do double-averaging to make sure that each residue is weighted in the same way
        # return torch.nanmean(mse_per_residue_N)

    def get_chi_angles_from_predictions(self, y_hat_N4, *args, chi=None):
        if chi is not None:
            raise NotImplementedError('chi is not implemented for this loss function')
        return (torch.remainder(1 + y_hat_N4, 2) - 1) * 180


class SinCosAngleLoss():
    def __init__(self, eps=1e-8):
        self.eps = eps
    
    def __call__(self, y_hat_N8, y_N4, aas_N, chi=None, weight_chis=False, epoch=None):
        y_sin_N4 = torch.sin(y_N4 * torch.pi/180)
        y_cos_N4 = torch.cos(y_N4 * torch.pi/180)
        y_sin_cos_N8 = torch.cat([y_sin_N4, y_cos_N4], dim=-1)

        flipped_y_N4 = torch.remainder(y_N4 + 180 + 180, 360) - 180
        flipped_y_sin_N4 = torch.sin(flipped_y_N4 * torch.pi/180)
        flipped_y_cos_N4 = torch.cos(flipped_y_N4 * torch.pi/180)
        flipped_y_sin_cos_N8 = torch.cat([flipped_y_sin_N4, flipped_y_cos_N4], dim=-1)

        y_hat_sin_cos_N8 = torch.tanh(y_hat_N8)

        if chi is not None:
            assert isinstance(chi, int), 'chi must be an integer, but is {}'.format(type(chi))
            assert chi >= 1 and chi <= 4, 'chi must be between 1 and 4, but is {}'.format(chi)
            # set predicted values equal to true values for all the OTHER chi angles
            # set the denominators to 1

            PADDING_VALUE = 1.0 # this value doesn't matter since they will be substracted from each other. but somehow I feel like if I set it to zero I will get trouble
            y_hat_sin_cos_N8 = torch.cat([torch.nn.functional.pad(y_hat_sin_cos_N8[:, 0].unsqueeze(0).T, (chi-1, 4-chi), mode='constant', value=PADDING_VALUE),
                                          torch.nn.functional.pad(y_hat_sin_cos_N8[:, 1].unsqueeze(0).T, (chi-1, 4-chi), mode='constant', value=PADDING_VALUE)], dim=-1)

        # handle nans by setting true values equal to predicted values so the loss is zero, but adjust the denominator accordingly
        isnan_mask_N8 = torch.isnan(y_sin_cos_N8) # it's the same for all three components
        n_N = torch.sum((~isnan_mask_N8).float(), dim=1) # it's the same for all three components
        VALUE = 1.0
        n_N[n_N == 0] = VALUE # this value will always be at the denominator of a zero so it doesn't matter what it is, it just can't be zero
        y_sin_cos_N8[isnan_mask_N8] = y_hat_sin_cos_N8[isnan_mask_N8]
        flipped_y_sin_cos_N8[isnan_mask_N8] = y_hat_sin_cos_N8[isnan_mask_N8]

        if chi is not None:
            other_chi_mask_8 = torch.cat([torch.logical_not(torch.eye(4, dtype=torch.bool, device=y_hat_sin_cos_N8.device)[chi-1]), torch.logical_not(torch.eye(4, dtype=torch.bool, device=y_hat_sin_cos_N8.device)[chi-1])], dim=-1)
            y_hat_sin_cos_N8[:, other_chi_mask_8] = y_sin_cos_N8[:, other_chi_mask_8]
            n_N = torch.ones_like(n_N)
            weight_chis = False # don't weight chis if we are only looking at one chi angle

        # normal loss
        base_diff_squared_N8 = torch.square(y_hat_sin_cos_N8 - y_sin_cos_N8)

        # loss with flipped angles
        flipped_diff_squared_N8 = torch.square(y_hat_sin_cos_N8 - flipped_y_sin_cos_N8)

        # need to store the minimum based on the minimum of the loss on the *angles*
        # (not on the sin and cos independently, as that might break sin and cos!)
        pred_angles_N4 = self.get_chi_angles_from_predictions(y_hat_N8, chi=chi)
        isnan_mask_N4 = torch.isnan(y_N4)
        y_N4[isnan_mask_N4] = pred_angles_N4[isnan_mask_N4]
        flipped_y_N4[isnan_mask_N4] = pred_angles_N4[isnan_mask_N4]

        base_diff_squared_N4 = torch.abs(pred_angles_N4 - y_N4)
        base_circular_diff_squared_N4 = torch.min(base_diff_squared_N4, 360 - base_diff_squared_N4)

        flipped_diff_squared_N4 = torch.abs(pred_angles_N4 - flipped_y_N4)
        flipped_circular_diff_squared_N4 = torch.min(flipped_diff_squared_N4, 360 - flipped_diff_squared_N4)

        is_flipped_lower = flipped_circular_diff_squared_N4 < base_circular_diff_squared_N4
        is_flipped_lower_N8 = torch.cat([is_flipped_lower, is_flipped_lower], dim=-1)

        min_diff_squared_N8 = torch.where(is_flipped_lower_N8, flipped_diff_squared_N8, base_diff_squared_N8)

        # use the flipped loss for the symmetric chi angles
        flipped_mask_N8 = make_flipped_mask(aas_N, 8)
        diff_squared_N8 = torch.where(flipped_mask_N8, min_diff_squared_N8, base_diff_squared_N8)

        if weight_chis:
            # up-weight later chi angles based on the frequency within which they occur
            # NOTE: lower chi angles for residues with a lot of chi angles will be downweighted I think
            weights = torch.tensor([1.0, 18.0/14.0, 18.0/5.0, 18.0/2.0, 1.0, 18.0/14.0, 18.0/5.0, 18.0/2.0], device=y_hat_sin_cos_N8.device).unsqueeze(0)
            weights = weights / torch.mean(weights)
            # weights = torch.tensor([3.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0], device=y_hat_sin_cos_N8.device).unsqueeze(0)
            diff_squared_N8 = diff_squared_N8 * weights

        mse_per_residue_N = torch.sum(diff_squared_N8, dim=-1) / n_N

        return torch.mean(mse_per_residue_N)

        # # average across the 4 angles, ignoring nans and thus keeping the scale for each angle the same
        # mse_per_residue_N = torch.nanmean(diff_squared_N8, dim=-1)

        # # average across the residues
        # # we do double-averaging to make sure that each residue is weighted in the same way
        # return torch.nanmean(mse_per_residue_N)
    
    def get_chi_angles_from_predictions(self, y_hat_N8, *args, chi=None):
        y_hat_sin_cos_N8 = torch.tanh(y_hat_N8)

        if chi is not None:
            PADDING_VALUE = 1.0 # this value doesn't matter since they will be substracted from each other. but somehow I feel like if I set it to zero I will get trouble
            y_hat_sin_cos_N8 = torch.tanh(y_hat_N8)
            y_hat_sin_cos_N8 = torch.cat([torch.nn.functional.pad(y_hat_sin_cos_N8[:, 0].unsqueeze(0).T, (chi-1, 4-chi), mode='constant', value=PADDING_VALUE),
                                          torch.nn.functional.pad(y_hat_sin_cos_N8[:, 1].unsqueeze(0).T, (chi-1, 4-chi), mode='constant', value=PADDING_VALUE)], dim=-1)
        
        y_sin_N4 = y_hat_sin_cos_N8[:, :4]
        y_cos_N4 = y_hat_sin_cos_N8[:, 4:]
        return torch.atan2(y_sin_N4, y_cos_N4) * 180 / torch.pi


class VectorLoss():
    def __init__(self, model_dir, loss_type, eps=1e-8):
        self.model_dir = model_dir
        self.loss_type = loss_type
        assert self.loss_type in {'mse', 'cosine'}
        self.eps = eps

        # compute and save reconstruction params if they haven't yet been computed
        if not os.path.exists(os.path.join(model_dir, '../reconstruction_params.json')):
            print('Computing reconstruction params from training data...')
            from .simple_reconstruction import compute_ideal_reconstruction_parameters_from_data_given_model_dir
            rec_params, rec_params__vectorized = compute_ideal_reconstruction_parameters_from_data_given_model_dir(model_dir)

            with open(os.path.join(model_dir, '../reconstruction_params.json'), 'w+') as f:
                json.dump(rec_params, f, indent=4)
            with open(os.path.join(model_dir, '../reconstruction_params__vectorized.json'), 'w+') as f:
                json.dump(rec_params__vectorized, f, indent=4)


    def __call__(self, y_hat_N43, y_N43, aas_N, chi=None, weight_chis=False, epoch=None):
        if chi is not None:
            raise NotImplementedError('chi is not implemented for this loss function')

        if y_N43.shape[1] == 5:
            # remove the first column, which is a backbone norm
            y_N43 = y_N43[:, 1:, :]
        
        flipped_y_N43 = - y_N43 # flipping the chi angle by 180 degrees is equal to flipping the sign of the normal vector
        
        isnan_mask_N43 = torch.isnan(y_N43) # it's the same for all three components
        n_N = torch.sum((~isnan_mask_N43[:, :, 0]).float(), dim=1) # it's the same for all three components
        VALUE = 1.0
        n_N[n_N == 0] = VALUE # this value will always be at the denominator of a zero so it doesn't matter what it is, it just can't be zero

        y_N43[isnan_mask_N43] = y_hat_N43[isnan_mask_N43]
        flipped_y_N43[isnan_mask_N43] = y_hat_N43[isnan_mask_N43]

        # make unit vectors
        y_hat_unit_N43 = y_hat_N43 / torch.sqrt(torch.sum(y_hat_N43 * y_hat_N43, dim=-1, keepdim=True))
        y_unit_N43 = y_N43 / torch.sqrt(torch.sum(y_N43 * y_N43, dim=-1, keepdim=True))
        flipped_y_unit_N43 = flipped_y_N43 / torch.sqrt(torch.sum(flipped_y_N43 * flipped_y_N43, dim=-1, keepdim=True))

        if self.loss_type == 'cosine':

            # compute dot product --> cosine similarity --> invert to get a distance
            # normal loss
            base_cos_dist_N4 = 1.0 - torch.sum(y_hat_unit_N43 * y_unit_N43, dim=-1)

            # loss with flipped angles
            flipped_cos_dist_N4 = 1.0 - torch.sum(y_hat_unit_N43 * flipped_y_unit_N43, dim=-1)

            # min of the two losses
            min_cos_dist_N4 = torch.min(base_cos_dist_N4, flipped_cos_dist_N4)

            # use the flipped loss for the symmetric chi angles
            flipped_mask_N4 = make_flipped_mask(aas_N, 4)
            cos_dist_N4 = torch.where(flipped_mask_N4, min_cos_dist_N4, base_cos_dist_N4)

            if weight_chis:
                # up-weight later chi angles based on the frequency within which they occur
                # NOTE: lower chi angles for residues with a lot of chi angles will be downweighted I think
                weights = torch.tensor([1.0, 18.0/14.0, 18.0/5.0, 18.0/2.0], device=y_hat_N43.device).unsqueeze(0)
                weights = weights / torch.mean(weights)
                cos_dist_N4 = cos_dist_N4 * weights

            # average over the 4 vectors
            cos_dist_avg_N = torch.sum(cos_dist_N4, dim=-1) / n_N

            # we do double-averaging to make sure that each residue is weighted in the same way
            return torch.mean(cos_dist_avg_N)

        elif self.loss_type == 'mse':

            # squared diff, averaged across coordinates
            base_mean_diff_squared_N4 = torch.mean(torch.square(y_hat_unit_N43 - y_unit_N43), dim=-1)

            # flipped loss
            flipped_mean_diff_squared_N4 = torch.mean(torch.square(y_hat_unit_N43 - flipped_y_N43), dim=-1)

            # min of the losses
            min_mean_diff_squared_N4 = torch.min(base_mean_diff_squared_N4, flipped_mean_diff_squared_N4)

            # use the flipped loss for the symmetric chi angles
            # need to temporarily flatten the tensors to use torch.where
            flipped_mask_N4 = make_flipped_mask(aas_N, 4)
            mean_diff_squared_N4 = torch.where(flipped_mask_N4, min_mean_diff_squared_N4, base_mean_diff_squared_N4)

            if weight_chis:
                # up-weight later chi angles based on the frequency within which they occur
                # NOTE: lower chi angles for residues with a lot of chi angles will be downweighted I think
                weights = torch.tensor([1.0, 18.0/14.0, 18.0/5.0, 18.0/2.0], device=y_hat_N43.device).unsqueeze(0)
                weights = weights / torch.mean(weights)
                cos_dist_N4 = cos_dist_N4 * weights

            # average over the 4 vectors
            mse_N = torch.sum(mean_diff_squared_N4, dim=-1) / n_N

            # we do double-averaging to make sure that each residue is weighted in the same way
            return torch.mean(mse_N)
        

    def get_chi_angles_from_predictions(self, y_hat_N43, aa_labels_N, backbone_atoms_N43, chi=None):
        if chi is not None:
            raise NotImplementedError('chi is not implemented for this loss function')

        from src.utils.protein_naming import ind_to_ol_size
        from src.sidechain_reconstruction.manual import Reconstructor

        reconstructor = Reconstructor(os.path.join(self.model_dir, '../reconstruction_params__vectorized.json'))

        # format the input in the way in which the reconstructor expects it
        atoms = [backbone_atoms_N43[:, 0, :].cpu(), backbone_atoms_N43[:, 1, :].cpu(), backbone_atoms_N43[:, 2, :].cpu(), backbone_atoms_N43[:, 3, :].cpu()]
        AA = [ind_to_ol_size[aa_idx.item()] for aa_idx in aa_labels_N]
        normal_vectors = y_hat_N43.cpu()

        # reconstruct the sidechains, which also gives the chi angles
        _, chi_angles = reconstructor.reconstruct_from_normal_vectors(atoms, AA, normal_vectors)

        return torch.stack(chi_angles, dim=1)


def angle_loss(pred_angles_N4, true_angles_N4, aas_N):

    flipped_true_angles_N4 = torch.remainder(true_angles_N4 + 180 + 180, 360) - 180

    base_naive_abs_diff_N4 = torch.abs(pred_angles_N4 - true_angles_N4)
    base_circular_abs_diff_N4 = torch.min(base_naive_abs_diff_N4, 360 - base_naive_abs_diff_N4)

    ## account for symmetric chi angles
    flipped_naive_abs_diff_N4 = torch.abs(pred_angles_N4 - flipped_true_angles_N4)
    flipped_circular_abs_diff_N4 = torch.min(flipped_naive_abs_diff_N4, 360 - flipped_naive_abs_diff_N4)

    # min of the two losses
    min_circular_abs_diff_N4 = torch.min(base_circular_abs_diff_N4, flipped_circular_abs_diff_N4)

    # use the flipped loss for the symmetric chi angles
    flipped_mask_N4 = make_flipped_mask(aas_N, 4)
    circular_abs_diff_N4 = torch.where(flipped_mask_N4, min_circular_abs_diff_N4, base_circular_abs_diff_N4)

    return circular_abs_diff_N4


def loss_per_chi_angle(y_hat_N4, y_N4, aas_N, return_global_accuracy=False, return_per_aa=False):
    # compute mae, and chi-angle accuracy with standard 20 degrees cutoff
    # angles will always be given in radians, so start by converting to degrees since they are more intuitively interpretable
    y_hat_deg_N4 = y_hat_N4.cpu()
    y_deg_N4 = y_N4.cpu()
    aas_N = aas_N.cpu()

    circular_abs_diff_N4 = angle_loss(y_hat_deg_N4, y_deg_N4, aas_N)

    # return error per residue-type
    aas_set = set(aas_N.numpy().tolist())
    mae_per_aa = {}
    error_per_aa = {}
    for aa in aas_set:
        curr_abs_diff_M4 = circular_abs_diff_N4[aas_N == aa]
        error_per_aa[aa] = curr_abs_diff_M4.detach().cpu().numpy()
        curr_denominators_4 = torch.logical_not(torch.isnan(curr_abs_diff_M4)).sum(dim=0) + 1e-9
        curr_mae_per_angle_4 = torch.nansum(curr_abs_diff_M4, dim=0) / curr_denominators_4
        mae_per_aa[aa] = curr_mae_per_angle_4

    # circular_abs_diff_N4 = base_circular_abs_diff_N4

    ## There will be an output NaN if there is no angle in this batch, nothing to be scared about though

    # compute mae loss
    denominators_4 = torch.logical_not(torch.isnan(circular_abs_diff_N4)).sum(dim=0) + 1e-9
    mae_per_angle_4 = torch.nansum(circular_abs_diff_N4, dim=0) / denominators_4

    # compute chi-angle accuracy
    denominators_4 = torch.logical_not(torch.isnan(circular_abs_diff_N4)).sum(dim=0) + 1e-9

    accuracy_per_angle_4 = torch.nansum((circular_abs_diff_N4 <= 20).float(), dim=0) / denominators_4

    if return_global_accuracy:

        has_error_N4 = circular_abs_diff_N4.detach().cpu().numpy() > 20
        is_good_N = ~np.logical_or.reduce(has_error_N4, axis=1)
        global_accuracy = np.mean(is_good_N)
        
        if return_per_aa:
            return mae_per_angle_4, accuracy_per_angle_4, global_accuracy, mae_per_aa, error_per_aa
        else:
            return mae_per_angle_4, accuracy_per_angle_4, global_accuracy
    
    if return_per_aa:
        return mae_per_angle_4, accuracy_per_angle_4, mae_per_aa, error_per_aa
    else:
        return mae_per_angle_4, accuracy_per_angle_4