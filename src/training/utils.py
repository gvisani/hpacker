
import os, sys
import json
import torch

NUM_AAS = 20

def one_hot_encode(x, n_classes):
    return torch.eye(n_classes, device=x.device)[x.to(torch.long)]

from src.so3_cnn.models import CGNet, SO3_ConvNet, SO3_ConvNet_WithInvariantConditioning, SO3_ConvNet_VecPrediction, SO3_ConvNet_VecPrediction_WithInvariantConditioning, SO3_ConvNet_WithInvariantConditioningAtTheBeginning, SO3_ConvNet_WithInvariantAndVectorConditioningAtTheBeginning
from src.so3_cnn.cg_coefficients import get_w3j_coefficients
from .losses import AngleLoss, SinCosAngleLoss, VectorLoss, loss_per_chi_angle

def general_model_init(model_dir, hparams, data_irreps):

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device), flush=True)

    # load w3j coefficients
    w3j_matrices = get_w3j_coefficients(hparams['lmax'])
    for key in w3j_matrices:
        # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
        if device is not None:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
        else:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
        w3j_matrices[key].requires_grad = False
    
    if hparams['model_type'] == 'cgnet':
        model = CGNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
    elif hparams['model_type'] == 'so3_convnet':
        model = SO3_ConvNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
    elif hparams['model_type'] == 'so3_convnet_cond':
        model = SO3_ConvNet_WithInvariantConditioning(
            data_irreps, 
            w3j_matrices, hparams['model_hparams'], 
            normalize_input_at_runtime=hparams['normalize_input']
        ).to(device)
    elif hparams['model_type'] == 'so3_convnet_cond_beginning':
        model = SO3_ConvNet_WithInvariantConditioningAtTheBeginning(
            data_irreps, 
            w3j_matrices, hparams['model_hparams'], 
            normalize_input_at_runtime=hparams['normalize_input']
        ).to(device)
    elif hparams['model_type'] == 'so3_convnet_cond_and_vec_cond_beginning':
        model = SO3_ConvNet_WithInvariantAndVectorConditioningAtTheBeginning(
            data_irreps, 
            w3j_matrices, hparams['model_hparams'], 
            normalize_input_at_runtime=hparams['normalize_input']
        ).to(device)
    elif hparams['model_type'] in {'so3_convnet_vec', 'so3_convnet_vec_mse'}:
        model = SO3_ConvNet_VecPrediction(
            data_irreps, 
            w3j_matrices, hparams['model_hparams'], 
            normalize_input_at_runtime=hparams['normalize_input']
        ).to(device)
        
    elif hparams['model_type'] in {'so3_convnet_vec_cond', 'so3_convnet_vec_mse_cond'}:
        model = SO3_ConvNet_VecPrediction_WithInvariantConditioning(
            data_irreps, 
            w3j_matrices, hparams['model_hparams'], 
            normalize_input_at_runtime=hparams['normalize_input']
        ).to(device)

    else:
        raise NotImplementedError()
    
    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params), flush=True)

    if hparams['model_type'] in {'so3_convnet_vec', 'so3_convnet_vec_cond'}:
        assert hparams['model_hparams']['output_dim'] in {1, 4}
        loss_fn = VectorLoss(model_dir, 'cosine')
    elif hparams['model_type'] in {'so3_convnet_vec_mse', 'so3_convnet_vec_mse_cond'}:
        assert hparams['model_hparams']['output_dim'] in {1, 4}
        loss_fn = VectorLoss(model_dir, 'mse')
    else: # angle model
        assert hparams['model_hparams']['output_dim'] in {1, 2, 4, 8}
        if hparams['model_hparams']['output_dim'] in {1, 4}:
            loss_fn = AngleLoss()
        else:
            loss_fn = SinCosAngleLoss()
    
    return model, loss_fn, device