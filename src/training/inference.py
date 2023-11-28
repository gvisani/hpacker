
import os, sys
from sys import argv
import gzip, pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from typing import *

from src.so3_cnn.models import CGNet, SO3_ConvNet, SO3_ConvNet_WithInvariantConditioning, SO3_ConvNet_VecPrediction, SO3_ConvNet_VecPrediction_WithInvariantConditioning
from src.so3_cnn.so3.functional import put_dict_on_device
from src.so3_cnn.cg_coefficients import get_w3j_coefficients

from .losses import AngleLoss, SinCosAngleLoss, VectorLoss, loss_per_chi_angle

from .utils import *


def run_inference(model_dir: str,
                  batch_size: int = 64, 
                  splits=['valid', 'test']):
    
    print('Running inference on splits: %s' % (str(splits)))

    # get hparams from json
    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # seed the random number generator
    if hparams['seed'] is not None:
        rng = torch.Generator().manual_seed(hparams['seed'])
    else:
        rng = torch.Generator() # random seed

    print('Loading data...')
    sys.stdout.flush()
    
    ########## THE CODE BLOCK BELOW MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########

    # get data and make dataloaders
    from .data import load_data
    datasets, data_irreps, _ = load_data(hparams, splits=splits)
    dataloaders = {split: DataLoader(datasets[split], batch_size=batch_size, generator=rng, shuffle=False, drop_last=False) for split in splits}

    ########## THIS CODE BLOCK ABOVE MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########
    
    print('Data Irreps: %s' % (str(data_irreps)))
    sys.stdout.flush()

    model, loss_fn, device = general_model_init(model_dir, hparams, data_irreps)

    model.load_state_dict(torch.load(os.path.join(model_dir, 'lowest_valid_loss_model.pt'), map_location=torch.device(device)))
    model.eval()

    
    summary_results_dict = {}
    per_example_results_dict = {}
    
    # inference loop
    for split, dataloader in dataloaders.items():
        y_hat_trace = []
        true_angles_trace = []
        predicted_angles_trace = []
        loss_trace = []
        res_ids_trace = []
        backbone_coords_trace = []
        aa_label_trace = []
        for i, (X, X_vec, aa_label, angles, vectors, backbone_coords, (rot, data_ids)) in tqdm(enumerate(dataloader), total=len(dataloader)):
            X = put_dict_on_device(X, device)
            aa_label = aa_label.to(device)
            angles = angles.to(device).float()
            vectors = vectors.to(device).float()
            model.eval()

            if '_cond' in hparams['model_type']:
                y_hat = model(X, one_hot_encode(aa_label, NUM_AAS).to(device).float())
            else:
                y_hat = model(X)
            
            if isinstance(loss_fn, VectorLoss):
                loss = loss_fn(y_hat, vectors.clone(), aa_label, chi=hparams['chi'])
            else:
                loss = loss_fn(y_hat, angles.clone(), aa_label, chi=hparams['chi'])
            
            predicted_chi_angles = loss_fn.get_chi_angles_from_predictions(y_hat, aa_label, backbone_coords, chi=hparams['chi'])
            true_angles_trace.append(angles.detach().cpu())
            predicted_angles_trace.append(predicted_chi_angles.detach().cpu())
            res_ids_trace.append(data_ids)
            y_hat_trace.append(y_hat.detach().cpu())
            backbone_coords_trace.append(backbone_coords.detach().cpu().numpy())
            aa_label_trace.append(aa_label.detach().cpu())

            loss_trace.append(loss.item())
        
        loss = np.mean(loss_trace)
        true_angles_N4 = torch.cat(true_angles_trace, dim=0)
        predicted_angles_N4 = torch.cat(predicted_angles_trace, dim=0)
        aa_label_trace_N = torch.cat(aa_label_trace, dim=0)

        # naive_error_on_angles_N4 = np.abs(true_angles_N4.detach().cpu().numpy() - predicted_angles_N4.detach().cpu().numpy())
        # error_on_angles_N4 = np.minimum(naive_error_on_angles_N4, 360 - naive_error_on_angles_N4) # remember circular error!
        res_ids_N = np.hstack(res_ids_trace)
        y_hat_N4_or_N8_or_N43 = torch.cat(y_hat_trace, dim=0).detach().cpu().numpy() # this could be angles, sines and cosines, or vectors
        backbone_coords_N43 = np.vstack(backbone_coords_trace)

        mae_per_angle_4, accuracy_per_angle_4 = loss_per_chi_angle(predicted_angles_N4, true_angles_N4, aa_label_trace_N)

        # save results
        summary_results_dict[split] = {
            'loss': loss,
            'mae_per_angle_4': mae_per_angle_4.tolist(),
            'accuracy_per_angle_4': accuracy_per_angle_4.tolist(),
        }

        per_example_results_dict[split] = {
            'res_ids': res_ids_N,
            'true_angles': true_angles_N4.detach().cpu().numpy(),
            'y_hat': y_hat_N4_or_N8_or_N43,
            'predicted_angles': predicted_angles_N4.detach().cpu().numpy(),
            'backbone_coords': backbone_coords_N43,
        }
    
    # save results
    with open(os.path.join(model_dir, 'summary_results_dict.json'), 'w+') as f:
        json.dump(summary_results_dict, f, indent=4)
    
    with gzip.open(os.path.join(model_dir, 'per_example_results_dict.gz'), 'wb') as f:
        pickle.dump(per_example_results_dict, f)


    



