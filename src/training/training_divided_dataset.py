

'''

Performs training by loading training data into RAM in large chunks, where each chunk is into a separate hdf5 file (and is the entire hdf5 file).

'''


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

from src.so3_cnn.so3.functional import put_dict_on_device

from .losses import AngleLoss, SinCosAngleLoss, VectorLoss, loss_per_chi_angle

from .utils import *



def run_training_divided_dataset(model_dir: str):
    '''
    Assumes that directory 'model_dir' exists and contains json file with data and model hyperprameters 
    '''

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
    # from .data import load_data
    # datasets, data_irreps, norm_factor = load_data(hparams, splits=['train', 'valid'])
    # train_dataset, valid_dataset = datasets['train'], datasets['valid']
    # train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=False, drop_last=False)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=False, drop_last=False)
    from .data import load_single_split_data
    train_dataset, data_irreps, norm_factor = load_single_split_data(hparams, 'training__0', get_norm_factor_if_training=True)
    valid_dataset, _, _ = load_single_split_data(hparams, 'validation')

    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], generator=rng, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True), drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=512, generator=rng, shuffle=False, drop_last=False)

    ########## THIS CODE BLOCK ABOVE MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########
    
    print('Data Irreps: %s' % (str(data_irreps)))
    sys.stdout.flush()

    # set norm factor in hparams, set chi if not present, save new hparams
    hparams['model_hparams']['input_normalizing_constant'] = norm_factor
    if 'chi' not in hparams:
        hparams['chi'] = None
    if 'weight_chis' not in hparams:
        hparams['weight_chis'] = True
    with open(os.path.join(model_dir, 'hparams.json'), 'w+') as f:
        json.dump(hparams, f, indent=2)
    
    model, loss_fn, device = general_model_init(model_dir, hparams, data_irreps)

    # setup learning algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    if hparams['lr_scheduler'] is None:
        lr_scheduler = None
    elif hparams['lr_scheduler'] == 'reduce_lr_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=hparams['factor'], patience=hparams['patience'], threshold=hparams['threshold'], threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)
    else:
        raise NotImplementedError()

    logfile = open(os.path.join(model_dir, 'log.txt'), 'w+')

    # training loop!
    num_validation_steps_per_epoch = 5
    divisor_for_validation_step = hparams['num_train_datasets'] // num_validation_steps_per_epoch
    train_loss_trace = []
    valid_loss_trace = []
    train_mae_trace, train_acc_trace = [[], [], [], []], [[], [], [], []]
    valid_mae_trace, valid_acc_trace = [[], [], [], []], [[], [], [], []]
    lowest_valid_loss = np.inf
    for epoch in range(hparams['n_epochs']):
        print('\nEpoch %d/%d\ttrain-val loss\t\ttrain-val 20-deg accuracy\t\t train-val MAE' % (epoch+1, hparams['n_epochs']))
        print('\nEpoch %d/%d\ttrain-val loss\t\ttrain-val 20-deg accuracy\t\t train-val MAE' % (epoch+1, hparams['n_epochs']), file=logfile, flush=True)
        sys.stdout.flush()
        temp_train_loss_trace = []
        train_mae_trace, temp_train_acc_trace = [[], [], [], []], [[], [], [], []]
        n_steps = 0
        start_time = time.time()

        if epoch == 0:
            shuffled_train_dataset_indices = np.hstack([np.array([0]), np.random.permutation(hparams['num_train_datasets']-1)+1]) # put the zero index first, as that's the first dataset no matter what
        else:
            shuffled_train_dataset_indices = np.random.permutation(hparams['num_train_datasets'])

        for train_dataset_i in range(hparams['num_train_datasets']):

            # get new training dataset
            if train_dataset_i > 0 or epoch > 0:
                # print('Loading new training dataset...', flush=True)
                del train_dataset
                del train_dataloader
                train_dataset, _, _ = load_single_split_data(hparams, f'training__{shuffled_train_dataset_indices[train_dataset_i]}', get_norm_factor_if_training=False)
                train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], generator=rng, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True), drop_last=False)
                # print('Done!')

            for train_i, (X_train, X_train_vec, aa_label_train, angles_train, vectors_train, backbone_coords_train, (rot, data_ids)) in enumerate(train_dataloader):
                            
                X_train = put_dict_on_device(X_train, device)
                aa_label_train = aa_label_train.to(device)
                angles_train = angles_train.to(device).float()
                vectors_train = vectors_train.to(device).float()
                
                model.train()

                optimizer.zero_grad()
                
                if '_cond_and_vec_cond' in hparams['model_type']:
                    y_train_hat = model(X_train, one_hot_encode(aa_label_train, NUM_AAS).to(device).float(), backbone_coords_train.to(device).float())
                elif '_cond' in hparams['model_type']:
                    y_train_hat = model(X_train, one_hot_encode(aa_label_train, NUM_AAS).to(device).float())
                else:
                    y_train_hat = model(X_train)
                            
                # compute loss
                # make copies of vectors and angles to avoid modifying the original tensors
                if isinstance(loss_fn, VectorLoss):
                    loss_train = loss_fn(y_train_hat, vectors_train.clone(), aa_label_train, epoch=epoch, weight_chis=hparams['weight_chis'], chi=hparams['chi'])
                else:
                    loss_train = loss_fn(y_train_hat, angles_train.clone(), aa_label_train, epoch=epoch, weight_chis=hparams['weight_chis'], chi=hparams['chi'])
                
                # print(loss_train)
                # if train_i == 2:
                #     exit(1)
                
                # keep track of chi-angle loss
                predicted_chi_angles_train = loss_fn.get_chi_angles_from_predictions(y_train_hat, aa_label_train, backbone_coords_train, chi=hparams['chi']).detach().cpu()
                mae_per_angle_train_temp, accuracy_per_angle_train_temp = loss_per_chi_angle(predicted_chi_angles_train, angles_train, aa_label_train)

                temp_train_loss_trace.append(loss_train.detach().cpu().item())
                for chi_angle_idx in range(4):
                    train_mae_trace[chi_angle_idx].append(mae_per_angle_train_temp[chi_angle_idx].item())
                    temp_train_acc_trace[chi_angle_idx].append(accuracy_per_angle_train_temp[chi_angle_idx].item())

                loss_train.backward()
                optimizer.step()

                n_steps += 1


            # validation step at the end of every slice of training data

            if not (train_dataset_i % divisor_for_validation_step == 0 or train_dataset_i == hparams['num_train_datasets'] - 1):
                continue

            temp_valid_loss_trace = []
            true_angles_trace, predicted_angles_trace = [], []
            aa_label_trace = []
            
            for valid_i, (X_valid, X_valid_vec, aa_label_valid, angles_valid, vectors_valid, backbone_coords_valid, (rot, data_ids)) in enumerate(valid_dataloader):
                X_valid = put_dict_on_device(X_valid, device)
                aa_label_valid = aa_label_valid.to(device)
                angles_valid = angles_valid.to(device).float()
                vectors_valid = vectors_valid.to(device).float()
                model.eval()

                if '_cond_and_vec_cond' in hparams['model_type']:
                    y_valid_hat = model(X_valid, one_hot_encode(aa_label_valid, NUM_AAS).to(device).float(), backbone_coords_valid.to(device).float())
                elif '_cond' in hparams['model_type']:
                    y_valid_hat = model(X_valid, one_hot_encode(aa_label_valid, NUM_AAS).to(device).float())
                else:
                    y_valid_hat = model(X_valid)
                
                if isinstance(loss_fn, VectorLoss):
                    loss_valid = loss_fn(y_valid_hat, vectors_valid.clone(), aa_label_valid, epoch=epoch, weight_chis=hparams['weight_chis'], chi=hparams['chi'])
                else:
                    loss_valid = loss_fn(y_valid_hat, angles_valid.clone(), aa_label_valid, epoch=epoch, weight_chis=hparams['weight_chis'], chi=hparams['chi'])
                
                predicted_chi_angles_valid = loss_fn.get_chi_angles_from_predictions(y_valid_hat, aa_label_valid, backbone_coords_valid, chi=hparams['chi'])
                true_angles_trace.append(angles_valid.detach().cpu())
                predicted_angles_trace.append(predicted_chi_angles_valid.detach().cpu())
                aa_label_trace.append(aa_label_valid.detach().cpu())

                temp_valid_loss_trace.append(loss_valid.item())
            
            true_angles = torch.cat(true_angles_trace, dim=0)
            predicted_angles = torch.cat(predicted_angles_trace, dim=0)
            aa_label_trace = torch.cat(aa_label_trace, dim=0)
            mae_per_angle_valid, accuracy_per_angle_valid = loss_per_chi_angle(predicted_angles, true_angles, aa_label_trace) # this guy might take a while to run on a big-ass dataset... we're talking operations between tensors of size 700_000
            
            curr_train_loss = np.mean(temp_train_loss_trace)
            curr_valid_loss = np.mean(temp_valid_loss_trace)

            end_time = time.time()
            print('%d/%d:\t\t%.4f - %.4f' % (train_dataset_i+1, hparams['num_train_datasets'], curr_train_loss, curr_valid_loss), end='\t\t')
            print('%d/%d:\t\t%.4f - %.4f' % (train_dataset_i+1, hparams['num_train_datasets'], curr_train_loss, curr_valid_loss), end='\t\t', file=logfile, flush=True)
            for chi_angle_idx in range(4):
                print('%.0f-%.0f' % (np.mean(temp_train_acc_trace[chi_angle_idx])*100, accuracy_per_angle_valid[chi_angle_idx].item()*100), end='\t')
                print('%.0f-%.0f' % (np.mean(temp_train_acc_trace[chi_angle_idx])*100, accuracy_per_angle_valid[chi_angle_idx].item()*100), end='\t', file=logfile, flush=True)
            print('\t', end='')
            print('\t', end='', file=logfile, flush=True)

            for chi_angle_idx in range(4):
                print('%.0f-%.0f' % (np.mean(train_mae_trace[chi_angle_idx]), mae_per_angle_valid[chi_angle_idx].item()), end='\t')
                print('%.0f-%.0f' % (np.mean(train_mae_trace[chi_angle_idx]), mae_per_angle_valid[chi_angle_idx].item()), end='\t', file=logfile, flush=True)
            print('\t', end='')
            print('\t', end='', file=logfile, flush=True)

            print('Time (s): %.2f' % (end_time - start_time))
            print('Time (s): %.2f' % (end_time - start_time), file=logfile, flush=True)
            sys.stdout.flush()
            
            # update lr with scheduler, only at the end of the epoch
            if lr_scheduler is not None:
                lr_scheduler.step(curr_valid_loss)

            # record best model so far
            if curr_valid_loss < lowest_valid_loss:
                lowest_valid_loss = curr_valid_loss
                
                save_attempts = 0
                while (True):
                    save_attempts += 1
                    if save_attempts > 50:
                        print("SAVING FAILED")
                        sys.exit()
                    try: # disk quota error work around?
                        torch.save(model.state_dict(), 
                                os.path.join(model_dir, 'lowest_valid_loss_model.pt'))
                        break
                    except:
                        time.sleep(10)

            train_loss_trace.append(curr_train_loss)
            valid_loss_trace.append(curr_valid_loss)

            for chi_angle_idx in range(4):
                train_mae_trace[chi_angle_idx].append(np.mean(train_mae_trace[chi_angle_idx]))
                valid_mae_trace[chi_angle_idx].append(mae_per_angle_valid[chi_angle_idx].detach().cpu().item())
                train_acc_trace[chi_angle_idx].append(np.mean(temp_train_acc_trace[chi_angle_idx]))
                valid_acc_trace[chi_angle_idx].append(accuracy_per_angle_valid[chi_angle_idx].detach().cpu().item())
            
            n_steps = 0
            temp_train_loss_trace = []
            temp_valid_loss_trace = []
            start_time = time.time()


    save_attempts = 0
    while save_attempts < 50:
        try:    
            # save last model
            torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pt'))

            # save loss traces, both as arrays and as plots
            if not os.path.exists(os.path.join(model_dir, 'loss_traces')):
                os.mkdir(os.path.join(model_dir, 'loss_traces'))

            np.save(os.path.join(model_dir, 'loss_traces', 'train_loss_trace.npy'), train_loss_trace)
            np.save(os.path.join(model_dir, 'loss_traces', 'valid_loss_trace.npy'), valid_loss_trace)
            np.save(os.path.join(model_dir, 'loss_traces', 'valid_acc_trace.npy'), valid_acc_trace)

            iterations = np.arange(len(train_loss_trace))

            plt.figure(figsize=(10, 4))
            plt.plot(iterations, train_loss_trace, label='train')
            plt.plot(iterations, valid_loss_trace, label='valid')
            plt.ylabel('MSE loss')
            plt.xlabel('Evaluation iterations (%d epochs)' % (hparams['n_epochs']))
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'loss_trace.png'))
            
            if hparams['model_type'] == 'so3_convnet_vec':
                plt.figure(figsize=(10, 4))
                for vi in range(4):
                    plt.plot(valid_acc_trace[vi], label=f'valid-chi{vi}')
                plt.ylabel('Accuracy')
                plt.xlabel('Evaluation iterations (%d epochs)' % (hparams['n_epochs']))
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, 'acc_trace.png'))
            break

        except:
            time.sleep(10)
    
    logfile.close()
    
