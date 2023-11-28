
'''

Performs training in the standard way of loading all data into RAM. Doesn't work for too large datasets.

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


def run_training(model_dir: str):
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
    from .data import load_data
    datasets, data_irreps, norm_factor = load_data(hparams, splits=['train', 'valid'])
    train_dataset, valid_dataset = datasets['train'], datasets['valid']
    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=False, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=hparams['batch_size'], generator=rng, shuffle=False, drop_last=False)

    ########## THIS CODE BLOCK ABOVE MAY BE CHANGED TO ACCOMODATE A DIFFERENT DATA-LOADING PIPELINE ##########
    
    print('Data Irreps: %s' % (str(data_irreps)))
    sys.stdout.flush()

    # set norm factor in hparams, save new hparams
    hparams['model_hparams']['input_normalizing_constant'] = norm_factor
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
    n_times_to_report = 5
    n_total_steps = len(train_dataloader)
    n_steps_until_report = n_total_steps // n_times_to_report
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
        reported_times = 0
        start_time = time.time()
        for train_i, (X_train, X_train_vec, aa_label_train, angles_train, vectors_train, backbone_coords_train, (rot, data_ids)) in enumerate(train_dataloader):
                        
            X_train = put_dict_on_device(X_train, device)
            aa_label_train = aa_label_train.to(device)
            angles_train = angles_train.to(device).float()
            vectors_train = vectors_train.to(device).float()
            
            model.train()

            optimizer.zero_grad()
            
            if '_cond' in hparams['model_type']:
                y_train_hat = model(X_train, one_hot_encode(aa_label_train, NUM_AAS).to(device).float())
            else:
                y_train_hat = model(X_train)
                        
            # compute loss
            if isinstance(loss_fn, VectorLoss):
                loss_train = loss_fn(y_train_hat, vectors_train.clone(), aa_label_train, epoch=epoch, weight_chis=hparams['weight_chis'], chi=hparams['chi'])
            else:
                loss_train = loss_fn(y_train_hat, angles_train.clone(), aa_label_train, epoch=epoch, weight_chis=hparams['weight_chis'], chi=hparams['chi'])
            
            # print(loss_train)
            # if train_i == 2:
            #     exit(1)
            
            # keep track of chi-angle loss
            predicted_chi_angles_train = loss_fn.get_chi_angles_from_predictions(y_train_hat, aa_label_train, backbone_coords_train, chi=hparams['chi'])
            mae_per_angle_train_temp, accuracy_per_angle_train_temp = loss_per_chi_angle(predicted_chi_angles_train, angles_train, aa_label_train)

            temp_train_loss_trace.append(loss_train.item())
            for chi_angle_idx in range(4):
                train_mae_trace[chi_angle_idx].append(mae_per_angle_train_temp[chi_angle_idx].item())
                temp_train_acc_trace[chi_angle_idx].append(accuracy_per_angle_train_temp[chi_angle_idx].item())

            loss_train.backward()
            optimizer.step()

            n_steps += 1

            # record train and validation loss
            if n_steps == n_steps_until_report or train_i == len(train_dataloader)-1:
                reported_times += 1
                temp_valid_loss_trace = []
                true_angles_trace, predicted_angles_trace = [], []
                aa_label_trace = []
                
                for valid_i, (X_valid, X_valid_vec, aa_label_valid, angles_valid, vectors_valid, backbone_coords_valid, (rot, data_ids)) in enumerate(valid_dataloader):
                    X_valid = put_dict_on_device(X_valid, device)
                    aa_label_valid = aa_label_valid.to(device)
                    angles_valid = angles_valid.to(device).float()
                    vectors_valid = vectors_valid.to(device).float()
                    model.eval()

                    if '_cond' in hparams['model_type']:
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
                mae_per_angle_valid, accuracy_per_angle_valid = loss_per_chi_angle(predicted_angles, true_angles, aa_label_trace)
                
                curr_train_loss = np.mean(temp_train_loss_trace)
                curr_valid_loss = np.mean(temp_valid_loss_trace)

                end_time = time.time()
                print('%d/%d:\t\t%.4f - %.4f' % (reported_times, n_times_to_report, curr_train_loss, curr_valid_loss), end='\t\t')
                print('%d/%d:\t\t%.4f - %.4f' % (reported_times, n_times_to_report, curr_train_loss, curr_valid_loss), end='\t\t', file=logfile, flush=True)
                for chi_angle_idx in range(4):
                    print('%d-%d' % (np.mean(temp_train_acc_trace[chi_angle_idx])*100, accuracy_per_angle_valid[chi_angle_idx].item()*100), end='\t')
                    print('%d-%d' % (np.mean(temp_train_acc_trace[chi_angle_idx])*100, accuracy_per_angle_valid[chi_angle_idx].item()*100), end='\t', file=logfile, flush=True)
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
                if train_i == len(train_dataloader)-1:
                    if lr_scheduler is not None:
                        lr_scheduler.step(curr_valid_loss)

                # record best model so far, only at the end of the epoch
                if train_i == len(train_dataloader)-1:
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
                    valid_mae_trace[chi_angle_idx].append(mae_per_angle_valid[chi_angle_idx].item())
                    train_acc_trace[chi_angle_idx].append(np.mean(temp_train_acc_trace[chi_angle_idx]))
                    valid_acc_trace[chi_angle_idx].append(accuracy_per_angle_valid[chi_angle_idx].item())
                
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
            plt.close()
            
            if hparams['model_type'] == 'so3_convnet_vec':
                plt.figure(figsize=(10, 4))
                for vi in range(4):
                    plt.plot(valid_acc_trace[vi], label=f'valid-chi{vi}')
                plt.ylabel('Accuracy')
                plt.xlabel('Evaluation iterations (%d epochs)' % (hparams['n_epochs']))
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, 'acc_trace.png'))
                plt.close()
            break

        except:
            time.sleep(10)
    
    logfile.close()
    
