
import os
import yaml
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from runtime.sidechain_prediction.src.training import run_training
from runtime.sidechain_prediction.src.training_divided_dataset import run_training_divided_dataset
from runtime.sidechain_prediction.src.inference import run_inference
from runtime.sidechain_prediction.src.simple_reconstruction import reconstruction_null, reconstruction_from_predictions, plot_mae_per_chi_binned_by_proportion_of_sidechains_removed
from runtime.sidechain_prediction.src.viz import plot_chi_angle_predictions_distributions_vs_true

from protein_holography_pytorch.utils.argparse import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='runs/throwaway')
    parser.add_argument('--training_config', type=optional_str, default='config/sime_task/so3_convnet_sin_cos.yaml',
                        help='Path to training config file. Ignored when --eval_only is set to True.')
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--tests_only', action='store_true', default=False,
                        help='If set to True, then only run the tests, and do not run training or inference.\
                              Intended to be used only when running new tests or changing/debuging existing ones, so inference does not have to be re-run.')
    parser.add_argument('--no_tests', action='store_true', default=False)
    parser.add_argument('--compute_null_reconstruction', action='store_true', default=False)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--eval_splits', type=comma_sep_str_list, default='valid,test', choices=['valid', 'test', 'valid,test'])
    args = parser.parse_args()

    # make directory if it does not already exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # load config if requested, if config is None, then use hparams within model_dir
    if args.training_config is not None and not (args.eval_only or args.tests_only):
        with open(args.training_config, 'r') as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)

        # save hparams as json file within expriment_dir
        with open(os.path.join(args.model_dir, 'hparams.json'), 'w+') as f:
            json.dump(hparams, f, indent=4)
        
    else:
        with open(os.path.join(args.model_dir, 'hparams.json'), 'r') as f:
            hparams = json.load(f)
    
    if args.compute_null_reconstruction:
        reconstruction_null(args.model_dir, args.eval_batch_size)

    if not (args.eval_only or args.tests_only):
        # launch training script
        if 'num_train_datasets' in hparams:
            run_training_divided_dataset(args.model_dir)
        else:
            run_training(args.model_dir)
    
    if not args.tests_only:
        # launch inference script
        run_inference(args.model_dir, batch_size=args.eval_batch_size, splits=args.eval_splits)

    if not args.no_tests:
        # launch tests
        plot_mae_per_chi_binned_by_proportion_of_sidechains_removed(args.model_dir)
        plot_chi_angle_predictions_distributions_vs_true(args.model_dir, splits=args.eval_splits)
        reconstruction_from_predictions(args.model_dir)





