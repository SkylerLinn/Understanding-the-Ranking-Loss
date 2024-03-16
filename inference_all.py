
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import numpy as np 
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
import gc
import argparse
import os
from pathlib import Path
import pandas as pd 
import torch
from tqdm import tqdm

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '#inference.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])


if __name__ == '__main__':
    ''' Usage: python inference.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform h5 data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    model.load_weights(model.checkpoint)
    logging.info(model.checkpoint)
    # create submission path 
    if not os.path.exists('./submission'):
        os.mkdir('./submission')
    submission_data = pd.DataFrame()

    params['shuffle'] = False
    train_gen,valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
    if train_gen:
        y_pred_train = model.predict(train_gen)
    if valid_gen:
        y_pred_valid = model.predict(valid_gen)
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    # test_result = {}
    if test_gen:
      y_pred_test = model.predict(test_gen)

    submission_data['infer'] = np.concatenate([y_pred_train, y_pred_valid, y_pred_test],axis=0)
    submission_data.to_csv('./submission/submission_{}.csv'.format(args['expid']),index=None,sep='\t')