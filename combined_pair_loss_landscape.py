
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
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '#inference.log')

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
    parser.add_argument('--xmin', type=float, default=-5.)
    parser.add_argument('--xmax', type=float, default=10.)
    parser.add_argument('--xnum', type=int, default=30)
    parser.add_argument('--ymin', type=float, default=-10.)
    parser.add_argument('--ymax', type=float, default=5.)
    parser.add_argument('--ynum', type=int, default=30)
    args = vars(parser.parse_args())
    
    xmin, xmax, xnum = args['xmin'], args['xmax'], args['xnum']
    ymin, ymax, ynum = args['ymin'], args['ymax'], args['ynum']

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
    
    # model.load_weights(model.checkpoint)
    params['shuffle'] = False
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    model.to(model.device)
    state_dict = torch.load(model.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    cat_weight = torch.cat([state_dict['fc.weight'],state_dict['fc.bias'].reshape(1,-1)],dim=1)
    l2_norm = torch.norm(cat_weight, p=2)
    fc_weight_shape = state_dict['fc.weight'].shape
    fc_bias_shape = state_dict['fc.bias'].shape

    def compute_loss(y_pred,logit,y_true):
        weight = torch.where(y_true==1., model.pos_weight, 1)
        index_pos = torch.nonzero(y_true.reshape(-1,)==1).reshape(-1,) # (1049,)
        index_neg = torch.nonzero(y_true.reshape(-1,)==0).reshape(-1,)
        cat = torch.cartesian_prod(index_pos, index_neg)
        pos_logit = logit[cat[:,0]] # (3196303,)
        neg_logit = logit[cat[:,1]] # (3196303,)
        left = torch.cat((pos_logit,neg_logit),dim=0)
        right = torch.cat((neg_logit,pos_logit),dim=0)
        cmp_true = torch.cat((
            torch.ones(len(pos_logit),1),
            torch.zeros(len(pos_logit),1)
        ),dim=0).to(model.device)
        pairwise_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(left-right),cmp_true, reduction='mean')
        pointwise_loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true, reduction='mean',weight=weight)
        loss = model.alpha* pairwise_loss+ (1-model.alpha)*pointwise_loss
        return loss, pointwise_loss, pairwise_loss

    def orthogonal_unit_vector(vec):
        vector = vec.reshape(-1,)
        unit_vector = vector / torch.norm(vector,p=2)
        random_vector = torch.randn_like(vector)
        dot_product = torch.dot(unit_vector, random_vector)
        orthogonal_vector = random_vector - dot_product * unit_vector
        orthogonal_unit_vector = orthogonal_vector / torch.norm(orthogonal_vector,p=2)
        return orthogonal_unit_vector.reshape(1,-1)

    alpha = np.linspace(xmin,xmax, xnum)
    beta = np.linspace(ymin,ymax, ynum)
    begin_vec = cat_weight  #（1，891），最后一位是bias，前面890位是weight
    delta_vec = cat_weight/l2_norm
    niu_vec = orthogonal_unit_vector(delta_vec)
    logging.info(torch.dot(niu_vec.reshape(-1,), delta_vec.reshape(-1,)))
    
    def calculate_added_vector(wa,va,wb,vb):
        return begin_vec + wa * va + wb * vb

    if test_gen:
        model.eval()  # set to evaluation mode
        
        alphas = []
        betas= []
        combined_pair_losses = []
        pointwise_losses = []
        pairwise_losses = []
        
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if model._verbose > 0:
                data_generator = tqdm(test_gen, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                # only consider the first batch
                for wb in tqdm(beta):
                    for wa in alpha: # moving x-axis
                        vec = calculate_added_vector(wa, delta_vec, wb, niu_vec)
                        
                        weight = vec[0,:-1]
                        bias = vec[0,-1]
                        
                        model.fc.weight.data = weight.reshape(fc_weight_shape).to(model.device)
                        model.fc.bias.data = bias.reshape(fc_bias_shape).to(model.device)
                        return_dict = model.forward(batch_data)
                        y_true = model.get_labels(batch_data)
                        y_pred = return_dict["y_pred"]
                        logit = return_dict["logit"]
                        combined_pair, pointwise_loss, pairwise_loss = compute_loss(y_pred,logit,y_true)
                        
                        alphas.append(wa)
                        betas.append(wb)
                        
                        combined_pair_losses.append(float(combined_pair))
                break
        alpha_grid, beta_grid = np.meshgrid(alpha, beta)
        np.save('./analysis_tools/loss_landscape/combined_pair_loss.npy',np.array(combined_pair_losses).reshape(alpha_grid.shape))