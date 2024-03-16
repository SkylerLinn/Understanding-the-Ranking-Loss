# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from tqdm import tqdm
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2, CrossNetMix
import logging
from fuxictr.metrics import evaluate_metrics
from sklearn.metrics import roc_auc_score
import sys
import os
import numpy as np
class DCNv2JRCGradFirstEpoch(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNv2", 
                 gpu=-1,
                 model_structure="parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None, 
                 jrc_weight=0.1,
                 pos_weight=1.,
                 **kwargs):
        super(DCNv2JRCGradFirstEpoch, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.jrc_weight=jrc_weight
        self.pos_weight = pos_weight
        self.model_root = kwargs['model_root']
        self.model_id = model_id
        input_dim = feature_map.sum_emb_out_dim()
        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(input_dim, num_cross_layers, low_rank=low_rank, num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"], \
               "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(input_dim=input_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=stacked_dnn_hidden_units,
                                         hidden_activations=dnn_activations,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            final_dim = stacked_dnn_hidden_units[-1]
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure == "crossnet_only": # only CrossNet
            final_dim = input_dim
        
        self.fc = nn.Linear(final_dim, 2)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        cross_out = self.crossnet(feature_emb)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(feature_emb)], dim=-1)
        logits = self.fc(final_out)

        logit_non_click = logits[:,0]
        logit_click = logits[:,1]
        diff_logit = logit_click-logit_non_click
        
        y_pred = self.output_activation(diff_logit)
        if logits.requires_grad:
            logit_non_click.retain_grad()
            logit_click.retain_grad()
            # diff_logit.retain_grad()
        return_dict = {
            "y_pred": y_pred,
            "logit_non_click": logit_non_click,
            "logit_click": logit_click,
            # "logit": diff_logit
        }
        return return_dict
    
    def compute_loss(self, return_dict, y_true):
        beta = self.pos_weight
        z1 = return_dict["logit_click"]
        z0 = return_dict["logit_non_click"]

        y = y_true.reshape(-1,)
        index_pos = torch.nonzero(y_true.reshape(-1,)==1).reshape(-1,) # (1049,)
        index_neg = torch.nonzero(y_true.reshape(-1,)==0).reshape(-1,)   
        exp_z1 = torch.exp(z1)
        exp_z0 = torch.exp(z0)
        
        numer = y*exp_z1+(1-y)*exp_z0
        denom_calib = exp_z0 + exp_z1
        
        denom_rank_pos = torch.sum(beta * y * exp_z1+(1-y)*exp_z1)
        denom_rank_neg = torch.sum(beta * y * exp_z0+(1-y)*exp_z0)
        denorm_rank = torch.where(y==1, denom_rank_pos, denom_rank_neg).reshape(-1,)

        # 为了可比，这里的Sum全部改成了Mean
        beta_vec = torch.where(y_true.reshape(-1,)==1, beta, 1).reshape(-1,)
        l_calib = -torch.mean(beta_vec * (torch.log(numer/denom_calib).reshape(-1,)))
        l_rank = -torch.mean(beta_vec* (torch.log(numer/denorm_rank).reshape(-1,)))
        grad_fac = (1-numer/denorm_rank)[index_neg]
        logging.info("Grad Fac: {}, {}, {}".format(
            torch.quantile(grad_fac, 0.1),
            grad_fac.mean(),
            torch.quantile(grad_fac, 0.9))
        )
        loss = self.jrc_weight * l_calib + (1-self.jrc_weight) * l_rank
        loss += self.regularization_loss()
       
        return loss, index_pos, index_neg, z0,z1 #, return_dict['logit']
    
    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        print(metrics)
        if 'wAUC' in metrics:
            sample_weight=np.where(y_true==1., self.pos_weight, 1.)
            print(sample_weight.mean())
            print(y_true.shape)
            print(y_pred.shape)
            ret_dict={'wAUC':roc_auc_score(y_true, y_pred, sample_weight=sample_weight, average='samples')}
        else:
            ret_dict = dict()
        tmp = [_ for _ in metrics if _ !='wAUC']
        ret_dict.update(evaluate_metrics(y_true, y_pred, tmp, group_id))
        return ret_dict
    
    def train_step(self, batch_data):
        def mean_abs(x):
            return torch.mean(torch.abs(x))
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss,index_pos, index_neg, non_click_logit, click_logit= self.compute_loss(return_dict, y_true) #logit
        loss.backward()
        non_click_logit_dim1 = non_click_logit.grad.reshape(-1,)
        click_logit_dim1 = click_logit.grad.reshape(-1,)
        logging.info("Pos Click Grad Mean: {}, Cnt:{}".format(mean_abs(click_logit_dim1[index_pos]), len(index_pos)))
        logging.info("Pos Non-Click Grad Mean: {}, Cnt:{}".format(mean_abs(non_click_logit_dim1[index_pos]), len(index_pos)))
        logging.info("Neg Click Grad Mean: {}, Cnt:{}".format(mean_abs(click_logit_dim1[index_neg]), len(index_neg)))
        logging.info("Neg Non-Click Grad Mean: {}, Cnt:{}".format(mean_abs(non_click_logit_dim1[index_neg]), len(index_neg)))

        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss