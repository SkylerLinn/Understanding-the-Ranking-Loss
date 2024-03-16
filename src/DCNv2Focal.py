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
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2, CrossNetMix
from fuxictr.metrics import evaluate_metrics
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import logging
class DCNv2Focal(BaseModel):
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
                 pos_weight=1.,
                 gamma=0.1,
                 **kwargs):
        super(DCNv2Focal, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.pos_weight = pos_weight
        self.gamma = gamma
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
        self.fc = nn.Linear(final_dim, 1)
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
        logit = self.fc(final_out)
        if logit.requires_grad:
            logit.retain_grad()
        y_pred = self.output_activation(logit)
        return_dict = {"y_pred": y_pred,'logit':logit}
        return return_dict

    def compute_loss(self, return_dict, y_true):
        weight = torch.where(y_true==1., self.pos_weight, 1)
        bceloss = self.loss_fn(return_dict["y_pred"], y_true, reduction='none',weight=weight).reshape(-1)
        y_dim1 = y_true.reshape(-1)
        y_hat_dim1 = return_dict["y_pred"].reshape(-1)
        
        index_pos = torch.nonzero(y_dim1==1.)
        if len(index_pos) == 0:
            pos_loss = 0
        else:
            index_pos = index_pos.reshape(-1,)
            gamma_pos = torch.pow((1-y_hat_dim1[index_pos]), self.gamma)
            gamma_pos = torch.where(torch.isnan(gamma_pos), torch.full_like(gamma_pos,0), gamma_pos)
            pos_loss = torch.dot(gamma_pos,bceloss[index_pos])
        
        index_neg = torch.nonzero(y_dim1==0.)
        if len(index_neg) == 0:
            neg_loss = 0
        else:
            index_neg = index_neg.reshape(-1,)
            gamma_neg = torch.pow(y_hat_dim1[index_neg], self.gamma)
            gamma_neg = torch.where(torch.isnan(gamma_neg), torch.full_like(gamma_neg,0), gamma_neg)
            gamma_neg_shift = gamma_neg + (1-gamma_neg.mean())
            neg_loss = torch.dot(gamma_neg_shift,bceloss[index_neg])
        loss = (pos_loss+neg_loss)/len(bceloss)
        loss += self.regularization_loss()
        return loss,index_pos,index_neg,return_dict['logit']
    
    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        print(metrics)
        ret_dict = dict()
        if 'wAUC' in metrics:
            sample_weight=np.where(y_true==1., self.pos_weight, 1.)
            ret_dict.update({'wAUC':roc_auc_score(y_true, y_pred, sample_weight=sample_weight, average='samples')})
        if 'wlogloss' in metrics:
            sample_weight=np.where(y_true==1., self.pos_weight, 1.)
            ret_dict.update({'wlogloss':log_loss(y_true=y_true, y_pred=y_pred,sample_weight=sample_weight)})
        tmp = [_ for _ in metrics if _ not in ['wAUC','wlogloss']]
        ret_dict.update(evaluate_metrics(y_true, y_pred, tmp, group_id))
        return ret_dict

    def train_step(self, batch_data):
        def mean_abs(x):
            return torch.mean(torch.abs(x))
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss,index_pos, index_neg, logit= self.compute_loss(return_dict, y_true)
        loss.backward()
        logit_grad_dim1 = logit.grad.reshape(-1,)
        logging.info("Pos Grad Mean: {}, Cnt:{}".format(mean_abs(logit_grad_dim1[index_pos]), len(index_pos)))
        logging.info("Neg Grad Mean: {}, Cnt:{}".format(mean_abs(logit_grad_dim1[index_neg]), len(index_neg)))
        logging.info("Loss: {}".format(float(loss)))     
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss