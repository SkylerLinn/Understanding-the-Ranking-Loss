# Understanding the Ranking Loss for Recommendation with Sparse User Feedback
This repository primarily focuses on disclosing experimental and analytical code pertaining to understanding ranking loss. The repository is currently under construction.
## Main Idea
Recent advancements suggest combining BCE loss with ranking loss has shown substantial performance improvement in many industrial deployments.
However, the efficacy of this combination loss is not fully understood.
We identify a novel challenge in CTR prediction with BCE loss: gradient vanishing in negative samples. 
We then propose a novel perspective on the effectiveness of ranking loss in CTR prediction, that it **leads to larger gradients on negative samples and hence mitigates their optimization issue, resulting in better classification ability**.

We conducted a comparative analysis of both ranking and classification performance, gradient examination, and loss landscape observation using the code from this repository.

# Getting Start
## Data Preparation

The Criteo dataset serves as a prevalent benchmark dataset in the field of click-through rate (CTR) prediction, encompassing approximately one week's worth of click-through data pertinent to display advertising.
Please download the [criteo_x1 data](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1) into `/data/criteo` folder. Kindly be informed that the usage of this dataset is restricted to academic research purposes exclusively and it must not be utilized for any commercial or illegal activities. 

If you want to reproduce the negative sampling experiment, please use the `/data/criteo/make_negative_sampling.ipynb` to produce the negative sampled data. If you have no need, just skip it.

## Configuration
We put our config files on `config/criteo` path. The basic settings of dataset and model are `dataset_config.yaml` and `model_config.yaml`, respectively.

## Reproduce the results
We use the DCN v2 as backbone and adjust the sparsity of positive feedback for experiments.
- BCE method (on sparse positive feedback scenario)
The realization file of BCE method is named as `DCNv2PosWeight.py`, you can run the following command to rerun the experiment using gpu (index 0):
```python3 run_expid.py --expid DCNv2_criteo_x1_posw0.1 --config ./config/criteo_x1 --gpu 0```
  - `expid`: the experiment id in `model_config.yaml`
  - `config`: the path containing `model_config.yaml`
  - `--gpu`: index of cuda

 For producing rebust results, we run the each experiment 5 times and culculate the average results. Thus, you can run the following command to reproduce the results with 5 gpus:
 ```python3 run_param_tuner.py  --config ./config/criteo_x1/DCNv2_criteo_x1_posw_tuner.yaml --gpu 0 1 2 3 4```
The contents of this yaml entail an exploration across six distinct levels of sparsity, with each experiment being repeated five times.

 - Combined-Pair
 ```python3 run_param_tuner.py  --config ./config/criteo_x1/DCNv2_criteo_x1_poswall_sample_tuner.yaml --gpu 0 1 2 3 4```

- JRC
 ```python3 run_param_tuner.py  --config ./config/criteo_x1/DCNv2_criteo_x1_posw0.1_jrc_alpha0.4_rerun_tuner.yaml --gpu 0 1 2 3 4```

- RCR (ListCE)
 ```python3 run_param_tuner.py  --config ./config/criteo_x1/DCNv2_criteo_x1_posw0.1_listce_alpha0.9.yaml --gpu 0 1 2 3 4```

- Combined-List
 ```python3 run_param_tuner.py  --config ./config/criteo_x1/DCNv2_criteo_x1_posw0.1_combined_list_alpha0.9.yaml --gpu 0 1 2 3 4```

- Focal Loss
 ```python3 run_param_tuner.py  --config ./config/criteo_x1/DCNv2_criteo_x1_posw0.1_focal_gamma_tuner1.yaml --gpu 0 1 2 3 4```

## Analysing
The project conducted a comprehensive experimental analysis, with the corresponding analysis code housed within the `analysis_pool`.  

## Citation
If you find our code or propcessed data helpful in your research, please kindly cite the following papers.
> Coming soon
Our code is based on the FuxiCTR and BARS.
> Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). The 30th ACM International Conference on Information and Knowledge Management (CIKM), 2021.
> Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, Rui Zhang. [BARS: Towards Open Benchmarking for Recommender Systems](https://arxiv.org/abs/2009.05794). The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2022.
 
