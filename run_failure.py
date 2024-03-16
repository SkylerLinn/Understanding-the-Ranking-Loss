import pandas as pd 
import argparse
import os 
import time
import hashlib
import yaml 
import itertools
import subprocess
from fuxictr import autotuner 
from datetime import datetime
from fuxictr.utils import print_to_json, load_model_config, load_dataset_config

# add this line to avoid weird characters in yaml files
yaml.Dumper.ignore_aliases = lambda *args : True
def grid_search(config_dir, gpu_list, exclude_expid=None, expid_tag=None, script='run_expid.py'):
    experiment_id_list = autotuner.load_experiment_ids(config_dir)
    new_experiment_id_list = [] 
    for exp_id in experiment_id_list:
        if exp_id in exclude_expid:
            continue
        new_experiment_id_list.append(exp_id)
    experiment_id_list = new_experiment_id_list
    print(len(experiment_id_list))
    if expid_tag is not None:
        experiment_id_list = [expid for expid in experiment_id_list if str(expid_tag) in expid]
        assert len(experiment_id_list) > 0, "tag={} does not match any expid."
    gpu_list = list(gpu_list)
    idle_queue = list(range(len(gpu_list)))
    processes = dict()
    while len(experiment_id_list) > 0:
        if len(idle_queue) > 0:
            idle_idx = idle_queue.pop(0)
            gpu_id = gpu_list[idle_idx]
            expid = experiment_id_list.pop(0)
            cmd = "python -u {} --config {} --expid {} --gpu {}"\
                    .format(script, config_dir, expid, gpu_id)
            p = subprocess.Popen(cmd.split())
            processes[idle_idx] = p
        else:
            time.sleep(3)
            for idle_idx, p in processes.items():
                if p.poll() is not None: # terminated
                    idle_queue.append(idle_idx)
    [p.wait() for p in processes.values()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/tuner_config.yaml', 
                        help='The config file for para tuning.')
    parser.add_argument('--tag', type=str, default=None, help='Use the tag to determine which expid to run (e.g. 001 for the first expid).')
    parser.add_argument('--gpu', nargs='+', default=[-1], help='The list of gpu indexes, -1 for cpu.')
    parser.add_argument('--exclude', type=str, default='', 
                        help='The experiment_result.csv file to exclude finished expid.')
    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    expid_tag = args['tag']
    exclude_expid = []
    if args['exclude'] != '' and os.path.exists(args['exclude']):
        result_df = pd.read_csv(args['exclude'], header=None)
        expid_df = result_df.iloc[:, 2].map(lambda x: x.replace('[exp_id] ', ''))
        exclude_expid = expid_df.tolist()
    # print(exclude_expid)

    # generate parameter space combinations
    config_dir = autotuner.enumerate_params(args['config'])
    grid_search(config_dir, gpu_list, exclude_expid, expid_tag)
