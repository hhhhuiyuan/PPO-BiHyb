import argparse
import torch
from torch import nn
import os
import time
import yaml
import subprocess
import sys
import random
import numpy as np
from torch.multiprocessing import Pool, cpu_count
from copy import deepcopy
import pickle


#from src.dag_ppo_bihyb_model import ActorNet, CriticNet, GraphEncoder
from utils.utils import print_args
from utils.tfboard_helper import TensorboardUtil
from utils.dag_graph import DAGraph
from dag_data.dag_generator import load_tpch_tuples, generate_diffusion_tpch, load_job
from dag_data.dag_addedge import new_graphs_addedge
from dag_ppo_bihyb_eval import evaluate


def parse_arguments():
    parser = argparse.ArgumentParser(description='DAG scheduler. You have two ways of setting the parameters: \n'
                                                 '1) set parameters by command line arguments \n'
                                                 '2) specify --config path/to/config.yaml')
    
    parser.add_argument('--config', default=None, type=str, help='path to config file,'
                        ' and command line arguments will be overwritten by the config file')
   
    # misc configs
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--scheduler_type', default='sft')
    parser.add_argument('--save_dir', default=None, type=str) 
    
    # dag generation
    parser.add_argument('--combo34', action='store_true')
    parser.add_argument('--combo52', action='store_true')
    parser.add_argument('--load_graph', default=100, type=int, help='number of graphs to load')
    parser.add_argument('--num_init_dags', default=5, type=int)
    parser.add_argument('--resource_limit', default=600, type=float)
    parser.add_argument('--resource_scale', default=0.0, type=float)
    parser.add_argument('--add_graph_features', action='store_true')
    
    # add edge config
    parser.add_argument('--add_edge', action='store_true')
    parser.add_argument('--addedge_per_graph', default=128, type=int, help='number of trials to add edges per graph')
    parser.add_argument('--max_add', default=5, type=int)
    parser.add_argument('--num_workers', default=1, type=int, help='number of processes to generate data')  
   
    args = parser.parse_args()

    if args.config:
        with open('config/' + args.config) as f:
            cfg_dict = yaml.load(f)
            for key, val in cfg_dict.items():
                assert hasattr(args, key), f'Unknown config key: {key}'
                setattr(args, key, val)
            f.seek(0)
            for line in f.readlines():
                print(line.rstrip())

    return args

def main(args):
    # initialize manual seed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        os.environ['PYTHONHASHSEED'] = str(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # create DAG graph environment
    resource_dim = 1
    raw_node_feature_dim = 1 + resource_dim  # (duration, resources)
    args.node_feature_dim = raw_node_feature_dim
    evaluator = DAGraph(resource_dim=resource_dim, feature_dim=raw_node_feature_dim)

    # load training/testing data
    vargs = (
        evaluator,
        args.num_init_dags,
        raw_node_feature_dim,
        resource_dim,
        args.resource_limit,
        args.resource_scale,
        args.add_graph_features,
        args.scheduler_type,
    )
    
    #use tid to replicate prevuious generated graphs if exist, generate new ones if tid == None
    #node feature:(duration, resource), total resource limit = 1
    #generate with bs=8
    if args.combo34:
        loaded_graphs = generate_diffusion_tpch(args.load_graph, "combo34", *vargs)
    elif args.combo52:
        loaded_graphs = generate_diffusion_tpch(args.load_graph, "combo52", *vargs)
    else:
        loaded_graphs = generate_diffusion_tpch(args.load_graph, False, *vargs)
    
    if args.split != 'train' or not args.add_edge:
        save_path = os.path.join(args.save_dir, f'{args.split}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(loaded_graphs, f)
    else:
        add_edge_vargs = (args.addedge_per_graph, args.max_add, args.save_dir, args.num_workers, args.random_seed)
        new_graphs_addedge(loaded_graphs, evaluator, *add_edge_vargs)


if __name__ == '__main__':
    main(parse_arguments())
