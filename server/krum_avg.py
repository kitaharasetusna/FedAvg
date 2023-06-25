import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import sys
sys.path.append('../client')
from client.client_opt import ClientOPT
from my_utils.utils import *
from my_utils.dataloader import DatasetSplit

import torch.multiprocessing as mp
from torch.multiprocessing import Process
import copy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from server.fedavg_server import ServerAVG
import os

class ServerKrum(ServerAVG):
    def __init__(self, dataset, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num, client_ratio, folder, args, algo='krum'):
        super().__init__(dataset=dataset, network=network, train_data=train_data,  \
            num_clients=num_clients, E=E, client_batch_size=client_batch_size, \
        learning_rate=learning_rate, device=device, shards_num=shards_num, \
            client_ratio=client_ratio, folder=folder, args=args, algo=algo)

    def server_update(self, w_locals):
        if self.args.com_ratio>0:
            c = max(int(self.args.com_ratio * len(w_locals)), 1)
        else:
            c = 0
        n = len(w_locals) - c
        distance = pairwise_distance(w_locals, self.args.device)
        sorted_idx = distance.sum(dim=0).argsort()[: n]
        chosen_idx = int(sorted_idx[0])
        
        return copy.deepcopy(w_locals[chosen_idx])
        

def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)
    
    
def pairwise_distance(w_locals, device):
    vectors = multi_vectorization(w_locals, device)
    distance = torch.zeros([len(vectors), len(vectors)]).to(device)
        
    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)
                    
    return distance    
    
def multi_vectorization(w_locals, device):
    vectors = copy.deepcopy(w_locals)
        
    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1]).to(device)
        vectors[i] = torch.cat(list(v.values()))

    return vectors