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
from my_utils.defense import *

import torch.multiprocessing as mp
from torch.multiprocessing import Process
import copy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from server.fedavg_server import ServerAVG
import os

from server.fedopt_server import ServerOPT


class ServerKrumOPT(ServerOPT):
    def __init__(self, dataset, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num, client_ratio, folder, args, eta,  algo='krumopt'):
        super().__init__(dataset, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num, client_ratio, folder, args, eta,  algo)
        
    def server_update(self, w_locals, glob):
        if self.args.com_ratio>0:
            c = max(int(self.args.com_ratio * len(w_locals)), 1)
        else:
            c = 0
        n = len(w_locals) - c
        distance = pairwise_distance(w_locals, self.args.device)
        sorted_idx = distance.sum(dim=0).argsort()[: n]
        chosen_idx = int(sorted_idx[0])
        new_dict = copy.deepcopy(w_locals[chosen_idx])
        
        ret_dict = {}
        for key in new_dict:
            ret_dict[key] = 0
            ret_dict[key] = glob[key].to(self.args.device) + self._eta*new_dict[key].to(self.args.device)
        return ret_dict