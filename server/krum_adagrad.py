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

from server.fedadagrad_server import ServerFedAdaGrade
import os



class ServerKrumadagrade(ServerFedAdaGrade):
    def __init__(self, dataset, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num, client_ratio, folder, args, initial_mom={}, \
        beta_1 = 0.9, eta= -1, tau=1e-3, algo='krumada'):
        super().__init__(dataset, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num, client_ratio, folder, args, initial_mom={}, \
        beta_1 = 0.9, eta= -1, tau=1e-3, algo=algo)
        
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
        
                
             
        # m_t = m_{t-1}*beta_1+(1-beta_1)*delta_t
        for key in new_dict.keys():
            self._cur_mom[key] = self._cur_mom[key]*self._beta_1 + (1-self._beta_1)*new_dict[key]

        # v_t = v_{t-1} + delta_t^2
        for key in new_dict.keys():
            self._veloc[key] = self._veloc[key] + new_dict[key]**2

        ret_dict = {}
        device = self.args.device
        # x_t = x_{t-1} + eta*m_t/(sqrt(v_t)+tau)
        for key in  new_dict.keys():
            ret_dict[key] =glob[key].to(device) + self._eta*self._cur_mom[key].to(device)/(torch.sqrt(self._veloc[key].to(device))+self._tau)
        
        return ret_dict