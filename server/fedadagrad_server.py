import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models.my_NN import TwoLayerNet
import sys
sys.path.append('../client')
from client.client_opt import ClientOPT
from my_utils.utils import *

import torch.multiprocessing as mp
from torch.multiprocessing import Process
import copy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from server.fedopt_server import ServerOPT
import math

class ServerFedAdaGrade(ServerOPT):
    # client learning rate: eta_l
    # eta: server learning rate
    def __init__(self, dataset, network, train_data, num_clients, E, client_batch_size, learning_rate, device, shards_num, client_ratio, folder, initial_mom={}, \
        beta_1 = 0.9, eta= -1, tau=1e-3, algo='fedada'):
        print(client_ratio)
        super().__init__(dataset, network, train_data, num_clients, E, client_batch_size, learning_rate, device, shards_num, client_ratio, folder, algo=algo, eta=eta)
        # m_0
        self._cur_mom = {}
        self._veloc = {}
        self._beta_1 = beta_1
        for key in self._global_model.state_dict().keys():
            self._cur_mom[key] =0
            self._veloc[key] = 0 
        self._eta = eta 
        self._tau = tau
    
    def server_update(self, deltas, glob):
        # delta_t = 1/S sum_{}
        new_dict = {}
        for key in deltas[0].keys():
            new_dict[key] = 0
            for delta in deltas:
                new_dict[key] += delta[key]
            new_dict[key] /= len(deltas)
        
                
             
        # m_t = m_{t-1}*beta_1+(1-beta_1)*delta_t
        for key in new_dict.keys():
            self._cur_mom[key] = self._cur_mom[key]*self._beta_1 + (1-self._beta_1)*new_dict[key]

        # v_t = v_{t-1} + delta_t^2
        for key in new_dict.keys():
            self._veloc[key] = self._veloc[key] + new_dict[key]**2

        ret_dict = {}
        # x_t = x_{t-1} + eta*m_t/(sqrt(v_t)+tau)
        for key in  new_dict.keys():
            ret_dict[key] =glob[key].to('cuda') + self._eta*self._cur_mom[key].to('cuda')/(torch.sqrt(self._veloc[key].to('cuda'))+self._tau)
        
        return ret_dict

        
        