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

from server.fedavg_server import ServerAVG



class ServerOPT(ServerAVG):
    def __init__(self, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num, client_ratio):
        
        #TODO: selcet data
        # intermediate parameters
        self._num_clients = num_clients
        self._subset_indices = torch.linspace(0, len(train_data)-1,  \
            steps=shards_num+1).round().tolist()
        self._client_datasets = [Subset(train_data, range(int(self._subset_indices[i]),  \
            int(self._subset_indices[i+2]))) for i in range(num_clients)]

        
        # parameters for Clients
        self._client_loaders = [DataLoader(self._client_datasets[i],  
                                           batch_size=client_batch_size, 
            shuffle=True) for i in range(num_clients)]
        self._clients_models = [function_map[network](input_size=28*28,  \
            hidden_size=200, output_size=10).to(device) for i in range(num_clients)]
        self._clients_optims = [optim.Adam(self._clients_models[i].parameters(), \
                                           lr=learning_rate) for i in range(num_clients)]
        
        self._clients = [ClientOPT(self._client_loaders[i], \
            self._clients_models[i], self._clients_optims[i],  \
                device, E, client_batch_size) for i in range(num_clients)]
        
        # parameters for the Server
        self._lr = learning_rate
        self._E = E
        self._global_model = function_map[network](input_size=28*28, hidden_size=200, output_size=10).to(device)
        self._client_ratio = client_ratio
        
        self._global_model.train() 
    def server_update(self, deltas, glob):
        new_dict = {}
        for key in deltas[0].keys():
            new_dict[key] = 0
            for delta in deltas:
                new_dict[key] += delta[key]
            new_dict[key] /= len(deltas)
        
        ret_dict = {}
        for key in new_dict:
            ret_dict[key] = 0
            ret_dict[key] = glob[key]+new_dict[key]
        return ret_dict
    
    def update_server_thread_res(self, T):
        '''
        FedOPT
        '''
        client_acc = []
        # 2: for t=0, ..., T-1 do
        for round in range(T):
            print(f"Round {round+1} started...")
            client_losses = []
            client_accs = []
            delta_t = []
            x_t = self._global_model.state_dict()
            # -multi-threading is here
            executor = ThreadPoolExecutor(max_workers=int(self._num_clients*self._client_ratio))
            processes = []

            random_ids = self.gen_client_id()
            for i in random_ids:
                x_t_temp = copy.deepcopy(x_t)
                processes.append(executor.submit(self._clients[i].client_update, round+1, i, x_t_temp))
            
            results = concurrent.futures.as_completed(processes)
            for res in results:
                delta_i_t, client_loss, client_acc = res.result()
                delta_t.append(delta_i_t)
                client_losses.append(client_loss)
                client_accs.append(client_acc)
            # --end
            
            global_state_dict = self.server_update(delta_t, x_t)
            self._global_model.load_state_dict(global_state_dict)
            print(f"Round {round+1} finished, global loss:  \
                {sum(client_losses)/len(client_losses):.4f},  \
                    global accuracy: {sum(client_accs)/len(client_accs): .4f}")
        return sum(client_accs)/len(client_accs)  


