import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models.my_NN import TwoLayerNet
import sys
sys.path.append('../client')
from client.client_avg import ClientAVG
from my_utils.utils import *

import torch.multiprocessing as mp
from torch.multiprocessing import Process
import copy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class ServerAVG():
    def __init__(self, network, train_data, num_clients, E, client_batch_size, learning_rate, device, \
        shards_num):
        
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
            hidden_size=32, output_size=10).to(device) for i in range(num_clients)]
        self._clients_optims = [optim.Adam(self._clients_models[i].parameters(), \
                                           lr=learning_rate) for i in range(num_clients)]
        
        self._clients = [ClientAVG(self._client_loaders[i], \
            self._clients_models[i], self._clients_optims[i],  \
                device, E, client_batch_size) for i in range(num_clients)]
        
        # parameters for the Server
        self._lr = learning_rate
        self._E = E
        self._global_model = function_map[network](input_size=28*28, hidden_size=32, output_size=10).to(device)
        
        self._global_model.train()
        
   
    def update_server_thread_res(self, T):
        '''
        FedAVG
        '''
        client_acc = []
        # 2: for t=0, ..., T-1 do
        for round in range(T):
            print(f"Round {round+1} started...")
            client_models = []
            client_losses = []
            client_accs = []
            x_t = self._global_model.state_dict()
            # -multi-threading is here
            executor = ThreadPoolExecutor(max_workers=self._num_clients)
            processes = []
            for i in range(self._num_clients):
                x_t_temp = copy.deepcopy(x_t)
                processes.append(executor.submit(self._clients[i].client_update, round+1, i, x_t_temp))
            
            results = concurrent.futures.as_completed(processes)
            for res in results:
                x_i_K_t, client_loss, client_acc = res.result()
                client_models.append(x_i_K_t)
                client_losses.append(client_loss)
                client_accs.append(client_acc)
            # --end
            
            global_state_dict = self.server_update(client_models)
            self._global_model.load_state_dict(global_state_dict)
            print(f"Round {round+1} finished, global loss:  \
                {sum(client_losses)/len(client_losses):.4f},  \
                    global accuracy: {sum(client_accs)/len(client_accs): .4f}")
        return sum(client_accs)/len(client_accs) 
    
    def server_update(self, models):
        new_state_dict = {}
        for key in models[0].keys():
            new_state_dict[key] = torch.stack([models[i][key] for i in range(len(models))], dim=0).mean(dim=0)
        return new_state_dict

    def test_acc(self, test_loader):
       self._global_model.eval() 
       with torch.no_grad():
           for x, y in test_loader:
               print(f' {self._global_model(x)} {y}')
               import sys
               sys.exit()



