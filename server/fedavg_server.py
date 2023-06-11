import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import sys
sys.path.append('../client')
from client.client_avg import ClientAVG
from my_utils.utils import *
from my_utils.dataloader import DatasetSplit

import torch.multiprocessing as mp
from torch.multiprocessing import Process
import copy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import random

import matplotlib.pyplot as plt
import numpy as np


class ServerAVG():
    def __init__(self, dataset, network, train_data, num_clients, E, client_batch_size, learning_rate, device, \
        shards_num, client_ratio):
        #TODO: selcet data
        # intermediate parameters
        self._num_clients = num_clients
        

        # parameters for Clients
        if dataset=='shakespeare':
            dict_users = train_data.get_client_dic()
            num_users = len(dict_users)
            self._client_datasets = [DatasetSplit(train_data, dict_users[idx]) for idx in range(num_users)]
            self._clients_models = [function_map[network]().to(device) for i in range(num_clients)]
            # print(f'{self._client_datasets[0][0][0].shape} {self._client_datasets[0][0][1]} {self._client_datasets[0][0][0].shape}')
            # print(self._client_datasets[1])
            # import sys; sys.exit() 
        elif dataset=='MNIST':
            self._subset_indices = torch.linspace(0, len(train_data)-1,  \
            steps=shards_num+1).round().tolist() 
            self._client_datasets = [Subset(train_data, range(int(self._subset_indices[i]),  \
            int(self._subset_indices[i+2]))) for i in range(num_clients)]
            self._clients_models = [function_map[network](input_size=28*28,  \
                hidden_size=200, output_size=10).to(device) for i in range(num_clients)]
        else:
            raise ValueError("The dataset you provides is no in 'MNIST', 'shakepeare' or things like that.") 
        
        # parameters for Clients
        self._client_loaders = [DataLoader(self._client_datasets[i],  
                                           batch_size=client_batch_size, 
            shuffle=True) for i in range(num_clients)]
        
        self._clients_optims = [optim.Adam(self._clients_models[i].parameters(), \
                                           lr=learning_rate) for i in range(num_clients)] 
        
        self._clients = [ClientAVG(self._client_loaders[i], \
            self._clients_models[i], self._clients_optims[i],  \
                device, E, client_batch_size) for i in range(num_clients)]
        
        # parameters for the Server
        self._lr = learning_rate
        self._E = E

        if dataset=='MNIST':
            self._global_model = function_map[network](input_size=28*28, hidden_size=200, output_size=10).to(device)
        elif dataset=="shakespeare":
            self._global_model = function_map[network]().to(device)

        self._client_ratio = client_ratio
        
        self._global_model.train()
    
    def gen_client_id(self):
        print (f'We are selecting {int(self._num_clients*self._client_ratio)} for fedavg...')
        random_numbers = random.sample(range(self._num_clients), int(self._num_clients*self._client_ratio))
        return random_numbers
   
    def plot_acc(self, rounds, acc, T):
        print(f'debuging... {rounds} {acc}')
        plt.scatter(rounds, acc, color='blue', label='accuracy')
        # Fit a line to the data points
        coefficients = np.polyfit(rounds, acc, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        # Create a line using the slope and intercept
        line_x = np.linspace(0, 1, 100)
        line_y = slope * line_x + intercept
        plt.xlabel('round')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
            
    def update_server_thread_res(self, T):
        '''
        FedAVG
        '''
        client_acc = []
        glob_acc = []
        rounds = np.arange(0, T, 20)
        print(T/20)
        # 2: for t=0, ..., T-1 do
        for round in range(T):
            print(f"Round {round+1} started..., picking {self._client_ratio} of all clients")
            client_models = []
            client_losses = []
            client_accs = []
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
            if (round+1)%20==0 and round!=0:
                glob_acc.append(sum(client_accs)/len(client_accs))
        self.plot_acc(rounds, glob_acc, T)
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



