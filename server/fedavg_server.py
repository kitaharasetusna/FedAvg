import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor

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
import pickle
import os

from torchvision.datasets import MNIST
from my_utils.dataset import ShakeSpeare


class ServerAVG():
    def __init__(self, dataset, network, train_data, num_clients, E, \
        client_batch_size, learning_rate, device, \
        shards_num, client_ratio, folder, args, algo='fedavg'):

        self.args = args
        self.algo = algo
        # TODO: clean the code and adjust heritance in fedavg
        #TODO: keep eye on MNIST model, it can be updated from scalar to variable 
        # intermediate parameters
        self._num_clients = num_clients
       
        # attack: selecting compromised clients
        if args.com_ratio > 0:
            self.compromised_idxs = compromised_clients(args)
        else:
            self.compromised_idxs = [] 

        
        # parameters for Clients
        self.dataset = dataset
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
        
        self._clients_optims = [optim.Adam(self._clients_models[i].parameters(), \
                                           lr=learning_rate) for i in range(num_clients)]  
        # parameters for Clients
        if client_batch_size=='inf':
            self._client_loaders = [DataLoader(self._client_datasets[i],  
                                           batch_size=len(self._client_datasets[0]), 
            shuffle=True) for i in range(num_clients)]
            self._clients = [client_map[algo](self._client_loaders[i], \
                self._clients_models[i], self._clients_optims[i],  \
                    device, E, len(self._client_datasets[0]), args) if i not in self.compromised_idxs else
                client_map[algo+'C'](self._client_loaders[i], \
                self._clients_models[i], self._clients_optims[i],  \
                    device, E, len(self._client_datasets[0]), args)
                             for i in range(num_clients)]
            self._B = len(self._client_datasets[0]) 
        else:
            batch_size = int(client_batch_size)
            self._client_loaders = [DataLoader(self._client_datasets[i],  
                                            batch_size=batch_size, 
                shuffle=True) for i in range(num_clients)]
            self._clients = [client_map[algo](self._client_loaders[i], \
                self._clients_models[i], self._clients_optims[i],  \
                    device, E, batch_size, args) if i not in self.compromised_idxs else
                            client_map[algo+'C'](self._client_loaders[i], \
                self._clients_models[i], self._clients_optims[i],  \
                    device, E, batch_size, args) for i in range(num_clients)]
            self._B = batch_size
        
        
        # parameters for the Server
        self._lr = learning_rate
        self._E = E

        if dataset=='MNIST':
            self._global_model = function_map[network](input_size=28*28, hidden_size=200, output_size=10).to(device)
        elif dataset=="shakespeare":
            self._global_model = function_map[network]().to(device)

        self._client_ratio = client_ratio
       
        self.folder = folder
        self.pkl_path = f'{dataset}_{network}_{algo}_num_Client_{num_clients}_T_{args.num_round}_eta_l_{learning_rate}'
        if args.attack_type:
            self.pkl_path = f'{dataset}_{args.attack_type}_{network}_{algo}_num_Client_{num_clients}_T_{args.num_round}_eta_l_{learning_rate}_compromised_ratio_{args.com_ratio}'
       
        print(self.folder)
        print(self.pkl_path+str(10)+'.png')

        self._global_model.train()
        
    '''
    TOOL FUNC
    help splitting dataset for federated learning
    TODO:
    ''' 
    def gen_client_id(self):
        print (f'We are selecting {int(self._num_clients*self._client_ratio)} for {self.algo}...')
        random_numbers = random.sample(range(self._num_clients), int(self._num_clients*self._client_ratio))
        return random_numbers
    '''
    TOOL FUNC
    draws pictures given 
    '''
    def plot_acc(self, rounds, acc, T):
        print(f'debuging... {len(rounds)} {len(acc)}')
        plt.figure()
        plt.plot(rounds, acc, color='blue', label='accuracy')
        
        plt.xlabel('round')
        plt.ylabel('accuracy')
        plt.legend()
        plt.xlim(0, 100)
        plt.savefig(self.folder+self.pkl_path+'/'+f'E_{self._E} B_{self._B}'+'.png')
    ''' 
    TOOL FUNC
    just averaging the all static models
    '''  
    def server_update(self, models):
        new_state_dict = {}
        for key in models[0].keys():
            new_state_dict[key] = torch.stack([models[i][key] for i in range(len(models))], dim=0).mean(dim=0)
        return new_state_dict

    def aggregate(self, glob_acc, T, test_loader):
        # 2: for t=0, ..., T-1 do
        early_stop_epoch = None
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
            fin_acc = self.test_acc(test_loader=test_loader)
            # lenth: T
            glob_acc.append(fin_acc)
            
            
            print(f"Round {round+1} finished, global loss:  \
                {sum(client_losses)/len(client_losses):.4f},  \
                    global accuracy: {sum(client_accs)/len(client_accs): .4f}, test_acc: {fin_acc: .4f}")
            
            if fin_acc>0.98:
                early_stop_epoch = round+1
                print('Congrates, Eearly stop, have reached 0.99 acc!')
                break
        
        pkl_path = f'{self.folder}{self.pkl_path}'
        os.makedirs(pkl_path, exist_ok=True)
        print(f'saving global model in {pkl_path}...')
        torch.save(self._global_model.state_dict(), f'{self.folder}{self.pkl_path}/{self._E}_{self._B}_model.pth')
        return glob_acc, client_accs, early_stop_epoch
    '''
    real aggregation that's called
    '''        
    def update_server_thread_res(self, T):
        '''
        FedAVG
        '''
        print('debuging... we are using fedavg')
        client_acc = []
        glob_acc = []
        # length: T
        rounds = np.arange(1, T+1, 1)
        # TODO: add other dataset
        if self.dataset == 'MNIST':
            test_loader = MNIST(root='./data', train=True, download=True, transform=ToTensor())
            test_loader = DataLoader(test_loader, batch_size=self._B)
        elif self.dataset == 'shakespeare':
            test_loader = ShakeSpeare(train=False)
            test_loader = DataLoader(test_loader, batch_size=self._B)
        
        glob_acc, client_accs, early_stop_epoch = self.aggregate(glob_acc=glob_acc, T=T, test_loader=test_loader) 

        print(f'rounds: {rounds}; test_acc: {glob_acc} T:{T}')
        pkl_path = f'{self.folder}{self.pkl_path}'
        os.makedirs(pkl_path, exist_ok=True)
        print(f'saving in {pkl_path}')
        with open(f'{self.folder}{self.pkl_path}/{self._E}_{self._B}.pkl','wb') as f:
            pickle.dump((rounds, glob_acc, T), f) 
        f.close()
        if early_stop_epoch:
            rounds = np.arange(1, early_stop_epoch+1, 1)  
        self.plot_acc(rounds, glob_acc, T)
        return sum(client_accs)/len(client_accs), glob_acc 

    '''
    TODO: finish the test past
    '''
    def test_acc(self, test_loader):
        acc_num = 0
        total_num = 0
        self._global_model.eval() 
        self._global_model.to('cpu')
        with torch.no_grad():
          for inputs, labels in test_loader:
                test_output = self._global_model(inputs)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                num_equal = (pred_y == labels).sum().item()
                acc_num += num_equal
                total_num += labels.size()[0]
        return acc_num/total_num



def compromised_clients(args):
    max_num = max(int(args.com_ratio * args.num_client), 1)
    tmp_idx = [i for i in range(args.num_client)]
    compromised_idxs = random.sample(tmp_idx, max_num)

    return compromised_idxs

