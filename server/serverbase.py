import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models.my_NN import TwoLayerNet
import sys
sys.path.append('../client')
from client.clientbase import ClientBase
from my_utils.utils import *

import torch.multiprocessing as mp

class ServerBase():
    def __init__(self, network, train_data, num_clients, E, client_batch_size, learning_rate, device, \
        shards_num):
        #TODO: select models
        #TODO: selcet data
        self._num_clients = num_clients
        self._subset_indices = torch.linspace(0, len(train_data)-1, steps=shards_num+1).round().tolist()
        self._client_datasets = [Subset(train_data, range(int(self._subset_indices[i]), int(self._subset_indices[i+2]))) for i in range(num_clients)]
        self._E = E
        self._client_loaders = [DataLoader(self._client_datasets[i], batch_size=client_batch_size, shuffle=True) for i in range(num_clients)]
        self._clients_models = [ function_map[network](input_size=28*28, hidden_size=32, output_size=10).to(device) for i in range(num_clients)]
        # self._clients_optims = [optim.SGD(self._clients_models[i].parameters(), lr=learning_rate) for i in range(num_clients)]
        self._clients_optims = [optim.Adam(self._clients_models[i].parameters(), \
                                           lr=learning_rate) for i in range(num_clients)]
        
        self._clients = [ClientBase(self._client_loaders[i], self._clients_models[i], self._clients_optims[i], device, E, client_batch_size) for i in range(num_clients)]

        self._global_model = function_map[network](input_size=28*28, hidden_size=32, output_size=10).to(device)
        self._global_model.train()
        
        # Initialize the optimizer
        self._optimizer = optim.SGD(self._global_model.parameters(), lr=learning_rate)

    def update_server(self, T):
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
            delta_is_t = []
           
            # --multi-pro pool
            # pool = mp.Pool(self._num_clients)
            # results = [] 
            # --
            
            # 3: TODO 2 : select a subset of client
            # 4: x{i, 0}^t = x_t
            x_t = self._global_model.state_dict()
            # 5: for each client i \in S in parallel do
            for i in range(self._num_clients):
                #TODO 1 : do this in parallel
                
                #  x_{i, 0}^t = x_t  : x_t is model from last epoch 
                # load_state_dict(x_t)
                # cur_user._model.load_state_dict(x_t)
                # 6-8： x{i, K}^t = CLIENTOPT
                # --
                # result = pool.apply_async(self._clients[i].client_update, args=(round+1,i, x_t))
                # --
                x_i_K_t, client_loss, client_acc = self._clients[i].client_update(epoch=round+1, id=i, global_model=x_t)
                # 9: \delta_{client i, epoch t} = x{i, K}^t - x_t
                # delta_i_t =  x_i_K_t - x_t
                # delta_i_t = {}
                # for key in x_i_K_t.keys():
                #     delta_i_t[key] = x_i_K_t[key] - x_t[key]
                # --
                # results.append(result)
                # --
                client_models.append(x_i_K_t)
                client_losses.append(client_loss)
                client_accs.append(client_acc)
                #delta_is_t.append(delta_i_t)
                # -- print(f"Client {i+1} loss: {client_loss:.4f}, accuracy {client_acc: .4f} ")
            # --
            # pool.close()
            # pool.join()
            # for result in results:
            #     client_state_dict, client_loss, client_acc = result.get()
            #     client_models.append(client_state_dict)
            #     client_losses.append(client_loss)
            #     client_accs.append(client_acc)
            # --

            # x_{t+1} = \frac{1}{S} \sum_{i \in S} \delta_i^t
            global_state_dict = self.server_update(client_models)
            self._global_model.load_state_dict(global_state_dict)
            print(f"Round {round+1} finished, global loss: {sum(client_losses)/len(client_losses):.4f}, global accuracy: {sum(client_accs)/len(client_accs): .4f}")
            
        return sum(client_accs)/len(client_accs) 

    # Define the server update function
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
    