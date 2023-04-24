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
from torch.multiprocessing import Process
import copy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class ServerBase():
    def __init__(self, network, train_data, num_clients, E, client_batch_size, learning_rate, device, \
        shards_num):
        #TODO: select models
        #TODO: selcet data
        self._num_clients = num_clients
        self._subset_indices = torch.linspace(0, len(train_data)-1,  \
            steps=shards_num+1).round().tolist()
        self._client_datasets = [Subset(train_data, range(int(self._subset_indices[i]),  \
            int(self._subset_indices[i+2]))) for i in range(num_clients)]
        self._E = E
        self._client_loaders = [DataLoader(self._client_datasets[i],  
                                           batch_size=client_batch_size, 
            shuffle=True) for i in range(num_clients)]
        self._clients_models = [function_map[network](input_size=28*28,  \
            hidden_size=32, output_size=10).to(device) for i in range(num_clients)]
        # self._clients_optims = [optim.SGD(self._clients_models[i].parameters(), lr=learning_rate) for i in range(num_clients)]
        self._clients_optims = [optim.Adam(self._clients_models[i].parameters(), \
                                           lr=learning_rate) for i in range(num_clients)]
        self._lr = learning_rate
        self._clients = [ClientBase(self._client_loaders[i], \
            self._clients_models[i], self._clients_optims[i],  \
                device, E, client_batch_size) for i in range(num_clients)]

        self._global_model = function_map[network](input_size=28*28, hidden_size=32, output_size=10).to(device)
        self._global_model.train()
        
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
            pool = mp.Pool(self._num_clients)
            results = [] 
            # --
            
            # 3: TODO 2 : select a subset of client
            # 4: x{i, 0}^t = x_t
            x_t = self._global_model.state_dict()
            # 5: for each client i \in S in parallel do
            for i in range(self._num_clients):
                #TODO 1 : do this in parallel
                
                #  x_{i, 0}^t = x_t  : x_t is model from last epoch 
                x_t_temp = copy.deepcopy(x_t)
                
                # cur_user._model.load_state_dict(x_t)
                # 6-8： x{i, K}^t = CLIENTOPT
                # --
                # result = pool.apply_async(self._clients[i].client_update, args=(round+1,i, x_t_temp))
                # x_i_K_t, client_loss, client_acc = \
                # pool.apply_async(self._clients[i].client_update, args=(round+1,i, x_t_temp)).get()
                # --
                x_i_K_t, client_loss, client_acc =  \
                   self._clients[i].client_update(epoch=round+1, id=i, global_model=x_t_temp)
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
                # delta_is_t.append(delta_i_t)
                # -- print(f"Client {i+1} loss: {client_loss:.4f}, accuracy {client_acc: .4f} ")
            # --
            
            # for result in results:
            #     client_state_dict, client_loss, client_acc = result.get()
            #     client_models.append(client_state_dict)
            #     client_losses.append(client_loss)
            #     client_accs.append(client_acc)
            # --

            # x_{t+1} = \frac{1}{S} \sum_{i \in S} \delta_i^t
            global_state_dict = self.server_update(client_models)
            self._global_model.load_state_dict(global_state_dict)
            print(f"Round {round+1} finished, global loss:  \
                {sum(client_losses)/len(client_losses):.4f},  \
                    global accuracy: {sum(client_accs)/len(client_accs): .4f}")
            #--
            pool.close()
            pool.join()
            #--
            
        return sum(client_accs)/len(client_accs) 

    # deprecated
    def update_server_multi(self, T):
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
           
            x_t = self._global_model.state_dict()
           
            
            for i in range(self._num_clients):
                self._clients[i]._model.load_state_dict(x_t)
                self._clients[i]._model.share_memory()
                

            # clients = [Client()]
            # [clients.models for client in clients]
            processes = []
            for i in range(self._num_clients):
                x_t_temp = copy.deepcopy(x_t)
                p = Process(target=client_update_multi, \
                        args=(self._clients[i]._model, round+1, i, \
                            self._clients[i]._device, self._clients[i]._E, \
                                self._clients[i]._dataloader, self._lr))
                p.start()
                processes.append(p) 
            for p in processes:
                p.join()
            for client in self._clients:
                client_models.append(client._model.state_dict())
            global_state_dict = self.server_update(client_models)
            self._global_model.load_state_dict(global_state_dict)
                 
        return  1
    # depracated
    def update_server_pool(self, T):
        #trying pool.map
        '''
        FedAVG
        '''
        # 2: for t=0, ..., T-1 do
        for round in range(T):
            print(f"Round {round+1} started...")
            client_models = []
           
            x_t = self._global_model.state_dict()
           
            
            # for i in range(self._num_clients):
            #     self._clients[i]._model.load_state_dict(x_t)
            #     self._clients[i]._model.share_memory()
                

            #--keep your multi-processing code here 
            processes = []
            # for i in range(self._num_clients):
            #    self._clients[i].client_update_multi_pool()
            with mp.Pool() as pool:
                pool.map(ClientBase.client_update_multi_pool, self._clients)
                # self._clients = list(pool.map(ClientBase.client_update_multi_pool, self._clients))
            #--keep your multi-processing code here 
            import sys
            sys.exit()
            for client in self._clients:
                client_models.append(client._model.state_dict())
            global_state_dict = self.server_update(client_models)
            self._global_model.load_state_dict(global_state_dict)
                 
        return  1
    # deprecated 
    def update_server_threading(self, T):
       #trying pool.map
        '''
        FedAVG
        '''
        # 2: for t=0, ..., T-1 do
        for round in range(T):
            print(f"Round {round+1} started...")
            client_models = []
           
            x_t = self._global_model.state_dict()
           
            
            # for i in range(self._num_clients):
            #     self._clients[i]._model.load_state_dict(x_t)
            #     self._clients[i]._model.share_memory()
                

            #--keep your multi-processing code here 
            executor = ThreadPoolExecutor(max_workers=self._num_clients) 
            for i in range(self._num_clients):
                executor.submit(self._clients[i].client_update, round+1, i, self._global_model.state_dict())
            #    self._clients[i].client_update_multi_pool()
            
            #--keep your multi-processing code here 
            import sys
            sys.exit()
            for client in self._clients:
                client_models.append(client._model.state_dict())
            global_state_dict = self.server_update(client_models)
            self._global_model.load_state_dict(global_state_dict)
                 
        return  1 
     
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
            delta_is_t = []
            x_t = self._global_model.state_dict()
            # -multi-threading is here
            executor = ThreadPoolExecutor(max_workers=self._num_clients)
            processes = []
            for i in range(self._num_clients):
                x_t_temp = copy.deepcopy(x_t)
                processes.append(executor.submit(self._clients[i].client_update, round+1, i, x_t_temp))
                # x_i_K_t, client_loss, client_acc =  \
                #    self._clients[i].client_update(epoch=round+1, id=i, global_model=x_t_temp)
                # client_models.append(x_i_K_t)
                # client_losses.append(client_loss)
                # client_accs.append(client_acc)
            
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


# deprecated 
def client_update_multi(_model, epoch, id, _device, _E, _dataloader, _lr):
        '''ClientUpdate in FedAVG;'''
        # print(f'client {id+1} is started to run.')
        _optimizer =  optim.Adam(_model.parameters(), \
                                           lr=_lr) 
        DataLoader(self._client_datasets[i],  
                                           batch_size=client_batch_size, 
            shuffle=True)
        _model.to(device=_device)
        criterion = nn.CrossEntropyLoss()
        running_loss = 0
        num_euqal = 0
        acc = None
        
        _model.train() 
        for _ in range(_E):
            for inputs, labels in _dataloader:
                inputs, labels = inputs.to(_device), labels.to(_device)
                _optimizer.zero_grad()
                outputs = _model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                _optimizer.step()
                running_loss += loss.item() * inputs.size(0)
        _model.eval()
        acc_num = 0
        total_num = 0
        with torch.no_grad():
            for inputs, labels in _dataloader:
                inputs, labels = inputs.to(_device), labels.to(_device)
                test_output = _model(inputs)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                num_equal = (pred_y == labels).sum().item()
                acc_num += num_equal
                total_num += labels.size()[0] 
        print(f"Client {id+1} Ended-loss:  \
            {running_loss / (len(_dataloader.dataset)*_E):.4f}, \
                accuracy {acc_num/total_num: .4f} ")