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



class ServerOPT(ServerAVG):
    def __init__(self, dataset, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num, client_ratio, folder, eta,  algo='fedopt'):
        super().__init__(dataset, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num, client_ratio, folder, algo=algo)
        self._eta = eta
        self.pkl_path = f'{dataset}_{network}_{algo}_num_Client_{num_clients}_eta_{eta}_eta_l_{learning_rate}'
        
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
            ret_dict[key] = glob[key].to('cuda') + self._eta*new_dict[key].to('cuda')
        return ret_dict
    
    def aggregate(self, glob_acc, T, test_loader):
        # 2: for t=0, ..., T-1 do
        early_stop_epoch = None
        for round in range(T):
            print(f"Round {round+1} started...")
            print('debuging... we are upating de')
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
            fin_acc = self.test_acc(test_loader=test_loader)
            # lenth: T
            glob_acc.append(fin_acc)
            
            
            global_state_dict = self.server_update(delta_t, x_t)
            self._global_model.load_state_dict(global_state_dict) 
            if fin_acc>0.95:
                early_stop_epoch = round+1
                print('Congrates, Eearly stop, have reached 0.99 acc!')
                break
            
            
            print(f"Round {round+1} finished, global loss:  \
                {sum(client_losses)/len(client_losses):.4f},  \
                    global accuracy: {sum(client_accs)/len(client_accs): .4f}")

        pkl_path = f'{self.folder}{self.pkl_path}'
        os.makedirs(pkl_path, exist_ok=True)
        print(f'saving global model in {pkl_path}...')
        torch.save(self._global_model.state_dict(), f'{self.folder}{self.pkl_path}/{self._E}_{self._B}_model.pth')

        return glob_acc, client_accs, early_stop_epoch

   


