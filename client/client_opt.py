import torch.nn as nn
import torch
import numpy as np
import copy

from client.client_avg import ClientAVG

class ClientOPT(ClientAVG):
    def get_static_dict_difference(self, A, B):
        diff_dict = {}
        for key in B.keys():
                diff_dict[key] = B[key].to('cpu') - A[key].to('cpu')
        return diff_dict
    
    def client_update(self, epoch, id, global_model):
        '''ClientUpdate in FedAVG;'''
        # print(f'client {id+1} is started to run.')
        
        self._model.load_state_dict(global_model)
        
        self._model.to(device=self._device)
        criterion = nn.CrossEntropyLoss()
        running_loss = 0
        num_euqal = 0
        acc = None
        
        self._model.train() 
        for _ in range(self._E):
            for inputs, labels in self._dataloader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()
                running_loss += loss.item() * inputs.size(0)
        self._model.eval()
        acc_num = 0
        total_num = 0
        with torch.no_grad():
            for inputs, labels in self._dataloader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                test_output = self._model(inputs)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                num_equal = (pred_y == labels).sum().item()
                acc_num += num_equal
                total_num += labels.size()[0] 
        print(f"Client {id+1} Ended-loss:  \
            {running_loss / len(self._dataloader.dataset)*self._E:.4f}, \
                accuracy {acc_num/total_num: .4f} ")
        
        # get diff
        diff_dict = self.get_static_dict_difference(global_model, self._model.state_dict())
        self._model.train()
        return diff_dict, \
            running_loss / (len(self._dataloader.dataset)*self._E), acc_num/total_num
        

class CompromisedClientOPT(ClientOPT):
     def client_update(self, epoch, id, global_model):
        attack_type = self.args.attack_type 
        print(f'this client{id} is compromised by{attack_type}')
        '''ClientUpdate in FedAVG;'''
        # print(f'client {id+1} is started to run.')
        
        self._model.load_state_dict(global_model)
        
        self._model.to(device=self._device)
        criterion = nn.CrossEntropyLoss()
        running_loss = 0
        num_euqal = 0
        acc = None

        if attack_type =='label_f':
            print('its label flipping attack')
        
        self._model.train() 
        for _ in range(self._E):
            for inputs, labels in self._dataloader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                if attack_type =='label_f' and self.args.dataset=='MNIST':
                    labels = 10-labels-1
                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()
                running_loss += loss.item() * inputs.size(0)
        self._model.eval()
        acc_num = 0
        total_num = 0
        with torch.no_grad():
            for inputs, labels in self._dataloader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                test_output = self._model(inputs)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                num_equal = (pred_y == labels).sum().item()
                acc_num += num_equal
                total_num += labels.size()[0] 
        print(f"Client {id+1} Ended-loss:  \
            {running_loss / len(self._dataloader.dataset)*self._E:.4f}, \
                accuracy {acc_num/total_num: .4f} ")
        
        # get diff
        diff_dict = self.get_static_dict_difference(global_model, self._model.state_dict())
        self._model.train()
        return diff_dict, \
            running_loss / (len(self._dataloader.dataset)*self._E), acc_num/total_num
    
    