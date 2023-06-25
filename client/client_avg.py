import torch.nn as nn
import torch
import numpy as np
import copy

class ClientAVG():
    '''base class for FL learning'''
    def __init__(self, dataloader, model, optimizer, device, E, B, args):
        self._dataloader = dataloader
        self._model = model
        self._optimizer  = optimizer
        self._device = device
        self._E = E
        self._B = B
        self.args = args

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
        self._model.train()
        return copy.deepcopy(self._model.cpu()).state_dict(), \
            running_loss / (len(self._dataloader.dataset)*self._E), acc_num/total_num

    
class ComproClientAVG(ClientAVG):
    def client_update(self, epoch, id, global_model):
        '''ClientUpdate in FedAVG;'''
        # print(f'client {id+1} is started to run.')
        attack_type = self.args.attack_type 
        print(f'client {id} is compromised') 
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

        # malicious code starts here----
        if attack_type =='partial_know':
            pass
        elif attack_type == 'full_know':
            pass
        # malicious code end here-- 
        
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
        self._model.train()
        return copy.deepcopy(self._model.cpu()).state_dict(), \
            running_loss / (len(self._dataloader.dataset)*self._E), acc_num/total_num