import torch.nn as nn
import torch
import numpy as np

class ClientBase():
    '''base class for FL learning'''
    def __init__(self, dataloader, model, optimizer, device, E, B):
        self._dataloader = dataloader
        self._model = model
        self._optimizer  = optimizer
        self._device = device
        self._E = E
        self._B = B

    def client_update(self, epoch):
        '''ClientUpdate in FedAVG;'''
        self._model.train()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0
        num_euqal = 0
        acc = None
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
        for inputs, labels in self._dataloader:
            inputs, labels = inputs.to(self._device), labels.to(self._device)
            test_output = self._model(inputs)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            num_equal = (pred_y == labels).sum().item()
            acc_num += num_equal
            total_num += labels.size()[0] 
        return self._model.state_dict(), running_loss / len(self._dataloader.dataset), acc_num/total_num
    