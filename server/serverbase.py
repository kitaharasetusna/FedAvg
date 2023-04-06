import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models.my_NN import TwoLayerNet
import sys
sys.path.append('../client')
from client.clientbase import ClientBase

class ServerBase():
    def __init__(self, train_data, num_clients, client_batch_size, learning_rate, device, \
        shards_num):
        #TODO: select models
        #TODO: selcet data
        self._num_clients = num_clients
        self._subset_indices = torch.linspace(0, len(train_data)-1, steps=shards_num+1).round().tolist()
        self._client_datasets = [Subset(train_data, range(int(self._subset_indices[i]), int(self._subset_indices[i+2]))) for i in range(num_clients)]
        self._client_loaders = [DataLoader(self._client_datasets[i], batch_size=client_batch_size, shuffle=True) for i in range(num_clients)]
        self._clients_models = [ TwoLayerNet(input_size=28*28, hidden_size=32, output_size=10).to(device) for i in range(num_clients)]
        self._clients_optims = [optim.SGD(self._clients_models[i].parameters(), lr=learning_rate) for i in range(num_clients)]
        self._clients = [ClientBase(self._client_loaders[i], self._clients_models[i], self._clients_optims[i], device, 1, client_batch_size) for i in range(num_clients)]

        self._global_model = TwoLayerNet(input_size=28*28, hidden_size=32, output_size=10).to(device)
        self._global_model.train()
        
        # Initialize the optimizer
        self._optimizer = optim.SGD(self._global_model.parameters(), lr=learning_rate)

    def update_server(self, T):
        for round in range(T):
            print(f"Round {round+1} started...")
            client_models = []
            client_losses = []
            for i in range(self._num_clients):
                # client_model = TwoLayerNet(input_size=28*28, hidden_size=32, output_size=10).to(device)
                cur_user = self._clients[i]
                # TODO: change this into a fucntion of class
                cur_user._model.load_state_dict(self._global_model.state_dict())
                client_state_dict, client_loss = cur_user.client_update()
                client_models.append(client_state_dict)
                client_losses.append(client_loss)
                print(f"Client {i+1} loss: {client_loss:.4f}")
            global_state_dict = self.server_update(client_models)
            self._global_model.load_state_dict(global_state_dict)
            print(f"Round {round+1} finished, global loss: {sum(client_losses)/len(client_losses):.4f}")

    # Define the server update function
    def server_update(self, models):
        new_state_dict = {}
        for key in models[0].keys():
            new_state_dict[key] = torch.stack([models[i][key] for i in range(len(models))], dim=0).mean(dim=0)
        return new_state_dict
    