import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# for dataset
from torchvision.datasets import MNIST
from my_utils.dataset import ShakeSpeare

from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from torch.utils.data import Dataset


from server.fedavg_server import ServerAVG
from server.fedopt_server import ServerOPT
from server.fedadagrad_server import ServerFedAdaGrade
from server.krum_avg import ServerKrum

from models.my_NN import TwoLayerNet
from my_utils.utils import ExpSetting



# Define the main Federated Learning function
def federated_learning(model,dataset,  T, train_data, num_clients, E, B, \
    learning_rate, device, algo, client_ratio, folder, args, \
    shards_num=200, beta_1=None, eta=None, tau=None):
    # Split the train data into subsets for each client
    # data, C?, B, lr, device, shard_num
    
    if algo =='fedopt':
        server = ServerOPT(dataset, model, train_data, num_clients, E, B, \
            learning_rate, device, shards_num, client_ratio, folder=folder, args=args, eta=eta)  
    elif algo == 'fedavg':
        server = ServerAVG(dataset,model, train_data, num_clients, E, B, \
            learning_rate, device, shards_num, client_ratio, folder=folder, args=args)  
    elif algo == 'krum':
       server = ServerKrum(dataset,model, train_data, num_clients, E, B, \
            learning_rate, device, shards_num, client_ratio, folder=folder, args=args)   
    elif algo == 'fedadag':
        server = ServerFedAdaGrade(dataset, model, train_data, num_clients, E, B, \
            learning_rate, device, shards_num, client_ratio, folder=folder, args=args,\
                initial_mom={}, beta_1=beta_1, eta=eta, tau=tau)
    fin_acc, glob_acc = server.update_server_thread_res(T)
    # TODO: add more other datasets
    return server._global_model , fin_acc, glob_acc[-1]

if __name__ == '__main__':
    # TODO: check the meaning of using seed bellow
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    
    
    # Set the hyperparameters
    exp_settings = ExpSetting()
    num_rounds, num_clients, E, B, learning_rate, algo, client_ratio, beta_1, eta, tau, dataset, model, folder, args = exp_settings.get_options()
    
    # Load the MNIST dataset
    if dataset == 'MNIST':
        train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    elif dataset == 'shakespeare':
        print('shakespeare')
        train_data = ShakeSpeare(train=True)
    else:
        print('ERROR: WRONG dataset'); import sys; sys.exit()
    # Check if GPU is available
    device_ava = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    print(f'{type(device)} {type(device_ava)} {device==device_ava}')
    assert device==device_ava, f"{device} is not available, please use {device_ava}"
    # Run the Federated Learning process
    global_model, fin_acc, test_acc = federated_learning(model, dataset, num_rounds, train_data, num_clients, E, B, learning_rate, device, \
        algo, client_ratio, folder, args, 
                                               beta_1=beta_1, eta=eta, tau=tau)
    print(f'final train accuracy: {fin_acc :.4f}, test accuracy: {test_acc}')
    print(test_acc)


