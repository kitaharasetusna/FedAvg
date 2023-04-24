import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from torch.utils.data import Dataset


from client.clientbase import ClientBase
from server.serverbase import ServerBase
from server.fedopt_server import ServerOPT

from models.my_NN import TwoLayerNet
from my_utils.utils import ExpSetting



# Define the main Federated Learning function
def federated_learning(T, train_data, num_clients, E, B, learning_rate, device, \
    shards_num=200):
    # Split the train data into subsets for each client
    # data, C?, B, lr, device, shard_num
    server = ServerOPT('TwoLayerNet', train_data, num_clients, E, B, learning_rate, device, shards_num)  
    fin_acc = server.update_server_thread_res(T)
    test_loader = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    return server._global_model , fin_acc

if __name__ == '__main__':
    # Set the hyperparameters
    exp_settings = ExpSetting()
    num_rounds, num_clients, E, B, learning_rate = exp_settings.get_options()
    # Load the MNIST dataset
    train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run the Federated Learning process
    global_model, fin_acc = federated_learning(num_rounds, train_data, num_clients, E, B, learning_rate, device)
    print(f'final train accuracy: {fin_acc :.4f}')

