import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Subset


from client.clientbase import ClientBase
from server.serverbase import ServerBase

from models.my_NN import TwoLayerNet



# Define the main Federated Learning function
def federated_learning(num_rounds, train_data, num_clients, client_batch_size, learning_rate, device, \
    shards_num=200):
    # Split the train data into subsets for each client
    server = ServerBase(train_data, num_clients, client_batch_size, learning_rate, device, shards_num)  
    server.update_server(num_rounds)
    return server._global_model 

# Set the hyperparameters
num_rounds = 10
num_clients = 100
client_batch_size = 32
learning_rate = 0.1

# Load the MNIST dataset
train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run the Federated Learning process
global_model = federated_learning(num_rounds, train_data, num_clients, client_batch_size, learning_rate, device)

