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

from models.my_NN import TwoLayerNet
from utils import ExpSetting



# Define the main Federated Learning function
def federated_learning(num_rounds, train_data, num_clients, client_batch_size, learning_rate, device, \
    shards_num=200):
    # Split the train data into subsets for each client
    server = ServerBase(train_data, num_clients, client_batch_size, learning_rate, device, shards_num)  
    server.update_server(num_rounds)
    test_loader = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    return server._global_model 


# Set the hyperparameters
exp_settings = ExpSetting()
num_rounds, num_clients, _, client_batch_size = exp_settings.get_options()



learning_rate = 0.001

# Load the MNIST dataset
train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run the Federated Learning process
global_model = federated_learning(num_rounds, train_data, num_clients, client_batch_size, learning_rate, device)

