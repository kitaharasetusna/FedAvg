import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import numpy as np
import subprocess
from main import federated_learning
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import glob
import pickle
import matplotlib.pyplot as plt


def fedopt_shakespeare_charLSTM_epoch_90_E_1_B_10():
    '''
    save time for wrting a bunch of commands
    '''
    # define the command to run script1.py with options
    command = 'python ./main.py --model LSTM --dataset shakespeare -T 20 -E 1 -B 2 --algo fedopt --num_client 100' 
    # run the command using subprocess
    os.system(command)

def fedavg_shakespeare_charLSTM_epoch_90_E_1_B_10():
    '''
    save time for wrting a bunch of commands
    '''
    # define the command to run script1.py with options
    command = 'python ./main.py --model LSTM --dataset shakespeare -T 20 -E 1 -B 2 --algo fedavg --num_client 100' 
    # run the command using subprocess
    os.system(command)

def fedada_shakespeare_charLSTM_epoch_90_E_1_B_10():
    '''
    save time for wrting a bunch of commands
    '''
    # define the command to run script1.py with options
    command = 'python ./main.py --model LSTM --dataset shakespeare -T 20 -E 1 -B 2 --algo fedadag --num_client 20 --folder data/test_acc/shakespeare/' 
    # run the command using subprocess
    os.system(command)

# -- 1. st. acc MNIST fedavg NN
def fedavg_MNIST_TwoLayerNet(T, E, B): 
    command = f'python ./main.py --model TwoLayerNet --dataset MNIST -T {T} -E {E} -B {B} --algo fedavg --num_client 100 --folder data/test_acc/MNIST/' 
    os.system(command)


def fedopt_MNIST_TwoLayerNet(T, E, B): 
    command = f'python ./main.py --model TwoLayerNet --dataset MNIST -T {T} -E {E} -B {B} --algo fedopt --num_client 100 --folder data/test_acc/MNIST/' 
    os.system(command)

def fedavg_MNIST_TwoLayerNet_acc_paint():
    E = [1, 5, 20]
    B = [600, 10, 50]
    for e in E:
        for b in B:
            fedavg_MNIST_TwoLayerNet(100, e, b)
    # folder_path = 'data/test_acc/MNIST/'

def fedavg_MNIST_TwoLayerNet_acc_paint_plot():
    my_path = 'data\\test_acc\MNIST\MNIST_TwoLayerNet_fedavg_num_Client_100_eta_l_0.001\\'
    E = [1, 5, 20]
    B = [600, 10, 50]
    for e in E:
        for b in B:
            file_name = f'{e}_{b}.pkl'
            file_path = my_path+file_name
            with open(file_path, 'rb') as file:
                rounds, glob_acc, T = pickle.load(file)
                rounds = np.arange(1, len(glob_acc)+1, 1)
                print(len(rounds), len(glob_acc)); import sys; sys.exit;
            plt.plot(rounds, glob_acc, label= f'E: {e} B: {b}')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.title('fedavg acc on MNIST(TwoLayerNN)')
    plt.legend()
    plt.xlim(0, 100)
    plt.savefig(my_path+'/acc_fedavg_MNIST_100_lr_0.001.jpg')

# 1. -- end. acc MNIST fedavg NN  

def fedopt_MNIST_TwoLayerNet(T, E, B):
    command = f'python ./main.py --model TwoLayerNet --dataset MNIST -T {T} -E {E} -B {B} --algo fedopt --num_client 100 --folder data/test_acc/MNIST/' 
    os.system(command)

def fedada_MNIST_TwoLayerNet(T, E, B):
    command = f'python ./main.py --model TwoLayerNet --dataset MNIST -T {T} -E {E} -B {B} --algo fedadag --num_client 100 --folder data/test_acc/MNIST/' 
    os.system(command)

# 2. --st  
def fedopt_grid_searching_MNIST_epoch_100(E, B, train=True):
    my_path = 'data\\grid_search\MNIST\MNIST_TwoLayerNet_fedopt_num_Client_100\\'
    os.makedirs(my_path, exist_ok=True)
    if train: 
        # TODO: 把所有参数写进config
        train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        algo = 'fedopt'
        model = 'TwoLayerNet'
        dataset = 'MNIST'
        # TODO: change this to 100
        num_rounds = 20
        num_clients = 100
        client_ratio = 0.1
        beta_1 = None
        tau = None
        import math

        y = [10**b for b in np.arange(-3.0, 1.0+0.5, 0.5)]
        x = [10**b for b in np.arange(-3.0, 0.5+0.5, 0.5)]
        
        y_len = len(y)
        x_len = len(x)
        print(f'{y_len} {x_len}')
        grid_arr = np.zeros((y_len, x_len))
        for index_eta, eta in enumerate(y):
            for index_eta_l, eta_l in enumerate(x):
                learning_rate = eta_l
                global_model, fin_acc, test_acc = federated_learning(model, dataset, num_rounds, train_data, num_clients, E, B, learning_rate, device, algo, client_ratio,
                                                folder='data/grid_search/MNIST/', beta_1=beta_1, eta=eta, tau=tau) 
                grid_arr[index_eta, index_eta_l] = test_acc
                print(f'eta: {eta}, eta_l: {eta_l}, test acc: {test_acc}'); 
        
        print(grid_arr)
        with open(my_path+'grid_arr.pkl', 'wb') as file:
            pickle.dump(grid_arr, file=file)
        
        file.close()
    
    with open(my_path+'grid_arr.pkl', 'rb') as file:
        grid_arr = pickle.load(file=file)

    grid_arr = np.round(grid_arr, decimals=2)
    
    # Create a figure and axis
    fig, ax = plt.subplots() 
    # Create the heatmap
    heatmap = ax.imshow(grid_arr, cmap='Blues')

    # Add the value annotations
    for i in range(grid_arr.shape[0]):
        for j in range(grid_arr.shape[1]):
            ax.text(j, i, grid_arr[i, j], ha='center', va='center', color='white')

    # Customize the plot
    ax.set_xticks(np.arange(grid_arr.shape[1]))
    ax.set_yticks(np.arange(grid_arr.shape[0]))
    ax.set_xticklabels([0.5*i for i in range(-6, 2, 1)])
    ax.set_yticklabels([0.5*i for i in range(-6, 3, 1)])
    ax.set_xlabel('Client Learning rate(log 10)')
    ax.set_ylabel('Server Learning rate(log 10)')
    ax.set_title('MNSIT Fedopt')
    # Show the colorbar
    cbar = plt.colorbar(heatmap)

    
    # plt.imshow(grid_arr, cmap='hot')
    # plt.colorbar()
    # plt.show()
    plt.savefig(my_path+'gird_search_MNIST_fedopt_num_of_client_100_E_1_B_10.png')
    # Display the plot
    plt.show()
    

# 2. --st  
def fedada_grid_searching_MNIST_epoch_100(E, B, train=True):
    my_path = 'data\\grid_search\MNIST\MNIST_TwoLayerNet_fedada_num_Client_100\\'
    os.makedirs(my_path, exist_ok=True)
    if train: 
        # TODO: 把所有参数写进config
        train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        algo = 'fedadag'
        model = 'TwoLayerNet'
        dataset = 'MNIST'
        # TODO: change this to 100
        num_rounds = 20
        num_clients = 100
        client_ratio = 0.1
        beta_1 = 0.9 
        tau = 1e-3
        import math

        y = [10**b for b in np.arange(-3.0, 1.0+0.5, 0.5)]
        x = [10**b for b in np.arange(-3.0, 0.5+0.5, 0.5)]
        
        y_len = len(y)
        x_len = len(x)
        print(f'{y_len} {x_len}')
        grid_arr = np.zeros((y_len, x_len))
        for index_eta, eta in enumerate(y):
            for index_eta_l, eta_l in enumerate(x):
                learning_rate = eta_l
                global_model, fin_acc, test_acc = federated_learning(model, dataset, num_rounds, train_data, num_clients, E, B, learning_rate, device, algo, client_ratio,
                                                folder='data/grid_search/MNIST/', beta_1=beta_1, eta=eta, tau=tau) 
                grid_arr[index_eta, index_eta_l] = test_acc
                print(f'eta: {eta}, eta_l: {eta_l}, test acc: {test_acc}'); 
        
        print(grid_arr)
        with open(my_path+'grid_arr.pkl', 'wb') as file:
            pickle.dump(grid_arr, file=file)
        
        file.close()
    
    with open(my_path+'grid_arr.pkl', 'rb') as file:
        grid_arr = pickle.load(file=file)

    grid_arr = np.round(grid_arr, decimals=2)
    
    # Create a figure and axis
    fig, ax = plt.subplots() 
    # Create the heatmap
    heatmap = ax.imshow(grid_arr, cmap='Blues')

    # Add the value annotations
    for i in range(grid_arr.shape[0]):
        for j in range(grid_arr.shape[1]):
            ax.text(j, i, grid_arr[i, j], ha='center', va='center', color='white')

    # Customize the plot
    ax.set_xticks(np.arange(grid_arr.shape[1]))
    ax.set_yticks(np.arange(grid_arr.shape[0]))
    ax.set_xticklabels([0.5*i for i in range(-6, 2, 1)])
    ax.set_yticklabels([0.5*i for i in range(-6, 3, 1)])
    ax.set_xlabel('Client Learning rate(log 10)')
    ax.set_ylabel('Server Learning rate(log 10)')
    ax.set_title('MNSIT FedAdagrad') 
    # Show the colorbar
    cbar = plt.colorbar(heatmap)

    
    # plt.imshow(grid_arr, cmap='hot')
    # plt.colorbar()
    # plt.show()
    plt.savefig(my_path+'gird_search_MNIST_fedopt_num_of_client_100_E_1_B_10.png')
    # Display the plot
    plt.show() 


def exp1():
    fedavg_MNIST_TwoLayerNet_acc_paint() 
    fedavg_MNIST_TwoLayerNet_acc_paint_plot()    

def exp2():
    # fedopt_MNIST_TwoLayerNet_acc_paint() 
    # fedopt_MNIST_TwoLayerNet_acc_paint_plot()
    # fedopt_MNIST_TwoLayerNet(T=5, E=5, B=50)
    # fedavg_MNIST_TwoLayerNet(T=5, E=5, B=50)
    fedopt_grid_searching_MNIST_epoch_100(1, 10, train=False)
    # fedada_grid_searching_MNIST_epoch_100(5, 50)

if __name__ == '__main__':
    # fedavg_shakespeare_charLSTM_epoch_90_E_1_B_10()
    # fedopt_shakespeare_charLSTM_epoch_90_E_1_B_10()
    # fedada_shakespeare_charLSTM_epoch_90_E_1_B_10()
    
    # fedavg_MNIST_TwoLayerNet(5, 1, 10)
    # fedopt_MNIST_TwoLayerNet(5, 1, 10)
    # fedada_MNIST_TwoLayerNet(5, 1, 10)

    # McMahan fig2.(b) non-IID acc MNIST fedavg 
    # exp1() 
    
    # grid_searching_MNIST_epoch_100(1, 10)
    # Reddit fig2. 
    # TODO: change this into validation set, now it's on test set
    exp2()
    
    # grid_searching_MNIST_epoch_100(1, 2)