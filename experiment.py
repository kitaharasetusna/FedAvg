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

def fed_defense(defense, dataset, model, T, E, B):
    command = f'python ./main.py --model {model} --dataset {dataset} -T {T} \
        -E {E} -B {B} --algo {defense} --num_client 100 --folder data/test_acc/{dataset}/ \
            --client_ratio 0.1 --com_ratio 0.5' 
    os.system(command)
    
# -- 1. st. acc MNIST fedavg NN
def fedavg(dataset, model,T, E, B): 
    command = f'python ./main.py --model {model} --dataset {dataset} -T {T} \
        -E {E} -B {B} --algo fedavg --num_client 100 --folder data/test_acc/{dataset}/ \
            --client_ratio 0.1' 
    os.system(command)


def fedopt(dataset, model,T, E, B): 
    eta = 1
    eta_l = 1e-2 
    command = f'python ./main.py --model {model} --dataset {dataset} -T {T} \
        -E {E} -B {B} --algo fedopt --num_client 100 --folder data/test_acc/{dataset}/' 
    os.system(command)

def fedadag(dataset, model,T, E, B): 
    eta = 10**-0.5
    eta_l = 1e-2
    command = f'python ./main.py --model {model} --dataset {dataset} -T {T} \
        -E {E} -B {B} --algo fedadag --num_client 100 --folder data/test_acc/{dataset}/ \
            --eta {eta} --eta_l {eta_l}' 
    os.system(command)

def acc_test(algo='fedavg', dataset='MNIST', model='TwoLayerNet', T=20):
    E = [1, 5, 20]
    B = [100, 10, 50]
    for e in E:
        for b in B:
            if algo=='fedavg': 
                fedavg(dataset=dataset,model=model,T=T, E=e, B=b)
            elif algo=='fedopt':
                fedopt(dataset=dataset, model=model, T=T, E=e, B=b)
            elif algo=='fedadag':
                fedadag(dataset=dataset, model=model, T=T, E=e, B=b)
    # folder_path = 'data/test_acc/MNIST/'

def MNIST_TwoLayerNet_acc_paint_plot(algo='fedavg'):
    if algo=='fedavg':
        my_path = 'data\\test_acc\MNIST\MNIST_TwoLayerNet_fedavg_num_Client_100_eta_l_0.001\\'
    elif algo == 'fedadag':
        my_path = 'data\\test_acc\MNIST\MNIST_TwoLayerNet_fedada_num_Client_100_eta_0.31622776601683794_eta_l_0.01\\'
    elif algo == 'fedopt':
        # TODO: change this
        my_path = 'data\\test_acc\MNIST\MNIST_TwoLayerNet_fedopt_num_Client_100_eta_0.6666666666666666_eta_l_0.001\\'
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
    plt.title(f'{algo} acc on MNIST(TwoLayerNN)')
    plt.legend()
    plt.xlim(0, 100)
    plt.savefig(my_path+f'/acc_{algo}_MNIST_100_lr_0.001.jpg')

# 1. -- end. acc MNIST fedavg NN  

def fedopt_MNIST_TwoLayerNet(T, E, B):
    command = f'python ./main.py --model TwoLayerNet --dataset MNIST -T {T} -E {E} -B {B} --algo fedopt --num_client 100 --folder data/test_acc/MNIST/' 
    os.system(command)

def fedada_MNIST_TwoLayerNet(T, E, B):
    eta = 10**-0.5
    command = f'python ./main.py --model TwoLayerNet --dataset MNIST -T {T} -E {E} -B {B} --algo fedadag --num_client 100 \
        --eta {eta} --eta_l 1e-2 --folder data/test_acc/MNIST/' 
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


# attack
# 3. --st
def attack_MNIST(algo, T, E, B,eta, eta_l, attack_type, comp_ratio):
    command = f'python ./main.py --model TwoLayerNet \
        --dataset MNIST -T {T} -E {E} -B {B} --algo {algo} \
            --num_client 100 --folder data/test_acc_attack/MNIST/ \
        --attack_type {attack_type} --com_ratio {comp_ratio} --eta {eta} --eta_l {eta_l}' 
    os.system(command)

def attack_MNIST_plot(E=5, B=50, algo='fedavg'):
    path = 'data\\test_acc_attack\\MNIST\\' 
    accs = [] 
    for comp_ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        if algo=='fedavg':
            sub = f'MNIST_label_f_TwoLayerNet_{algo}_num_Client_100_eta_l_0.001_compromised_ratio_{comp_ratio}\\{E}_{B}.pkl'
        elif algo=='fedopt':
            sub = f'MNIST_label_f_TwoLayerNet_{algo}_num_Client_100_eta_0.6666666666666666_eta_l_0.001_compromised_ratio_{comp_ratio}\\{E}_{B}.pkl'
        elif algo=='fedadag':
            # sub = f'MNIST_label_f_TwoLayerNet_fedada_num_Client_100_eta_0.6666666666666666_eta_l_0.001_compromised_ratio_{comp_ratio}\\{E}_{B}.pkl'
            sub = f'MNIST_label_f_TwoLayerNet_fedada_num_Client_100_eta_0.6666666666666666_eta_l_0.001_compromised_ratio_{comp_ratio}\\{E}_{B}.pkl'
        with open(path+sub, 'rb') as temp_file:
            rounds, acc, T = pickle.load(temp_file)
            accs.append(acc[-1])
        temp_file.close()
    x= [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    accs = [1-acc for acc in accs]

    with open(path+f'test_error_of_{algo}_on_MNIST_for_label_flipping.pkl', 'wb') as file:
        pickle.dump((x, accs), file=file)
    file.close()
    plt.figure()
    plt.plot(x, accs, '--o')
    plt.title(f'Testing error rates of {algo} on MNIST for label flipping.')
    plt.xlabel('Percentage of compromised clients')
    plt.ylabel('Error Rate')
    plt.savefig(path+f'impace_of_compromised_ratio_{algo}.jpg')
    plt.show() 
    print(accs)

def exp1(dataset='MNIST', model='TwoLayerNet', T=20):
    for algo in ['fedadag', 'fedavg', 'fedopt']:
        acc_test(algo=algo, dataset=dataset, model=model, T=T) 
    # MNIST_TwoLayerNet_acc_paint_plot(algo=algo)    

def exp2():
    # fedopt_grid_searching_MNIST_epoch_100(1, 10, train=False)
    fedada_grid_searching_MNIST_epoch_100(1, 10)

def exp3():
    # algos = ['krum', 'fedavg', 'fedopt', 'fedadag']
    # algos = ['krum']
    # algos = ['krumadag']
    algos = ['fedadag']
    for algo in algos :
        for comp_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
            attack_MNIST(algo=algo,T=20, E=5, B=50, eta=10**-0.5, eta_l=1e-2,attack_type='label_f', comp_ratio=comp_ratio)

def exp3_plot():
    plt.figure()
    my_path = 'data/test_acc_attack/MNIST/'
    color = ['purple', 'aqua', 'coral']
    i = 0
    for algo in ['fedavg', 'fedopt', 'fedadag']:
        sub = f'test_error_of_{algo}_on_MNIST_for_label_flipping.pkl'
        with open(my_path+sub, 'rb') as temp_file:
            x, accs = pickle.load(temp_file) 
            plt.plot(x, accs, '--o', color=color[i], label=algo)
            plt.xlabel('Percentage of compromised clients')
            plt.ylabel('Error Rate')
        temp_file.close()
        i += 1
    plt.title('Testing error rates on MNIST for label flipping.')
    plt.legend()
    plt.savefig(my_path+f'impace_of_compromised_ratio.jpg')
    plt.show()
    

    
if __name__ == '__main__':
    

    # McMahan fig2.(b) non-IID acc MNIST fedavg 
    # exp1(dataset='shakespeare', model='LSTM', T=20) 
    # exp1(algo='fedopt')
    
    # grid_searching_MNIST_epoch_100(1, 10)
    # Reddit fig2. 
    # exp2()
   
    # Impact of  
    exp3()
    # attack_MNIST_plot(algo='fedadag') 
    # exp3_plot()
    # fed_defense(defense='krumopt', dataset='MNIST', model='TwoLayerNet', T=2, E=5, B=50)



    
    