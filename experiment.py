import os

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

def fedavg_shakespeare_charLSTM_epoch_90_E_1_B_10():
    '''
    save time for wrting a bunch of commands
    '''
    # define the command to run script1.py with options
    command = 'python ./main.py --model LSTM --dataset shakespeare -T 20 -E 1 -B 2 --algo fedadag --num_client 100' 
    # run the command using subprocess
    os.system(command)

if __name__ == '__main__':
    fedavg_shakespeare_charLSTM_epoch_90_E_1_B_10()
    # fedopt_shakespeare_charLSTM_epoch_90_E_1_B_10()