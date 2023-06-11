import argparse
import sys
sys.path.append('../')
from models.my_NN import TwoLayerNet, CharLSTM

function_map = {
    "TwoLayerNet": TwoLayerNet,
    "LSTM": CharLSTM
}


class ExpSetting():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='experiment settings')
        self.parser.add_argument('--algo', choices=['fedavg', 'fedopt', 'fedadag'], \
            help='choose algorithm for FL learning')
        self.parser.add_argument('--dataset', choices=['MNIST', 'shakespeare'], \
            help='name of dataset, available choices: MNIST, shakespeare')
        self.parser.add_argument('--model', choices=['TwoLayerNet', 'LSTM']) 
        
        # for client
        self.parser.add_argument('--num_client', type=int, default=10, \
            help='number of clients')
        self.parser.add_argument('-C', '--client_ratio', type=float, default=0.1, \
            help='number of clients')
        self.parser.add_argument('-E', '--round_client', type=int, default=200, \
            help='number of rounds on clients')
        self.parser.add_argument('-B', '--size_batch', type=int, default=32, \
            help='batch size b on client')
        self.parser.add_argument('--eta_l', type=float, default=1e-3, \
            help='learning rate in client update, f_l in the paper')
        
        # for server 
        self.parser.add_argument('-T', '--num_round', type=int, default=10, \
            help='number of rounds')
        self.parser.add_argument('--beta_1', type=float, default=0.9, \
            help='hyper-param for momentum')
        self.parser.add_argument('--eta', type=float, default=2/3, \
            help='server learning rate')
        self.parser.add_argument('--tau', type=float, default=1e-3, \
            help='hyper-param for updating')
        
        
    def get_options(self):
        args = self.parser.parse_args()
        print(f'running algorithm {args.algo}..., dataset={args.dataset},  model={args.model}, T(server epoch)={args.num_round} \
            , num_client={args.num_client},  \
               C={args.client_ratio}, E={args.round_client}, B={args.size_batch}, client lr={args.eta_l} \
                   eta={args.eta}, tau = {args.tau}')
        return args.num_round, args.num_client, args.round_client, args.size_batch, args.eta_l, args.algo, args.client_ratio, args.beta_1, \
            args.eta, args.tau, args.dataset, args.model
    