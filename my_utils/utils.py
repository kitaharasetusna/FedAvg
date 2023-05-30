import argparse
import sys
sys.path.append('../')
from models.my_NN import TwoLayerNet

function_map = {
    "TwoLayerNet": TwoLayerNet
}


class ExpSetting():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='experiment settings')
        self.parser.add_argument('-T', '--num_round', type=int, default=10, \
            help='number of rounds')
        self.parser.add_argument('--num_client', type=int, default=10, \
            help='number of clients')
        self.parser.add_argument('-E', '--round_client', type=int, default=200, \
            help='number of rounds on clients')
        self.parser.add_argument('-B', '--size_batch', type=int, default=32, \
            help='batch size b on client')
        self.parser.add_argument('--eta_l', type=float, default=1e-3, \
            help='learning rate in client update')
        self.parser.add_argument('--algo', choices=['fedavg', 'fedopt'], \
            help='choose algorithm for FL learning')
        
        
    def get_options(self):
        args = self.parser.parse_args()
        print(f'running algorithm {args.algo}...')
        return args.num_round, args.num_client, args.round_client, args.size_batch, args.eta_l, args.algo
    
 