import argparse

class ExpSetting():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='experiment settings')


        self.parser.add_argument('-T', '--num_round', type=int, default=64, \
            help='number of rounds')
        self.parser.add_argument('--num_client', type=int, default=10, \
            help='number of clients')
        self.parser.add_argument('-K', '--round_client', type=int, default=1, \
            help='number of rounds on clients')
        self.parser.add_argument('-b', '--size_batch', type=int, default=32, \
            help='batch size b on client')
        
        
    def get_options(self):
        args = self.parser.parse_args()
        return args.num_round, args.num_client, args.round_client, args.size_batch
    
 