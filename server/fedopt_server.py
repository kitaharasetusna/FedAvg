from server.serverbase import ServerBase


class ServerOPT(ServerBase):
    def __init__(self, network, train_data, num_clients, E, client_batch_size, \
        learning_rate, device, shards_num):
        super().__init__(network, train_data, num_clients, E, client_batch_size, \
            learning_rate, device, shards_num)


