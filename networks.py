import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp



# define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10) 
        
    def forward(self, x):
       x = F.relu(F.max_pool2d(self.conv1(x), 2))
       x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
       x = x.view(-1, 320)
       x = F.relu(self.fc1(x))
       x = F.dropout(x, training=self.training)
       x = self.fc2(x)
       return F.log_softmax(x) 

# define the dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x, y = self.data[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# define the training function
def train(model, dataloader, optimizer):
    model.train()
    for batch_idx, (inputs, labels) in  enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        output_size = outputs.size()
        labels_size  = labels.size()
        print(f'{output_size} {labels_size}')
        loss = F.nll_loss(outputs, labels)
        print(f'loss: {loss.item()}')
        loss.backward()
        optimizer.step()

# define the Federated Averaging function
def fed_avg(model, data, lr=0.01, num_epochs=10, num_clients=4):
    # split the data into num_clients clients
    data_per_client = len(data) // num_clients
    client_data = [data[i:i+data_per_client] for i in range(0, len(data), data_per_client)]
    
    # initialize shared model and optimizer
    shared_model = model
    shared_optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.5) 
    
    # define the training function for each client
    def train_client(client_model, client_data):
        client_dataset = CustomDataset(client_data)
        client_dataloader = DataLoader(client_dataset, batch_size=10, shuffle=True)
        train(client_model, client_dataloader, shared_optimizer)
        
    # define the function to update the shared model with the average of the client models
    def update_shared_model():
        with torch.no_grad():
            for param in shared_model.parameters():
                param.fill_(0)
            for client_model in client_models:
                for param in client_model.parameters():
                    param /= num_clients
                for shared_param, client_param in zip(shared_model.parameters(), client_model.parameters()):
                    shared_param += client_param
    
    # create the client models
    client_models = [Net() for _ in range(num_clients)]
    
    # train each client model
    for i in range(num_epochs):
        processes = []
        for j in range(num_clients):
            
            train_client(client_models[j], client_data[j])
        
        # update the shared model with the average of the client models
        update_shared_model()
        
    return shared_model

