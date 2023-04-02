import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from torch.multiprocessing import Process, Manager

# Define the two-layer neural network as the local model
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the client update function
def client_update(model, optimizer, train_loader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return model.state_dict(), running_loss / len(train_loader.dataset)

# Define the server update function
def server_update(models):
    new_state_dict = {}
    for key in models[0].keys():
        new_state_dict[key] = torch.stack([models[i][key] for i in range(len(models))], dim=0).mean(dim=0)
    return new_state_dict

# Define the main Federated Learning function
def federated_learning(num_rounds, train_data, num_clients, client_batch_size, learning_rate, device, stopping_event, global_model_dict):
    # Split the train data into subsets for each client
    subset_indices = torch.linspace(0, len(train_data)-1, steps=num_clients+1).round().tolist()
    client_datasets = [Subset(train_data, range(int(subset_indices[i]), int(subset_indices[i+1]))) for i in range(num_clients)]
    client_loaders = [DataLoader(client_datasets[i], batch_size=client_batch_size, shuffle=True) for i in range(num_clients)]

    # Initialize the global model
    global_model = TwoLayerNet(input_size=28*28, hidden_size=32, output_size=10).to(device)
    global_model.train()

    # Initialize the optimizer
    optimizer = optim.SGD(global_model.parameters(), lr=learning_rate)

    for round in range(num_rounds):
        if stopping_event.is_set():
            break
        print(f"Round {round+1} started...")
        client_models = []
        client_losses = []
        for i in range(num_clients):
            client_model = TwoLayerNet(input_size=28*28, hidden_size=32, output_size=10).to(device)
            client_model.load_state_dict(global_model.state_dict())
            client_optimizer = optim.SGD(client_model.parameters(), lr=learning_rate)
            client_state_dict, client_loss = client_update(client_model, client_optimizer, client_loaders[i], device)
            client_models.append(client_state_dict)
            client_losses.append(client_loss)
            print(f"Client {i+1} loss: {client_loss:.4f}")
        global_state_dict = server_update(client_models)
        global_model.load_state_dict(global_state_dict)
        print(f"Round {round+1} finished, global loss: {sum(client_losses)/len(client_losses):.4f}")
        global_model_dict[round+1] = global_model.state_dict()

    return global_model

if __name__ == '__main__':
    # Set the hyperparameters
    num_rounds = 10
    num_clients = 10
    client_batch_size = 32
    learning_rate = 0.1

    # Load the MNIST dataset
    train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a Manager to share state between processes
    manager = Manager()
    stopping_event = manager.Event()
    global_model_dict = manager.dict()

    # Run the Federated Learning process on multiple processes
    processes = []
    for i in range(4):
        p = Process(target=federated_learning, args=(num_rounds, train_data, num_clients, client_batch_size, learning_rate, device, stopping_event, global_model_dict))
        p.start()
        processes.append(p)

    # Wait for user to stop the program
    try:
        while True:
            pass
    except KeyboardInterrupt:
        stopping_event.set()

    # Join the processes and retrieve the final global model
    for p in processes:
        p.join()

    global_model_state_dict = server_update(list(global_model_dict.values()))
    global_model = TwoLayerNet(input_size=28*28, hidden_size=32, output_size=10).to(device)
    global_model.load_state_dict(global_model_state_dict)
    print("Training completed!")
