import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a function to train a model on a dataset
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Define a function to test a model on a dataset
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy

# Define a function to run in each process
def run_process(train_set, test_set, process_id, num_processes):
    # Initialize a new model
    model = MyModel().cuda()

    # Define the optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Create a data loader for the subset of the dataset assigned to this process
    train_sampler = SubsetRandomSampler(range(process_id, len(train_set), num_processes))
    train_loader = DataLoader(train_set, batch_size=64, sampler=train_sampler)

    # Train the model on the subset of the dataset assigned to this process
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, epoch, 'cuda')

    # Test the model on the full test set
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
    accuracy = test(model, test_loader, 'cuda')

    # Return the accuracy and state_dict to the main process
    return (process_id, accuracy, model.state_dict())

if __name__ == '__main__':
    # Initialize the dataset and data loaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Initialize a process pool with one process per available GPU
    num_processes = torch.cuda.device_count()
    pool = mp.Pool(num_processes)

    # Launch a process for each available GPU
    processes = []
    for i in range(num_processes):
        process = pool.apply_async(run_process, args=(train_set, test_set, i, num_processes))
        processes.append(process)

    # Collect the results from the subprocesses
    state_dicts = []
    accuracies = []
    for process in processes:
        pid, accuracy, state_dict = process.get()
        accuracies.append(accuracy)
        state_dicts.append(state_dict)

    # Average the state_dicts
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = torch.mean(torch.stack([sd[key] for sd in state_dicts]), dim=0)

    # Load the averaged state_dict back into the original model
    model = MyModel().cuda()
    model.load_state_dict(avg_state_dict)

    # Test the model on the full test set
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)
    accuracy = test(model, test_loader, 'cuda')

    # Print the average accuracy across all GPUs
    print('Average accuracy: {:.2f}%'.format(sum(accuracies) / len(accuracies)))

