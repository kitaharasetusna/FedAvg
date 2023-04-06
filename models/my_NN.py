
import torch
import torch.nn as nn

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