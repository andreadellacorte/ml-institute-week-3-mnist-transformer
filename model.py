import torch
import torch.nn as nn
import torch.nn.init as init
import math

class MNISTModel(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 10)

        # Initialize weights and biases
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.zeros_(self.fc1.bias)

        init.zeros_(self.fc2.weight)
        init.constant_(self.fc2.bias, -math.log(num_classes))  # Set bias for balanced logits

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x