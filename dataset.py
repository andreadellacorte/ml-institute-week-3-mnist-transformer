import torch
import numpy as np

# Define a custom Dataset class
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, num_patches):
        self.images = torch.tensor(np.array([np.array(img) for img in images])).float().view(-1, 28*28) / 255.0  # normalize to [0,1]
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Train an input-independent baseline
# Create a DataLoader with all inputs set to zero
class ZeroInputDataset(torch.utils.data.Dataset):
    def __init__(self, size, num_classes):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        zero_input = torch.zeros(28*28)  # Input is all zeros
        label = torch.randint(0, self.num_classes, (1,)).item()  # Random label
        return zero_input, label