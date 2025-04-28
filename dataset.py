import torch
import numpy as np

# Define a custom Dataset class
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(np.array([np.array(img) for img in images])).float().view(-1, 28*28) / 255.0  # normalize to [0,1]
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]