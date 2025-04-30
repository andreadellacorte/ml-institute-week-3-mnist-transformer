import torch
import math
import numpy as np
from typing import Tuple

# Define a custom Dataset class
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, normalize_dataset, num_patches):
        self.num_patches = num_patches
        self.images = np.array(images)
        self.labels = labels

        if normalize_dataset:
            self.mean = np.mean(self.images)
            self.std = np.std(self.images)
            self.images = (self.images - self.mean) / self.std  # Normalize images

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self,
                    idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunks_per_size = math.sqrt(self.num_patches)

        # throw error if num_patches is not a perfect square
        if chunks_per_size != int(chunks_per_size):
            raise ValueError("num_patches must be a perfect square")
        
        if self.images.shape[1] % chunks_per_size != 0:
            raise ValueError("Image size must be divisible by the square root of num_patches")
        
        # Cast to int without changing the value
        chunks_per_size = int(chunks_per_size)
        pixels_per_chunk = self.images.shape[1] // int(chunks_per_size)

        image = self.images[idx]
        label = self.labels[idx]

        patches = []
        for j in range(self.num_patches):
            row_start = (j // chunks_per_size) * pixels_per_chunk
            col_start = (j % chunks_per_size) * pixels_per_chunk
            image = np.array(image)  # Convert to NumPy array
            patch = image[row_start:row_start+pixels_per_chunk, col_start:col_start+pixels_per_chunk]
            patches.append(patch.flatten())

        # Convert patches list to a single numpy array before creating a tensor
        patches = np.array(patches)
        return torch.tensor(patches, dtype=torch.float32), torch.tensor(label, dtype=torch.long)