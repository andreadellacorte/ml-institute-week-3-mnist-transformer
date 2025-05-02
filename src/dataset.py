import torch
import math
import numpy as np
from typing import Tuple

# Define a custom Dataset class
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, normalize, rescale):
        self.images = np.array(images)
        self.labels = labels

        if rescale:
            self.images = self.images / 255.0

        if normalize:
            self.mean = np.mean(self.images)
            self.std = np.std(self.images)
            self.images = (self.images - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self,
                    idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        image = self.images[idx]
        label = self.labels[idx]

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Define a custom Dataset class
class MNISTDatasetCluster(torch.utils.data.Dataset):
    def __init__(self, images, labels, word_to_index, max_sequence_length, normalize, rescale, cluster_size_min=1, cluster_size_max=1):
        self.images = np.array(images)
        self.labels = np.array(labels)  # Convert labels to a NumPy array

        self.word_to_index = word_to_index
        self.max_sequence_length = max_sequence_length

        self.cluster_size_min = cluster_size_min
        self.cluster_size_max = cluster_size_max

        if cluster_size_min < 1 or cluster_size_min > cluster_size_max:
            raise ValueError("cluster_size_min must be >= 1 and <= cluster_size_max")

        if rescale:
            self.images = self.images / 255.0

        if normalize:
            self.mean = np.mean(self.images)
            self.std = np.std(self.images)
            self.images = (self.images - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate a random cluster size normally distributed from cluster_size_min to cluster_size_max
        cluster_size = np.random.randint(self.cluster_size_min, self.cluster_size_max + 1)
        
        # Randomly select a cluster of images
        cluster_indices = np.random.choice(len(self.images), size=cluster_size, replace=False)
        cluster_images = self.images[cluster_indices]
        cluster_labels = self.labels[cluster_indices]

        # Concatenate the images along the width (axis=1)
        image = np.concatenate(cluster_images, axis=1)
        
        # extend image to max_sequence_length * 28 (since each digit is 28 pixels wide)
        target_width = self.max_sequence_length * 28
        current_width = image.shape[1]
        
        if current_width < target_width:
            padding = np.zeros((28, target_width - current_width))
            image = np.concatenate((image, padding), axis=1)
        elif current_width > target_width:
            raise ValueError(f"Image width {current_width} exceeds target width {target_width}. " +
                           f"Cluster size: {cluster_size}, Cluster indices: {cluster_indices}")

        # Check that image has the correct dimensions
        if image.shape != (28, self.max_sequence_length * 28):
            raise ValueError(f"Final image shape is {image.shape}, expected (28, {self.max_sequence_length*28}). " +
                           f"Started with {len(reshaped_images)} images of shapes {[img.shape for img in reshaped_images]}")

        # add end token at beginning of cluster_labels
        end_token = self.word_to_index["<end>"]
        cluster_labels = np.concatenate((cluster_labels, [end_token]))

        # Convert the image and label to PyTorch tensors
        return torch.tensor(image, dtype=torch.float32), torch.tensor(cluster_labels, dtype=torch.long)