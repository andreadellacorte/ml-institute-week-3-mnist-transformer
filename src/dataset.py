import torch
import math
import numpy as np
import random
from typing import Tuple, List, Dict

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
            
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self,
                    idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        image = self.images[idx]
        label = self.labels[idx]

        return torch.tensor(image, dtype=torch.float32, device=self.device), torch.tensor(label, dtype=torch.long, device=self.device)

# Define a custom Dataset class
class MNISTDatasetCluster(torch.utils.data.Dataset):
    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        word_to_index: Dict[str, int],
        max_sequence_length: int,
        normalize: bool = True,
        rescale: bool = True,
        cluster_size_min: int = 1,
        cluster_size_max: int = None,
    ):
        self.images = images
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_sequence_length = max_sequence_length
        self.normalize = normalize
        self.rescale = rescale
        self.cluster_size_min = cluster_size_min
        self.cluster_size_max = cluster_size_max or max_sequence_length

        if cluster_size_min < 1 or cluster_size_min > cluster_size_max:
            raise ValueError("cluster_size_min must be >= 1 and <= cluster_size_max")

        if rescale:
            self.images = self.images / 255.0

        if normalize:
            self.mean = np.mean(self.images)
            self.std = np.std(self.images)
            self.images = (self.images - self.mean) / self.std
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get a random cluster size
        cluster_size = random.randint(self.cluster_size_min, self.cluster_size_max)
        
        # Get random indices for the cluster
        cluster_indices = random.sample(range(len(self.images)), cluster_size)
        
        # Get the images and labels for this cluster
        cluster_images = [self.images[i] for i in cluster_indices]
        cluster_labels = [self.labels[i] for i in cluster_indices]
        
        # Concatenate images horizontally
        image = np.concatenate(cluster_images, axis=2)
        current_width = image.shape[2]
        target_width = self.max_sequence_length * 28
        
        # Add padding if needed
        if current_width < target_width:
            padding_width = target_width - current_width
            padding = np.zeros((28, 1, padding_width))
            image = np.concatenate((image, padding), axis=2)
            
            # Calculate how many pad tokens we need (one for each digit-width of padding)
            num_pad_tokens = (target_width - current_width) // 28
            pad_token = self.word_to_index["<pad>"]
            cluster_labels = np.concatenate((cluster_labels, [pad_token] * num_pad_tokens))
        elif current_width > target_width:
            raise ValueError(f"Image width {current_width} exceeds target width {target_width}. " +
                           f"Cluster size: {cluster_size}, Cluster indices: {cluster_indices}")

        # Check that image has the correct dimensions
        if image.shape != (28, 1, self.max_sequence_length * 28):
            raise ValueError(f"Final image shape is {image.shape}, expected (28, 1, {self.max_sequence_length*28}). " +
                           f"Started with {len(cluster_images)} images of shapes {[img.shape for img in cluster_images]}")

        # add end token at end of cluster_labels
        end_token = self.word_to_index["<end>"]
        cluster_labels = np.concatenate((cluster_labels, [end_token]))

        # Convert to PyTorch tensors and move to correct device
        image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(cluster_labels, dtype=torch.long, device=self.device)

        return image_tensor, labels_tensor