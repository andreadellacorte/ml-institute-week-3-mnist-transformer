import torch
import random
import numpy as np
from dataset import MNISTDataset
from model import MNISTModel
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import math

# Set random seed for reproducibility
random_seed = 3407 # https://arxiv.org/abs/2109.08203
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Verify loss at initialization

# Calculate expected initial loss for CrossEntropyLoss
n_classes = 10
expected_loss = -math.log(1 / n_classes)

# Verify initial loss
model = MNISTModel(emb_dim=32)
model.eval()
with torch.no_grad():
    dummy_input = torch.zeros((1, 1, 28, 28))  # Batch size 1, single-channel 28x28 image
    logits = model(dummy_input)
    loss_fn = torch.nn.CrossEntropyLoss()
    dummy_target = torch.tensor([0])  # Arbitrary class label
    initial_loss = loss_fn(logits, dummy_target)

print(f"Expected initial loss: {expected_loss:.6f}, Measured initial loss: {initial_loss.item():.6f}")

# Load MNIST dataset at hugging face ylecun/mnist
mnist = load_dataset("ylecun/mnist")

# Extract training data
train_data = mnist['train']
test_data = mnist['test']

# Create Dataset instances
train_dataset = MNISTDataset(train_data['image'], train_data['label'])
test_dataset = MNISTDataset(test_data['image'], test_data['label'])

# DataLoaders
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
emb_dim = 32
learning_rate = 1e-3

model = MNISTModel(emb_dim)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training and evaluation loop
epochs = 5
for epoch in range(epochs):
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f'Pre-Epoch {epoch+1} | Test Accuracy: {accuracy:.6f}')
    
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Perform evaluation over the entire test set for correctness
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)  # Accumulate loss weighted by batch size
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_test_loss = total_loss / total
    print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | Test Accuracy: {accuracy:.6f}')