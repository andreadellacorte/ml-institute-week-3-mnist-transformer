# Import necessary libraries
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torchvision import datasets, transforms  # Datasets and image transformations
from torch.utils.data import DataLoader  # Data loading utilities
from simp_transform_setup import MNISTTransformer  # Import only MNISTTransformer
import wandb  # Weights & Biases for experiment tracking
import numpy as np  # Numerical computing

# Ask user to choose model type
print("\nChoose which model to use:")
print("1. Library Transformer")
print("2. Custom MyEncoder")
choice = input("Enter 1 or 2: ")

while choice not in ['1', '2']:
    print("Invalid choice. Please enter 1 or 2.")
    choice = input("Enter 1 or 2: ")

option = 'my_own' if choice == '2' else 'library'

# Initialize Weights & Biases for experiment tracking
# This creates a new run in the specified project
wandb.init(project="mnist-transformer", entity=None)  # Replace None with your wandb username if needed

# Set the device (GPU if available, otherwise CPU)
# This determines where the computations will be performed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameters in a configuration dictionary
# These parameters control the model architecture and training process
config = {
    "batch_size": 64,  # Number of samples processed in one forward/backward pass
    "learning_rate": 0.001,  # Step size for optimization
    "num_epochs": 10,  # Number of complete passes through the training dataset
    "img_size": 28,  # Size of MNIST images (28x28)
    "patch_size": 7,  # Size of patches for the transformer
    "emb_dim": 64,  # Dimension of the embedding space
    "num_heads": 4,  # Number of attention heads in the transformer
    "num_layers": 2,  # Number of transformer encoder layers
    "num_classes": 10,  # Number of output classes (digits 0-9)
    "option": option  # Add the option parameter
}

# Log the configuration to Weights & Biases
# This allows tracking of hyperparameters across experiments
wandb.config.update(config)

# Define image transformations for preprocessing
# These transformations are applied to each image before training
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean and std
])

# Load the MNIST dataset
# train=True for training set, train=False for test set
# download=True will download the dataset if not already present
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders for efficient batch processing
# DataLoader handles batching, shuffling, and parallel loading
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize the model with the chosen option
print(f"Using {'Custom MyEncoder' if option == 'my_own' else 'Library Transformer'} model")
model = MNISTTransformer(
    img_size=config["img_size"],
    patch_size=config["patch_size"],
    emb_dim=config["emb_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    num_classes=config["num_classes"],
    option=option  # Pass the option parameter
).to(device)

# Set up Weights & Biases to watch the model
# This enables tracking of gradients and parameters
wandb.watch(model, log="all")

# Define the loss function and optimizer
# CrossEntropyLoss is suitable for multi-class classification
# Adam optimizer is used for parameter updates
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Training function for one epoch
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()  # Set model to training mode
    total_loss = 0  # Accumulator for total loss
    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total samples
    
    # Iterate over batches in the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to the appropriate device
        data, target = data.to(device), target.to(device)
        
        # Zero the gradients before forward pass
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        output = model(data)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        _, predicted = output.max(1)  # Get predicted class
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Log batch metrics every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                "train_batch_loss": loss.item(),
                "train_batch_accuracy": 100. * correct / total,
                "epoch": epoch,
                "batch": batch_idx
            })
            print(f'Train Epoch: [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
    
    # Return average loss and accuracy for the epoch
    return total_loss / len(train_loader), 100. * correct / total

# Evaluation function for the test set
def evaluate(model, test_loader, criterion, device, epoch):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Return average loss and accuracy for the test set
    return total_loss / len(test_loader), 100. * correct / total

# Main training loop
for epoch in range(config["num_epochs"]):
    print(f'\nEpoch: {epoch+1}/{config["num_epochs"]}')
    
    # Training phase
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
    
    # Evaluation phase
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, epoch)
    
    # Log epoch metrics to Weights & Biases
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })
    
    # Print epoch results
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# Save the trained model
# This saves only the model parameters, not the entire model
model_save_name = f'mnist_transformer_{option}.pth'
torch.save(model.state_dict(), model_save_name)
print(f"Model saved to {model_save_name}")

# Finish the Weights & Biases run
wandb.finish()
