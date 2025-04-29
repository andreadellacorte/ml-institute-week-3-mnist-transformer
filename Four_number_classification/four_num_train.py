import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
from PIL import Image
import random
import matplotlib.pyplot as plt

# Custom dataset for multiple MNIST digits
class MultiMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, num_digits=4, transform=None):
        self.mnist_dataset = mnist_dataset
        self.num_digits = num_digits
        self.transform = transform
        
    def __len__(self):
        return len(self.mnist_dataset) // self.num_digits
        
    def __getitem__(self, idx):
        # Create a blank canvas
        canvas = Image.new('L', (56, 56), 0)  # 2x2 grid of 28x28 digits
        
        # Select random digits and their positions
        digits = []
        positions = [(0, 0), (0, 28), (28, 0), (28, 28)]  # 2x2 grid positions
        random.shuffle(positions)
        
        for i in range(self.num_digits):
            # Get a random MNIST image
            img, label = self.mnist_dataset[idx * self.num_digits + i]
            digits.append(label)
            
            # Convert tensor to PIL Image for pasting
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            
            # Define the region to paste (x1, y1, x2, y2)
            x, y = positions[i]
            region = (x, y, x + 28, y + 28)
            
            # Paste the digit onto the canvas at the specified position
            canvas.paste(img, region)
        
        if self.transform:
            canvas = self.transform(canvas)
            
        return canvas, torch.tensor(digits)

# Multi-digit Transformer model
class MultiDigitTransformer(nn.Module):
    def __init__(self, img_size=56, patch_size=7, emb_dim=128, num_heads=8, num_layers=4, num_digits=4):
        super().__init__()
        self.num_digits = num_digits
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Compress patches into smaller dimension
        self.patch_compression = nn.Sequential(
            nn.Linear(patch_size * patch_size, 32),  # Compress 49 -> 32
            nn.GELU(),
            nn.Linear(32, emb_dim)  # Project to final embedding dimension
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))
        
        # Position-specific CLS tokens
        self.cls_tokens = nn.Parameter(torch.randn(num_digits, 1, emb_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification heads (one for each digit position)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, 10)
            ) for _ in range(num_digits)
        ])
        
    def forward(self, x):
        # x shape: (B, 1, 56, 56)
        batch_size = x.size(0)
        
        # Split into patches and flatten
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)
        
        # Compress patches and project to embedding space
        x = self.patch_compression(x)  # (B, num_patches, emb_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Add CLS tokens for each position
        cls_tokens = self.cls_tokens.repeat(batch_size, 1, 1)  # (B, num_digits, emb_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_digits + num_patches, emb_dim)
        
        # Transformer encoding
        x = self.transformer(x)  # (B, num_digits + num_patches, emb_dim)
        
        # Get predictions for each digit position using their specific CLS tokens
        outputs = []
        for i in range(self.num_digits):
            cls_token = x[:, i, :]  # Get the i-th CLS token
            output = self.cls_heads[i](cls_token)  # (B, 10)
            outputs.append(output)
            
        return torch.stack(outputs, dim=1)  # (B, num_digits, 10)

# Initialize wandb
wandb.init(project="multi-digit-mnist", entity=None)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 20,
    "img_size": 56,  # 2x2 grid of 28x28 digits
    "patch_size": 7,
    "emb_dim": 64,  # Reduced to be smaller than patch_size * patch_size (49)
    "num_heads": 8,
    "num_layers": 4,
    "num_digits": 4,
    "train_frac": 0.5  # Use half of the data for training
}

wandb.config.update(config)

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)  # No transform initially

# Create multi-digit dataset
multi_dataset = MultiMNISTDataset(mnist_dataset, num_digits=config["num_digits"], transform=transform)
train_size = int(config["train_frac"] * len(multi_dataset))
val_size = len(multi_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(multi_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize model
model = MultiDigitTransformer(
    img_size=config["img_size"],
    patch_size=config["patch_size"],
    emb_dim=config["emb_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    num_digits=config["num_digits"]
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# After creating the data loaders, before training starts
# Get a single batch
sample_batch = next(iter(train_loader))
images, labels = sample_batch

# Take the first image from the batch
sample_image = images[0].squeeze().numpy()  # Remove channel dimension and convert to numpy

# Create a figure to display the original image and its patches
plt.figure(figsize=(15, 10))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(sample_image, cmap='gray')
plt.title('Original Image (56x56)')
plt.axis('off')

# Plot the patches
plt.subplot(1, 2, 2)
patch_size = 7
num_patches = 8  # 56/7 = 8 patches per dimension
patches = np.zeros((num_patches, num_patches, patch_size, patch_size))

for i in range(num_patches):
    for j in range(num_patches):
        patches[i, j] = sample_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

# Create a grid of patches
grid = np.vstack([np.hstack([patches[i, j] for j in range(num_patches)]) for i in range(num_patches)])

plt.imshow(grid, cmap='gray')
plt.title('Patches (7x7 each)')
plt.axis('off')

# Add grid lines to separate patches
for i in range(1, num_patches):
    plt.axvline(x=i*patch_size-0.5, color='red', linestyle='-', alpha=0.3)
    plt.axhline(y=i*patch_size-0.5, color='red', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('patch_visualization.png')
plt.close()

print("Patch visualization saved as 'patch_visualization.png'")
print(f"Image shape: {sample_image.shape}")
print(f"Number of patches: {num_patches}x{num_patches} = {num_patches*num_patches}")
print(f"Patch size: {patch_size}x{patch_size}")

# Training function
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)  # (B, num_digits, 10)
        
        # Reshape for loss calculation
        outputs = outputs.view(-1, 10)  # (B * num_digits, 10)
        targets = targets.view(-1)  # (B * num_digits)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
            wandb.log({
                "train_batch_loss": loss.item(),
                "train_batch_accuracy": 100. * total_correct / total_samples,
                "epoch": epoch,
                "batch": batch_idx
            })
    
    return total_loss / len(train_loader), 100. * total_correct / total_samples

# Validation function
def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            outputs = outputs.view(-1, 10)
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
    
    return total_loss / len(val_loader), 100. * total_correct / total_samples

# Training loop
for epoch in range(config["num_epochs"]):
    print(f'\nEpoch: {epoch+1}/{config["num_epochs"]}')
    
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
    val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

# Save the model
torch.save(model.state_dict(), 'multi_digit_transformer.pth')
print("Model saved to multi_digit_transformer.pth")

wandb.finish() 