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
        canvas = Image.new('L', (28 * self.num_digits, 28), 0)
        
        # Select random digits and their positions
        digits = []
        positions = list(range(self.num_digits))
        random.shuffle(positions)
        
        for i in range(self.num_digits):
            # Get a random MNIST image
            img, label = self.mnist_dataset[idx * self.num_digits + i]
            digits.append(label)
            
            # Paste the digit onto the canvas
            canvas.paste(img, (positions[i] * 28, 0))
        
        if self.transform:
            canvas = self.transform(canvas)
            
        return canvas, torch.tensor(digits)

# Multi-digit Transformer model
class MultiDigitTransformer(nn.Module):
    def __init__(self, img_size=112, patch_size=7, emb_dim=128, num_heads=8, num_layers=4, num_digits=4):
        super().__init__()
        self.num_digits = num_digits
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding with Conv2D
        self.patch_embed = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification heads (one for each digit position)
        self.cls_heads = nn.ModuleList([
            nn.Linear(emb_dim, 10) for _ in range(num_digits)
        ])
        
    def forward(self, x):
        # x shape: (B, 1, 112, 112)
        batch_size = x.size(0)
        
        # Create patch embeddings using Conv2D
        x = self.patch_embed(x)  # (B, emb_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, emb_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)  # (B, num_patches, emb_dim)
        
        # Get predictions for each digit position
        outputs = []
        for head in self.cls_heads:
            # Use the first token for classification (like BERT's [CLS] token)
            cls_token = x[:, 0, :]  # (B, emb_dim)
            output = head(cls_token)  # (B, 10)
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
    "img_size": 112,  # 28 * 4
    "patch_size": 7,
    "emb_dim": 128,
    "num_heads": 8,
    "num_layers": 4,
    "num_digits": 4
}

wandb.config.update(config)

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create multi-digit dataset
multi_dataset = MultiMNISTDataset(mnist_dataset, num_digits=config["num_digits"], transform=transform)
train_size = int(0.8 * len(multi_dataset))
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