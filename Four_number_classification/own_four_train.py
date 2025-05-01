import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

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
from tqdm import tqdm

# Configuration
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 20,
    "img_size": 56,  # 2x2 grid of 28x28 digits
    "patch_size": 7,
    "emb_dim": 64,
    "num_heads": 8,
    "num_layers": 4,
    "num_digits": 4,
    "train_frac": 0.9
}

# Custom dataset for multiple MNIST digits
class MultiMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, num_digits=4, transform=None):
        self.mnist_dataset = mnist_dataset
        self.num_digits = num_digits
        self.transform = transform
        self.finish_token = 11  # Index for finish token
        
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
        
        # Append finish token to digits
        digits.append(self.finish_token)
            
        return canvas, torch.tensor(digits)

# Custom encoder class (similar to MyEncoder from simp_transform_setup.py)
class MyEncoder(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.num_layers = num_layers
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"
        
        # Create N encoder layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # Multi-head attention
                'attention': nn.ModuleDict({
                    'wV': nn.Linear(emb_dim, emb_dim),
                    'wQ': nn.Linear(emb_dim, emb_dim),
                    'wK': nn.Linear(emb_dim, emb_dim),
                    'wO': nn.Linear(emb_dim, emb_dim)
                }),
                # Layer normalization
                'norm1': nn.LayerNorm(emb_dim),
                'norm2': nn.LayerNorm(emb_dim),
                # Feed-forward network
                'ffn': nn.Sequential(
                    nn.Linear(emb_dim, emb_dim * 4),  # Expansion
                    nn.GELU(),                        # Non-linear activation
                    nn.Linear(emb_dim * 4, emb_dim)   # Contraction
                )
            }) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process through multiple transformer layers
        for layer in self.layers:
            # Store input for residual connection this helps with training deep neural networks, learning small corrections and avoid the vanishing gradient problem
            #output of a sub-layer (like a multi-head attention or feed-forward network) is added to the original input, allowing gradients to flow through the residual connection
            residual = x
            
            # Multi-head attention
            V = layer['attention']['wV'](x)
            Q = layer['attention']['wQ'](x)
            K = layer['attention']['wK'](x)
            
            # Reshape for multi-head attention
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention scores
            scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            A = torch.softmax(scores, dim=-1)
            
            # Apply attention to values
            H = A @ V
            
            # Reshape back to original dimensions
            H = H.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)
            
            # Final projection (output of attention head)
            O = layer['attention']['wO'](H)
            
            # Add residual connection and apply layer norm
            x = layer['norm1'](residual + O)
            
            # Feed-forward network with residual connection
            residual = x
            x = layer['ffn'](x)
            x = layer['norm2'](residual + x)
        
        return x

def create_mask(size):
    # Create upper triangular matrix of ones
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    # Convert to -inf where mask is 1 (positions to mask out) we do this to avoid the model from seeing the future tokens its no 0 becasue
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Multi-digit Transformer model using custom encoder
class MultiDigitTransformer(nn.Module):
    def __init__(self, img_size=56, patch_size=7, emb_dim=64, num_heads=8, num_layers=4, num_digits=4):
        super().__init__()
        self.num_digits = num_digits
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.num_classes = 12  # 0-9 digits + start + finish
        self.token_dim = 16    # Dimension for digit token embeddings

        # Input embedding for encoder
        self.patch_embed = nn.Linear(patch_size * patch_size, emb_dim)
        self.pos_embed_enc = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))
        
        # Embeddings for each possible digit (0-10) with token_dim
        # Shape: (12, 16) - each digit gets a 16-dimensional embedding
        self.digit_embeddings = nn.Parameter(torch.randn(self.num_classes, self.token_dim))
        
        # Linear projection from token_dim to emb_dim
        # Weight matrix shape: (emb_dim, token_dim) e.g., (64, 16)
        self.token_projection = nn.Linear(self.token_dim, emb_dim)

        # Encoder
        self.encoder = MyEncoder(emb_dim=emb_dim, num_heads=num_heads, num_layers=num_layers)
        
        # Decoder
        self.decoder = MyDecoder(emb_dim=emb_dim, num_heads=num_heads, num_layers=num_layers)
        
        # Final layers
        self.final_layer = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, self.num_classes)
        )
        
    def forward(self, x, targets=None):
        batch_size = x.size(0)
        
        # Encoder processing
        # Split into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, -1, self.patch_size * self.patch_size)
        x = x.squeeze(1)  # shape: (batch_size, num_patches, patch_size*patch_size)

        # Input embedding + positional encoding for encoder
        x = self.patch_embed(x) + self.pos_embed_enc  # shape: (batch_size, num_patches, emb_dim)

        # Encoder processing
        encoder_output = self.encoder(x)
        
        # Prepare decoder input sequence
        # Index 10: start token, Index 11: finish token
        start_token_idx = 10
        
        if targets is not None:
            # Teacher forcing: prepend start token to target sequence except last token
            # targets shape is (B, num_digits + 1) because of finish token
            decoder_input_indices = torch.cat([
                torch.full((batch_size, 1), start_token_idx, dtype=targets.dtype, device=targets.device),
                targets[:, :-1]  # exclude finish token for input
            ], dim=1)
        else:
            # Inference: just use start token as first input
            decoder_input_indices = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=x.device)

        # Get embeddings for decoder input
        decoder_input = self.digit_embeddings[decoder_input_indices]
        decoder_input = self.token_projection(decoder_input)

        # Create mask for decoder (now needs to be size of input sequence)
        mask = create_mask(decoder_input.size(1)).to(x.device)

        # Decoder processing
        decoder_output = self.decoder(decoder_input, encoder_output, mask)

        # Final prediction
        outputs = self.final_layer(decoder_output)
        return outputs  # (B, seq_len, num_classes) where seq_len = num_digits + 1

class MyDecoder(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.num_layers = num_layers
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"
        
        # Create N decoder layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # Masked Multi-head attention
                'masked_attention': nn.ModuleDict({
                    'wV': nn.Linear(emb_dim, emb_dim),
                    'wQ': nn.Linear(emb_dim, emb_dim),
                    'wK': nn.Linear(emb_dim, emb_dim),
                    'wO': nn.Linear(emb_dim, emb_dim)
                }),
                # Encoder-Decoder attention
                'enc_dec_attention': nn.ModuleDict({
                    'wV': nn.Linear(emb_dim, emb_dim),
                    'wQ': nn.Linear(emb_dim, emb_dim),
                    'wK': nn.Linear(emb_dim, emb_dim),
                    'wO': nn.Linear(emb_dim, emb_dim)
                }),
                # Layer normalizations
                'norm1': nn.LayerNorm(emb_dim),
                'norm2': nn.LayerNorm(emb_dim),
                'norm3': nn.LayerNorm(emb_dim),
                # Feed-forward network
                'ffn': nn.Sequential(
                    nn.Linear(emb_dim, emb_dim * 4),  # Expansion
                    nn.GELU(),                        # Non-linear activation
                    nn.Linear(emb_dim * 4, emb_dim)   # Contraction
                )
            }) for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output, mask=None):
        batch_size = x.size(0)
        
        # Process through multiple decoder layers
        for layer in self.layers:
            # Step 1: Masked Multi-head Attention
            residual = x
            
            # Project to queries, keys, and values
            V = layer['masked_attention']['wV'](x)
            Q = layer['masked_attention']['wQ'](x)
            K = layer['masked_attention']['wK'](x)
            
            # Reshape for multi-head attention
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention scores
            scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            
            # Apply mask if provided
            if mask is not None:
                scores = scores + mask
            else:
                print ("mask is none")
            
            A = torch.softmax(scores, dim=-1)
            
            # Apply attention to values
            H = A @ V
            
            # Reshape back to original dimensions
            H = H.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)
            
            # Final projection
            O = layer['masked_attention']['wO'](H)
            
            # Add residual connection and apply layer norm
            x = layer['norm1'](residual + O)
            
            # Step 2: Encoder-Decoder Attention
            residual = x
            
            # Project encoder output to keys and values
            V_enc = layer['enc_dec_attention']['wV'](encoder_output)
            K_enc = layer['enc_dec_attention']['wK'](encoder_output)
            
            # Project decoder output to queries
            Q_dec = layer['enc_dec_attention']['wQ'](x)
            
            # Reshape for multi-head attention
            V_enc = V_enc.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K_enc = K_enc.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            Q_dec = Q_dec.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention scores
            scores = (Q_dec @ K_enc.transpose(-2, -1)) / np.sqrt(self.head_dim)
            A = torch.softmax(scores, dim=-1)
            
            # Apply attention to values
            H = A @ V_enc
            
            # Reshape back to original dimensions
            H = H.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)
            
            # Final projection
            O = layer['enc_dec_attention']['wO'](H)
            
            # Add residual connection and apply layer norm
            x = layer['norm2'](residual + O)
            
            # Step 3: Feed Forward Network
            residual = x
            x = layer['ffn'](x)
            x = layer['norm3'](residual + x)
        
        return x

# Initialize wandb
wandb.init(project="multi-digit-mnist-custom", entity=None)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

wandb.config.update(config)

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

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

# Training function
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (data, targets) in progress_bar:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data, targets)  # (B, num_digits + 1, num_classes)
        
        # Reshape for loss calculation
        batch_size = outputs.size(0)
        seq_len = outputs.size(1)  # num_digits + 1 (including finish token)
        outputs = outputs.reshape(batch_size * seq_len, -1)  # (B * (num_digits + 1), num_classes)
        targets = targets.reshape(-1)  # (B * (num_digits + 1))
        
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)
        
        progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(train_loader), 100. * total_correct / total_samples

# Validation function
def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1} [Val]")
    with torch.no_grad():
        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data, targets)  # (B, seq_len, num_classes)
            
            # Reshape for loss calculation
            batch_size = outputs.size(0)
            seq_len = outputs.size(1)  # num_digits + 1 (including finish token)
            outputs = outputs.reshape(batch_size * seq_len, -1)  # (B * (num_digits + 1), num_classes)
            targets = targets.reshape(-1)  # (B * (num_digits + 1))
                 
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            progress_bar.set_postfix(loss=loss.item())
    
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
torch.save(model.state_dict(), 'multi_digit_transformer_custom.pth')
print("Model saved to multi_digit_transformer_custom.pth")

wandb.finish()
