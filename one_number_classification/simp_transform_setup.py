import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MyEncoder(nn.Module):
    def __init__(self, emb_dim=128, weight_col=24, num_heads=8, num_layers=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension of each head
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"
        
        # Linear projections for queries, keys, and values for all heads
        self.wV = nn.Linear(emb_dim, emb_dim)  # Project to emb_dim for all heads
        self.wQ = nn.Linear(emb_dim, emb_dim)  # Project to emb_dim for all heads
        self.wK = nn.Linear(emb_dim, emb_dim)  # Project to emb_dim for all heads
        self.wO = nn.Linear(emb_dim, emb_dim)  # Project back to emb_dim
        
    def forward(self, x):
        batch_size = x.size(0)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        # Project inputs to queries, keys, and values
        V = self.wV(x)  # (batch_size, seq_len, emb_dim)
        Q = self.wQ(x)  # (batch_size, seq_len, emb_dim)
        K = self.wK(x)  # (batch_size, seq_len, emb_dim)
        
        # Reshape for multi-head attention
        # Split the embedding dimension into num_heads heads: This allows for parallel attention computations
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        A = torch.softmax(scores, dim=-1)  # attention weights
        
        # Apply attention to values
        H = A @ V  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back to original dimensions
        H = H.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)  # (batch_size, seq_len, emb_dim)
        
        # Final projection
        O = self.wO(H)  # (batch_size, seq_len, emb_dim)
        return O

class MNISTTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, emb_dim=64, num_heads=4, num_layers=2, num_classes=10, option='library'):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(patch_size * patch_size, emb_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        # Choose transformer implementation based on option
        if option == 'my_own':
            self.transformer = MyEncoder(emb_dim=emb_dim, num_heads=num_heads, num_layers=num_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.cls_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        batch_size = x.size(0)

        # Split into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 1, -1, self.patch_size * self.patch_size)
        x = x.squeeze(1)  # shape: (batch_size, num_patches, patch_size*patch_size)

        # Patch embedding + positional embedding
        x = self.patch_embed(x) + self.pos_embed  # shape: (batch_size, num_patches, emb_dim)

        # Transformer encoding
        x = self.transformer(x)  # shape: (batch_size, num_patches, emb_dim)

        # Aggregate (mean pooling)
        x = x.mean(dim=1)  # shape: (batch_size, emb_dim)

        # Classification
        logits = self.cls_head(x)  # shape: (batch_size, num_classes)
        return logits

    # @staticmethod
    # def loss_fn(predictions, targets):
    #     return F.cross_entropy(predictions, targets)

    # @staticmethod
    # def get_digit_embeddings(model, dataloader, device):
    #     model.eval()
    #     embeddings = [[] for _ in range(10)]

    #     with torch.no_grad():
    #         for images, labels in dataloader:
    #             images = images.to(device)
    #             labels = labels.to(device)

    #             # Same forward logic up to transformer output
    #             batch_size = images.size(0)
    #             x = images.unfold(2, model.patch_size, model.patch_size).unfold(3, model.patch_size, model.patch_size)
    #             x = x.contiguous().view(batch_size, 1, -1, model.patch_size * model.patch_size)
    #             x = x.squeeze(1)

    #             x = model.patch_embed(x) + model.pos_embed
    #             x = model.transformer(x)
    #             pooled = x.mean(dim=1)  # Get the embedding

    #             for emb, label in zip(pooled, labels):
    #                 embeddings[label.item()].append(emb.cpu())

        # Average embeddings per digit
        digit_embeddings = [torch.stack(e).mean(0) if e else torch.zeros(model.emb_dim) for e in embeddings]
        return digit_embeddings
    
    
