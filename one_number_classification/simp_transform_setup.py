import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, emb_dim=64, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(patch_size * patch_size, emb_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        # Transformer encoder
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

    @staticmethod
    def loss_fn(predictions, targets):
        return F.cross_entropy(predictions, targets)

    @staticmethod
    def get_digit_embeddings(model, dataloader, device):
        model.eval()
        embeddings = [[] for _ in range(10)]

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Same forward logic up to transformer output
                batch_size = images.size(0)
                x = images.unfold(2, model.patch_size, model.patch_size).unfold(3, model.patch_size, model.patch_size)
                x = x.contiguous().view(batch_size, 1, -1, model.patch_size * model.patch_size)
                x = x.squeeze(1)

                x = model.patch_embed(x) + model.pos_embed
                x = model.transformer(x)
                pooled = x.mean(dim=1)  # Get the embedding

                for emb, label in zip(pooled, labels):
                    embeddings[label.item()].append(emb.cpu())

        # Average embeddings per digit
        digit_embeddings = [torch.stack(e).mean(0) if e else torch.zeros(model.emb_dim) for e in embeddings]
        return digit_embeddings
    
    
