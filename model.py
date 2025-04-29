import torch
import torch.nn.init as init
import torch.nn.functional as F
import math

class ProjectionLayer(torch.nn.Module):
    def __init__(self, emb_dim, num_patches):
        super(ProjectionLayer, self).__init__()
        self.linear = torch.nn.Linear(int(28*28/num_patches), emb_dim)

        # Initialize weights and biases
        init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

class PositionalLayer(torch.nn.Module):
    def __init__(self, emb_dim, num_patches):
        super(PositionalLayer, self).__init__()

        self.pos_embed = torch.nn.Parameter(torch.randn(1, num_patches, emb_dim))

    def forward(self, x):
        return x + self.pos_embed

class TransformerEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            batch_first=True)

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

class OutputLayer(torch.nn.Module):
    def __init__(self, emb_dim, num_classes):
        super(OutputLayer, self).__init__()
        self.linear = torch.nn.Linear(emb_dim, num_classes)

        init.zeros_(self.linear.weight)
        init.constant_(self.linear.bias, -math.log(num_classes))  # Set bias for balanced logits

    def forward(self, x):
        # x is a matrix of (batch_size, num_patches, emb_dim)
        # For every batch, take the first emb_dim out of num_patches
        # and apply the linear layer
        x = x[:, 0, :]
        return self.linear(x)