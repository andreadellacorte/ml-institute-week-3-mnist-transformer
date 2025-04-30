import torch
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
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

class EncoderLayer(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers):
        super(EncoderLayer, self).__init__()
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            batch_first=True)

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

class MyEncoderLayer(torch.nn.Module):
    def __init__(self, emb_dim, encoder_emb_dim):
        super(MyEncoderLayer, self).__init__()

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.wV = torch.nn.Linear(emb_dim, encoder_emb_dim)
        self.wK = torch.nn.Linear(emb_dim, encoder_emb_dim)
        self.wQ = torch.nn.Linear(emb_dim, encoder_emb_dim)
        self.w0 = torch.nn.Linear(encoder_emb_dim, emb_dim)
        self.encoder_emb_dim = encoder_emb_dim

        # Initialize weights and biases
        init.kaiming_uniform_(self.wV.weight, nonlinearity='relu')
        init.zeros_(self.wV.bias)

        init.kaiming_uniform_(self.wK.weight, nonlinearity='relu')
        init.zeros_(self.wK.bias)

        init.kaiming_uniform_(self.wQ.weight, nonlinearity='relu')
        init.zeros_(self.wQ.bias)

        init.kaiming_uniform_(self.w0.weight, nonlinearity='relu')
        init.zeros_(self.w0.bias)

    def forward(self, x):
        # normalise x
        x = self.layer_norm(x)

        q = self.wQ(x)
        k = self.wK(x)

        v = self.wV(x)        

        # calculate a from Q * K^T
        scores = q @ k.transpose(-2, -1) / np.sqrt(self.encoder_emb_dim)

        # Create a mask to set the bottom-left section of weights to -inf
        mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1)), diagonal=-1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        a = F.softmax(scores, dim=-1)

        h = a @ v

        return self.w0(h)

class OutputLayer(torch.nn.Module):
    def __init__(self, emb_dim, num_classes):
        super(OutputLayer, self).__init__()
        self.linear = torch.nn.Linear(emb_dim, num_classes)

        init.zeros_(self.linear.weight)
        init.constant_(self.linear.bias, -math.log(num_classes))  # Set bias for balanced logits

    def forward(self, x):
        # x is a matrix of (batch_size, num_patches, emb_dim)
        # needs to be flattened to (batch_size, emb_dim)

        # Either do the mean across the num_patches dimension
        # x = x.mean(dim=1)

        # or take the first patch
        x = x[:, 0, :]
        return self.linear(x)