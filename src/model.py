import torch
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math

class TransformerModel(torch.nn.Module):
    def __init__(self, num_classes, emb_dim, num_patches, num_heads, num_layers, encoder_emb_dim, masking, self_attending, output_mechanism):
        super(TransformerModel, self).__init__()

        self.model = torch.nn.Sequential(
            ProjectionLayer(emb_dim, num_patches),
            PositionalLayer(emb_dim, num_patches),
            *[MyEncoderLayer(emb_dim, encoder_emb_dim, masking, self_attending) for _ in range(num_layers)],
            OutputLayer(emb_dim, num_classes, output_mechanism)
        )

    def forward(self, x):
        return self.model(x)

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
    def __init__(self, emb_dim, encoder_emb_dim, masking, self_attending):
        super(MyEncoderLayer, self).__init__()

        self.encoder_emb_dim = encoder_emb_dim
        self.self_attending = self_attending
        self.masking = masking

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.wQ = torch.nn.Linear(emb_dim, encoder_emb_dim)
        self.wK = torch.nn.Linear(emb_dim, encoder_emb_dim)
        self.wV = torch.nn.Linear(emb_dim, encoder_emb_dim)
        self.w0 = torch.nn.Linear(encoder_emb_dim, emb_dim)

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

        # Calculate attention scores from Q * K^T
        scores = q @ k.transpose(-2, -1) / (math.sqrt(self.encoder_emb_dim) + 1e-9)

        # Clamp attention scores to a reasonable range to prevent extreme values
        scores = torch.clamp(scores, min=-1e9, max=1e9)

        if self.masking:
            if self.self_attending:
                # Don't mask the diagonal to -inf
                mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1)), diagonal=-1).bool()
            else:
                # Mask the diagonal to -inf
                mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1))).bool()

            scores = scores.masked_fill(mask, float('-inf'))

            # If a column is all -inf, set it to 0 to avoid
            # NaN values in the softmax
            if self.self_attending is False:
                scores = torch.where(torch.isinf(scores), torch.zeros_like(scores), scores)

        a = F.softmax(scores, dim=-1)

        h = a @ v

        return self.w0(h)

class OutputLayer(torch.nn.Module):
    def __init__(self, emb_dim, num_classes, output_mechanism):
        super(OutputLayer, self).__init__()

        if output_mechanism not in ["mean", "first"]:
            raise ValueError("output_mechanism must be either 'mean' or 'first'")

        self.linear = torch.nn.Linear(emb_dim, num_classes)
        self.output_mechanism = output_mechanism

        init.zeros_(self.linear.weight)
        init.constant_(self.linear.bias, -math.log(num_classes))  # Set bias for balanced logits

    def forward(self, x):
        # x is a matrix of (batch_size, num_patches, emb_dim)
        # needs to be flattened to (batch_size, emb_dim)

        # Either do the mean across the num_patches dimension
        if self.output_mechanism == "mean":
            x = x.mean(dim=1)

        # or take the first patch
        if self.output_mechanism == "first":
            x = x[:, 0, :]

        return self.linear(x)