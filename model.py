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

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute the scaled dot-product attention.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value), attn_weights

def multi_head_attention(x, emb_dim, num_heads):
    """
    Multi-head attention mechanism.
    """
    head_dim = emb_dim // num_heads
    assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

    # Linear layers for query, key, and value
    query_proj = torch.nn.Linear(emb_dim, emb_dim)
    key_proj = torch.nn.Linear(emb_dim, emb_dim)
    value_proj = torch.nn.Linear(emb_dim, emb_dim)
    output_proj = torch.nn.Linear(emb_dim, emb_dim)

    # Split into multiple heads
    query = query_proj(x).view(x.size(0), x.size(1), num_heads, head_dim).transpose(1, 2)
    key = key_proj(x).view(x.size(0), x.size(1), num_heads, head_dim).transpose(1, 2)
    value = value_proj(x).view(x.size(0), x.size(1), num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention
    attn_output, _ = scaled_dot_product_attention(query, key, value)

    # Concatenate heads and project
    attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), emb_dim)
    return output_proj(attn_output)

def feed_forward(x, emb_dim, ff_dim):
    """
    Feed-forward network.
    """
    linear1 = torch.nn.Linear(emb_dim, ff_dim)
    linear2 = torch.nn.Linear(ff_dim, emb_dim)
    return linear2(F.relu(linear1(x)))

def transformer_encoder_layer(x, emb_dim, num_heads, ff_dim):
    """
    Single transformer encoder layer.
    """
    # Multi-head attention with residual connection
    attn_output = multi_head_attention(x, emb_dim, num_heads)
    x = x + attn_output  # Residual connection
    x = F.layer_norm(x, x.size()[1:])

    # Feed-forward network with residual connection
    ff_output = feed_forward(x, emb_dim, ff_dim)
    x = x + ff_output  # Residual connection
    x = F.layer_norm(x, x.size()[1:])
    return x

def transformer_encoder(x, emb_dim, num_heads, ff_dim, num_layers):
    """
    Stack multiple transformer encoder layers.
    """
    for _ in range(num_layers):
        x = transformer_encoder_layer(x, emb_dim, num_heads, ff_dim)
    return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

    def forward(self, x):
        return transformer_encoder(x, self.emb_dim, self.num_heads, self.ff_dim, self.num_layers)

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