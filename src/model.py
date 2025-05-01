import torch
import torch.nn.init as init
import torch.nn.functional as F
import math

class Encoder(torch.nn.Module):
    def __init__(self, num_classes, emb_dim, num_patches, num_heads, num_layers, encoder_emb_dim, masking, self_attending, output_mechanism):
        super(Encoder, self).__init__()

        self.model = torch.nn.Sequential(
            InputEmbedding(emb_dim, num_patches),
            PositionalEncoding(emb_dim, num_patches),
            *[MultiHeadAttention(num_heads, emb_dim, encoder_emb_dim, masking, self_attending) for _ in range(num_layers)],
            FeedForward(emb_dim),
            OutputLayer(emb_dim, num_classes, output_mechanism)
        )

    def forward(self, x):
        return self.model(x)

class Transformer(torch.nn.Module):
    def __init__(self, num_classes, emb_dim, num_patches, num_heads, num_layers, encoder_emb_dim, masking, self_attending, output_mechanism):
        super(Transformer, self).__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "input_embedding": InputEmbedding(emb_dim, num_patches),
                "input_positional_encoding": PositionalEncoding(emb_dim, num_patches),
                "encoder": Encoder(num_classes, emb_dim, num_patches, num_heads, num_layers, encoder_emb_dim, masking, self_attending, output_mechanism),
                "output_embedding": OutputEmbedding(emb_dim, num_classes),
                "multi_head_attention_1": MultiHeadAttention(num_heads, emb_dim, encoder_emb_dim, masking, self_attending),
                "multi_head_attention_2": MultiHeadAttention(num_heads, emb_dim, encoder_emb_dim, masking, self_attending),
                "output_layer": OutputLayer(emb_dim, num_classes, output_mechanism)
            })
        ])

    def forward(self, x):
        self.layers[0]["input_embedding"](x)

class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim):
        super(FeedForward, self).__init__()

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.linear1 = torch.nn.Linear(emb_dim, emb_dim * 4)
        self.linear2 = torch.nn.Linear(emb_dim * 4, emb_dim)

        # Initialize weights and biases
        init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        init.zeros_(self.linear1.bias)

        init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        init.zeros_(self.linear2.bias)

    def forward(self, x):
        input = x

        x = self.layer_norm(x)
        return self.linear2(F.relu(self.linear1(x))) + input

class OutputEmbedding(torch.nn.Module):
    def __init__(self, emb_dim, num_classes):
        super(OutputEmbedding, self).__init__()

        self.linear = torch.nn.Linear(emb_dim, num_classes)

        init.zeros_(self.embs.bias)
        init.kaiming_uniform_(self.embs.weight, nonlinearity='relu')

    def forward(self, x):
        return self.linear(x)

class InputEmbedding(torch.nn.Module):
    def __init__(self, emb_dim, num_patches):
        super(InputEmbedding, self).__init__()
        
        patch_size = int(math.sqrt(28*28/num_patches))
        
        self.fold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

        self.linear = torch.nn.Linear(int(28*28/num_patches), emb_dim)

        # Initialize weights and biases
        init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        init.zeros_(self.linear.bias)

    def forward(self, x):
        # Hack to make both torchivision and huggingface datasets work
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.fold(x).transpose(1, 2)

        return self.linear(x)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_dim, num_patches):
        super(PositionalEncoding, self).__init__()

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

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, emb_dim, encoder_emb_dim, masking, self_attending):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.encoder_emb_dim = encoder_emb_dim
        self.self_attending = self_attending
        self.masking = masking

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.wQ = torch.nn.Linear(emb_dim, encoder_emb_dim * num_heads)
        self.wK = torch.nn.Linear(emb_dim, encoder_emb_dim * num_heads)
        self.wV = torch.nn.Linear(emb_dim, encoder_emb_dim * num_heads)
        self.w0 = torch.nn.Linear(encoder_emb_dim * num_heads, emb_dim)

        # Initialize weights and biases
        init.kaiming_uniform_(self.wV.weight, nonlinearity='relu')
        init.zeros_(self.wV.bias)

        init.kaiming_uniform_(self.wK.weight, nonlinearity='relu')
        init.zeros_(self.wK.bias)

        init.kaiming_uniform_(self.wQ.weight, nonlinearity='relu')
        init.zeros_(self.wQ.bias)

        init.kaiming_uniform_(self.w0.weight, nonlinearity='relu')
        init.zeros_(self.w0.bias)

    def forward(self, x, encoder_input=None):
        input = x

        # Normalize x
        x = self.layer_norm(x)

        batch_size, seq_len, _ = x.size()

        # Use encoder_input for wQ if provided, otherwise use x
        q_input = encoder_input if encoder_input is not None else x
        q = self.wQ(q_input).view(batch_size, seq_len, self.num_heads, self.encoder_emb_dim).transpose(1, 2)

        # Use x for wK and wV
        k = self.wK(x).view(batch_size, seq_len, self.num_heads, self.encoder_emb_dim).transpose(1, 2)
        v = self.wV(x).view(batch_size, seq_len, self.num_heads, self.encoder_emb_dim).transpose(1, 2)

        # Calculate attention scores from Q * K^T
        scores = q @ k.transpose(-2, -1) / (math.sqrt(self.encoder_emb_dim))

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
        h = h.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.w0(h) + input  # Residual connection

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