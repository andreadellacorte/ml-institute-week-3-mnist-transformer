import torch
import torch.nn.init as init
import torch.nn.functional as F
import math

class ClassifierEncoder(torch.nn.Module):
    def __init__(self, num_classes, emb_dim, patch_size, num_patches, num_heads, num_layers, encoder_emb_dim, masking, self_attending, output_mechanism):
        super(ClassifierEncoder, self).__init__()

        self.model = torch.nn.Sequential(
            Encoder(emb_dim, patch_size, num_patches, num_heads, num_layers, encoder_emb_dim, masking, self_attending),
            OutputLayer(emb_dim, num_classes, output_mechanism)
        )

    def forward(self, x):
        return self.model(x)

class Encoder(torch.nn.Module):
    def __init__(self, emb_dim, patch_size, num_patches, num_heads, num_layers, encoder_emb_dim, masking, self_attending):
        super(Encoder, self).__init__()

        self.model = torch.nn.Sequential(
            InputEmbedding(emb_dim, patch_size),
            PositionalEncoding(emb_dim, num_patches),
            *[MultiHeadAttention(num_heads, emb_dim, emb_dim, encoder_emb_dim, masking, self_attending) for _ in range(num_layers)],
            FeedForward(emb_dim),
        )

    def forward(self, x):
        return self.model(x)

class Transformer(torch.nn.Module):
    def __init__(
            self,
            emb_dim,
            patch_size,
            num_patches_per_digit,
            num_heads,
            num_layers,
            encoder_emb_dim,
            decoder_emb_dim,
            internal_decoder_emb_dim,
            output_vocab_size,
            max_sequence_length,
            masking,
            self_attending,
            output_mechanism
        ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(emb_dim, patch_size, num_patches_per_digit * max_sequence_length, num_heads, num_layers, encoder_emb_dim, masking, self_attending)

        # For decoder, we need to account for start and end tokens in max length
        actual_max_length = max_sequence_length + 2  # +2 for <start> and <end> tokens
        
        # Move embedding and positional encoding outside layers
        self.output_embedding = OutputEmbedding(decoder_emb_dim, output_vocab_size)
        self.positional_encoding = PositionalEncoding(decoder_emb_dim, actual_max_length)
        
        # Ensure decoder layers maintain consistent embedding dimensions
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "multi_head_attention_1": MultiHeadAttention(num_heads, decoder_emb_dim, decoder_emb_dim, internal_decoder_emb_dim, True, self_attending),
                "multi_head_attention_2": MultiHeadAttention(num_heads, decoder_emb_dim, emb_dim, internal_decoder_emb_dim, False, self_attending),
                "feed_forward": FeedForward(decoder_emb_dim),
            }) for _ in range(num_layers)
        ])

        # Output layer should match the vocabulary size exactly
        self.output_layer = OutputLayer(decoder_emb_dim, output_vocab_size, output_mechanism)

    def forward(self, inputs_x, outputs_x):
        inputs_x = self.encoder(inputs_x)
        
        # Apply embedding and positional encoding once before the layers
        outputs_x = self.output_embedding(outputs_x)
        outputs_x = self.positional_encoding(outputs_x)

        for layer in self.layers:
            outputs_x = layer["multi_head_attention_1"](outputs_x)
            outputs_x = layer["multi_head_attention_2"](outputs_x, inputs_x)
            outputs_x = layer["feed_forward"](outputs_x)

        return self.output_layer(outputs_x)

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

class InputEmbedding(torch.nn.Module):
    def __init__(self, emb_dim, patch_size):
        super(InputEmbedding, self).__init__()
        
        self.fold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

        self.linear = torch.nn.Linear(patch_size*patch_size, emb_dim)

        # Initialize weights and biases
        init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        init.zeros_(self.linear.bias)

    def forward(self, x):
        # Hack to make both torchivision and huggingface datasets work
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.fold(x).transpose(1, 2)

        return self.linear(x)

class OutputEmbedding(torch.nn.Module):
    def __init__(self, emb_dim, vocab_size):
        super(OutputEmbedding, self).__init__()
        # Use exact vocab_size since indices are 0 through vocab_size-1
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        
        # Initialize weights using normal distribution
        init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Ensure input is long tensor for embedding lookup
        x = x.long()
        return self.embedding(x)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_dim, num_embeddings):
        super(PositionalEncoding, self).__init__()
        self.num_embeddings = num_embeddings

        self.pos_embed = torch.nn.Parameter(torch.randn(1, num_embeddings, emb_dim))

    def forward(self, x):
        # Get the actual sequence length from the input
        actual_len = x.size(1)
        
        # Use only the positional embeddings up to the actual sequence length
        pos_embed_used = self.pos_embed[:, :actual_len, :]
        
        return x + pos_embed_used

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, q_input_emb_dim, kv_input_emb_dim, internal_emb_dim, masking, self_attending):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.internal_emb_dim = internal_emb_dim
        self.self_attending = self_attending
        self.masking = masking

        self.layer_norm = torch.nn.LayerNorm(q_input_emb_dim)
        self.wQ = torch.nn.Linear(q_input_emb_dim, internal_emb_dim * num_heads)
        self.wK = torch.nn.Linear(kv_input_emb_dim, internal_emb_dim * num_heads)
        self.wV = torch.nn.Linear(kv_input_emb_dim, internal_emb_dim * num_heads)
        self.w0 = torch.nn.Linear(internal_emb_dim * num_heads, q_input_emb_dim)

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
        x = self.layer_norm(x)
        batch_size, seq_len, _ = x.size()

        # Use encoder_input for K,V if provided, otherwise use x
        kv_input = encoder_input if encoder_input is not None else x
        kv_batch_size, kv_seq_len, _ = kv_input.size()
        
        # Project and reshape Q, K, V with their respective sequence lengths
        q = self.wQ(x).view(batch_size, seq_len, self.num_heads, self.internal_emb_dim).transpose(1, 2)
        k = self.wK(kv_input).view(kv_batch_size, kv_seq_len, self.num_heads, self.internal_emb_dim).transpose(1, 2)
        v = self.wV(kv_input).view(kv_batch_size, kv_seq_len, self.num_heads, self.internal_emb_dim).transpose(1, 2)

        # Calculate attention scores from Q * K^T
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.internal_emb_dim)

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
        
        # Reshape back to original dimensions
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