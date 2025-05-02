import torch
import torch.nn as nn

vocab_size = 100
embedding_dim = 64
max_len = 10

# Example batch: sequences of variable length
# Tokenised sequences (as indices)
sequences = [
    [4, 15, 23],                 # length 3
    [10, 22, 35, 42, 7, 19, 88], # length 7
]
print(f"Original sequences: {sequences}")

# Pad sequences to max_len
padded = torch.full((len(sequences), max_len), fill_value=0)  # 0 = padding index
for i, seq in enumerate(sequences):
    padded[i, :len(seq)] = torch.tensor(seq)
print(f"\nPadded tensor shape: {padded.shape}")
print(f"Padded tensor:\n{padded}")

# Attention mask: 1 for real tokens, 0 for padding
attention_mask = (padded != 0).int()
print(f"\nAttention mask shape: {attention_mask.shape}")
print(f"Attention mask (1 = token, 0 = padding):\n{attention_mask}")

# Embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
embedded = embedding(padded)  # shape: [batch_size, 10, embedding_dim]
print(f"\nEmbedded tensor shape: {embedded.shape}")
print(f"First sequence, first token embedding (first few values): {embedded[0,0,:10]}")
