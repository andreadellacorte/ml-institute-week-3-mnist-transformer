import torch

class MNISTModel(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, emb_dim),  # project to embedding space
        )
        self.emb_to_logits = torch.nn.Linear(emb_dim, 10)  # predict digits
        self.emb_dim = emb_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.emb_to_logits(features)
        return logits