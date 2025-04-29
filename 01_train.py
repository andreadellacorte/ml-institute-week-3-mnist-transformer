import torch
import random
import pickle
import numpy as np
from dataset import MNISTDataset, ZeroInputDataset
from model import ProjectionLayer
from model import PositionalLayer
from model import EncoderLayer
from model import MyEncoderLayer
from model import OutputLayer
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import wandb

def main():
    # Define hyperparameters

    config = {
        'random_seed': 3407, # https://arxiv.org/abs/2109.08203
        'normalize_dataset': True,
        'batch_size': 256,
        'emb_dim': 64,
        'encoder_emb_dim': 64,
        'learning_rate': 0.003,
        'num_classes':  10,
        'num_patches': 4,
        'num_heads': 1,
        'num_layers': 4,
        'epochs': 1000,
        'train_dataset_size': 'full',
        'test_dataset_size': 'full',
    }

    # Initialize Weights & Biases (wandb)
    wandb.init(project="mlx7-week-3-mnist-transformer", config=config)

    # Set random seed for reproducibility

    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    
    # Extract training data

    train_data = pickle.load(open(f"data/raw/{config['train_dataset_size']}_train_data.pkl", "rb"))
    test_data = pickle.load(open(f"data/raw/{config['test_dataset_size']}_test_data.pkl", "rb"))

    # Create Dataset instances

    train_dataset = MNISTDataset(train_data['image'], train_data['label'], config['normalize_dataset'], config['num_patches'])
    test_dataset = MNISTDataset(test_data['image'], test_data['label'], config['normalize_dataset'], config['num_patches'])

    print("\n# Dataset statistics")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Label shape: {train_dataset[0][1].shape}")

    # Flight checks
    # print("\n# Pre-training checks")
    # verify_expected_loss(num_classes, emb_dim)
    # verify_input_independent_baseline(train_dataset, test_dataset, batch_size, emb_dim, learning_rate)
    # verify_overfit_small_dataset(train_dataset, emb_dim, learning_rate)

    # Initialize the model
    projectionLayer = ProjectionLayer(config['emb_dim'], config['num_patches'])
    positionalLayer = PositionalLayer(config['emb_dim'], config['num_patches'])
    # encoderLayer = EncoderLayer(config['emb_dim'], config['num_heads'])
    encoderLayers = []
    for i in range(config['num_layers']):
        encoderLayers.append(MyEncoderLayer(config['emb_dim'], config['encoder_emb_dim']))
    outputLayer = OutputLayer(config['emb_dim'], config['num_classes'])

    # Log metrics to wandb
    wandb.watch([projectionLayer, positionalLayer, *encoderLayers, outputLayer], log="all")

    # DataLoaders

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(
        list(projectionLayer.parameters())
        + list(positionalLayer.parameters())
        + [param for encoderLayer in encoderLayers for param in encoderLayer.parameters()]
        + list(outputLayer.parameters()), config['learning_rate'])
    
    print("\n# Model statistics")
    print(f"Projection Layer: {projectionLayer}")
    print(f"Positional Layer: {positionalLayer}")
    print(f"Encoder Layer: {encoderLayers[0]}")
    print(f"Output Layer: {outputLayer}")
    print(f"Optimizer: {optimizer}")
    print(f"Number of parameters: {sum(p.numel() for p in projectionLayer.parameters()) + sum(p.numel() for p in positionalLayer.parameters()) + sum(p.numel() for encoderLayer in encoderLayers for p in encoderLayer.parameters()) + sum(p.numel() for p in outputLayer.parameters())}")
    print(f"Loss function: CrossEntropyLoss")

    loss_fn = torch.nn.CrossEntropyLoss()

    print("\n# Training the model")

    # Training and evaluation loop
    for epoch in range(config['epochs']):
        projectionLayer.train()
        positionalLayer.train()
        for encoderLayer in encoderLayers:
            encoderLayer.train()
        outputLayer.train()
        total_loss = 0
        for images, labels in train_loader:
            projections = projectionLayer(images)

            positionalLayer_out = positionalLayer(projections)

            encoders_out = positionalLayer_out

            for encoderLayer in encoderLayers:
                encoders_out = encoderLayer(encoders_out)

            logits = outputLayer(encoders_out)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Log training loss to wandb
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # Perform evaluation over the entire test set for correctness
        # Evaluation
        projectionLayer.eval()
        positionalLayer.train()
        for encoderLayer in encoderLayers:
            encoderLayer.eval()
        outputLayer.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                projections = projectionLayer(images)

                positionalLayer_out = positionalLayer(projections)

                encoders_out = positionalLayer_out
                for encoderLayer in encoderLayers:
                    encoders_out = encoderLayer(encoders_out)

                logits = outputLayer(encoders_out)

                loss = loss_fn(logits, labels)
                total_loss += loss.item() * labels.size(0)  # Accumulate loss weighted by batch size
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_test_loss = total_loss / total

        # Log test metrics to wandb
        wandb.log({"epoch": epoch + 1, "test_loss": avg_test_loss, "test_accuracy": accuracy})

        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | Test Accuracy: {accuracy:.6f}')

if __name__ == "__main__":
    main()