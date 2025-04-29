import torch
import random
import pickle
import numpy as np
from dataset import MNISTDataset, ZeroInputDataset
from model import ProjectionLayer
from model import PositionalLayer
from model import EncodersLayer
from model import OutputLayer
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

def main():
    # Define hyperparameters

    config = {
        'random_seed': 3407, # https://arxiv.org/abs/2109.08203
        'batch_size': 1024,
        'emb_dim': 64,
        'learning_rate': 0.003,
        'num_classes':  10,
        'num_patches': 4,
        'num_heads': 4,
        'num_layers': 2,
        'epochs': 1000,
    }

    # Set random seed for reproducibility

    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    
    # Extract training data

    train_data = pickle.load(open("data/raw/small_train_data.pkl", "rb"))
    test_data = pickle.load(open("data/raw/small_test_data.pkl", "rb"))

    # Create Dataset instances

    train_dataset = MNISTDataset(train_data['image'], train_data['label'], config['num_patches'])
    test_dataset = MNISTDataset(test_data['image'], test_data['label'], config['num_patches'])

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
    encodersLayer = EncodersLayer(config['emb_dim'], config['num_heads'], config['num_layers'])
    outputLayer = OutputLayer(config['emb_dim'], config['num_classes'])

    # DataLoaders

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(
        list(projectionLayer.parameters())
        + list(positionalLayer.parameters())
        + list(encodersLayer.parameters())
        + list(outputLayer.parameters()), config['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss()

    print("\n# Training the model")

    # Training and evaluation loop
    for epoch in range(config['epochs']):
        projectionLayer.train()
        encodersLayer.train()
        outputLayer.train()
        total_loss = 0
        for images, labels in train_loader:
            projections = projectionLayer(images)

            positionalLayer_out = positionalLayer(projections)

            encoders_out = encodersLayer(positionalLayer_out)

            logits = outputLayer(encoders_out)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Perform evaluation over the entire test set for correctness
        # Evaluation
        projectionLayer.eval()
        outputLayer.eval()
        encodersLayer.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                projections = projectionLayer(images)

                positionalLayer_out = positionalLayer(projections)

                encoders_out = encodersLayer(positionalLayer_out)

                logits = outputLayer(encoders_out)

                loss = loss_fn(logits, labels)
                total_loss += loss.item() * labels.size(0)  # Accumulate loss weighted by batch size
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_test_loss = total_loss / total
        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | Test Accuracy: {accuracy:.6f}')

if __name__ == "__main__":
    main()