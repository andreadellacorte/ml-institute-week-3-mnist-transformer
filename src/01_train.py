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
import matplotlib.pyplot as plt
import wandb

# Define the sweep configuration
sweep_config_massive = {
    'method': 'grid',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'normalize_dataset': {'values': [True, False]},
        'batch_size': {'values': [32, 64, 128, 256]},
        'emb_dim': {'values': [8, 16, 32, 64]},
        'encoder_emb_dim': {'values': [8, 16, 32, 64]},
        'learning_rate': {'values': [0.001, 0.003, 0.005]},
        'weight_decay': {'values': [0.0, 0.001, 0.01, 0.01, 0.1]},
        'self_attending' : {'values': [False, True]},
        'num_patches': {'values': [1, 4, 16, 196]},
        'num_heads': {'values': [1]},
        'num_layers': {'values': [1, 2, 3, 4]},
        'epochs': {'values': [5]},
        'train_dataset_size': {'values': ['m', 'l', 'xl', 'xxl', 'full']},
        'test_dataset_size': {'values': ['full']}
    }
}

sweep_config_single = {
    'method': 'grid',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'normalize_dataset': {'values': [True]},
        'batch_size': {'values': [32]},
        'emb_dim': {'values': [64]},
        'encoder_emb_dim': {'values': [32]},
        'learning_rate': {'values': [0.001]},
        'weight_decay': {'values': [0.0]},
        'num_patches': {'values': [4]},
        'self_attending' : {'values': [False, True]},
        'num_heads': {'values': [1]},
        'num_layers': {'values': [1]},
        'epochs': {'values': [5]},
        'train_dataset_size': {'values': ['full']},
        'test_dataset_size': {'values': ['full']}
    }
}

# Comment this
sweep_config = sweep_config_single

print("Loading datasets...")

# Extract all required training and testing data based on sweep_config
required_train_sizes = set(sweep_config['parameters']['train_dataset_size']['values'])
required_test_sizes = set(sweep_config['parameters']['test_dataset_size']['values'])

train_data = {
    size: pickle.load(open(f"data/raw/{size}_train_data.pkl", "rb"))
    for size in required_train_sizes
}
test_data = {
    size: pickle.load(open(f"data/raw/{size}_test_data.pkl", "rb"))
    for size in required_test_sizes
}

# Print the number of datasets loaded
print(f"Loaded {len(train_data)} training datasets and {len(test_data)} testing datasets.")

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="mlx7-week-3-mnist-transformer")

# Set parameters
num_classes = 10
random_seed = 3407

def train():
    # Initialize a new wandb run
    wandb.init()
    config = wandb.config

    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    sweep_train_dataset = train_data[config['train_dataset_size']]
    sweep_test_dataset = test_data[config['test_dataset_size']]
    
    # Extract training data
    train_dataset = MNISTDataset(sweep_train_dataset['image'], sweep_train_dataset['label'], config['normalize_dataset'], config['num_patches'])
    test_dataset = MNISTDataset(sweep_test_dataset['image'], sweep_test_dataset['label'], config['normalize_dataset'], config['num_patches'])

    # Initialize the model
    projectionLayer = ProjectionLayer(config['emb_dim'], config['num_patches'])
    positionalLayer = PositionalLayer(config['emb_dim'], config['num_patches'])
    encoderLayers = []
    for i in range(config['num_layers']):
        encoderLayers.append(
            MyEncoderLayer(
                config['emb_dim'],
                config['encoder_emb_dim'],
                config['self_attending']
            )
        )
    outputLayer = OutputLayer(config['emb_dim'], num_classes)

    # Log metrics to wandb
    wandb.watch([projectionLayer, positionalLayer, *encoderLayers, outputLayer], log="all")

    # DataLoaders
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(
        list(projectionLayer.parameters())
        + list(positionalLayer.parameters())
        + [param for encoderLayer in encoderLayers for param in encoderLayer.parameters()]
        + list(outputLayer.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    
    loss_fn = torch.nn.CrossEntropyLoss()

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

# Run the sweep agent
wandb.agent(sweep_id, function=train)