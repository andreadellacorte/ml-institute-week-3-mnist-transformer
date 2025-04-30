import torch
import random
import pickle
import numpy as np
from dataset import MNISTDataset
from model import TransformerModel
from torch.utils.data import DataLoader
import wandb

# Define the sweep configuration
sweep_config_massive = {
    'method': 'grid',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'train_dataset_size': {'values': ['m', 'l', 'xl', 'xxl', 'full']},
        'test_dataset_size': {'values': ['full']},
        'normalize_dataset': {'values': [True, False]},
        'batch_size': {'values': [32, 64, 128, 256]},
        'emb_dim': {'values': [8, 16, 32, 64]},
        'encoder_emb_dim': {'values': [8, 16, 32, 64]},
        'learning_rate': {'values': [0.001, 0.003, 0.005]},
        'weight_decay': {'values': [0.0, 0.001, 0.01, 0.01, 0.1]},
        'output_mechanism': {'values': ['mean', 'first']},
        'num_patches': {'values': [1, 4, 16, 196]},
        'num_heads': {'values': [1]},
        'num_layers': {'values': [1, 2, 3, 4]},
        'epochs': {'values': [5]},
        'masking': {'values': [False, True]},
        'self_attending': {'values': [False, True]},
    }
}

sweep_config_single = {
    'method': 'grid',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'train_dataset_size': {'values': ['full']},
        'test_dataset_size': {'values': ['full']},
        'normalize_dataset': {'values': [True]},
        'batch_size': {'values': [32]},
        'emb_dim': {'values': [64]},
        'encoder_emb_dim': {'values': [32]},
        'learning_rate': {'values': [0.001]},
        'weight_decay': {'values': [0.0]},
        'num_patches': {'values': [4]},
        'output_mechanism' : {'values': ['mean']},
        'num_heads': {'values': [1, 2, 4, 8, 16]},
        'num_layers': {'values': [1]},
        'epochs': {'values': [5]},
        'masking': {'values': [False]},
        'self_attending': {'values': [True]},
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

    transformerModel = TransformerModel(
        num_classes,
        config['emb_dim'],
        config['num_patches'],
        config['num_heads'],
        config['num_layers'],
        config['encoder_emb_dim'],
        config['masking'],
        config['self_attending'],
        config['output_mechanism']
    )

    # Log metrics to wandb
    wandb.watch(transformerModel, log="all", log_graph=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(
        list(transformerModel.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training and evaluation loop
    for epoch in range(config['epochs']):
        transformerModel.train()
        total_loss = 0
        for images, labels in train_loader:
            logits = transformerModel(images)

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
        transformerModel.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                logits = transformerModel(images)

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