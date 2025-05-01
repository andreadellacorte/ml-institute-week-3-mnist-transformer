import torch
import random
import pickle
import numpy as np
from dataset import MNISTDataset
from model import ClassifierEncoder
from torch.utils.data import DataLoader
import wandb
import time

# Define the sweep configuration
sweep_config_full = {
    'method': 'grid',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'dataset_type': {'values': ['huggingface']},
        'train_dataset_size': {'values': ['5000']},
        'test_dataset_size': {'values': ['full']},
        'rescale_dataset': {'values': [True, False]},
        'normalize_dataset': {'values': [True, False]},
        'batch_size': {'values': [512]},
        'emb_dim': {'values': [16]},
        'encoder_emb_dim': {'values': [16]},
        'learning_rate': {'values': [0.001, 0.002]},
        'weight_decay': {'values': [0.001]},
        'output_mechanism': {'values': ['mean', 'first']},
        'num_patches': {'values': [16]},
        'num_heads': {'values': [1, 2, 4, 8, 16]},
        'num_layers': {'values': [1, 2, 4, 8, 16]},
        'epochs': {'values': [10]},
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
        'dataset_type': {'values': ['huggingface']},
        'train_dataset_size': {'values': ['5000']},
        'test_dataset_size': {'values': ['full']},
        'normalize_dataset': {'values': [True]},
        'rescale_dataset': {'values': [True]},
        'batch_size': {'values': [512]},
        'emb_dim': {'values': [16]},
        'encoder_emb_dim': {'values': [16]},
        'learning_rate': {'values': [0.001]},
        'weight_decay': {'values': [0.01]},
        'num_patches': {'values': [4]},
        'output_mechanism' : {'values': ['mean']},
        'num_heads': {'values': [1]},
        'num_layers': {'values': [1]},
        'epochs': {'values': [5]},
        'masking': {'values': [False]},
        'self_attending': {'values': [True]},
    }
}

sweep_config = sweep_config_single

print("Loading datasets...")

# Extract all required training and testing data based on sweep_config
required_dataset_types = set(sweep_config['parameters']['dataset_type']['values'])
required_train_sizes = set(sweep_config['parameters']['train_dataset_size']['values'])
required_test_sizes = set(sweep_config['parameters']['test_dataset_size']['values'])

train_data = {
    dataset_type: {
        size: pickle.load(open(f"data/raw/{dataset_type}/{size}_train_data.pkl", "rb"))
        for size in required_train_sizes
    }
    for dataset_type in required_dataset_types
}
test_data = {
    dataset_type: {
        size: pickle.load(open(f"data/raw/{dataset_type}/{size}_test_data.pkl", "rb"))
        for size in required_test_sizes
    }
    for dataset_type in required_dataset_types
}

# Print the number of datasets loaded
print(f"Loaded {len(train_data)} training datasets and {len(test_data)} testing datasets.")

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="mlx7-week-3-mnist-transformer")

# Set parameters
NUM_CLASSES = 10
RANDOM_SEED = 3407

def train():
    # Initialize a new wandb run
    wandb.init()
    config = wandb.config

    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sweep_train_dataset = train_data[config['dataset_type']][config['train_dataset_size']]
    sweep_test_dataset = test_data[config['dataset_type']][config['test_dataset_size']]

    # Extract training data
    train_dataset = MNISTDataset(
        sweep_train_dataset['image'],
        sweep_train_dataset['label'],
        config['normalize_dataset'],
        config['rescale_dataset']
        #config['num_patches']
    )
    test_dataset = MNISTDataset(
        sweep_test_dataset['image'],
        sweep_test_dataset['label'],
        config['normalize_dataset'],
        config['rescale_dataset']
        #config['num_patches']
    )

    # Initialize the model

    classifier = ClassifierEncoder(
        NUM_CLASSES,
        config['emb_dim'],
        config['num_patches'],
        config['num_heads'],
        config['num_layers'],
        config['encoder_emb_dim'],
        config['masking'],
        config['self_attending'],
        config['output_mechanism']
    )

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)

    # Log metrics to wandb
    wandb.watch(classifier, log="all", log_graph=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(
        list(classifier.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    
    print(f"Num of trainable parameters: {sum(p.numel() for p in classifier.parameters() if p.requires_grad)}")

    # Log the number of trainable parameters to wandb
    num_trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    wandb.log({"num_trainable_params": num_trainable_params})

    loss_fn = torch.nn.CrossEntropyLoss()

    # Timing the training and evaluation loop
    start_time = time.time()

    # Training and evaluation loop
    for epoch in range(config['epochs']):
        classifier.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = classifier(images)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Log training loss to wandb
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # Perform evaluation over the entire test set for correctness
        classifier.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = classifier(images)

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

    # Log the total runtime to wandb
    total_runtime = time.time() - start_time
    wandb.log({"total_runtime": total_runtime})

# Run the sweep agent
wandb.agent(sweep_id, function=train)