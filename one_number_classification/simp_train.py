# Import necessary libraries
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torchvision import datasets, transforms  # Datasets and image transformations
from torch.utils.data import DataLoader  # Data loading utilities
from simp_transform_setup import MNISTTransformer  # Import only MNISTTransformer
import wandb  # Weights & Biases for experiment tracking
import numpy as np  # Numerical computing
import datetime

# Define base hyperparameters
hyperparameters = {
    'option': 'library',
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 1,
    'img_size': 28,
    'patch_size': 7,
    'emb_dim': 64,
    'num_heads': 4,
    'num_layers': 2,
    'train_frac': 0.5,
    'weight_decay': 0.01,
    'patience': 3
}

def train():
    with wandb.init(config=hyperparameters):
        config = wandb.config
        
        # Update hyperparameters with sweep values
        hyperparameters.update({
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'patch_size': config.patch_size,
            'emb_dim': config.emb_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'train_frac': config.train_frac
        })
        
        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Print current configuration
        print("\n" + "="*50)
        print("CURRENT CONFIGURATION:")
        print("="*50)
        for key, value in hyperparameters.items():
            print(f"{key:15}: {value}")
        print("="*50 + "\n")
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Split training data
        train_size = int(len(train_dataset) * hyperparameters['train_frac'])
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders with adjusted batch sizes to maintain same number of batches
        effective_batch_size = int(hyperparameters['batch_size'] / hyperparameters['train_frac'])
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False)
        
        print(f"\nData Loading Summary:")
        print(f"Total training samples: {len(train_dataset)}")
        print(f"Total validation samples: {len(val_dataset)}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Number of batches per epoch: {len(train_loader)}\n")
        
        # Initialize model
        model = MNISTTransformer(
            img_size=hyperparameters['img_size'],
            patch_size=hyperparameters['patch_size'],
            emb_dim=hyperparameters['emb_dim'],
            num_heads=hyperparameters['num_heads'],
            num_layers=hyperparameters['num_layers'],
            num_classes=10,
            option=hyperparameters['option']
        ).to(device)
        
        # Print model summary
        print(f"\nModel Summary:")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Transformer type: {'Library' if hyperparameters['option'] == 'library' else 'Custom'}")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay']
        )
        
        # Training function
        def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
            
            return total_loss / len(train_loader), 100. * correct / total
        
        # Validation function
        def validate(model, val_loader, criterion, device):
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    total_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            return total_loss / len(val_loader), 100. * correct / total
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        
        for epoch in range(1, hyperparameters['num_epochs'] + 1):
            print(f'\nEpoch: {epoch}/{hyperparameters["num_epochs"]}')
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'mnist_transformer_{hyperparameters["option"]}_{ts}_best.pth')
            else:
                epochs_no_improve += 1
                print(f"No improvement. Early stop patience: {epochs_no_improve}/{hyperparameters['patience']}")
            
            if epochs_no_improve >= hyperparameters['patience']:
                print("Early stopping triggered.")
                break
        
        # Save final model
        torch.save(model.state_dict(), f'mnist_transformer_{hyperparameters["option"]}_{ts}_final.pth')
        print(f"Model saved to mnist_transformer_{hyperparameters['option']}_{ts}_final.pth")

# Define sweep configuration
sweep_config = {
    'method': 'grid',  # or 'random', 'bayes'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'weight_decay': {
            'values': [0.01, 0.001]
        },
        'patch_size': {
            'values': [7, 14]
        },
        'emb_dim': {
            'values': [64, 128]
        },
        'num_heads': {
            'values': [4, 8]
        },
        'num_layers': {
            'values': [2, 4]
        },
        'train_frac': {
            'values': [0.2, 0.5, 0.8]
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="mnist-transformer-sweep")

# Run the sweep
wandb.agent(sweep_id, function=train)
