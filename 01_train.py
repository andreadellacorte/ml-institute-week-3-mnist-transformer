import torch
import random
import numpy as np
from dataset import MNISTDataset, ZeroInputDataset
from model import MNISTModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

def verify_expected_loss(num_classes, emb_dim):
    print("\nVerify expected initial loss")

    model = MNISTModel(emb_dim, num_classes)

    expected_loss = -math.log(1 / num_classes)

    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros((1, 1, 28, 28))  # Batch size 1, single-channel 28x28 image
        logits = model(dummy_input)
        loss_fn = torch.nn.CrossEntropyLoss()
        dummy_target = torch.tensor([0])  # Arbitrary class label
        initial_loss = loss_fn(logits, dummy_target)

    print(f"Expected initial loss: {expected_loss:.6f}, Measured initial loss: {initial_loss.item():.6f}")

def verify_input_independent_baseline(train_dataset, test_dataset, batch_size, emb_dim, learning_rate):
    print("\nVerify input-independent baseline")

    zero_train_dataset = ZeroInputDataset(len(train_dataset), 10)
    zero_test_dataset = ZeroInputDataset(len(test_dataset), 10)

    zero_train_loader = DataLoader(zero_train_dataset, batch_size=batch_size, shuffle=True)
    zero_test_loader = DataLoader(zero_test_dataset, batch_size=batch_size, shuffle=False)

    # Train and evaluate the input-independent baseline
    baseline_model = MNISTModel(emb_dim, num_classes=10)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), learning_rate)
    baseline_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
    # Training loop for baseline
        baseline_model.train()
        total_loss = 0
        for images, labels in zero_train_loader:
            logits = baseline_model(images)
            loss = baseline_loss_fn(logits, labels)

            baseline_optimizer.zero_grad()
            loss.backward()
            baseline_optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(zero_train_loader)

    # Evaluation loop for baseline
        baseline_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in zero_test_loader:
                logits = baseline_model(images)
                loss = baseline_loss_fn(logits, labels)
                total_loss += loss.item() * labels.size(0)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_test_loss = total_loss / total
        print(f'Baseline Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | Test Accuracy: {accuracy:.6f}')

def verify_overfit_small_dataset(train_dataset, emb_dim, learning_rate):
    print("\nVerify we can Overfit a single batch of data")

    # Create a small dataset with only two examples
    small_train_dataset = torch.utils.data.Subset(train_dataset, [0, 1])
    small_train_loader = DataLoader(small_train_dataset, batch_size=2, shuffle=False)

    # Initialize the overfit model
    overfit_model = MNISTModel(emb_dim, num_classes=10)
    overfit_optimizer = torch.optim.Adam(overfit_model.parameters(), learning_rate)
    overfit_loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model on the small dataset
    for epoch in range(100):  # Train for more epochs to ensure overfitting
        overfit_model.train()
        total_loss = 0
        for images, labels in small_train_loader:
            logits = overfit_model(images)
            loss = overfit_loss_fn(logits, labels)

            overfit_optimizer.zero_grad()
            loss.backward()
            overfit_optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(small_train_loader)

        # Evaluate the model on the same small dataset
        overfit_model.eval()
        correct = 0
        total = 0
        predictions_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in small_train_loader:
                logits = overfit_model(images)
                predictions = logits.argmax(dim=1)
                predictions_list.extend(predictions.tolist())
                labels_list.extend(labels.tolist())
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f'Overfit Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Accuracy: {accuracy:.6f}')

        # Stop training if loss is near zero
        if avg_train_loss < 1e-6:
            break

    # Visualize predictions and labels
    print("Labels:", labels_list)
    print("Predictions:", predictions_list)

def main():
    # Set random seed for reproducibility
    random_seed = 3407 # https://arxiv.org/abs/2109.08203
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Load MNIST dataset at hugging face ylecun/mnist

    mnist = load_dataset("ylecun/mnist")

    # Extract training data

    train_data = mnist['train']
    test_data = mnist['test']

    # Create Dataset instances

    train_dataset = MNISTDataset(train_data['image'], train_data['label'])
    test_dataset = MNISTDataset(test_data['image'], test_data['label'])

    # Define hyperparameters
    
    batch_size = 64
    emb_dim = 32
    learning_rate = 1e-3
    num_classes = 10

    # Flight checks

    print("\n# Pre-training checks")

    # Calculate expected initial loss for CrossEntropyLoss

    verify_expected_loss(num_classes, emb_dim)

    verify_input_independent_baseline(train_dataset, test_dataset, batch_size, emb_dim, learning_rate)

    verify_overfit_small_dataset(train_dataset, emb_dim, learning_rate)

    # Initialize the model
    model = MNISTModel(emb_dim, num_classes)

    # DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("\n# Training the model")

    # Training and evaluation loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Perform evaluation over the entire test set for correctness
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images)
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