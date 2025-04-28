import torch
import random
import pickle
import numpy as np
from dataset import MNISTDataset, ZeroInputDataset
from model import MNISTModel
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
    # Define hyperparameters

    random_seed = 3407 # https://arxiv.org/abs/2109.08203
    batch_size = 64
    emb_dim = 32
    learning_rate = 1e-3
    num_classes = 10
    num_patches = 16

    # Set random seed for reproducibility

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Extract training data

    train_data = pickle.load(open("data/raw/small_train_data.pkl", "rb"))
    test_data = pickle.load(open("data/raw/small_test_data.pkl", "rb"))

    # Split each train_data['image'] of 28x28 pixles into 16 patches of 7x7 pixels
    for i in range(len(train_data['image'])):
        image = train_data['image'][i]
        patches = []
        for j in range(num_patches):
            row_start = (j // 4) * 7
            col_start = (j % 4) * 7
            image = np.array(image)  # Convert to NumPy array
            patch = image[row_start:row_start+7, col_start:col_start+7]
            patches.append(patch)
        train_data['image'][i] = np.array(patches)
    
    # Split each train_data['image'] of 28x28 pixles into 16 patches of 7x7 pixels
    for i in range(len(test_data['image'])):
        image = test_data['image'][i]
        patches = []
        for j in range(num_patches):
            row_start = (j // 4) * 7
            col_start = (j % 4) * 7
            image = np.array(image)  # Convert to NumPy array
            patch = image[row_start:row_start+7, col_start:col_start+7]
            patches.append(patch)
        test_data['image'][i] = np.array(patches)

    # Visualize the first image and its 16 patches in both train_data and test_data
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('First Image and its 16 Patches (Train Data on Left, Test Data on Right)', fontsize=16)

    for i in range(num_patches):
        # Train data patches
        axes[i // 4, i % 4].imshow(train_data['image'][0][i], cmap='gray')
        axes[i // 4, i % 4].axis('off')

        # Test data patches
        axes[i // 4, i % 4 + 4].imshow(test_data['image'][0][i], cmap='gray')
        axes[i // 4, i % 4 + 4].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("Train data #0:", train_data['image'][0])
    print("Train data #0 shape:", train_data['image'][0].shape)    

    # Create Dataset instances

    train_dataset = MNISTDataset(train_data['image'], train_data['label'], num_patches)
    test_dataset = MNISTDataset(test_data['image'], test_data['label'], num_patches)

    # Flight checks
    # print("\n# Pre-training checks")
    # verify_expected_loss(num_classes, emb_dim)
    # verify_input_independent_baseline(train_dataset, test_dataset, batch_size, emb_dim, learning_rate)
    # verify_overfit_small_dataset(train_dataset, emb_dim, learning_rate)

    # Initialize the model
    model = MNISTModel(emb_dim, num_classes, num_patches)

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