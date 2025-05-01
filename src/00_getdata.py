from datasets import load_dataset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, Grayscale
import os
import pickle
import shutil

def get_huggingface_mnist():
    # Load the MNIST dataset
    dataset = load_dataset("mnist")
    
    # Split the dataset into training and testing sets
    train_data = dataset['train']
    test_data = dataset['test']
    
    return train_data, test_data

def clear_mnist_cache():
    mnist_cache_path = "data/raw/MNIST"
    if os.path.exists(mnist_cache_path):
        shutil.rmtree(mnist_cache_path)
        print(f"Cleared cache at {mnist_cache_path}")
    else:
        print(f"No cache found at {mnist_cache_path}")

def get_torchvision_mnist():
    # Define the transform to resize and convert images to grayscale
    transform = Compose([
        Resize((28, 28)),  # Ensure images are 28x28
        Grayscale(num_output_channels=1)  # Convert to grayscale
    ])

    # Load the MNIST dataset from torchvision
    train = datasets.MNIST(root='data/raw', train=True, download=True)
    test = datasets.MNIST(root='data/raw', train=False, download=True)

    # Convert the dataset into a list of dictionaries with 'image' and 'label'
    train_data = [
        {'image': transform(train.data[i].unsqueeze(0).float()), 'label': int(train.targets[i])}
        for i in range(len(train))
    ]
    test_data = [
        {'image': transform(test.data[i].unsqueeze(0).float()), 'label': int(test.targets[i])}
        for i in range(len(test))
    ]

    # clear_mnist_cache()

    return train_data, test_data

def save_dataset(type, train_data, test_data):
    if not os.path.exists(f"data/raw/{type}"):
        os.makedirs(f"data/raw/{type}")

    dataset_sizes = {
        "100": 100,
        "500": 500,
        "1000": 1000,
        "5000": 5000,
        "10000": 10000,
        "25000": 25000,
    }

    for size in dataset_sizes:
        # Create a new dataset with the specified size
        new_train_data = train_data[:dataset_sizes[size]]
        # Save the new dataset to a pickle file
        with open(f"data/raw/{type}/{size}_train_data.pkl", "wb") as f:
            pickle.dump(new_train_data, f)

        if dataset_sizes[size] <= len(test_data):
            new_test_data = test_data[:dataset_sizes[size]]
            with open(f"data/raw/{type}/{size}_test_data.pkl", "wb") as f:
                pickle.dump(new_test_data, f)

    with open(f"data/raw/{type}/full_train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)

    with open(f"data/raw/{type}/full_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

def main():
    print("Loading datasets...")

    print("Getting MNIST dataset from Hugging Face...")

    # Get the MNIST dataset from Hugging Face
    train_data, test_data = get_huggingface_mnist()

    # Save the dataset to pickle files
    save_dataset("huggingface", train_data, test_data)

    print("Hugging Face dataset saved successfully.")

    print("Getting MNIST dataset from torchvision...")

    # Get the MNIST dataset from torchvision
    train_data, test_data = get_torchvision_mnist()

    # Save the dataset to pickle files
    save_dataset("torchvision", train_data, test_data)

    print("torchvsion datasets saved successfully.")

if __name__ == "__main__":
    main()