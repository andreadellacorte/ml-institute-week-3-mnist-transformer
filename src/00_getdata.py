from datasets import load_dataset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, Grayscale
import os
import pickle
import shutil

def convert_huggingface_dataset(dataset):
    return {
        'image': [example['image'] for example in dataset],
        'label': [example['label'] for example in dataset]
    }

def get_huggingface_mnist():
    # Load the MNIST dataset
    dataset = load_dataset("ylecun/mnist")
    
    # Split the dataset into training and testing sets
    train_data = convert_huggingface_dataset(dataset['train'])
    test_data = convert_huggingface_dataset(dataset['test'])

    # Check the size of the dataset
    print("Train data size:", len(train_data['image']))
    print("Test data size:", len(test_data['image']))
    
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
    train = datasets.MNIST(root='data/raw', train=True, download=True, transform=None)
    test = datasets.MNIST(root='data/raw', train=False, download=True, transform=None)

    # Convert the dataset into a dictionary with 'image' and 'label' as lists
    train_data = {
        'image': [transform(train.data[i].unsqueeze(0).float()) for i in range(len(train))],
        'label': [int(train.targets[i]) for i in range(len(train))]
    }
    test_data = {
        'image': [transform(test.data[i].unsqueeze(0).float()) for i in range(len(test))],
        'label': [int(test.targets[i]) for i in range(len(test))]
    }

    # print ("Train data size:", len(train_data['image']))
    # print ("Test data size:", len(test_data['image']))
    # print the first 5 images and labels
    for i in range(5):
        print(f"Train Image {i}: {train_data['image'][i]}")
        print(f"Train Label {i}: {train_data['label'][i]}")
    
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

    # Save the full datasets
    with open(f"data/raw/{type}/full_train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)

    with open(f"data/raw/{type}/full_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    for size in dataset_sizes:
        print(f"Creating {size} samples for {type} dataset...")

        new_train_data = {key: value[:dataset_sizes[size]] for key, value in train_data.items()}

        with open(f"data/raw/{type}/{size}_train_data.pkl", "wb") as f:
            pickle.dump(new_train_data, f)

        if dataset_sizes[size] < len(test_data['image']):
            new_test_data = {key: value[:dataset_sizes[size]] for key, value in test_data.items()}
            with open(f"data/raw/{type}/{size}_test_data.pkl", "wb") as f:
                pickle.dump(new_test_data, f)

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