from datasets import load_dataset
from torchvision import datasets
import pickle

def get_huggingface_mnist():
    # Load the MNIST dataset
    dataset = load_dataset("mnist")
    
    # Split the dataset into training and testing sets
    train_data = dataset['train']
    test_data = dataset['test']
    
    return train_data, test_data

def get_torchvision_mnist():
    # Load the MNIST dataset from torchvision
    train = datasets.MNIST(root='data/raw', train=True, download=True, transform=None)
    test = datasets.MNIST(root='data/raw', train=False, download=True, transform=None)

    train_data = {
        'image': train.data,
        'label': train.targets
    }
    test_data = {
        'image': test.data,
        'label': test.targets
    }

    return train_data, test_data

def save_dataset(type, train_data, test_data):
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
        new_train_data = train_data.select(range(dataset_sizes[size]))
        # Save the new dataset to a pickle file
        with open(f"data/raw/{type}/{size}_train_data.pkl", "wb") as f:
            pickle.dump(new_train_data, f)

        if dataset_sizes[size] <= len(test_data):
            new_test_data = test_data.select(range(dataset_sizes[size]))
            with open(f"data/raw/{type}/{size}_test_data.pkl", "wb") as f:
                pickle.dump(new_test_data, f)

    with open(f"data/raw/{type}/full_train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)

    with open(f"data/raw/{type}/full_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

def main():
    # Get the MNIST dataset from Hugging Face
    train_data, test_data = get_huggingface_mnist()
    
    # Save the dataset to pickle files
    save_dataset("huggingface", train_data, test_data)

    # Get the MNIST dataset from torchvision
    train_data, test_data = get_torchvision_mnist()
    
    # Save the dataset to pickle files
    save_dataset("torchvision", train_data, test_data)

if __name__ == "__main__":
    main()