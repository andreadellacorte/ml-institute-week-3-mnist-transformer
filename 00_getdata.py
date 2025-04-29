from datasets import load_dataset
import pickle

mnist = load_dataset("ylecun/mnist")

train_data = mnist["train"]
test_data = mnist["test"]

# pickle.dump an xs, s, m, l, xl, xxl, and full dataset

dataset_sizes = {
    "xs": 100,
    "s": 500,
    "m": 1000,
    "l": 5000,
    "xl": 10000,
    "xxl": 50000,
}

for size in dataset_sizes:
    # Create a new dataset with the specified size
    new_train_data = train_data.select(range(dataset_sizes[size]))
    # Save the new dataset to a pickle file
    with open(f"data/raw/{size}_train_data.pkl", "wb") as f:
        pickle.dump(new_train_data, f)

    if dataset_sizes[size] <= len(test_data):
        new_test_data = test_data.select(range(dataset_sizes[size]))
        with open(f"data/raw/{size}_test_data.pkl", "wb") as f:
            pickle.dump(new_test_data, f)

with open(f"data/raw/full_train_data.pkl", "wb") as f:
    pickle.dump(train_data, f)

with open(f"data/raw/full_test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)

