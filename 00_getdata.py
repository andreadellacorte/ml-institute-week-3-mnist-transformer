from datasets import load_dataset
import pickle

mnist = load_dataset("ylecun/mnist")
    
# Extract training data

train_data = pickle.load(open("data/raw/train_data.pkl", "rb"))
test_data = pickle.load(open("data/raw/test_data.pkl", "rb"))

pickle.dump(train_data, open("data/raw/train_data.pkl", "wb"))
pickle.dump(test_data, open("data/raw/test_data.pkl", "wb"))

# dump 100 images and labels in a smaller dataset

small_train_data = {
    'image': train_data['image'][:100],
    'label': train_data['label'][:100]
}
small_test_data = {
    'image': test_data['image'][:100],
    'label': test_data['label'][:100]
}

pickle.dump(small_train_data, open("data/raw/small_train_data.pkl", "wb"))
pickle.dump(small_test_data, open("data/raw/small_test_data.pkl", "wb"))