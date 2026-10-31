
import pickle
import numpy as np
import os

DATA_DIR = "dataset/cifar-10-batches-py"

def load_batch(file_path):
    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, labels

# Load all training batches
x_train_list = []
y_train_list = []

for i in range(1, 6):
    x, y = load_batch(os.path.join(DATA_DIR, f"data_batch_{i}"))
    x_train_list.append(x)
    y_train_list.extend(y)

x_train = np.concatenate(x_train_list)
y_train = np.array(y_train_list)

# Load test batch
x_test, y_test = load_batch(os.path.join(DATA_DIR, "test_batch"))
y_test = np.array(y_test)

print("Complete CIFAR-10 dataset loaded!")
print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

# Normalize image pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nPreprocessing completed!")
print("Training data range:", x_train.min(), "to", x_train.max())
print("Testing data range:", x_test.min(), "to", x_test.max())

import matplotlib.pyplot as plt

# CIFAR-10 class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Display first 10 images
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()