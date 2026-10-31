import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

from cnn_model import build_cnn

# Dataset path
DATA_DIR = "dataset/cifar-10-batches-py"

# Function to load CIFAR-10 batches
def load_batch(file_path):
    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, labels


# Load training data
x_train_list = []
y_train_list = []

for i in range(1, 6):
    x, y = load_batch(
        os.path.join(DATA_DIR, f"data_batch_{i}")
    )
    x_train_list.append(x)
    y_train_list.extend(y)

x_train = np.concatenate(x_train_list).astype("float32") / 255.0
y_train = np.array(y_train_list)

# Load testing data
x_test, y_test = load_batch(
    os.path.join(DATA_DIR, "test_batch")
)

x_test = x_test.astype("float32") / 255.0
y_test = np.array(y_test)

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# Build CNN model
model = build_cnn()

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("Test Accuracy:", test_accuracy * 100)

# Save model
model.save("cifar10_cnn.keras")

print("CNN model saved successfully!")

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN Model Accuracy")
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("CNN Model Loss")
plt.legend()
plt.show()