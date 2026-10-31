import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Load trained model
model = tf.keras.models.load_model("cifar10_cnn.keras")

# Load test batch
with open("dataset/cifar-10-batches-py/test_batch", "rb") as file:
    batch = pickle.load(file, encoding="bytes")

x_test = batch[b"data"]
y_test = np.array(batch[b"labels"])

# Reshape images
x_test = x_test.reshape(-1, 3, 32, 32)
x_test = x_test.transpose(0, 2, 3, 1)

# Normalize images
x_test = x_test.astype("float32") / 255.0

# Select one image
index = 10
image = x_test[index]

# Predict
prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)

predicted_class = np.argmax(prediction)
actual_class = y_test[index]

print("Predicted:", class_names[predicted_class])
print("Actual:", class_names[actual_class])
print("Confidence:", round(float(np.max(prediction)) * 100, 2), "%")

# Display image
plt.imshow(image)
plt.title("Predicted: " + class_names[predicted_class])
plt.axis("off")
plt.show()