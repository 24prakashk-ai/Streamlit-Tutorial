
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# CIFAR-10 classes
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_cnn.keras")

model = load_model()

# Dashboard heading
st.title("🤖 CIFAR-10 AI Image Assistant")
st.write("Upload an image and let the CNN model identify the object.")

# Sidebar
st.sidebar.title("Dashboard")
st.sidebar.success("CNN Model: Online")
st.sidebar.write("Dataset: CIFAR-10")
st.sidebar.write("Test Accuracy: 71.54%")
st.sidebar.write("Classes: 10")

# Chat-style interface
st.subheader("💬 Chat with your AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Identify Image"):
        # Resize image to CIFAR-10 input size
        image_resized = image.resize((32, 32))

        image_array = np.array(image_resized)
        image_array = image_array.astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # CNN prediction
        prediction = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        # Display chatbot conversation
        st.session_state.messages.append({
            "role": "user",
            "content": "Please identify this image."
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"I predict this image is a **{predicted_class}**. "
                f"My model confidence is **{confidence:.2f}%**."
            )
        })

        st.rerun()

# Prediction information
st.divider()
st.caption("Powered by TensorFlow, CNN and Streamlit")
