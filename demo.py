import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
#model = tf.keras.models.load_model("potato_disease_model.h5")
model = tf.keras.models.load_model("potato_disease_model.keras")

# Define class names (update according to your training dataset)
class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# Streamlit app title
st.title("🌿 Potato Disease Classification")
st.write("Upload a potato leaf image to detect disease and see confidence score.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)
    img_resized = tf.image.resize(img_array, (224, 224)) / 255.0  # same size as training
    img_batch = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_batch)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Display results
    st.markdown(f"### 🏷 Prediction: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")
