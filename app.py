import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

file_id = "1NLj9MTHYH5z3N6d5YaXfMEG3qWmRyPNH"
url = f"https://drive.google.com/file/d/1NLj9MTHYH5z3N6d5YaXfMEG3qWmRyPNH/view?usp=sharing"
# Define model path
model_path = "trained_plant_disease_model.keras"

# Check if model file exists, if not, download it first
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Now load the trained model
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop execution if model fails to load

# Define class labels for potato leaf diseases
class_labels = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']

# Custom CSS for styling
st.markdown(
    """
    <style>
        body, .stApp {
            background: linear-gradient(135deg, #D0F0C0, #98FB98, #2E8B57) !important;
            color: black;
        }
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .uploadedFile {
            max-width: 400px;
        }
        img {
            max-width: 300px;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #2E8B57;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown('<h1 style="text-align: center;">🌿 Potato Leaf Disease Detection</h1>', unsafe_allow_html=True)
st.write("Upload an image of a potato leaf to classify its disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=False)

    # Ensure image is in RGB mode
    image = image.convert("RGB")

    # Add a "Predict" button
    if st.button("Predict"):
        # Preprocess the image
        image = image.resize((128, 128))  # Resize to match model input size
        image_array = np.array(image)  # Convert to array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get highest probability class
        confidence = np.max(predictions)  # Get confidence score

        # Display prediction results
        st.subheader("Prediction")
        st.write(f"Predicted Class: {class_labels[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}")

        # Display additional message based on prediction
        if class_labels[predicted_class] == 'Potato_Early_blight':
            st.warning("⚠ This leaf has Early Blight. Consider using fungicides and improving field management.")
        elif class_labels[predicted_class] == 'Potato_Late_blight':
            st.error("🚨 This leaf has Late Blight. Immediate action is needed to prevent crop loss!")
        else:
            st.success("✅ This potato leaf is healthy!")
