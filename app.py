import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image
import io

# Model file URL and path
file_id = "1NLj9MTHYH5z3N6d5YaXfMEG3qWmRyPNH"
url = f'https://drive.google.com/file/d/1NLj9MTHYH5z3N6d5YaXfMEG3qWmRyPNH/view?usp=sharing'  # Direct download URL

model_path = "trained_plant_disease_model.keras"

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
else:
    st.success("Model file found.")

# Model prediction function
def model_prediction(test_image):
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    # Check if an image has been uploaded
    if test_image is not None:
        # Open the image from the file uploader
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize image to match model input size
        input_arr = np.array(image)  # Convert to numpy array
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max prediction
    else:
        st.error("No image uploaded!")
        return None

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Home page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Disease recognition page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        st.image(test_image, width=400, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.snow()  # Show snow effect during prediction
        st.write("Our Prediction:")
        
        result_index = model_prediction(test_image)
        
        if result_index is not None:
            # Define the classes for the prediction
            class_name = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']
            st.success(f"Model predicts: {class_name[result_index]}")
        else:
            st.error("Prediction failed. Please try again.")
