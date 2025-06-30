# Etana Disasa \\
# Department of Data Science \\
# College of Computer and Information Sciences \\
# Regis University \\
# 3333 Regis Boulevard, Denver, CO 80221 \\
#----------------------------------------------------

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load model
MODEL_PATH = 'Models/APTOS_best_model.keras' # After several attepts, it appears that the APTOS model predicted better. 

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Then recompile it manually (if needed)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Prevent further errors if model not loaded

# Define image size and class labels mapping
IMAGE_SIZE = (224, 224)
label_map = {'0': 'No Disease Risk', '1': 'Disease Risk'}

# Title
st.title("Retinal Disease Classifier")
st.write("Upload a retinal image (PNG/JPG) to classify the disease risk level.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) 

if uploaded_file is not None:
    if model is None:
        st.error("Model failed to load, prediction not possible.")
    else:
        # Display uploaded image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        img_resized = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict with spinner and error handling
        try:
            with st.spinner('Classifying...'):
                preds = model.predict(img_batch)

            pred_index = np.argmax(preds, axis=1)[0]
            pred_class = label_map[str(pred_index)]
            confidence = np.max(preds) * 100

            # Show result
            st.subheader("Prediction")
            st.write(f"**Class:** {pred_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")

        except Exception as e:
            st.error(f"Prediction error: {e}")
