import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("lung_cancer_model.h5")  # Ensure correct path

# Define class labels
CATEGORIES = ["Lung Adenocarcinoma", "Normal Lung", "Lung Squamous Cell Carcinoma"]

# Function to preprocess image
def preprocess_image(image):
    IMG_SIZE = 224  # Ensure it matches the model's training size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize image
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# --- STREAMLIT UI ---
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #F0F2F6;
        }
        .stApp {
            background-color: #F8F9FA;
            padding: 20px;
            border-radius: 15px;
        }
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
        }
        .upload-box {
            border: 2px dashed #2E86C1;
            padding: 15px;
            border-radius: 10px;
            background-color: #FFFFFF;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="title">Lung Cancer Detection using CNN</p>', unsafe_allow_html=True)
st.write("💡 Upload a histopathological image to classify lung cancer type.")

# File uploader section with styling
st.markdown('<div class="upload-box">📤 Upload an Image (JPG, PNG, JPEG)</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to NumPy array
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as an image

    # Display uploaded image
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Prediction with loading effect
    with st.spinner("🔍 Analyzing Image..."):
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    # Display result with formatting
    st.markdown(f"""
        <div style="text-align:center; background-color:#2E86C1; color:white; padding:10px; border-radius:10px;">
            <h2>🧬 Prediction: {CATEGORIES[predicted_class]}</h2>
            <h4>Confidence: {confidence:.2f}%</h4>
        </div>
    """, unsafe_allow_html=True)

    # Add a progress bar to show confidence level
    st.progress(int(confidence))
