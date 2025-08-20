import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your saved model
@st.cache_resource  # cache so it loads once
def load_flower_model():
    return load_model("flower_model.keras")

model = load_flower_model()

# Set image size (depends on your training setup, ResNet50 expects 224x224)
IMG_SIZE = (128, 128)

# Your class names (update these with the correct ones you used in training)
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Streamlit UI
st.title("🌸 Flower Classifier")
st.write("Upload a flower photo and let the model predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0  # normalize if you trained with rescale=1./255

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.success(f"🌼 Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")
