import streamlit as st
from PIL import Image
import numpy as np
from image_gen import generate_image
from inference import inference

# Title of the app
st.title("Image Upload and Processing App")

# File uploader allows users to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    with open("./utils/sample.png", "wb") as f:
      f.write(uploaded_file.getbuffer())
    result_text = inference()    
    result_text = str(result_text)
    st.write("Processing Result:")
    st.write(result_text)

else:
    st.write("No image uploaded. You can generate a sample image instead:")
    image_health = st.toggle("Healthy", value=True)
    if st.button("Generate Sample Image"):
        sample_image = generate_image(image_health=image_health)
        if isinstance(sample_image, np.ndarray):
          sample_image = Image.fromarray(sample_image)
        st.image(sample_image, caption='Generated Sample Image', use_container_width=True)
        result_text = inference()
        result_text = str(result_text)
        st.write("Processing Result:")
        st.write(result_text)
