import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import streamlit as st
import numpy as np

# Load model function
@st.cache_resource()
def load_model():
    model_id = "nitrosocke/Ghibli-Diffusion"  
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.write("⏳ Loading model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    pipe.enable_attention_slicing()  # Optimize memory
    st.write("✅ Model loaded successfully!")
    return pipe

# Function to generate Ghibli-style image
def generate_ghibli_image(image, pipe, strength=0.6):
    image = image.convert("RGB")
    width, height = image.size
    
    # Resize while maintaining aspect ratio
    ratio = min(512 / width, 512 / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    image = image.resize((new_width, new_height))

    prompt = "A beautiful Ghibli-style anime painting, soft lighting, magical world, highly detailed"
    output_image = pipe(prompt=prompt, image=image, strength=strength).images[0]

    return output_image

# Streamlit UI
st.title("🎨 Studio Ghibli Art Generator ✨")

# Load Model
pipe = load_model()

# File Upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg", "heic"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Set strength slider
    strength = st.slider("🎛 Adjust Stylization Strength", min_value=0.3, max_value=0.8, value=0.6, step=0.05)

    if st.button("🎨 Convert to Ghibli Style"):
        st.write("🔄 Processing...")
        
        # Generate Ghibli-style image
        output = generate_ghibli_image(image, pipe, strength)

        # Show & Save the Output
        st.image(output, caption="🌟 Ghibli-style Artwork", use_container_width=True)
        output.save("ghibli_output.png")
        st.write("📥 [Download your artwork](ghibli_output.png)")
