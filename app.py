import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from typing import Union, Tuple
import io

# Assuming 'loadimg' is a pip-installable library as in your original requirements
# If not, its functionality (loading image from path/URL to PIL) needs to be ensured.
# For example, using:
# from PIL import Image
# import requests
# from io import BytesIO
# def load_img_alternative(path_or_url, output_type="pil"):
#     if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
#         response = requests.get(path_or_url)
#         response.raise_for_status()
#         img = Image.open(BytesIO(response.content))
#     else:
#         img = Image.open(path_or_url)
#     if output_type == "pil":
#         return img.convert("RGB") if img.mode != 'RGB' else img
#     # Add other output_type handling if needed
#     return img

# Using the original load_img, ensure it's in your environment
try:
    from loadimg import load_img
except ImportError:
    st.error("The 'loadimg' library is not installed. Please add it to your requirements.txt or ensure it's available.")
    st.stop() # Stop execution if loadimg is not found


# --- Model Loading and Configuration ---
@st.cache_resource # Cache the model resource across reruns
def load_model():
    """Loads the BiRefNet model and moves it to the appropriate device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    model.to(device)
    return model, device

torch.set_float32_matmul_precision(["high", "highest"][0])
birefnet, device = load_model()

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# --- Core Processing Functions ---
def process_image_core(image: Image.Image) -> Image.Image:
    """
    Apply BiRefNet-based image segmentation to remove the background.
    Args:
        image (PIL.Image): The input RGB image.
    Returns:
        PIL.Image: The image with the background removed, using the segmentation mask as transparency.
    """
    image_size = image.size
    # Ensure image is RGB before transform
    input_img_rgb = image.convert("RGB")
    input_tensor = transform_image(input_img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()
    pred_mask_tensor = preds[0].squeeze()
    
    pred_pil_mask = transforms.ToPILImage()(pred_mask_tensor)
    mask_resized = pred_pil_mask.resize(image_size, Image.LANCZOS) # Use LANCZOS for better quality resize

    # Create a copy to put alpha to avoid modifying the original if it's from cache or input
    output_image = image.copy()
    if output_image.mode != 'RGBA':
        output_image = output_image.convert('RGBA') # Ensure image is RGBA to accept alpha
    
    output_image.putalpha(mask_resized)
    return output_image

def fn_streamlit(image_input: Union[Image.Image, str]) -> Tuple[Image.Image, Image.Image]:
    """
    Remove background for Streamlit app.
    Args:
        image_input (PIL.Image or str): PIL image or path/URL.
    Returns:
        tuple: (processed_image, original_image)
    """
    if isinstance(image_input, str): # If it's a path or URL
        im = load_img(image_input, output_type="pil")
    else: # If it's already a PIL image
        im = image_input
    
    original_rgb = im.convert("RGB") # Keep a clean RGB version of the original
    processed_image = process_image_core(original_rgb.copy()) # Process a copy
    return processed_image, original_rgb

# --- Streamlit UI ---
st.title("Background Removal Tool")

tab1, tab2, tab3 = st.tabs(["Upload Image", "Process from URL", "Upload & Download PNG"])

# --- Tab 1: Image Upload ---
with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        input_image_pil = Image.open(uploaded_file)
        
        with st.spinner("Processing image..."):
            processed_image, original_image = fn_streamlit(input_image_pil)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
        with col2:
            st.subheader("Background Removed")
            st.image(processed_image, use_column_width=True)

            # Download button for processed image
            buf = io.BytesIO()
            processed_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Processed PNG",
                data=byte_im,
                file_name=f"bg_removed_{uploaded_file.name.split('.')[0]}.png",
                mime="image/png"
            )

# --- Tab 2: URL Input ---
with tab2:
    st.header("Process Image from URL")
    url_example = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg" #
    url = st.text_input("Enter image URL:", value=url_example)

    if url and st.button("Process from URL"):
        try:
            with st.spinner("Loading and processing image from URL..."):
                # load_img should handle URL, if not, use requests+PIL alternative shown in comments
                input_image_from_url = load_img(url, output_type="pil")
                processed_image_url, original_image_url = fn_streamlit(input_image_from_url)
            
            col1_url, col2_url = st.columns(2)
            with col1_url:
                st.subheader("Original Image")
                st.image(original_image_url, use_column_width=True)
            with col2_url:
                st.subheader("Background Removed")
                st.image(processed_image_url, use_column_width=True)

                # Download button
                buf_url = io.BytesIO()
                processed_image_url.save(buf_url, format="PNG")
                byte_im_url = buf_url.getvalue()
                st.download_button(
                    label="Download Processed PNG",
                    data=byte_im_url,
                    file_name="bg_removed_from_url.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Error processing URL: {e}")
            st.error("Please ensure the URL points directly to an image file (e.g., .jpg, .png).")

# --- Tab 3: File Output (Upload, Process, Download) ---
with tab3:
    st.header("Upload Image for Direct PNG Download")
    uploaded_file_direct = st.file_uploader("Upload image file...", type=["png", "jpg", "jpeg"], key="direct_upload")

    if uploaded_file_direct is not None:
        input_image_direct_pil = Image.open(uploaded_file_direct)
        
        with st.spinner("Processing for download..."):
            # The original process_file function saved to disk.
            # Here, we'll process and offer for download directly.
            # Ensure input_image_direct_pil is RGB for processing
            img_rgb = input_image_direct_pil.convert("RGB")
            transparent_image = process_image_core(img_rgb)

        st.subheader("Processed Image Preview")
        st.image(transparent_image, use_column_width=True, caption="Background Removed")

        buf_direct = io.BytesIO()
        transparent_image.save(buf_direct, format="PNG")
        byte_im_direct = buf_direct.getvalue()
        
        st.download_button(
            label="Download Processed PNG File",
            data=byte_im_direct,
            file_name=f"processed_{uploaded_file_direct.name.split('.')[0]}.png",
            mime="image/png"
        )

st.markdown("---")
st.markdown("Powered by BiRefNet and Streamlit.")

# To run this app:
# 1. Save as app_st.py (or any other .py name)
# 2. Ensure all dependencies from the updated requirements.txt are installed.
# 3. Run `streamlit run app_st.py` in your terminal.
