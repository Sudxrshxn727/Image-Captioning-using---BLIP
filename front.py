import streamlit as st
from IC import get_caption, get_detailed_description, load_image, caption_model, caption_tokenizer, image_processor, description_model, description_processor
from IPython.display import display
from PIL import Image
import requests
import os


st.set_page_config(page_title="Create Captions for Images",
                   page_icon='🖼️',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Image Caption Generator 🖼️")

custom_css = """
<style>
body {
    background-color: #FFE4B5; /* Lightest orange background color */
    color: #000000; /* Black text color */
}
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

input_url = st.text_input("URL :")

uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if input_url:
    image = load_image(input_url)
    display(image)
    if image:
        # Caption
        st.image(image, caption='Input Image from URL', use_column_width=True)
        description = get_detailed_description(description_model, description_processor, input_url)
        st.write(f"Caption : {description}")
    else:
        st.write("Failed to load image from URL. Please check the URL or path.")
elif uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded file to a temporary path for processing
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    # Detailed Description
    description = get_detailed_description(description_model, description_processor, temp_path)
    st.write(f"Caption : {description}")

    # Optionally, remove the temporary file after processing
    os.remove(temp_path)