import requests

from IPython.display import display

import torch

# Image Processing
from PIL import Image

# Transformer and Pretrained Model
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast,BlipProcessor, BlipForConditionalGeneration

import urllib.parse as parse
import os

# Managing loading processsing
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# Assign available GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
caption_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load a more advanced model for detailed descriptions

description_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
description_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Verify url
def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


# Load an image
def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)


def get_caption(caption_model, image_processor,caption_tokenizer, image_path):
    image = load_image(image_path)

    # Preprocessing the Image
    img = image_processor(image, return_tensors="pt").to(device)

    # Generating captions
    output = caption_model.generate(**img)

    # decode the output
    caption = caption_tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return caption


def get_detailed_description(description_model, description_processor, image_path):
    image = load_image(image_path)
    if image is None:
        return "Invalid image path or URL."

    # Preprocessing the Image
    inputs = description_processor(images=image, return_tensors="pt").to(device)

    # Generating detailed description
    output = description_model.generate(**inputs)

    # Decode the output
    description = description_processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return description


#url = "https://images.pexels.com/photos/101667/pexels-photo-101667.jpeg?auto=compress&cs=tinysrgb&w=600"

# Display Image
#display(load_image(url))

# Display Caption
#get_caption(model, image_processor, tokenizer, url)