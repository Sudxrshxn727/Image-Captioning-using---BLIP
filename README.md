# Image Captioning using BLIP (Transformers)

This project demonstrates an image captioning system using BLIP (Bootstrapping Language-Image Pre-training) models from the Hugging Face Transformers library. The application uses Streamlit for the user interface and processes images to generate captions and detailed descriptions.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [File Descriptions](#file-descriptions)
  - [front.py](#frontpy)
  - [IC.py](#icpy)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Overview

This project leverages BLIP models to generate captions for images. The main components include a Streamlit-based user interface (`front.py`) and the core image captioning logic (`IC.py`). 

## Setup

To set up the project, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run the Streamlit application.

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
streamlit run front.py
```

## File Descriptions

### front.py

This file contains the Streamlit code for the user interface.

**Description:**

- `front.py` sets up a web interface using Streamlit.
- Users can upload an image through the interface.
- The uploaded image is displayed, and a caption and detailed description are generated using functions from `IC.py`.

### IC.py

This file contains the core functions and model loading logic for image captioning.

**Description:**

- `IC.py` handles loading of the models and tokenizers.
- Contains functions for image processing and generating captions.

**Functions:**

1. **load_image(image_path):**
   - Loads an image from the given path.

2. **get_caption(image):**
   - Generates a caption for the given image using the loaded BLIP model.

3. **get_detailed_description(image):**
   - Provides a more detailed description of the image.

**Models and Tokenizers:**

- `caption_model`: BLIP model for generating image captions.
- `caption_tokenizer`: Tokenizer corresponding to the BLIP model.
- `image_processor`: Processor for preparing images for the model.
- `description_model`: Additional model for generating detailed descriptions.
- `description_processor`: Processor corresponding to the description model.

## How to Run

1. Ensure you have the required dependencies installed.
2. Run the Streamlit application using the command:
   ```bash
   streamlit run front.py
   ```
3. Upload an image through the web interface.
4. View the generated caption and detailed description.

## Dependencies

The project relies on the following libraries:

- `streamlit`
- `torch`
- `PIL`
- `transformers`
- `requests`
- `tqdm`
