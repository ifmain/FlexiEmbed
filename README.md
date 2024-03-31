# FlexiEmbed
Dynamic Image Embedding Extractor
# Non-Standard Size Image Embedding Extraction

This project provides a Python script for extracting embeddings from images of non-standard sizes using a pre-trained Vision Transformer (ViT) model. The script segments an input image into 224x224 squares, processes each square through the ViT model to obtain embeddings, and then reassembles these embeddings into a final, unified embedding structure. This approach allows for the processing of images of arbitrary size, addressing the common constraint of needing standard-sized inputs for deep learning models.

## Overview

The script consists of several key functions:

- `auto_overlap_crop(image, square_size)`: Segments the input image into 224x224 squares, adding padding if necessary.
- `get_embeddings_from_squares(squares, model, feature_extractor)`: Processes each square through the ViT model to obtain embeddings.
- `reshape_embeddings(embeddings)`: Reshapes and reassembles the embeddings into a single structure.
- `trim_edges_embeddings(image, final_embeddings)`: Trims the padded edges from the final embedding structure to match the original image dimensions.

## Installation

To use this script, you will need Python 3.6 or later. You'll also need to install the required dependencies:

```bash
pip install torch torchvision transformers Pillow
```

Ensure you have a suitable environment for running PyTorch models, possibly configuring CUDA for GPU acceleration if supported and desired.

## Usage

1. Clone this repository to your local machine.
2. Place the image you wish to process into the same directory as the script or specify its path.
3. Run the script with Python:

```bash
python main.py
```

### Code Example

Here's a brief example of how to use the provided functions in a script:

```python
from FlexiEmbed import *

model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)

img_path = 'image.jpg'
image = Image.open(img_path)

square_size = 224

squares = auto_overlap_crop(image, square_size)
embeddings = get_embeddings_from_squares(squares, model, feature_extractor)
final_embeddings = reshape_embeddings(embeddings)
trimmed_final_embeddings=trim_edges_embeddings(image,final_embeddings)

print("Size of the final embedding array:", trimmed_final_embeddings.shape)
```

## Contribution

Contributions are welcome! If you have suggestions for improvement or have found bugs, please open an issue or a pull request.
