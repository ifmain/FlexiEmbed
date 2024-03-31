from PIL import Image, ImageOps
import math
import torch
from transformers import ViTFeatureExtractor, ViTModel
import numpy as np

def auto_overlap_crop(image, square_size):
    width, height = image.size
    num_squares_width = math.ceil(width / square_size)
    num_squares_height = math.ceil(height / square_size)
    new_width = square_size * num_squares_width
    new_height = square_size * num_squares_height
    new_image = ImageOps.expand(image, border=(0, 0, new_width-width, new_height-height), fill='black')
    
    squares = []
    for i in range(num_squares_height):
        for j in range(num_squares_width):
            x = j * square_size
            y = i * square_size
            square = new_image.crop((x, y, x + square_size, y + square_size))
            squares.append(square)
    
    return squares


def get_embeddings_from_squares(squares, model, feature_extractor):
    embeddings = []
    i=0
    for square in squares:
        print(f'{i+1}/{len(squares)}')
        inputs = feature_extractor(images=square, return_tensors="pt")
        outputs = model(**inputs)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :].detach().numpy()  # Исключаем [CLS] токен
        embeddings.append(patch_embeddings)
        i+=1
    
    return np.array(embeddings)

def reshape_embeddings(embeddings):
    reshaped = [e.reshape(1, 14, 14, 768) for e in embeddings]
    reshaped = np.concatenate(reshaped, axis=0)
    squares_per_side = int(np.sqrt(reshaped.shape[0]))
    reshaped = reshaped.reshape(squares_per_side, squares_per_side, 14, 14, 768)
    final_shape = (squares_per_side * 14, squares_per_side * 14, 768)
    final_embeddings = np.zeros(final_shape)
    
    for i in range(squares_per_side):
        for j in range(squares_per_side):
            final_embeddings[i*14:(i+1)*14, j*14:(j+1)*14, :] = reshaped[i, j]
            
    return final_embeddings

def trim_edges_embeddings(image, final_embeddings):
    width_tiles_target = image.size[0] // 224
    height_tiles_target = image.size[1] // 224

    w_for_remove = (image.size[0] - (width_tiles_target * 224)) // 16
    h_for_remove = (image.size[1] - (height_tiles_target * 224)) // 16

    trimmed_embeddings = final_embeddings[h_for_remove:-h_for_remove, w_for_remove:-w_for_remove, :]
    return trimmed_embeddings