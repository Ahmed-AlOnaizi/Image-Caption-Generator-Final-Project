# preprocessing.py

from tensorflow.keras.applications import InceptionV3 # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from transformers import BertTokenizer
import numpy as np
import os
import pandas as pd

BASE_IMAGE_DIR = 'data/dataset/flickr30k_images/images/'


# Initialize the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_image(image_path, target_size=(299, 299)):
    """
    Load and preprocess an image for the model.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_image_path(image_name):
    """
    Construct the full path for an image given its name.
    Args:
        image_name (str): The image file name (e.g., '12345.jpg').
    Returns:
        str: The full path to the image file.
    """
    return os.path.join(BASE_IMAGE_DIR, image_name)

def load_captions(caption_file):
    """
    Load captions from a CSV file and organize them by image ID.
    Args:
        caption_file (str): Path to the CSV file containing captions.
    Returns:
        dict: A dictionary mapping image IDs to lists of captions.
    """
    captions = {}
    # Load CSV file
    df = pd.read_csv(caption_file)
    
    # Assuming the CSV has columns 'image_id' and 'caption'
    for _, row in df.iterrows():
        image_id, caption = row['image_name'], row['comment']
        
        # Organize captions by image_id
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)
    
    return captions

def tokenize_captions(captions):
    """
    Tokenize captions using BERT's tokenizer.
    """
    tokenized_captions = {}
    for image_id, caption_list in captions.items():
        tokenized_captions[image_id] = [bert_tokenizer.encode(caption, add_special_tokens=True) for caption in caption_list]
    return tokenized_captions

def pad_sequences_custom(sequences, max_len):
    """
    Pad sequences to a fixed length for consistency.
    """
    return pad_sequences(sequences, maxlen=max_len, padding='post')
