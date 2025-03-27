
from tensorflow.keras.applications import InceptionV3 # type: ignore
from tensorflow.keras.applications import ResNet50 #type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input #type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess # type: ignore
from transformers import BertTokenizer
import numpy as np
import os
import pandas as pd
import pickle

BASE_IMAGE_DIR = 'data/dataset/flickr30k_images/images/'


# Initialize the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')



def preprocess_image(image_path, model_type='inception'):
    
    if model_type == 'inception':
        target_size = (299, 299)
        preprocess = inception_preprocess
    elif model_type == 'resnet':
        target_size = (224, 224)
        preprocess = resnet_preprocess
    else:
        raise ValueError("Invalid model_type. Choose either 'inception' or 'resnet'.")

    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess(img_array)
    return img_array


def get_image_path(image_name):
    
    return os.path.join(BASE_IMAGE_DIR, image_name)

import os

def extract_image_features(image_dir, model, model_type='inception', batch_size=32):
    
    features = {}
    image_list = [f for f in os.listdir(image_dir) 
                  if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(image_list)

    for i in range(0, num_images, batch_size):
        batch_filenames = image_list[i:i + batch_size]
        batch_images = []

        for filename in batch_filenames:
            try:
                image_path = os.path.join(image_dir, filename)
                img_array = preprocess_image(image_path, model_type=model_type)
                batch_images.append(img_array)
            except PermissionError:
                print(f"PermissionError: Skipping {filename} due to access restrictions.")
                continue

        if batch_images:
            batch_images = np.vstack(batch_images)
            batch_features = model.predict(batch_images, verbose=1)
            for j, filename in enumerate(batch_filenames):
                features[filename] = batch_features[j]

    return features



def load_captions(caption_file):
    
    captions = {}
    # Load CSV file
    df = pd.read_csv(caption_file, delimiter='|')
    
    # print("Column names:", df.columns)

    df.columns = df.columns.str.strip()
    
    for _, row in df.iterrows():
        image_id, caption = row['image_name'], row['comment']
        
        
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)
    
    return captions

def tokenize_captions(captions):
   
    tokenized_captions = {}
    for image_id, caption_list in captions.items():
        
        cleaned_captions = [caption for caption in caption_list if isinstance(caption, str) and caption.strip()]

        
        tokenized_captions[image_id] = [bert_tokenizer.encode(caption, add_special_tokens=True) for caption in cleaned_captions]
    
    return tokenized_captions

def save_image_features(features, output_path='data/preprocessed/image_features.npy'):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features)
    print(f"Image features saved to {output_path}")

def save_tokenized_captions(captions, output_path='data/preprocessed/tokenized_captions.pkl'):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(captions, f)
    print(f"Tokenized captions saved to {output_path}")


def load_image_features(input_path='data/preprocessed/image_features.npy'):
    
    features = np.load(input_path, allow_pickle=True).item()
    print(f"Image features loaded from {input_path}")
    return features

def load_tokenized_captions(input_path='data/preprocessed/tokenized_captions.pkl'):
    
    with open(input_path, 'rb') as f:
        captions = pickle.load(f)
    print(f"Tokenized captions loaded from {input_path}")
    return captions

def pad_sequences_custom(sequences, max_len):
    
    return pad_sequences(sequences, maxlen=max_len, padding='post')
