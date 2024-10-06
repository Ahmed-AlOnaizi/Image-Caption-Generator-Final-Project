from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def preprocess_image(image_path):
    # Load image and resize
    img = image.load_img(image_path, target_size=(299, 299))
    
    # Convert the image to array and preprocess for InceptionV3
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array

# Use pre-trained InceptionV3 as the image feature extractor
def extract_image_features(image_path):
    model = InceptionV3(weights='imagenet')
    img_array = preprocess_image(image_path)
    features = model.predict(img_array)
    return features


import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')

def preprocess_caption(caption):
    # Tokenize and clean the captions
    tokens = nltk.word_tokenize(caption.lower())
    return ' '.join(tokens)

def tokenize_captions(captions):
    tokenizer = Tokenizer(num_words=5000, oov_token='<UNK>')
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    sequences = pad_sequences(sequences, padding='post')
    return sequences, tokenizer