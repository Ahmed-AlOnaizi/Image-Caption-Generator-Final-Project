# generate_captions.py
import sys
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import h5py

# === GPU Check (Move to very top of the script) ===
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU for inference.")
    except RuntimeError as e:
        print("Error setting up GPU:", e)
else:
    print("No GPU detected. Inference will use CPU.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import resnet_model




bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = load_model('models/image_captioning_model_inceptionv3.h5')
print("Model loaded successfully.")

max_len = 30


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image to extract features using ResNet50.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def extract_features(image_path):
    img_array = preprocess_image(image_path)
    features = resnet_model.predict(img_array)
    return features


def generate_caption(model, image_features, tokenizer, max_len=30):
    """
    Generate a caption for a given image using the trained model.
    """
    in_text = "[CLS]"  
    for _ in range(max_len):
        sequence = tokenizer.encode(in_text, add_special_tokens=False)
        sequence = pad_sequences([sequence], maxlen=max_len)
        
        
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        
        word = tokenizer.decode([yhat])
        
        
        if word == "[SEP]":
            break
        
        
        in_text += " " + word
    
    
    caption = in_text.replace("[CLS]", "").replace("[SEP]", "").strip()
    return caption

def beam_search_decoder(model, image_features, tokenizer, max_len=30, beam_width=3):
    """
    Beam Search Decoder for caption generation.
    """
    
    start_token = "[CLS]"
    sequences = [[start_token, 0.0]]  
    
    
    for _ in range(max_len):
        all_candidates = []
        
        
        for seq, score in sequences:
            sequence = tokenizer.encode(seq, add_special_tokens=False)
            sequence = pad_sequences([sequence], maxlen=max_len)
            
            
            yhat = model.predict([image_features, sequence], verbose=0)
            yhat = np.log(yhat[0] + 1e-10)  
            
            
            top_candidates = np.argsort(yhat)[-beam_width:]
            
            
            for word_id in top_candidates:
                word = tokenizer.decode([word_id])
                new_seq = seq + " " + word
                new_score = score + yhat[word_id]
                all_candidates.append([new_seq, new_score])
        
        
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]
    
    
    best_seq = sequences[0][0]
    
    caption = best_seq.replace("[CLS]", "").replace("[SEP]", "").strip()
    return caption

def top_k_sampling(model, image_features, tokenizer, max_len, k=5, temperature=1.0):
    
    
    in_text = ["[CLS]"]  
    end_token = "[SEP]"

    for _ in range(max_len):
        # Encode 
        sequence = tokenizer.encode(in_text, add_special_tokens=False)
        sequence = pad_sequences([sequence], maxlen=max_len)

        
        predictions = model.predict([image_features, sequence], verbose=0)[0]

        
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        
        top_k_indices = np.argsort(predictions)[-k:]  
        top_k_probs = predictions[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)  

        
        next_index = np.random.choice(top_k_indices, p=top_k_probs)
        next_word = tokenizer.decode([next_index])

        
        if next_word == end_token:
            break

        
        in_text.append(next_word)

    
    return ' '.join(in_text[1:])




def display_image_and_caption(image_path, caption):
    
    img = load_img(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()


def test_image_captioning(image_path):
    
    image_features = extract_features(image_path)
    
    
    
    generated_caption = top_k_sampling(model, image_features, bert_tokenizer, max_len=30, k=5, temperature=1.0)
    print("Generated caption:", generated_caption)
    
    
    display_image_and_caption(image_path, generated_caption)


if __name__ == "__main__":
    
    test_image_path = 'data/images/testingimg4.jpg'
    test_image_captioning(test_image_path)
