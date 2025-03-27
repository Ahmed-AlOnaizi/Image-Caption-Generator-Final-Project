from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

import numpy as np


inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def extract_features_for_new_image(image_path):
   
    
    img = load_img(image_path, target_size=(299, 299))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features
    features = inception_model.predict(img_array)
    return features  

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from transformers import BertTokenizer
import pickle
import h5py

# # custom load function
# def custom_load_model(h5_path):
#     with h5py.File(h5_path, mode='r') as f:
#         if 'time_major' in str(f.attrs):
#             del f.attrs['time_major']
#     return load_model(h5_path)


model = load_model('models/image_captioning_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

    

#Function to generate caption
def generate_caption(model, image_features, tokenizer, max_len):
    
    
    in_text = ["[CLS]"]  

    for _ in range(max_len):
        
        sequence = tokenizer.encode(in_text, add_special_tokens=False)
        sequence = pad_sequences([sequence], maxlen=max_len)

        
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = yhat.argmax()  

        # Convert the index to a word 
        word = tokenizer.decode([yhat])
        if word == "[SEP]":  
            break
        in_text.append(word)

    # Join words to form the final caption
    return ' '.join(in_text[1:])

def beam_search(model, image_features, tokenizer, max_len, beam_width=3):
    
    
    start_token = "[CLS]"
    end_token = "[SEP]"
    sequences = [[start_token]]  
    probabilities = [0.0]  

    for _ in range(max_len):
        all_candidates = []

        
        for i in range(len(sequences)):
            seq = sequences[i]
            prob = probabilities[i]

            
            encoded_seq = tokenizer.encode(seq, add_special_tokens=False)
            padded_seq = pad_sequences([encoded_seq], maxlen=max_len)

            
            predictions = model.predict([image_features, padded_seq], verbose=0)[0]
            top_indices = np.argsort(predictions)[-beam_width:]  

            
            for index in top_indices:
                word = tokenizer.decode([index])
                new_seq = seq + [word]
                new_prob = prob + np.log(predictions[index])  
                all_candidates.append((new_seq, new_prob))

        
        all_candidates = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences, probabilities = zip(*all_candidates[:beam_width])

        
        if any(seq[-1] == end_token for seq in sequences):
            break

    
    best_sequence = sequences[0]
    best_sequence = [word for word in best_sequence if word != start_token and word != end_token]
    
    return ' '.join(best_sequence)

def top_k_sampling(model, image_features, tokenizer, max_len, k=5, temperature=1.0):
   
    
    in_text = ["[CLS]"]  
    end_token = "[SEP]"

    for _ in range(max_len):
        
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





def load_image_features(image_id, features_path='data/preprocessed/image_features.npy'):
    features = np.load(features_path, allow_pickle=True).item()
    return features[image_id].reshape((1, 2048))  

# Generate a caption for a test image
sample_image_id = '3367399.jpg'  
sample_image_features = load_image_features(sample_image_id)
generated_caption_sample = top_k_sampling(model, sample_image_features, tokenizer, max_len=30, k=5, temperature=1.0)

print("Generated caption:", generated_caption_sample)

new_image_path = 'data/images/testingimg4.jpg'  


new_image_features = extract_features_for_new_image(new_image_path)


generated_caption = top_k_sampling(model, new_image_features, tokenizer, max_len=30, k=5, temperature=1.0)
generated_caption_beam = beam_search(model, new_image_features, tokenizer, max_len=30, beam_width=3)
# generated_caption = generate_caption(model, new_image_features, tokenizer, max_len=30)
print("Generated caption:", generated_caption)
print("Generated Caption: ", generated_caption_beam)
