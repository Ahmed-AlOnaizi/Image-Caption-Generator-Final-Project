# test_preprocessing.py
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.preprocessing import load_captions, get_image_path, preprocess_image, tokenize_captions, extract_image_features, base_model, save_image_features, save_tokenized_captions

from src.preprocessing import (
    load_captions, 
    get_image_path, 
    preprocess_image, 
    tokenize_captions, 
    extract_image_features, 
    base_model, 
    resnet_model,           
    save_image_features, 
    save_tokenized_captions
)
image_dir = 'data/dataset/flickr30k_images/images'


try:
    files = os.listdir(image_dir)
    print("Files in dataset directory:", files)
except PermissionError:
    print("PermissionError: Python doesn't have access to this directory.")

# Test caption loading
caption_file = 'data/dataset/flickr30k_images/results.csv'
captions = load_captions(caption_file)
print(f"Loaded captions for {len(captions)} images.")
print("Sample captions:", captions[list(captions.keys())[0]])  # Print captions for one image

tokenized_captions = tokenize_captions(captions)
save_tokenized_captions(tokenized_captions)

# image_features = extract_image_features(image_dir, base_model)
# print(f"Extracted features for {len(image_features)} images.")

# save_image_features(image_features)  # Saves to 'data/preprocessed/image_features.npy' by default

# print("Preprocessing Complete")

# # Test image path construction
# image_name = list(captions.keys())[0]
# image_path = get_image_path(image_name)
# print(f"Image path for '{image_name}': {image_path}")

# # Test image preprocessing
# preprocessed_image = preprocess_image(image_name)
# print(f"Preprocessed image shape: {preprocessed_image.shape}")

# def extract_features_for_dataset(image_dir, model, output_path='data/preprocessed/image_features_resnet.npy'):
#     """
#     Extract features for all images in a directory using a given model.
#     Args:
#         image_dir (str): Path to the directory containing images.
#         model: Feature extraction model (e.g., ResNet).
#         output_path (str): Path to save the extracted features.
#     """
#     features = {}
#     image_list = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     for i, image_name in enumerate(image_list):
#         try:
#             image_path = os.path.join(image_dir, image_name)
#             features[image_name] = extract_features_with_resnet(image_path)
#             print(f"Extracted features for {i+1}/{len(image_list)} images", end='\r')
#         except Exception as e:
#             print(f"Error processing {image_name}: {e}")
#             continue

#     # Save features to a file
#     np.save(output_path, features)
#     print(f"Features saved to {output_path}")

# # Example usage
# image_dir = 'data/dataset/flickr30k_images/images'  # Update this path
# extract_features_for_dataset(image_dir, resnet_model)

# **InceptionV3 Features**
image_features = extract_image_features(image_dir, base_model, model_type='inception')
print(f"Extracted InceptionV3 features for {len(image_features)} images.")
save_image_features(image_features, output_path='data/preprocessed/image_features_inception.npy')

# **ResNet50 Features**
image_features_resnet = extract_image_features(image_dir, resnet_model, model_type='resnet')
print(f"Extracted ResNet50 features for {len(image_features_resnet)} images.")
save_image_features(image_features_resnet, output_path='data/preprocessed/image_features_resnet.npy')

print("Preprocessing Complete")
