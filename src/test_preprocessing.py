# test_preprocessing.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.preprocessing import load_captions, get_image_path, preprocess_image

# Test caption loading
caption_file = 'data/dataset/flickr30k_images/results.csv'
captions = load_captions(caption_file)
print(f"Loaded captions for {len(captions)} images.")
print("Sample captions:", captions[list(captions.keys())[0]])  # Print captions for one image

# Test image path construction
image_name = list(captions.keys())[0]
image_path = get_image_path(image_name)
print(f"Image path for '{image_name}': {image_path}")

# Test image preprocessing
preprocessed_image = preprocess_image(image_name)
print(f"Preprocessed image shape: {preprocessed_image.shape}")
