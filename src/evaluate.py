import os
from src.preprocessing import preprocess_image
from src.utils import rank_and_select_best_caption
from tensorflow.keras.models import load_model  # type: ignore
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

image_dir = 'data/dataset/flickr30k_images/images'

def generate_caption(image, model, tokenizer, max_len):
    
    
    pass

def evaluate_model(image_dir, model, tokenizer, max_len):
    
    for image_file in os.listdir(image_dir):
        image = preprocess_image(os.path.join(image_dir, image_file))
        caption = generate_caption(image, model, tokenizer, max_len)
        print(f"Generated Caption for {image_file}: {caption}")

# Load model and evaluate
model = load_model('models/image_captioning_model.h5')
evaluate_model('data/dataset/flickr30k_images/images', model, bert_tokenizer, max_len=30)
