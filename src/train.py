# train.py
import os
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore
from src.preprocessing import preprocess_image, load_captions, tokenize_captions, pad_sequences_custom
from src.model import build_encoder, build_captioning_model

caption_file = 'data/dataset/flickr30k_images/results.csv'
captions = load_captions(caption_file)



def prepare_data(captions, image_features, max_len, vocab_size):
    """
    Create input-output sequences for the model.
    """
    X1, X2, y = [], [], []
    for image_id, caption_list in captions.items():
        for caption in caption_list:
            for i in range(1, len(caption)):
                in_seq, out_seq = caption[:i], caption[i]
                in_seq = pad_sequences_custom([in_seq], max_len)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(image_features[image_id][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def train_model(caption_file, image_dir, max_len, vocab_size):
    """
    Train the image captioning model.
    """
    # Load and preprocess captions
    captions = load_captions(caption_file)
    tokenized_captions = tokenize_captions(captions)
    
    # Load and preprocess images
    encoder = build_encoder()
    image_features = {img: encoder.predict(preprocess_image(f"{image_dir}/{img}")) for img in os.listdir(image_dir)}
    
    # Prepare training data
    X1, X2, y = prepare_data(tokenized_captions, image_features, max_len, vocab_size)
    
    # Build model
    model = build_captioning_model(vocab_size, max_len)
    
    # Train model
    model.fit([X1, X2], y, epochs=10, batch_size=64)
    model.save('models/image_captioning_model.h5')

# Example usage
if __name__ == "__main__":
    train_model('data/flickr8k/captions.txt', 'data/flickr8k/images', max_len=30, vocab_size=30522)
