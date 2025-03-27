# train.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Add  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import tensorflow as tf

#  GPU Check 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU for training.")
    except RuntimeError as e:
        print("Error setting up GPU:", e)
else:
    print("No GPU detected. Training will use CPU.")

#  Path Setup 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import (
    preprocess_image, 
    load_captions, 
    tokenize_captions, 
    pad_sequences_custom, 
    extract_image_features, 
    base_model,   # Using InceptionV3 model
    load_image_features, 
    load_tokenized_captions
)

# Load Preprocessed Data 
image_features_path = 'data/preprocessed/image_features_inception.npy'  # Using InceptionV3 features
tokenized_captions_path = 'data/preprocessed/tokenized_captions.pkl'

image_features = load_image_features(image_features_path)
tokenized_captions = load_tokenized_captions(tokenized_captions_path)


vocab_size = 30522  
max_len = 30        
embedding_dim = 256 

def build_captioning_model(vocab_size, max_len, embedding_dim=256):
    # Encoder 
    image_input = Input(shape=(2048,), name="image_input")
    image_embedding = Dense(embedding_dim, activation='relu')(image_input)

    # Decoder
    caption_input = Input(shape=(max_len,), name="caption_input")
    caption_embedding = Embedding(vocab_size, embedding_dim)(caption_input)
    lstm_output = LSTM(256)(caption_embedding)

    # Combine 
    combined = Add()([image_embedding, lstm_output])
    output = Dense(vocab_size, activation='softmax')(combined)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    
    return model

# Build the Model
model = build_captioning_model(vocab_size, max_len, embedding_dim)
model.summary()  # Print model architecture

def create_sequences(captions, image_features, max_len, vocab_size):
    X1, X2, y = [], [], []

    for img_id, caption_list in captions.items():
        if img_id in image_features:
            for caption in caption_list:
                for i in range(1, len(caption)):
                    in_seq, out_seq = caption[:i], caption[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    X1.append(image_features[img_id])  
                    X2.append(in_seq)                  
                    y.append(out_seq)                  

    return np.array(X1), np.array(X2), np.array(y)

def data_generator(captions, image_features, max_len, vocab_size, batch_size=64):
    """
    Generator to yield batches of input and output data for training.
    """
    X1, X2, y = [], [], []
    while True:
        for img_id, caption_list in captions.items():
            if img_id in image_features:
                for caption in caption_list:
                    for i in range(1, len(caption)):
                        
                        in_seq, out_seq = caption[:i], caption[i]
                        in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                        
                        # Append to the batch
                        X1.append(image_features[img_id])  # Image features
                        X2.append(in_seq)                  # Partial caption sequence
                        y.append(out_seq)                  # Next word as an integer index
                        
                        
                        if len(X1) == batch_size:
                            yield [np.array(X1), np.array(X2)], np.array(y)
                            X1, X2, y = [], [], []  # Reset batch arrays


X1, X2, y = create_sequences(tokenized_captions, image_features, max_len, vocab_size)
print("Training data prepared:", X1.shape, X2.shape, y.shape)


epochs = 14  
batch_size = 64
steps_per_epoch = len(tokenized_captions) // batch_size


early_stopping = EarlyStopping(
    monitor='loss',
    patience=3,             # Stop training if no improvement for 3 epochs
    restore_best_weights=True
)


print("Starting model training with data generator...")
history = model.fit(
    data_generator(tokenized_captions, image_features, max_len, vocab_size, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[early_stopping]
)
print("Model training complete.")


model_save_path = 'models/image_captioning_model_inceptionv3.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('models/loss_plot_inceptionv3.png')
plt.show()
