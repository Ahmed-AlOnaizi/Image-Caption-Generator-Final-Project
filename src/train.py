from tensorflow.keras.utils import to_categorical

def train_model(model, images, captions, vocab_size):
    for epoch in range(epochs):
        for i in range(len(images)):
            image_feature = images[i]
            caption_sequence = captions[i]
            
            # Convert the caption sequence to categorical format (one-hot encoding)
            y = to_categorical(caption_sequence[1:], num_classes=vocab_size)
            
            # Train the model on each (image, caption) pair
            model.fit([image_feature, caption_sequence], y, epochs=1)
