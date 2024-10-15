# model.py

from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Add # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.applications import InceptionV3 # type: ignore

def build_encoder():
    """
    Build the CNN encoder to extract image features.
    """
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

def build_captioning_model(vocab_size, max_len, embedding_dim=256):
    """
    Define the image captioning model with a CNN encoder and an RNN-based decoder.
    """
    # Encoder
    image_input = Input(shape=(2048,))
    image_embedding = Dense(embedding_dim, activation='relu')(image_input)

    # Decoder
    caption_input = Input(shape=(max_len,))
    caption_embedding = Embedding(vocab_size, embedding_dim)(caption_input)
    lstm_output = LSTM(256)(caption_embedding)

    # Combined model
    combined = Add()([image_embedding, lstm_output])
    output = Dense(vocab_size, activation='softmax')(combined)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
