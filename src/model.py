from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, Add

def create_model(vocab_size, max_caption_len, embedding_dim=256):
    # Encoder
    image_input = Input(shape=(2048,))  # InceptionV3 output is (2048,)
    image_embedding = Dense(embedding_dim, activation='relu')(image_input)
    
    # Decoder
    caption_input = Input(shape=(max_caption_len,))
    caption_embedding = Embedding(vocab_size, embedding_dim)(caption_input)
    lstm_output = LSTM(256)(caption_embedding)
    
    # Combine the encoded image and the caption
    combined = Add()([image_embedding, lstm_output])
    output = Dense(vocab_size, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model