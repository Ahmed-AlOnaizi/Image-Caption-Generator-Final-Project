def generate_caption(model, image, tokenizer, max_length):
    caption = '<START>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([image, sequence], verbose=0)
        word = tokenizer.index_word[np.argmax(y_pred)]
        caption += ' ' + word
        if word == '<END>':
            break
    return caption
