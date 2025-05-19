import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('models/seq2seq_model.h5')

def generate_reply(input_text, tokenizer, max_len=20):
    seq = tokenizer.texts_to_sequences([input_text])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post')

    # Use dummy decoder input for simplicity
    decoder_input = np.zeros((1, max_len))
    pred = model.predict([pad_seq, decoder_input])
    response = []

    for i in np.argmax(pred[0], axis=-1):
        for word, index in tokenizer.word_index.items():
            if index == i:
                response.append(word)
                break
    return ' '.join(response)
