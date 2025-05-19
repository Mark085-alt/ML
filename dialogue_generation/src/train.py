from src.data_loader import load_data, preprocess
from src.model import build_seq2seq
from tensorflow.keras.utils import to_categorical
import numpy as np

def train():
    contexts, responses = load_data('data/dialogues.csv')
    x, y, tokenizer = preprocess(contexts, responses)

    y_input = y[:, :-1]
    y_target = y[:, 1:]
    y_target_cat = to_categorical(y_target, num_classes=10000)

    model = build_seq2seq(vocab_size=10000)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([x, y_input], y_target_cat, epochs=10, batch_size=32)

    model.save('models/seq2seq_model.h5')
    print("Model saved.")
