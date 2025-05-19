import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(path):
    df = pd.read_csv(path)
    return df['context'].astype(str), df['response'].astype(str)

def preprocess(contexts, responses, vocab_size=10000, max_len=20):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(list(contexts) + list(responses))

    context_seq = tokenizer.texts_to_sequences(contexts)
    response_seq = tokenizer.texts_to_sequences(responses)

    context_pad = pad_sequences(context_seq, maxlen=max_len, padding='post')
    response_pad = pad_sequences(response_seq, maxlen=max_len, padding='post')

    return context_pad, response_pad, tokenizer
