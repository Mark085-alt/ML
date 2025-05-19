from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def build_seq2seq(vocab_size, embedding_dim=128, latent_dim=256, max_len=20):
    enc_input = Input(shape=(max_len,))
    enc_emb = Embedding(vocab_size, embedding_dim)(enc_input)
    _, h, c = LSTM(latent_dim, return_state=True)(enc_emb)

    dec_input = Input(shape=(max_len,))
    dec_emb = Embedding(vocab_size, embedding_dim)(dec_input)
    dec_output = LSTM(latent_dim, return_sequences=True)(dec_emb, initial_state=[h, c])
    output = Dense(vocab_size, activation='softmax')(dec_output)

    model = Model([enc_input, dec_input], output)
    return model
