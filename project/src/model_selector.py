
# Modelo selector para elegir entre diferentes arquitecturas

import torch.nn as nn

class ModelSelector:
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, model_type='RNN'):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type

    def get_model(self):
        if self.model_type == 'RNN':
            return self._build_rnn_model()
        elif self.model_type == 'LSTM':
            return self._build_lstm_model()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

    def _build_rnn_model(self):
        return nn.Sequential(
            nn.Embedding(self.vocab_size, self.embed_dim),
            nn.RNN(self.embed_dim, self.hidden_dim, batch_first=True),
            nn.Linear(self.hidden_dim, self.vocab_size)
        )

    def _build_lstm_model(self):
        return nn.Sequential(
            nn.Embedding(self.vocab_size, self.embed_dim),
            nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True),
            nn.Linear(self.hidden_dim, self.vocab_size)
        )
