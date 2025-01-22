
# Modelo de lenguaje basado en PyTorch para autocompletado de código

import torch.nn as nn

class CodeCompletionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CodeCompletionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
