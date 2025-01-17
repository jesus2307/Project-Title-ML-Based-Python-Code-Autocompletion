
# Entrenamiento del modelo de autocompletado
import torch
from torch.utils.data import DataLoader, Dataset
from src.model import CodeCompletionModel
import torch
import torch.nn as nn  # Importación de torch.nn
from torch.utils.data import DataLoader, Dataset
from src.model import CodeCompletionModel


class CodeDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens) - 1

    def __getitem__(self, idx):
        return self.tokens[idx], self.tokens[idx + 1]

def train_model():
    # Datos y configuración
    tokens = [1, 2, 3, 4, 5]  # Sustituir con datos reales tokenizados
    dataset = CodeDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Modelo
    vocab_size = 10
    model = CodeCompletionModel(vocab_size, embed_dim=32, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Entrenamiento
    for epoch in range(5):  # Sustituir con más épocas si es necesario
        for x, y in dataloader:
            y_pred = model(x)
            loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Época {epoch+1}, Pérdida: {loss.item()}")

if __name__ == "__main__":
    train_model()
