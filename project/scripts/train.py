import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Asegurar que src está en el PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import CodeCompletionModel

class CodeDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokens = tokenized_data

    def __len__(self):
        return len(self.tokens) - 1

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx]), torch.tensor(self.tokens[idx + 1])

def load_training_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Aplanar la lista si es una lista de listas
    if isinstance(data, list):
        flattened_tokens = [token for sublist in data for token in sublist] if all(isinstance(sublist, list) for sublist in data) else data
        return flattened_tokens
    else:
        raise ValueError("El archivo JSON no contiene una lista válida de tokens.")

def train_model():
    # Cargar datos reales
    train_data_path = os.path.join(os.path.dirname(__file__), '../data/train.json')
    tokens = load_training_data(train_data_path)

    if not tokens or not all(isinstance(i, int) for i in tokens):
        raise ValueError("Los datos de entrenamiento deben ser una lista de enteros.")

    dataset = CodeDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Modelo
    vocab_size = max(tokens) + 1  # Determinar el vocabulario a partir de los datos
    model = CodeCompletionModel(vocab_size, embed_dim=32, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Entrenamiento
    for epoch in range(5):  # Sustituir con más épocas si es necesario
        for x, y in dataloader:
            x = x.unsqueeze(0)  # Asegurar que la entrada tiene la forma adecuada
            y_pred = model(x)
            loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Época {epoch+1}, Pérdida: {loss.item()}")

if __name__ == "__main__":
    train_model()
