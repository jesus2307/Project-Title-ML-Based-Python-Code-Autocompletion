import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Asegurar que 'src' está en el PYTHONPATH para permitir la importación del modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import CodeCompletionModel  # Importar el modelo de autocompletado de código

# Definición de la clase Dataset para el entrenamiento
class CodeDataset(Dataset):
    """
    Conjunto de datos basado en una secuencia de tokens.
    Cada muestra consiste en:
    - Entrada: token actual
    - Objetivo: siguiente token en la secuencia
    """
    def __init__(self, tokenized_data):
        self.tokens = tokenized_data  # Lista de tokens

    def __len__(self):
        """Devuelve el número total de ejemplos en el conjunto de datos."""
        return len(self.tokens) - 1  # Se usa len() - 1 para evitar el desbordamiento

    def __getitem__(self, idx):
        """
        Retorna un par (entrada, objetivo).
        - Entrada: token actual
        - Objetivo: siguiente token esperado
        """
        return torch.tensor(self.tokens[idx]), torch.tensor(self.tokens[idx + 1])

# Función para cargar datos de entrenamiento desde un archivo JSON
def load_training_data(file_path):
    """
    Carga datos tokenizados desde un archivo JSON y los aplanará si es necesario.

    Parámetros:
        file_path (str): Ruta del archivo JSON con los datos tokenizados.

    Retorna:
        flattened_tokens (list[int]): Lista de tokens en formato plano.

    Excepciones:
        ValueError: Si el archivo JSON no contiene una lista válida de tokens.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)  # Cargar los datos en formato JSON

    # Aplanar la lista si los datos están anidados
    if isinstance(data, list):
        flattened_tokens = (
            [token for sublist in data for token in sublist] if all(isinstance(sublist, list) for sublist in data) else data
        )
        return flattened_tokens
    else:
        raise ValueError("El archivo JSON no contiene una lista válida de tokens.")

# Función principal para entrenar el modelo
def train_model():
    """
    Carga los datos de entrenamiento, define el modelo y lo entrena.
    El entrenamiento optimiza la predicción del siguiente token en una secuencia de código.
    """
    # Ruta del archivo de entrenamiento
    train_data_path = os.path.join(os.path.dirname(__file__), '../data/train.json')
    
    # Cargar los tokens desde el archivo JSON
    tokens = load_training_data(train_data_path)

    # Verificar que los datos sean válidos
    if not tokens or not all(isinstance(i, int) for i in tokens):
        raise ValueError("Los datos de entrenamiento deben ser una lista de enteros.")

    # Crear el conjunto de datos y el DataLoader
    dataset = CodeDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Lotes de tamaño 2, con barajado aleatorio

    # Definir la arquitectura del modelo
    vocab_size = max(tokens) + 1  # El tamaño del vocabulario se determina dinámicamente a partir de los datos
    model = CodeCompletionModel(vocab_size, embed_dim=32, hidden_dim=64)  # Crear la red neuronal

    # Definir el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam con tasa de aprendizaje 0.001
    criterion = nn.CrossEntropyLoss()  # Función de pérdida para clasificación

    # Iniciar el entrenamiento del modelo
    for epoch in range(5):  # Entrenar por 5 épocas (puede aumentarse según necesidad)
        for x, y in dataloader:
            x = x.unsqueeze(0)  # Ajustar la forma de entrada para que coincida con la red neuronal
            y_pred = model(x)  # Realizar predicción con el modelo

            # Calcular la pérdida
            loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))
            
            # Optimización
            optimizer.zero_grad()  # Reiniciar gradientes
            loss.backward()  # Retropropagación
            optimizer.step()  # Actualizar parámetros del modelo

        # Imprimir el valor de la pérdida en cada época
        print(f"Época {epoch+1}, Pérdida: {loss.item()}")

# Ejecutar el entrenamiento si el script se ejecuta directamente
if __name__ == "__main__":
    train_model()
