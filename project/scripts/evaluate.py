import sys
import os

# Agregar el directorio padre al PYTHONPATH para permitir la importación de módulos desde 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from src.model import CodeCompletionModel  # Importar el modelo de autocompletado de código

# Definición del conjunto de datos (Dataset) para el modelo
class CodeDataset(Dataset):
    """
    Clase que representa un conjunto de datos basado en secuencias de tokens.
    Cada muestra es una secuencia de tokens donde:
    - Entrada: todos los tokens excepto el último.
    - Objetivo: todos los tokens excepto el primero (para predecir el siguiente).
    """
    def __init__(self, sequences):
        self.sequences = sequences  # Lista de secuencias de tokens

    def __len__(self):
        return len(self.sequences)  # Número total de secuencias en el conjunto de datos

    def __getitem__(self, idx):
        sequence = self.sequences[idx]  # Obtener la secuencia en la posición idx
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[1:], dtype=torch.long)
        # Retorna la secuencia de entrada y la secuencia objetivo desplazada en un token

# Función de padding para manejar secuencias de diferentes longitudes
def collate_fn(batch):
    """
    Función para hacer padding a las secuencias en un batch.
    - Rellena con ceros (<PAD>) para igualar la longitud de todas las secuencias en el lote.
    """
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # Aplicar padding en la entrada
    targets = pad_sequence(targets, batch_first=True, padding_value=0)  # Aplicar padding en los objetivos
    return inputs, targets  # Retorna las secuencias con padding

# Función de evaluación del modelo
def evaluate_model():
    """
    Evalúa el modelo de autocompletado de código basado en una red neuronal recurrente (RNN).
    """
    print("Evaluating RNN model...")

    # Ruta del archivo de datos de prueba
    test_data_path = os.path.join("data", "test.json")
    
    # Cargar datos de prueba desde un archivo JSON
    try:
        with open(test_data_path, "r") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Test data file not found at {test_data_path}. Please ensure it exists.")
        return

    # Crear el conjunto de datos y el DataLoader
    dataset = CodeDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Definir parámetros del modelo
    model_path = os.path.join("models", "rnn_model.pth")
    vocab_size = 27  # Número de palabras en el vocabulario (ajustar según vocab.json)
    embed_dim = 64  # Dimensión de los embeddings
    hidden_dim = 128  # Dimensión de la capa oculta de la RNN

    # Cargar el modelo
    model = CodeCompletionModel(vocab_size, embed_dim, hidden_dim)

    # Verificar si el modelo ya ha sido entrenado y sus pesos guardados
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Initializing a new model with random weights.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Initialized model weights saved to {model_path}.")

    # Cargar los pesos entrenados del modelo si existen
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Configurar el modelo en modo evaluación
    print("Model loaded successfully.")

    # Evaluación del modelo
    total_loss = 0
    total_correct = 0
    total_count = 0
    criterion = torch.nn.CrossEntropyLoss()  # Función de pérdida para clasificación

    with torch.no_grad():  # Desactivar el cálculo de gradientes (mejora la eficiencia en inferencia)
        for inputs, targets in dataloader:
            outputs = model(inputs)  # Obtener las predicciones del modelo
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))  # Calcular la pérdida
            total_loss += loss.item()

            # Obtener las predicciones más probables
            predictions = torch.argmax(outputs, dim=-1)

            # Calcular el número de aciertos
            total_correct += (predictions == targets).sum().item()
            total_count += targets.numel()  # Número total de elementos en los objetivos

    # Calcular la precisión del modelo en el conjunto de prueba
    accuracy = total_correct / total_count * 100
    print(f"Evaluation complete. Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Ejecutar la evaluación si el script se ejecuta directamente
if __name__ == "__main__":
    evaluate_model()
