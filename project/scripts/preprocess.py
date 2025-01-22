# Script para preprocesar datos de un dataset público como Py150
# Convierte código fuente en secuencias tokenizadas y genera archivos JSON con datos preprocesados.

import os
import json
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba

def build_vocab(data):
    """
    Construye un vocabulario a partir de los datos de entrada.

    Parámetros:
        data (list[str]): Lista de líneas de código en formato texto.

    Retorna:
        vocab (dict): Diccionario con los tokens únicos y sus respectivos índices.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}  # Añadir tokens especiales: <PAD> para padding y <UNK> para desconocidos
    idx = 2  # Comenzamos el índice en 2 porque 0 y 1 están reservados

    # Recorrer cada línea y dividir en tokens
    for line in data:
        for token in line.split():
            if token not in vocab:
                vocab[token] = idx  # Asignar un índice único a cada token
                idx += 1
    return vocab  # Retornar el diccionario de vocabulario

def tokenize_and_convert_to_ids(data, vocab):
    """
    Tokeniza las líneas de código y las convierte en listas de identificadores numéricos.

    Parámetros:
        data (list[str]): Lista de líneas de código en formato texto.
        vocab (dict): Diccionario de vocabulario con tokens mapeados a índices.

    Retorna:
        tokenized_data (list[list[int]]): Lista de líneas de código tokenizadas como secuencias de IDs.
    """
    tokenized_data = []  # Lista para almacenar los datos tokenizados

    for line in data:
        # Convertir cada token en su ID correspondiente o en <UNK> si no existe en el vocabulario
        tokenized_line = [vocab.get(token, vocab["<UNK>"]) for token in line.split()]
        tokenized_data.append(tokenized_line)  # Agregar la línea tokenizada

    return tokenized_data  # Retornar los datos en formato numérico

def load_and_tokenize_data():
    """
    Carga y tokeniza un conjunto de datos de ejemplo.
    Este conjunto de datos se puede reemplazar con datos reales.

    Retorna:
        data (list[str]): Lista de líneas de código en texto plano.
    """
    data = [
        "def suma ( a , b ) : return a + b",
        "for i in range ( 10 ) : print ( i )",
        "if x > 0 : print ( 'positivo' )",
        "while n > 0 : n -= 1"
    ]
    return data  # Retornar las líneas de código en texto

def save_preprocessed_data():
    """
    Procesa los datos, crea el vocabulario, tokeniza las líneas de código y guarda los archivos en JSON.
    Genera archivos:
    - `train.json`: Datos de entrenamiento tokenizados.
    - `test.json`: Datos de prueba tokenizados.
    - `vocab.json`: Vocabulario con tokens e IDs.

    Los archivos se almacenan en la carpeta `data/`.
    """
    # Cargar los datos sin procesar
    raw_data = load_and_tokenize_data()

    # Construir vocabulario a partir de los datos
    vocab = build_vocab(raw_data)

    # Convertir las líneas de código en secuencias de IDs
    tokenized_data = tokenize_and_convert_to_ids(raw_data, vocab)

    # Dividir los datos en entrenamiento (80%) y prueba (20%)
    train, test = train_test_split(tokenized_data, test_size=0.2, random_state=42)

    # Crear la carpeta 'data' si no existe
    os.makedirs("data", exist_ok=True)

    # Guardar los datos tokenizados en formato JSON
    with open("data/train.json", "w") as f:
        json.dump(train, f)  # Guardar datos de entrenamiento
    with open("data/test.json", "w") as f:
        json.dump(test, f)  # Guardar datos de prueba
    with open("data/vocab.json", "w") as f:
        json.dump(vocab, f)  # Guardar el vocabulario

    # Mensaje de confirmación
    print("Datos preprocesados y guardados en la carpeta 'data'.")
    print(f"Tamaño del vocabulario: {len(vocab)} tokens.")  # Imprimir la cantidad de tokens en el vocabulario

# Ejecutar el script si se llama directamente
if __name__ == "__main__":
    save_preprocessed_data()
