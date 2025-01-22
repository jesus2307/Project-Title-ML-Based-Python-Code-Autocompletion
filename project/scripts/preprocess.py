
# Script para preprocesar datos de un dataset público como Py150

import os
import json
from sklearn.model_selection import train_test_split

def build_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}  # Añadir tokens especiales
    idx = 2
    for line in data:
        for token in line.split():
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def tokenize_and_convert_to_ids(data, vocab):
    tokenized_data = []
    for line in data:
        tokenized_line = [vocab.get(token, vocab["<UNK>"]) for token in line.split()]
        tokenized_data.append(tokenized_line)
    return tokenized_data

def load_and_tokenize_data():
    # Dataset simulado; puedes reemplazarlo con datos reales
    data = [
        "def suma ( a , b ) : return a + b",
        "for i in range ( 10 ) : print ( i )",
        "if x > 0 : print ( 'positivo' )",
        "while n > 0 : n -= 1"
    ]
    return data

def save_preprocessed_data():
    raw_data = load_and_tokenize_data()
    vocab = build_vocab(raw_data)
    tokenized_data = tokenize_and_convert_to_ids(raw_data, vocab)
    train, test = train_test_split(tokenized_data, test_size=0.2, random_state=42)
    
    os.makedirs("data", exist_ok=True)
    with open("data/train.json", "w") as f:
        json.dump(train, f)
    with open("data/test.json", "w") as f:
        json.dump(test, f)
    with open("data/vocab.json", "w") as f:
        json.dump(vocab, f)
    
    print("Datos preprocesados y guardados en la carpeta 'data'.")
    print(f"Tamaño del vocabulario: {len(vocab)} tokens.")

if __name__ == "__main__":
    save_preprocessed_data()
