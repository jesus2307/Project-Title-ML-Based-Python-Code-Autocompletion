import sys
import os
import torch
from src.model import CodeCompletionModel

# Asegurar que src está en el PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_model(model_path):
    model = CodeCompletionModel(vocab_size=10, embed_dim=32, hidden_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_next_token(model, input_sequence):
    # Implementación real de la predicción
    input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # Convertir en tensor
    with torch.no_grad():
        output = model(input_tensor)
    predicted_token = torch.argmax(output, dim=-1).item()
    return predicted_token

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for code autocompletion.")
    parser.add_argument("--input", type=str, required=True, help="Input code snippet.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model.")
    args = parser.parse_args()

    model = load_model(args.model)
    input_sequence = [int(x) for x in args.input.split()]  # Convertir entrada en lista de enteros
    prediction = predict_next_token(model, input_sequence)
    print(f"Input: {args.input}")
    print(f"Prediction: {prediction}")
