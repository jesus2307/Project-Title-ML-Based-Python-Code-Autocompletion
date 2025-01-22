import torch
from src.model import CodeCompletionModel  # Importar la definición del modelo

def load_model(model_path):
    """Carga el modelo con las dimensiones correctas desde el checkpoint"""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Cargar modelo en CPU
    
    # Obtener las dimensiones correctas desde el checkpoint
    vocab_size = checkpoint["embedding.weight"].shape[0]
    embed_dim = checkpoint["embedding.weight"].shape[1]
    hidden_dim = checkpoint["lstm.weight_ih_l0"].shape[0] // 4  # LSTM tiene 4 puertas
    
    # Instanciar el modelo con las dimensiones correctas
    model = CodeCompletionModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for code autocompletion.")
    parser.add_argument("--input", type=str, required=True, help="Input sequence for prediction")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    
    args = parser.parse_args()
    model = load_model(args.model)
    print("Model loaded successfully with correct dimensions.")
