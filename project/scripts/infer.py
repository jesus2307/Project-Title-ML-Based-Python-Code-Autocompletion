import torch
from src.model import CodeCompletionModel  # Importar la definición del modelo

def load_model(model_path):
    """
    Carga un modelo de autocompletado de código desde un checkpoint guardado.

    Parámetros:
        model_path (str): Ruta al archivo del modelo guardado en formato PyTorch (.pth).

    Retorna:
        model (CodeCompletionModel): Modelo cargado en modo evaluación.
    """
    # Cargar los pesos del modelo desde el checkpoint, usando CPU para compatibilidad
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Extraer las dimensiones correctas desde los pesos del checkpoint
    vocab_size = checkpoint["embedding.weight"].shape[0]  # Tamaño del vocabulario
    embed_dim = checkpoint["embedding.weight"].shape[1]  # Dimensión de embeddings
    hidden_dim = checkpoint["lstm.weight_ih_l0"].shape[0] // 4  # Dimensión oculta (LSTM tiene 4 puertas)

    # Crear una instancia del modelo con los parámetros extraídos
    model = CodeCompletionModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    
    # Cargar los pesos del modelo entrenado
    model.load_state_dict(checkpoint)
    
    # Poner el modelo en modo evaluación (desactiva dropout, batch norm, etc.)
    model.eval()

    return model  # Devolver el modelo listo para inferencia

if __name__ == "__main__":
    import argparse

    # Definir argumentos de línea de comandos para la inferencia
    parser = argparse.ArgumentParser(description="Inference script for code autocompletion.")
    
    # Argumento obligatorio: secuencia de entrada para la predicción
    parser.add_argument("--input", type=str, required=True, help="Input sequence for prediction")
    
    # Argumento obligatorio: ruta al modelo guardado
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    
    # Parsear los argumentos ingresados por el usuario
    args = parser.parse_args()
    
    # Cargar el modelo usando la función load_model
    model = load_model(args.model)
    
    print("Model loaded successfully with correct dimensions.")
