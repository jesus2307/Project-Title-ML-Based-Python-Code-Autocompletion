
import torch
from src.model import CodeCompletionModel

def load_model(model_path):
    model = CodeCompletionModel(vocab_size=10, embed_dim=32, hidden_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_next_token(model, input_sequence):
    # Example function for prediction
    # Replace with real implementation
    return "<next_token>"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for code autocompletion.")
    parser.add_argument("--input", type=str, required=True, help="Input code snippet.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model.")
    args = parser.parse_args()

    model = load_model(args.model)
    prediction = predict_next_token(model, args.input)
    print(f"Input: {args.input}")
    print(f"Prediction: {prediction}")
    