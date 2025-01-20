import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from src.model import CodeCompletionModel

# Define dataset
class CodeDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[1:], dtype=torch.long)

# Padding function
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # <PAD> token
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

# Evaluation function
def evaluate_model():
    print("Evaluating RNN model...")
    # Load test data
    test_data_path = os.path.join("data", "test.json")
    try:
        with open(test_data_path, "r") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Test data file not found at {test_data_path}. Please ensure it exists.")
        return

    dataset = CodeDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Load the model
    model_path = os.path.join("models", "rnn_model.pth")
    vocab_size = 27  # Adjust based on vocab.json
    embed_dim = 64
    hidden_dim = 128
    model = CodeCompletionModel(vocab_size, embed_dim, hidden_dim)

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Initializing a new model with random weights.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Initialized model weights saved to {model_path}.")

    # Load model weights
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # Evaluate the model
    total_loss = 0
    total_correct = 0
    total_count = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_count += targets.numel()

    accuracy = total_correct / total_count * 100
    print(f"Evaluation complete. Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
