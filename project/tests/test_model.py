
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from src.model import CodeCompletionModel

import unittest

class TestModel(unittest.TestCase):
    def test_forward_pass(self):
        vocab_size = 27
        embed_dim = 64
        hidden_dim = 128
        model = CodeCompletionModel(vocab_size, embed_dim, hidden_dim)

        # Dummy input for testing
        dummy_input = torch.randint(0, vocab_size, (1, 5))
        output = model(dummy_input)

        self.assertEqual(output.shape, (1, 5, vocab_size))

if __name__ == "__main__":
    unittest.main()
