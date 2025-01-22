# Pruebas unitarias para el modelo de autocompletado de código en PyTorch

import sys
import os

# Asegurar que 'src' está en el PYTHONPATH para permitir la importación del modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # Importar PyTorch
from src.model import CodeCompletionModel  # Importar el modelo a probar

import unittest  # Importar el módulo de pruebas unitarias

class TestModel(unittest.TestCase):
    """
    Clase de pruebas unitarias para verificar el correcto funcionamiento del modelo CodeCompletionModel.

    Métodos:
        - test_forward_pass(): Verifica que la salida del modelo tiene la forma esperada.
    """

    def test_forward_pass(self):
        """
        Prueba un pase hacia adelante (forward pass) del modelo para verificar que la salida
        tiene la forma correcta.

        Procedimiento:
            1. Define un modelo con un tamaño de vocabulario, dimensión de embeddings y dimensión oculta.
            2. Genera una entrada ficticia de tokens aleatorios.
            3. Pasa la entrada a través del modelo.
            4. Verifica que la salida tiene la forma esperada: (batch_size, sequence_length, vocab_size).

        Se espera que la forma de la salida sea (1, 5, vocab_size),
        donde:
        - `1` es el tamaño del batch.
        - `5` es la longitud de la secuencia de entrada.
        - `vocab_size` es el número total de tokens posibles.
        """
        vocab_size = 27  # Tamaño del vocabulario (ejemplo)
        embed_dim = 64  # Dimensión de embeddings
        hidden_dim = 128  # Dimensión de la capa oculta

        # Instanciar el modelo
        model = CodeCompletionModel(vocab_size, embed_dim, hidden_dim)

        # Crear una entrada ficticia: una secuencia de 5 tokens aleatorios dentro del vocabulario
        dummy_input = torch.randint(0, vocab_size, (1, 5))  # (batch_size=1, sequence_length=5)

        # Pasar la entrada a través del modelo
        output = model(dummy_input)

        # Verificar que la forma de la salida es la esperada
        self.assertEqual(output.shape, (1, 5, vocab_size))

# Ejecutar las pruebas si el script se ejecuta directamente
if __name__ == "__main__":
    unittest.main()
