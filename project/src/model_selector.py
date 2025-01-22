# Modelo selector para elegir entre diferentes arquitecturas de redes neuronales recurrentes

import torch.nn as nn  # Importar PyTorch para definir las arquitecturas de los modelos

class ModelSelector:
    """
    Clase para seleccionar y construir diferentes modelos de redes neuronales recurrentes.

    Permite elegir entre:
    - RNN (Red Neuronal Recurrente)
    - LSTM (Long Short-Term Memory)

    Parámetros:
        vocab_size (int): Tamaño del vocabulario del modelo.
        embed_dim (int, opcional): Dimensión del espacio de embeddings (por defecto 64).
        hidden_dim (int, opcional): Dimensión de la capa oculta de la red neuronal (por defecto 128).
        model_type (str, opcional): Tipo de modelo a construir ('RNN' o 'LSTM', por defecto 'RNN').

    Métodos:
        - get_model(): Devuelve el modelo seleccionado basado en el tipo especificado.
        - _build_rnn_model(): Construye un modelo basado en RNN.
        - _build_lstm_model(): Construye un modelo basado en LSTM.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, model_type='RNN'):
        """
        Inicializa la clase con los parámetros del modelo.

        Parámetros:
            vocab_size (int): Número de tokens en el vocabulario.
            embed_dim (int, opcional): Tamaño de la capa de embeddings.
            hidden_dim (int, opcional): Tamaño de la capa oculta en la red neuronal.
            model_type (str, opcional): Tipo de modelo ('RNN' o 'LSTM').
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type

    def get_model(self):
        """
        Devuelve el modelo seleccionado en función del tipo especificado.

        Retorna:
            nn.Sequential: Un modelo de red neuronal basado en RNN o LSTM.

        Excepciones:
            ValueError: Si el tipo de modelo especificado no es soportado.
        """
        if self.model_type == 'RNN':
            return self._build_rnn_model()
        elif self.model_type == 'LSTM':
            return self._build_lstm_model()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

    def _build_rnn_model(self):
        """
        Construye un modelo basado en una Red Neuronal Recurrente (RNN).

        Arquitectura:
            1. Capa de Embedding: Convierte tokens en vectores de dimensión embed_dim.
            2. Capa RNN: Captura relaciones entre tokens en la secuencia.
            3. Capa Lineal (Fully Connected): Predice el siguiente token en la secuencia.

        Retorna:
            nn.Sequential: Modelo basado en RNN.
        """
        return nn.Sequential(
            nn.Embedding(self.vocab_size, self.embed_dim),  # Capa de embeddings
            nn.RNN(self.embed_dim, self.hidden_dim, batch_first=True),  # Capa RNN
            nn.Linear(self.hidden_dim, self.vocab_size)  # Capa de salida
        )

    def _build_lstm_model(self):
        """
        Construye un modelo basado en Long Short-Term Memory (LSTM).

        Arquitectura:
            1. Capa de Embedding: Convierte tokens en vectores de dimensión embed_dim.
            2. Capa LSTM: Maneja dependencias a largo plazo en la secuencia.
            3. Capa Lineal (Fully Connected): Predice el siguiente token en la secuencia.

        Retorna:
            nn.Sequential: Modelo basado en LSTM.
        """
        return nn.Sequential(
            nn.Embedding(self.vocab_size, self.embed_dim),  # Capa de embeddings
            nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True),  # Capa LSTM
            nn.Linear(self.hidden_dim, self.vocab_size)  # Capa de salida
        )
