# Modelo de lenguaje basado en PyTorch para autocompletado de código

import torch.nn as nn  # Importar las capas necesarias de PyTorch

class CodeCompletionModel(nn.Module):
    """
    Modelo de autocompletado de código basado en una red neuronal recurrente (LSTM).
    
    Este modelo toma una secuencia de tokens, la procesa con embeddings y una LSTM,
    y predice el siguiente token en la secuencia.
    
    Parámetros:
        vocab_size (int): Número total de tokens en el vocabulario.
        embed_dim (int): Dimensión del espacio de embeddings.
        hidden_dim (int): Dimensión de la capa oculta de la LSTM.
    
    Arquitectura:
        1. Capa de embeddings: Convierte tokens en vectores densos.
        2. Capa LSTM: Procesa las secuencias para capturar dependencias temporales.
        3. Capa Fully Connected (FC): Genera la predicción del siguiente token.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CodeCompletionModel, self).__init__()

        # Capa de embeddings: Convierte índices de tokens en vectores de dimensión embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Capa LSTM: Recibe los embeddings y aprende patrones en la secuencia
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Capa totalmente conectada (FC): Transforma la salida de la LSTM en una predicción de token
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Propagación hacia adelante del modelo.

        Parámetros:
            x (Tensor): Tensor de entrada con tokens representados como índices.

        Retorna:
            Tensor: Predicción de la probabilidad de cada token en la siguiente posición.
        """
        x = self.embedding(x)  # Convertir los tokens en embeddings
        x, _ = self.lstm(x)  # Pasar la secuencia por la LSTM
        x = self.fc(x)  # Aplicar la capa totalmente conectada para predecir el siguiente token
        return x  # Retornar la predicción
