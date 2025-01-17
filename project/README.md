
# Proyecto de Autocompletado de Código Python

Este proyecto utiliza aprendizaje automático con PyTorch y un modelo basado en transformadores para desarrollar una herramienta de autocompletado de código Python.

## Requisitos

1. **Python 3.x**.
2. Bibliotecas: PyTorch, numpy, pandas, matplotlib, transformers.

## Instrucciones

1. **Instalar dependencias**:
---
 ![Training Loss](image/Imagen1.png)


```
pip install -r requirements.txt
```

2. **Preprocesar los datos**:
---
![Training Loss](image/Imagen2.png)

Ejecutar el script de preprocesamiento:

```
python scripts/preprocess.py
```

3. **Entrenar el modelo**:
---
![Training Loss](image/Imagen3.png)
```
python scripts/train.py
```

4. **Evaluar el modelo**:
---
![Training Loss](image/Imagen4.png)
```
python scripts/evaluate.py
```

5. **Inferencia**:

Ejecutar el script de inferencia:

```
python scripts/infer.py --input "def mi_funcion("
```

## Estructura del Proyecto
![Training Loss](image/Imagen5.PNG)
```
code_autocomplete_project/

