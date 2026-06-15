import torch
import pandas as pd
from datasets import Dataset, load_from_disk
import os

# Función que obtiene de un DataFrame de BGCs los tokens especiales de cada tipo de BGC, en forma de strings formateados tal que: "[BGC_TYPE]".
# Se utiliza la columna "class".
def get_bgc_special_tokens(df):
    bgc_types = df["class"].dropna().unique().tolist()
    return list(map(lambda x: f"[{x.upper()}]", bgc_types))

# Función que tokenizará las secuencias (previamente procesadas) de los genes, dejándolos listos para ser utilizados por el modelo.
def tokenize_sequences(dataset, base_model, base_tokenizer, bgc_class_tokens):
    # Se añaden los tokens especiales de tipos de BGC.
    base_tokenizer.add_tokens(bgc_class_tokens)
    base_model.resize_token_embeddings(len(base_tokenizer))
    # Se reinicializa el bias del modelo para evitar errores.
    base_model.lm_head.bias = torch.nn.Parameter(torch.zeros(len(base_tokenizer)))

    # Mapeo de la función de tokenización sobre el dataset, eliminando la columna "formatted_sequence".
    # Aunque el preprocesado no lo debería permitir, por robustez se establece un máximo de 1024 tokens por secuencia.
    tokenized_dataset = dataset.map(lambda x: base_tokenizer(x["formatted_sequence"], 
                                                        truncation=True,
                                                        max_length=1024),
                                                    batched=True,
                                                    remove_columns=["formatted_sequence"])

    return tokenized_dataset