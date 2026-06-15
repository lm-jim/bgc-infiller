import logging
import os
from pathlib import Path

import pandas as pd
import ast
import yaml
import random
import numpy as np
import torch

# Establecimiento del sistema de logging.
def setup_logging(nivel : str):
    # Los logs se guardarán en el directorio "logs/bgc-infiller.log".
    path_final = Path(__file__).resolve().parents[1]
    Path(path_final/"logs").mkdir(exist_ok=True)
    path_final = path_final / "logs"/ "bgc-infiller.log"

    # Configuración del formato de los logs
    logging.basicConfig(
        level = getattr(logging, nivel.upper(), logging.DEBUG),
        format = "%(asctime)s | %(levelname)-8s | %(funcName)s.%(lineno)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(path_final)
        ]
    )

# Lectura del archivo CSV con los datos de los BGCs preprocesados, leyendo las columnas que contienen listas como objetos de tipo lista en lugar de strings
def read_bgcs_from_csv(file_path):    
    return pd.read_csv(file_path).map(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)

# Función para cargar la configuración principal desde un archivo YAML. 
# Esta función también carga la configuración específica del modelo a entrenar, obtenida del parámetro 'model_config' dentro de la configuración principal.
def load_config(config_file):
    with open(config_file, "r") as f1:
        config = yaml.safe_load(f1)

        # Extraemos la configuración propia del modelo a entrenar desde la configuración principal.
        # La establecemos como un diccionario anidado dentro de la configuración principal, facilitando el acceso posterior.
        with open(f"config/{config['model_config']}", "r") as f2:
            config['model_config'] = yaml.safe_load(f2)
        return config

# Establecimiento de semilla global en las diferentes librerías utilizadas
def gbl_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Función que comprueba si existe un checkpoint de entrenamiento anterior para el modelo especificado.
def model_has_checkpoint(model_name_version):
    dir = f"./models/{model_name_version}"
    if os.path.exists(dir):
        return any(
            name.startswith("checkpoint") and
            os.path.isdir(os.path.join(dir, name))
            for name in os.listdir(dir)
        )
    return False