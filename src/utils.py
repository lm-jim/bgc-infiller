import os

import pandas as pd
import ast
import yaml
import random
import numpy as np
import torch

def read_bgcs_from_csv(file_path):    
    return pd.read_csv(file_path).map(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)

def load_config(config_file):
    with open(config_file, "r") as f1:
        config = yaml.safe_load(f1)
        with open(f"config/{config['model_config']}", "r") as f2:
            config['model_config'] = yaml.safe_load(f2)
        return config

def gbl_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def model_has_checkpoint(model_name_version):
    dir = f"./models/{model_name_version}"
    if os.path.exists(dir):
        return any(
            name.startswith("checkpoint") and
            os.path.isdir(os.path.join(dir, name))
            for name in os.listdir(dir)
        )
    return False