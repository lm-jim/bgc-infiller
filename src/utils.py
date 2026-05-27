import pandas as pd
import ast
import yaml
import random
import numpy as np
import torch

def read_bgcs_from_csv(file_path):    
    return pd.read_csv(file_path).map(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)

def get_bgc_special_tokens(df):
    bgc_types = df["class"].dropna().unique().tolist()
    return list(map(lambda x: f"[{x.upper()}]", bgc_types))
  

def format_gene_sequences(df, 
                          max_sequence_length=1024, 
                          overlapping=250, 
                          min_chunk_length=50):
    
    df_exploded = df.explode("gene_sequences").reset_index(drop=True).dropna()
    formatted_rows = []
    extra = 4

    for _, row in df_exploded.iterrows():
        bgc_type = f"[{str(row['class']).upper()}]"
        seq = str(row["gene_sequences"])

        if len(seq) <= max_sequence_length - extra:
            formatted_rows.append((bgc_type, seq))
        else:
            start = 0
            while start < len(seq):
                end = start + max_sequence_length - extra
                chunk = seq[start:end]

                if len(chunk) >= min_chunk_length:
                    formatted_rows.append((bgc_type, chunk))

                start += max_sequence_length - extra - overlapping

                if end >= len(seq):
                    break

    return pd.DataFrame({"formatted_sequence": formatted_rows})

def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def gbl_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)