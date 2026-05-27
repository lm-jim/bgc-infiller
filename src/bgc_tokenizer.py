import torch
import pandas as pd
from datasets import Dataset, load_from_disk
import os

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

def tokenize_sequences(dataset, base_model, base_tokenizer, bgc_class_tokens, config):
    model_config = config['model_config']

    base_tokenizer.add_tokens(bgc_class_tokens)
    base_model.resize_token_embeddings(len(base_tokenizer))
    base_model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(len(base_tokenizer))
    )
    print(base_tokenizer.get_vocab())

    tokenization_path = f"{config['data_path']}/tokenized_bgc_dataset"

    if os.path.exists(tokenization_path):
        tokenized_dataset = load_from_disk(tokenization_path)
    else:
        tokenized_dataset = dataset.map(lambda x: base_tokenizer(x["formatted_sequence"], 
                                                            truncation=True,
                                                            max_length=1024),
                                                        batched=True,
                                                        remove_columns=["formatted_sequence"])
        tokenized_dataset.save_to_disk(tokenization_path)

    return tokenized_dataset