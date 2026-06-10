import os
import json
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

def extract_biosynthetic_genes(record):
    biosynthetic_genes = []
    
    for feature in record.features:
        if feature.type in ["CDS", "gene"]:
            qualifiers = feature.qualifiers
            if any(f.strip().lower() == "biosynthetic" for f in qualifiers.get('gene_kind', [])):
                sequence = qualifiers.get('translation', [''])[0]
                biosynthetic_genes.append((feature, sequence))
                continue

    return biosynthetic_genes

def build_bgc_dataframe(folder_path):
    all_data = []

    files = os.listdir(f"{folder_path}/raw/mibig_json_4.0")
    
    base_names = [f[:-5] for f in files if f.endswith('.json')]

    for name in tqdm(base_names, desc="Procesando archivos", unit="BGC"):
        json_path = os.path.join(f"{folder_path}/raw/mibig_json_4.0", f"{name}.json")
        gbk_path = os.path.join(f"{folder_path}/raw/mibig_gbk_4.0", f"{name}.gbk")

        if os.path.exists(gbk_path):
            data_row = {}
            with open(json_path, 'r') as f:
                j = json.load(f)
                data_row['compounds'] = [i['name'] for i in j['compounds']]
                data_row['class'] = [i['class'] for i in j['biosynthesis']['classes']][0]     # Ignoramos subclase de biosíntesis porque la mayoría de los BGCs no la tienen anotada, lo que haría que el modelo se sesgue hacia la clase "Unknown"
                
            record = SeqIO.read(gbk_path, "genbank")

            data_row['BGC'] = record.id
            data_row['organism'] = record.annotations.get('organism', 'Unknown')
            data_row['core_genes'] = extract_biosynthetic_genes(record)
            data_row['gene_names'] = [f[0].qualifiers.get('gene', [f'UnnamedGene-{i+1}'])[0] for i, f in enumerate(data_row['core_genes'])]
            data_row['gene_sequences'] = [f[1] for f in data_row['core_genes']]

            all_data.append(data_row)

    return pd.DataFrame(all_data)[['BGC', 'class', 'organism', 'compounds', 'gene_names', 'gene_sequences']]

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