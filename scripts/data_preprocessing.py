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

    files = os.listdir(f"{folder_path}/mibig_json_4.0")
    
    base_names = [f[:-5] for f in files if f.endswith('.json')]

    for name in tqdm(base_names, desc="Procesando archivos", unit="BGC"):
        json_path = os.path.join(f"{folder_path}/mibig_json_4.0", f"{name}.json")
        gbk_path = os.path.join(f"{folder_path}/mibig_gbk_4.0", f"{name}.gbk")

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

df = build_bgc_dataframe('./data')
df.to_csv('./data/BGC_Data.csv', index=False)