import os
import json
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# En los datos "en crudo" se pueden encontrar varias categorías de genes. 
# Esta función, dado un registro de un BGC, extraerá únicamente los genes que tengan función biosintética principal.
# Se devolverá un objeto con forma List[n_biosynthetic_genes] en la que cada posición es una tupla de dos elementos tal que (gene_name, full_sequence).
def extract_biosynthetic_genes(record):
    biosynthetic_genes = []
    
    for feature in record.features:
        if feature.type in ["CDS", "gene"]:
            qualifiers = feature.qualifiers
            # Si el cualificador 'gene_kind' contiene la palabra "biosynthetic", se trata de un gen biosintético principal.
            if any(f.strip().lower() == "biosynthetic" for f in qualifiers.get('gene_kind', [])):
                # Se extrae la secuencia de aminoácidos completa del gen y la añadimos a la lista. 
                # Cada elemento de la lista será un gen biosintético completo.
                sequence = qualifiers.get('translation', [''])[0]
                biosynthetic_genes.append((feature, sequence))
                continue

    return biosynthetic_genes

# Función principal de preprocesamiento de los datos crudos de BGC's, transformándolos en un CSV estructurado.
# El CSV resultante se compondrá de las siguientes columnas:
# 
# # 'BGC': número identificativo del BGC en la forma BGCxxxxxx
# # 'class': tipo o clase del BGC
# # 'organism': nombre del organismo del que proviene este BGC
# # 'compounds': metabolitos secundarios producidos por el BGC
# # 'gene_names': nomenclatura interna de MiBIG para cada gen presente en el BGC
# # 'gene_sequences': secuencia de aminoácidos completa de cada gen biosintético principal en el BGC

def build_bgc_dataframe(folder_path):

    # Lectura de archivos JSON con el fin de indexación.
    files = os.listdir(f"{folder_path}/raw/mibig_json_4.0")
    base_names = [f[:-5] for f in files if f.endswith('.json')]

    all_data = []

    # Bucle ejecutado una vez por cada entrada BGC. Se utilizará la función tqdm() para mostrar una barra de progreso del preprocesamiento.
    for name in tqdm(base_names, desc="Procesando archivos", unit="BGC"):
        # Directorio del archivo JSON (metadatos) y archivo GBK (GenBank) con las secuencias protéicas.
        json_path = os.path.join(f"{folder_path}/raw/mibig_json_4.0", f"{name}.json")
        gbk_path = os.path.join(f"{folder_path}/raw/mibig_gbk_4.0", f"{name}.gbk")

        if os.path.exists(gbk_path):
            data_row = {}
            with open(json_path, 'r') as f:
                # Se extraen del archivo JSON los compuestos producidos por el BGC y su clase.
                j = json.load(f)
                data_row['compounds'] = [i['name'] for i in j['compounds']]
                # Ignoramos subclase de biosíntesis porque la mayoría de los BGCs no la tienen anotada, lo que haría que el modelo se sesgue hacia la clase "Unknown"
                data_row['class'] = [i['class'] for i in j['biosynthesis']['classes']][0]
            
            # Utilizando la libería Biopython, se lee el archivo GBK correspondiente a este BGC.
            record = SeqIO.read(gbk_path, "genbank")

            # Extracción del ID del registro GBK y organismo de origen.
            data_row['BGC'] = record.id
            data_row['organism'] = record.annotations.get('organism', 'Unknown')

            # Utilizando la función auxiliar extract_biosynthetic_genes() se extraerán las secuencias completas de los genes biosintéticos principales de este BGC.
            # Esta función devolverá un objeto con forma List[n_biosynthetic_genes] en la que cada posición es una tupla de dos elementos tal que (gene_name, full_sequence).
            data_row['core_genes'] = extract_biosynthetic_genes(record)

            # Obtención de los nombres de los genes. Si un BGC no los tiene anotados, se generarán nombres genéricos en la forma UnnamedGene-X.
            data_row['gene_names'] = [f[0].qualifiers.get('gene', [f'UnnamedGene-{i+1}'])[0] for i, f in enumerate(data_row['core_genes'])]
            # Obtención de las secuencias protéicas completas.
            data_row['gene_sequences'] = [f[1] for f in data_row['core_genes']]

            all_data.append(data_row)

    # Se devuelve el DataFrame preprocesado con las columnas en el orden anotado.
    return pd.DataFrame(all_data)[['BGC', 'class', 'organism', 'compounds', 'gene_names', 'gene_sequences']]

# Las secuencias de genes almacenadas en formatted_bgc_data.csv son demasiado extensas para ser procesadas directamente por el modelo.
# Esta función, dado el DataFrame preprocesado por la función build_bgc_dataframe(), dividirá las secuencias de genes en fragmentos de tamaño máximo max_sequence_length.
# Las divisiones incluirán un solapamiento con los otros fragmentos adyacentes para intentar mantener el contexto.
# Cada entrada del DataFrame resultante será una tupla de forma: (BGC_CLASS, GENE_SEQUENCE) 
def format_gene_sequences(df, 
                          max_sequence_length=1024, 
                          overlapping=250, 
                          min_chunk_length=50):
    
    # Se expande el DataFrame separando cada BGC con una única de sus secuencia protéica.
    df_exploded = df.explode("gene_sequences").reset_index(drop=True).dropna()

    # Margen reductorio de max_sequence_length.
    # Asegura que ninguna entrada acabe igualando o excediendo el límite. 
    extra = 4

    formatted_rows = []

    # Procesamiento por cada entrada en el DataFrame.
    for _, row in df_exploded.iterrows():

        # Tipo de BGC en la forma: "[BGC_TYPE]".
        bgc_type = f"[{str(row['class']).upper()}]"
        # Secuencia protéica completa
        seq = str(row["gene_sequences"])

        # Si la longitud de la secuencia es menor que el máximo, se obtiene la secuencia completa.
        if len(seq) <= max_sequence_length - extra:
            formatted_rows.append((bgc_type, seq))

        # En caso contrario, se recortará y dividirá en una nueva entrada, compartiendo solapamiento con la anterior.
        else:
            start = 0
            while start < len(seq):
                # Cálculo del límite superior y extracción del chunk correspondiente desde la secuencia completa.
                end = start + max_sequence_length - extra
                chunk = seq[start:end]

                # Si el chunk es demasiado pequeño, se descartará. 
                # En caso contrario, se añade como entrada.
                if len(chunk) >= min_chunk_length:
                    formatted_rows.append((bgc_type, chunk))
                
                # Se avanza el puntero manteniendo el solapamiento, de forma que se incluya parte del fragmento actual para conservar algo de contexto.
                start += max_sequence_length - extra - overlapping

                # Si se alcanza el final de la secuencia, se finaliza el procesamiento.
                if end >= len(seq):
                    break
    
    return pd.DataFrame({"formatted_sequence": formatted_rows})