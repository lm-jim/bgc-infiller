from datasets import Dataset
import wandb
import os
import pytest
from src import utils, model, train, bgc_tokenizer, data_preprocessing

# Función de test que comprueba la existencia del fichero de configuración de test.
def test_config_exists():
    os.environ.get("config/main_config_test.yaml")

# Función de test que comprueba la existencia de los datos JSON y GBK obtenidos de MiBIG en el directorio de trabajo.
def test_raw_bgc_data():
    config = utils.load_config("config/main_config_test.yaml")
    assert os.path.exists(f"{config['data_path']}/raw/mibig_gbk_4.0")
    assert os.path.exists(f"{config['data_path']}/raw/mibig_json_4.0")

# Función de test que comprueba el correcto funcionamiento del preprocesamiento inicial.
def test_preprocess_data():
    config = utils.load_config("config/main_config_test.yaml")
    data_preprocessing.build_bgc_dataframe(f"{config['data_path']}")

# Función de test que comprueba la conexión con Weights & Biases.
def test_wandb_integration():
    config = utils.load_config("config/main_config_test.yaml")
    wandb.init(
        project=config['model_config']["model_info"]["wandb_project"],
        name="PyTestExample",
        config=config,
        reinit=True,
        mode="disabled" # Se establece mode="disabled" para evitar registrar actividad real.
    )
    assert wandb.run is not None

# Función de test que comprueba la descarga del modelo base desde la plataforma Hugging Face.
def test_model_creation():
    config = utils.load_config("config/main_config_test.yaml")
    base_model, _ = model.get_base_model(config)
    assert base_model is not None

# Función de test que comprueba la descarga del tokenizador base desde la plataforma Hugging Face.
def test_tokenizer_creation():
    config = utils.load_config("config/main_config_test.yaml")
    _, base_tokenizer = model.get_base_model(config)
    assert base_tokenizer is not None

# Función de test que comprueba el correcto funcionamiento de la división de las secuencias y su correcta tokenización.
def test_tokenize_sequences():
    config = utils.load_config("config/main_config_test.yaml")
    df = utils.read_bgcs_from_csv(f"{config['data_path']}/formatted_bgc_data.csv")

    base_model, base_tokenizer = model.get_base_model(config)

    bgc_types = bgc_tokenizer.get_bgc_special_tokens(df)
    formatted_df = data_preprocessing.format_gene_sequences(df)

    dataset = Dataset.from_pandas(formatted_df)
    tokenized_dataset = bgc_tokenizer.tokenize_sequences(dataset, base_model, base_tokenizer, bgc_types)

    assert tokenized_dataset is not None

# Función de test que ejecuta el pipeline completo de entrenamiento, con el objetivo de comprobar el correcto funcionamiento completo.
def test_train_pipeline_execution():
    #train.run_training_pipeline(config_file="config/main_config_test.yaml", preprocess_data=False)