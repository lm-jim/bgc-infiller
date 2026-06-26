import logging
import os
import torch
import wandb
from src import data_preprocessing, utils, model, bgc_tokenizer
from datasets import Dataset
from transformers import Trainer
from transformers import AutoModelForMaskedLM

# Función principal que ejecutará el pipeline completo de entrenamiento. Desde el preprocesado de datos, hasta el entrenamiento del modelo y su posterior guardado en Weights & Biases.
def run_training_pipeline(config_file, preprocess_data=True):
    print(f"--- BGC INFILLER TRAINING PIPELINE BEGIN ---")

    # Carga de la configuración principal.
    print(f"Loading configuration from {config_file}")
    config = utils.load_config(config_file)
    model_config = config['model_config']
    
    # Nombre completo del modelo a entrenar, en la forma de: {model_name}-v{model_version}. (P.ej: bgc-infiller-esm2-v1.0.0)
    model_name_version = f"{model_config['model_info']['model_name']}-v{model_config['model_info']['model_version']}"
    print(f"Configuration loaded successfully. Model Config: {model_name_version}")
    
    # Establecimiento del sistema de logging.
    print(f"Setting up logging with level: {config['log_level']}")
    utils.setup_logging(config['log_level'])
    logger = logging.getLogger(__name__)
    print(f"Logging setup complete at level: {config['log_level']}")
    logger.info(f"Switching to logging mode from now on.")
    
    # Inicialización de Weights & Biases. 
    # Se deberá definir la variable de entorno WANDB_API_KEY previamente a la ejecución del pipeline.
    logger.info(f"Initializing Weights & Biases...")
    wandb.init(
        project=model_config["model_info"]["wandb_project"],
        name=model_name_version,
        config=model_config,
        reinit=True
    )

    # Establecimiento de la semilla global.
    utils.gbl_set_seed(model_config["train_params"]["seed"])

    # Lectura y preprocesado de los datos "crudos" de los BGC's.
    # Si preprocess_data es verdadera, se sustituirá el archivo formatted_bgc_data.csv por un preprocesamiento nuevo.
    # Si es falsa, deberá existir un archivo formatted_bgc_data.csv previamente generado.
    logger.info(f"Loading and formatting raw BGC data...")
    if preprocess_data:
        # Preprocesado de datos crudos de BGC's a archivo CSV formateado...
        df = data_preprocessing.build_bgc_dataframe(f"{config['data_path']}")
        df.to_csv(f"{config['data_path']}/formatted_bgc_data.csv", index=False)

    try:
        df = utils.read_bgcs_from_csv(f"{config['data_path']}/formatted_bgc_data.csv")
    except Exception as e:
        logger.error(f"Error reading formatted BGC data: {e}")
        raise e
    logger.info(f"BGC data formatted and loaded successfully. Total BGCs: {len(df)}")

    # Extraemos el modelo base a fine-tunear de Hugging Face, especificado en la configuración del modelo, junto con su tokenizer.
    logger.info(f"Retrieving base model {model_config['model_info']['base_model']}...")
    base_model, base_tokenizer = model.get_base_model(config)
    logger.info(f"Base model retrieved: {base_model.__class__.__name__}")

    # Desde los BGC's formateados, extraemos los tokens especiales que se utilizarán para especificar la clase de cada BGC. (P.ej: [NRPS], [PKS]... etc.)
    logger.info(f"Beginning special tokenization process...")
    bgc_types = bgc_tokenizer.get_bgc_special_tokens(df)
    logger.debug(f"Found Special BGC Type Tokens:\n{bgc_types}")

    # Las secuencias de genes almacenadas en formatted_bgc_data.csv son demasiado extensas para ser procesadas directamente por el modelo.
    # Se dividirán las secuencias de genes en fragmentos de tamaño máximo max_sequence_length, incluyendo un solapamiento con los fragmentos adyacentes para mantener el contexto.
    logger.info(f"Overlapping gene sequences and adding special class tokens...")
    formatted_df = data_preprocessing.format_gene_sequences(df)
    logger.debug(f"Overlapping successful. Sample of formatted gene sequences:\n{formatted_df.head(5)}")

    # Se crea un objeto dataset a partir del DataFrame formateado en el paso anterior.
    dataset = Dataset.from_pandas(formatted_df)

    # Tokenización de las secuencias de genes, incluyendo los tokens especiales de clases de BGC.
    logger.info(f"Tokenizing gene sequences...")
    tokenized_dataset = bgc_tokenizer.tokenize_sequences(dataset, base_model, base_tokenizer, bgc_types)
    logger.info(f"Successfully tokenized gene sequences.")

    # División del dataset en conjuntos de entrenamiento, validación y test.
    logger.info(f"Splitting dataset into training, validation, and test sets...")
    if config['val_split_size'] + config['test_split_size'] >= 1.0:
        logger.error(f"Invalid split sizes: val_split_size + test_split_size is or exceeds 1.0")
        raise ValueError("Invalid split sizes: val_split_size + test_split_size is or exceeds 1.0")

    first_split = config['val_split_size']
    second_split = config['test_split_size'] / (1 - config['val_split_size'])

    validation_split = tokenized_dataset.train_test_split(test_size=first_split, 
                                                           seed=model_config["train_params"]["seed"])
    test_split = validation_split["train"].train_test_split(test_size=second_split, 
                                                           seed=model_config["train_params"]["seed"])
                                                           
    train_data = test_split["train"]
    eval_data = validation_split["test"]
    test_data = test_split["test"]

    # Se guardará el conjunto de test generado en disco para posterior evaluación.
    test_data.save_to_disk(f"{config['data_path']}/test")
    logger.info(f"Split completed. Training samples: {len(train_data)}, Validation samples: {len(eval_data)}, Test samples: {len(test_data)}")

    # Extracción de hiperparámetros de entrenamiento del modelo según la configuración especificada.
    logger.info(f"Running training for model {model_name_version}")
    data_collator, training_args = model.get_model_training_hyperparameters(config, base_tokenizer, config['fp16'])

    # Se obtendrá un objeto data_collator y training_args que se incluyen al crear el objeto Trainer de la librería de Hugging Face.
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator
    )

    # Si existe un checkpoint previo, se reanudará el entrenamiento desde ese punto. En caso contrario, se comenzará desde cero.
    if config['resume_from_checkpoint'] and utils.model_has_checkpoint(model_name_version):
        logger.info(f"Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=f"./models/{model_name_version}/checkpoint-1000")
    else:
        trainer.train()

    logger.info(f"Training completed. Saving and uploading model and tokenizer...")

    # Ruta en la que se guardarán los archivos generados durante el entrenamiento del modelo.
    model_path = f"./models/{model_name_version}"
    trainer.save_model(model_path)
    base_tokenizer.save_pretrained(model_path)

    # Se subirá este directorio a Weights & Biases como un artifact.
    artifact = wandb.Artifact(
        name=model_name_version,
        type="model"
    )
    artifact.add_dir(model_path)
    wandb.log_artifact(artifact)

    # Finalización de la sesión de Weights & Biases.
    wandb.finish()

    # Finalización del pipeline de entrenamiento.
    print(f"--- BGC INFILLER TRAINING PIPELINE END ---")