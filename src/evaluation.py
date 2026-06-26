import logging
import math
import os

import torch
import yaml
from src import model, utils
import pandas as pd
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Función que computará las métricas de Accuracy para los tokens de máscara predichos.
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Los logits tienen forma [batch=3101, seq_len=1024, vocab_size=39].
    # Se reducirá la dimensión quedándonos con el token con mayor probabilidad predicha por el modelo.
    predictions = predictions.argmax(axis=-1)
    
    # DataCollatorForLanguageModeling marca con -100 las posiciones que NO fueron enmascaradas. Se obtienen dichas posiciones en forma booleana y el total de estas.
    mask = labels != -100
    n_masked = mask.sum()

    # Se calcula el Accuracy basándose en los aciertos en las posiciones enmascaradas.
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / n_masked

    return {
        "masked_token_accuracy": float(accuracy),
        "n_masked_tokens": int(n_masked),
    }


# Función que evalúa una versión del modelo proporcionado.
def evaluate_model(config, logger, eval_model, eval_tokenizer, model_name_version, test_data):
    # Se obtiene la misma configuración con la que se entrenó el modelo para poder evaluarlo correctamente.
    model_config = config["model_config"]
    logger.info(f"Model {model_name_version} loaded successfully.")

    # Definición de objetos DataCollatorForLanguageModeling, TrainingArguments y Trainer con los que se realizará la evaluación con los datos de Test.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=eval_tokenizer,
        mlm=True,
        mlm_probability=model_config["train_params"]["mlm"]
    )

    eval_args = TrainingArguments(
        per_device_eval_batch_size=model_config["train_params"]["batch_size"],
        fp16=config["fp16"],
        report_to="none"
    )

    # Se utiliza el conjunto de test y se llama a la función compute_metrics tras finalizar.
    trainer = Trainer(
        model=eval_model,
        args=eval_args,
        eval_dataset=test_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Ejecución de la evaluación...
    logger.info(f"Running test split on {model_name_version}...")
    metrics = trainer.evaluate()

    # Cálculo de la perplejidad a partir de la pérdida media de evaluación (eval_loss).
    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(metrics["eval_loss"])

    metrics["model_name_version"] = model_name_version
    return metrics

def run_evaluation_pipeline(config_file, m_conf="", baseline=False):
    print("--- BGC INFILLER EVALUATION PIPELINE BEGIN ---")

    # Carga de la configuración principal.
    print(f"Loading configuration from {config_file}")
    config = utils.load_config(config_file)
    
    # Carga de la configuración del modelo. Si se define el parámetro m_conf, se sustituirá la definida en la configuración principal.
    if m_conf != "":    
        with open(f"config/{m_conf}", "r") as f2:
            config['model_config'] = yaml.safe_load(f2)
    model_config = config['model_config']

    # Nombre completo del modelo a validar, en la forma de: {model_name}-v{model_version}. (P.ej: bgc-infiller-esm2-v1.0.0)
    model_name_version = f"{model_config['model_info']['model_name']}-v{model_config['model_info']['model_version']}"

    # Establecimiento del sistema de logging.
    print(f"Setting up logging with level: {config['log_level']}")
    utils.setup_logging(config['log_level'])
    logger = logging.getLogger(__name__)
    print(f"Logging setup complete at level: {config['log_level']}")
    logger.info(f"Switching to logging mode from now on.")

    # Establecimiento de la semilla global.
    utils.gbl_set_seed(model_config["train_params"]["seed"])

    test_data_path = f"{config['data_path']}/test"
    logger.info(f"Loading test dataset from {test_data_path}...")
    try:
        test_data = load_from_disk(test_data_path)
    except Exception as e:
        logger.error(f"Could not load test dataset from {test_data_path}: {e}")
        raise e
    logger.info(f"Test dataset loaded successfully. Total samples: {len(test_data)}")
    
    # Descarga del modelo y tokenizador desde el repositorio de Hugging Face.
    logger.info(f"Downloading {model_name_version} from Hugging Face...")
    eval_model, eval_tokenizer = model.download_model_from_hf("lm-jim/bgc-infiller", model_name_version)
    logger.info(f"{model_name_version} model and tokenizer downloaded successfully.")
    
    # Si el parámetro baseline se define a True, se evaluará el modelo base definido, sin fine-tuning
    if baseline:
        eval_model, _ = model.get_base_model(config)
        eval_model.resize_token_embeddings(len(eval_tokenizer))
        eval_model.lm_head.bias = torch.nn.Parameter(torch.zeros(len(eval_tokenizer)))
        logger.info(f"Baseline model {config['model_config']['model_info']['base_model']} downloaded successfully.")
    

    # Evaluación de métricas de rendimiento del modelo.
    logger.info(f"Evaluating model...")
    metrics = evaluate_model(config, logger, eval_model, eval_tokenizer, model_name_version, test_data)
    logger.info(f"Results for {model_name_version}:\n\n{metrics}")
    print("--- BGC INFILLER EVALUATION PIPELINE END ---")
    return metrics

# Conjunto de todas las métricas generadas.
all_metrics = []

# Configuración de los modelos a evaluar.
model_configs = [
    "bgc-infiller-v1.0.0.yaml",
    "bgc-infiller-v1.5.0.yaml",
    "bgc-infiller-v2.0.0.yaml",
    "bgc-infiller-v2.5.0.yaml",
]

# Modelos que utilizan ESM-2 8M y 35M para evaluar ambos baselines.
baseline_models = [
    "bgc-infiller-v1.5.0.yaml",
    "bgc-infiller-v2.5.0.yaml",
]

baseline=True

# Ejecución de pipeline de evaluación, una vez por modelo a evaluar.
for m_conf in baseline_models:
    metrics = run_evaluation_pipeline(os.environ.get('MAIN_CONFIG'), m_conf, baseline)
    all_metrics.append(metrics)
    
# Guardado en CSV de métricas generadas.
results_df = pd.DataFrame(all_metrics)[['model_name_version', 'eval_loss', 'perplexity', 'eval_runtime', 'eval_masked_token_accuracy', 'eval_n_masked_tokens']]
if not baseline:
    results_df.to_csv("./eval_results/evaluation_results.csv", index=False)
else:
    results_df.to_csv("./eval_results/baseline_results.csv", index=False)