import logging
import os
import torch
import wandb
from src import data_loader, utils, model, bgc_tokenizer
from datasets import Dataset
from transformers import Trainer
from transformers import AutoModelForMaskedLM

def run_training_pipeline(config_file, preprocess_data=True, use_fp16=True):
    print(f"--- BGC INFILLER TRAINING PIPELINE BEGIN ---")

    print(f"Loading configuration from {config_file}")
    config = utils.load_config(config_file)
    model_config = config['model_config']
    model_name_version = f"{model_config['model_info']['model_name']}-v{model_config['model_info']['model_version']}"
    print(f"Configuration loaded successfully. Model Config: {model_name_version}")
    
    print(f"Setting up logging with level: {config['log_level']}")
    utils.setup_logging(config['log_level'])
    logger = logging.getLogger(__name__)
    print(f"Logging setup complete at level: {config['log_level']}")
    logger.info(f"Switching to logging mode from now on.")

    logger.info(f"Initializing Weights & Biases...")
    wandb.init(
        project=model_config["model_info"]["wandb_project"],
        name=model_name_version,
        config=model_config,
        reinit=True
    )

    logger.info(f"Loading and formatting raw BGC data...")
    utils.gbl_set_seed(model_config["train_params"]["seed"])
    
    if preprocess_data:
        df = data_loader.build_bgc_dataframe(f"{config['data_path']}")
        df.to_csv(f"{config['data_path']}/BGC_Data.csv", index=False)

    df = utils.read_bgcs_from_csv(f"{config['data_path']}/BGC_Data.csv")

    logger.info(f"BGC data formatted and loaded successfully. Total BGCs: {len(df)}")

    logger.info(f"Retrieving base model {model_config['model_info']['base_model']}...")
    base_model, base_tokenizer = model.get_base_model(config)
    logger.info(f"Base model retrieved: {base_model.__class__.__name__}")

    logger.info(f"Beginning special tokenization process...")
    bgc_types = bgc_tokenizer.get_bgc_special_tokens(df)
    logger.debug(f"Found Special BGC Type Tokens:\n{bgc_types}")
    logger.info(f"Overlapping gene sequences and adding special class tokens...")
    formatted_df = bgc_tokenizer.format_gene_sequences(df)
    logger.debug(f"Overlapping successful. Sample of formatted gene sequences:\n{formatted_df.head(5)}")

    logger.info(f"Tokenizing gene sequences...")
    dataset = Dataset.from_pandas(formatted_df)
    tokenized_dataset = bgc_tokenizer.tokenize_sequences(dataset, base_model, base_tokenizer, bgc_types, config)
    logger.info(f"Successfully tokenized gene sequences.")

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=model_config["train_params"]["split_size"], 
                                                           seed=model_config["train_params"]["seed"])
    train_data = tokenized_dataset["train"]
    eval_data = tokenized_dataset["test"]

    logger.info(f"Running training for model {model_name_version}")
    data_collator, training_args = model.get_model_training_hyperparameters(config, base_tokenizer, use_fp16)

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    if config['resume_from_checkpoint'] and utils.model_has_checkpoint(model_name_version):
        logger.info(f"Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=f"./models/{model_name_version}/checkpoint-1000")
    else:
        trainer.train()

    logger.info(f"Training completed. Saving and uploading model and tokenizer...")

    model_path = f"./models/{model_name_version}"

    trainer.save_model(model_path)
    base_tokenizer.save_pretrained(model_path)

    artifact = wandb.Artifact(
        name=model_name_version,
        type="model"
    )
    artifact.add_dir(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

    print(f"--- BGC INFILLER TRAINING PIPELINE END ---")