import os
import torch
import wandb
import utils, model, bgc_tokenizer
from datasets import Dataset
from transformers import Trainer
from transformers import AutoModelForMaskedLM

def run_training_pipeline(config_file):
    print(f"--- BGC INFILLER TRAINING PIPELINE BEGIN ---")

    print(f"Loading configuration from {config_file}")
    config = utils.load_config(config_file)
    model_config = config['model_config']
    model_name_version = f"{model_config['model_info']['model_name']}-v{model_config['model_info']['model_version']}"
    print(f"Configuration loaded successfully. Model Config: {model_name_version}")
    
    print(f"Initializing Weights & Biases...")
    wandb.init(
        project=model_config["model_info"]["wandb_project"],
        name=model_name_version,
        config=model_config,
        reinit=True
    )

    print(f"Loading and formatting raw BGC data...")

    utils.gbl_set_seed(model_config["train_params"]["seed"])
    df = utils.read_bgcs_from_csv(f"{config['data_path']}/BGC_Data.csv")

    print(f"BGC data loaded and formatted successfully. Total BGCs: {len(df)}")
    
    print(f"Retrieving base model {model_config['model_info']['base_model']}...")
    base_model, base_tokenizer = model.get_base_model(config)
    print(f"Base model retrieved: {base_model.__class__.__name__}")

    print(f"Beginning special tokenization process...")
    bgc_types = bgc_tokenizer.get_bgc_special_tokens(df)
    print(f"Found Special BGC Type Tokens:\n{bgc_types}")
    print(f"Overlapping gene sequences and adding special class tokens...")
    formatted_df = bgc_tokenizer.format_gene_sequences(df)
    print(f"Overlapping successful. Sample of formatted gene sequences:\n{formatted_df.head(5)}")
    
    print(f"Tokenizing gene sequences...")
    dataset = Dataset.from_pandas(formatted_df)
    tokenized_dataset = bgc_tokenizer.tokenize_sequences(dataset, base_model, base_tokenizer, bgc_types, config)
    print(f"Successfully tokenized gene sequences.")

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=model_config["train_params"]["split_size"], 
                                                           seed=model_config["train_params"]["seed"])
    train_data = tokenized_dataset["train"]
    eval_data = tokenized_dataset["test"]

    print(f"Running training for model {model_name_version}")
    data_collator, training_args = model.get_model_training_hyperparameters(config, base_tokenizer)

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    if config['resume_from_checkpoint'] and utils.model_has_checkpoint(model_name_version):
        print(f"Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=f"./models/{model_name_version}/checkpoint-1000")
    else:
        trainer.train()

    print(f"Training completed. Saving and uploading model and tokenizer...")

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

run_training_pipeline("config/main_config.yaml")