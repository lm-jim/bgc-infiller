from datasets import Dataset
import wandb
import os
import pytest
from src import data_loader, utils, model, train, bgc_tokenizer

def test_load_config():
    config = utils.load_config("config/main_config.yaml")
    assert "model_config" in config
    assert "data_path" in config
    assert "resume_from_checkpoint" in config
    assert "log_level" in config

    model_info = config['model_config']["model_info"]
    assert "model_name" in model_info
    assert "model_version" in model_info
    assert "wandb_project" in model_info
    assert "base_model" in model_info

    train_params = config['model_config']["train_params"]
    assert "seed" in train_params
    assert "split_size" in train_params
    assert "epochs" in train_params
    assert "lr" in train_params
    assert "batch_size" in train_params
    assert "weight_decay" in train_params
    assert "mlm" in train_params
    assert "save_steps" in train_params
    assert "save_total_limit" in train_params
    assert "logging_steps" in train_params
    assert "eval_steps" in train_params


def test_config_formats():
    config = utils.load_config("config/main_config.yaml")

    assert isinstance(config["data_path"], str)
    assert isinstance(config["resume_from_checkpoint"], bool)
    assert isinstance(config["log_level"], str)

    model_info = config['model_config']["model_info"]
    assert isinstance(model_info["model_name"], str)
    assert isinstance(model_info["model_version"], str)
    assert isinstance(model_info["wandb_project"], str)
    assert isinstance(model_info["base_model"], str)

    train_params = config['model_config']["train_params"]
    assert isinstance(train_params["seed"], int)
    assert isinstance(train_params["split_size"], float)
    assert isinstance(train_params["epochs"], float)
    assert isinstance(train_params["lr"], float)
    assert isinstance(train_params["batch_size"], int)
    assert isinstance(train_params["weight_decay"], float)
    assert isinstance(train_params["mlm"], float)
    assert isinstance(train_params["save_steps"], int)
    assert isinstance(train_params["save_total_limit"], int)
    assert isinstance(train_params["logging_steps"], int)
    assert isinstance(train_params["eval_steps"], int)

def test_raw_bgc_data():
    config = utils.load_config("config/main_config.yaml")
    assert os.path.exists(f"{config['data_path']}/mibig_gbk_4.0")
    assert os.path.exists(f"{config['data_path']}/mibig_json_4.0")

def test_preprocess_data():
    config = utils.load_config("config/main_config.yaml")
    data_loader.build_bgc_dataframe(f"{config['data_path']}")

def test_wandb_integration():
    config = utils.load_config("config/main_config.yaml")
    wandb.init(
        project=config['model_config']["model_info"]["wandb_project"],
        name="PyTestExample",
        config=config,
        reinit=True,
        mode="disabled"
    )
    assert wandb.run is not None

def test_tokenizer_creation():
    config = utils.load_config("config/main_config.yaml")
    _, base_tokenizer = model.get_base_model(config)
    assert base_tokenizer is not None

def test_model_creation():
    config = utils.load_config("config/main_config.yaml")
    base_model, _ = model.get_base_model(config)
    assert base_model is not None

def test_tokenize_sequences():
    config = utils.load_config("config/main_config.yaml")
    df = utils.read_bgcs_from_csv(f"{config['data_path']}/BGC_Data.csv")

    base_model, base_tokenizer = model.get_base_model(config)

    bgc_types = bgc_tokenizer.get_bgc_special_tokens(df)
    formatted_df = bgc_tokenizer.format_gene_sequences(df)

    dataset = Dataset.from_pandas(formatted_df)
    tokenized_dataset = bgc_tokenizer.tokenize_sequences(dataset, base_model, base_tokenizer, bgc_types, config)

    assert tokenized_dataset is not None

def test_train_pipeline_execution():
    train.run_training_pipeline(config_file="config/main_config_test.yaml", preprocess_data=False)