import utils
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments

def get_base_model(config):
    model_checkpoint = config["model_config"]["model_info"]["base_model"]
    return AutoModelForMaskedLM.from_pretrained(model_checkpoint), AutoTokenizer.from_pretrained(model_checkpoint)


def get_model_training_hyperparameters(config, tokenizer):
    model_config = config['model_config']
    model_name_version = f"{model_config['model_info']['model_name']}-v{model_config['model_info']['model_version']}"

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=model_config["train_params"]["mlm"]
    )

    training_args = TrainingArguments(
        output_dir=f"./models/{model_name_version}",
        num_train_epochs=model_config["train_params"]["epochs"],
        per_device_train_batch_size=model_config["train_params"]["batch_size"],
        save_steps=model_config["train_params"]["save_steps"],
        save_total_limit=model_config["train_params"]["save_total_limit"],
        learning_rate=model_config["train_params"]["lr"],
        weight_decay=model_config["train_params"]["weight_decay"],
        logging_steps=model_config["train_params"]["logging_steps"],
        fp16=True,
        report_to="wandb",
        run_name=model_name_version,
        eval_strategy="steps",
        eval_steps=model_config["train_params"]["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    return data_collator, training_args