import os
import torch
import wandb
import utils
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset, load_from_disk

main_config = utils.load_config("config/main_config.yaml")
model_config = utils.load_config(f"config/{main_config['model_config']}")

utils.gbl_set_seed(model_config["train_params"]["seed"])

df = utils.read_bgcs_from_csv(f"{main_config['data_path']}/BGC_Data.csv")

bgc_types = utils.get_bgc_special_tokens(df)
print(f"Found BGC Type Tokens:\n{bgc_types}")

model_checkpoint = model_config["model_info"]["base_model"]
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

tokenizer.add_tokens(bgc_types)
model.resize_token_embeddings(len(tokenizer))
model.lm_head.bias = torch.nn.Parameter(
    torch.zeros(len(tokenizer))
)
print(tokenizer.get_vocab())

formatted_df = utils.format_gene_sequences(df)
dataset = Dataset.from_pandas(formatted_df)

tokenization_path = f"{main_config['data_path']}/tokenized_bgc_dataset"

if os.path.exists(tokenization_path):
    tokenized_dataset = load_from_disk(tokenization_path)
else:
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["formatted_sequence"], 
                                                        truncation=True,
                                                        max_length=1024),
                                                    batched=True,
                                                    remove_columns=["formatted_sequence"])
    tokenized_dataset.save_to_disk(tokenization_path)

tokenized_dataset = tokenized_dataset.train_test_split(test_size=model_config["train_params"]["split_size"])
train_data = tokenized_dataset["train"]
eval_data = tokenized_dataset["test"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=model_config["train_params"]["mlm"]
)

model_name_version = f"{model_config['model_info']['model_name']}-v{model_config['model_info']['model_version']}"

os.environ['WANDB_API_KEY'] = os.getenv("WANDB_API_KEY")
wandb.init(
    project=model_config["model_info"]["wandb_project"],
    name=model_name_version,
    config=model_config,
    reinit=True
)

print(f"Running training pipeline for model {model_name_version}")

training_args = TrainingArguments(
    output_dir="./models/esm2_trainer",
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
)

trainer.train()

model_path = f"./models/{model_name_version}"

trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

artifact = wandb.Artifact(
    name=model_name_version,
    type="model"
)
artifact.add_dir(model_path)
wandb.log_artifact(artifact)

wandb.finish()