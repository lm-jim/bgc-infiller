import train

train.run_training_pipeline("main_config.yaml", preprocess_data=True, use_fp16=True)