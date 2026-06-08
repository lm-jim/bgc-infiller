import os
import train

train.run_training_pipeline(os.environ.get('MAIN_CONFIG'), preprocess_data=True)