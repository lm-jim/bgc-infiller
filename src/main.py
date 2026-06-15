import os
import train

# Ejecución del pipeline de entrenamiento. Se obtendrá el archivo de configuración principal a través de la variable de entorno MAIN_CONFIG. 
# Si se desea ejecutar el pipeline sin preprocesar de nuevo los datos, se puede establecer preprocess_data=False (para ello, deberá existir un archivo formatted_bgc_data.csv preprocesado anteriormente)
train.run_training_pipeline(os.environ.get('MAIN_CONFIG'), preprocess_data=True)