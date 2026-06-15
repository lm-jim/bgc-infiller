![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-white.svg)

![Tests](https://github.com/lm-jim/bgc-infiller/actions/workflows/tests.yml/badge.svg)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Active-gold.svg)](https://wandb.ai/lm-jim-universidad-polit-cnica-de-madrid/bgc-infiller/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/lm-jim/mdl-mlops)

---

# Infiller de Clusteres de Genes Biosintéticos (BGC's)

Trabajo de fin de Máster del Máster en Deep Learning de la Universidad Politécnica de Madrid. Un modelo basado en ESM2, fine-tuneado para poder realizar infilling de BGC's (Clusteres de Genes Biosintéticos).

## Contenido del proyecto

- Datos crudos originales obtenidos del repositorio MiBIG.
- Funciones de preprocesado de datos.
- Pipeline de entrenamiento integrado con Weights & Biases.
- Funciones de test de la librería Pytest.
- Endpoint Gradio local y online, accesible en Hugging Face Spaces.
- Archivos de configuración generales y de los modelos entrenados.
- Dockerfile para creación de imagen y contenedorización.

## Instrucciones de contenedorización

Se pueden utilizar los scripts `scripts/build-docker.bat` (Windows) o `scripts/build-docker.sh` (Linux) para iniciar la construcción de la imagen automáticamente. También se podrá introducir el siguiente
comando en la raíz del proyecto:

> `docker build -t bgc-infiller .`

Una vez construida la imagen, se podrá acceder a ella a través de Docker Desktop o siguiendo las instrucciones a continuación.

# Instrucciones de entrenamiento

## Archivos de configuración

Es obligatorio incluir en la variable de entorno `MAIN_CONFIG` la ruta a la configuración principal para el entrenamiento. Esta será un fichero YAML con los siguientes campos:

| Campo                    | Tipo    | Descripción                                                                                             |
| :----------------------- | :------ | :------------------------------------------------------------------------------------------------------ |
| `model_config`           | String  | Ruta del archivo a la **configuración del modelo** a entrenar (P. ej: `config/bgc-infiller-v2.5.yaml`). |
| `data_path`              | String  | Ruta del directorio que contendrá los datos (Se recomienda establecerla a `./data`).                    |
| `resume_from_checkpoint` | Boolean | Activar o desactivar la reanudación del entrenamiento en caso de encontrarse un checkpoint.             |
| `log_level`              | String  | Nivel de profundidad de los logs durante la ejecución (P. ej: `INFO`, `DEBUG`, `WARNING`, `ERROR`).     |
| `fp16`                   | Boolean | Activa la precisión mixta de 16 bits para mejorar la eficiencia en caso de tener GPU.                   |
| `val_split_size`         | Float   | Proporción del dataset dedicada a los datos de validación. Deberá ser un valor entre 0.0 y 1.0.         |
| `test_split_size`        | Float   | Proporción del dataset dedicada a los datos de test. Deberá ser un valor entre 0.0 y 1.0.               |

Para la **configuración específica del modelo** a entrenar, se deberán incluir en el fichero YAML estas sub-secciones con los siguietes parámetros:

### model_info:

| Campo           | Tipo   | Descripción                                                    |
| :-------------- | :----- | :------------------------------------------------------------- |
| `model_name`    | String | Nombre identificativo del modelo (P. ej: `bgc-infiller-esm2`). |
| `model_version` | String | Versión del modelo (P. ej: `2.5.0`).                           |
| `wandb_project` | String | Nombre del proyecto en Weights & Biases.                       |
| `base_model`    | String | Modelo base para fine-tunear, a obtener desde Hugging Face.    |

### train_params:

| Campo              | Tipo  | Descripción                                                        |
| :----------------- | :---- | :----------------------------------------------------------------- |
| `seed`             | Int   | Semilla de aleatoriedad, para asegurar la reproducibilidad.        |
| `epochs`           | Int   | Número de épocas de entrenamiento.                                 |
| `lr`               | Float | Tasa de aprendizaje.                                               |
| `batch_size`       | Int   | Tamaño de batch.                                                   |
| `weight_decay`     | Float | Factor de regularización para evitar _Overfitting_.                |
| `mlm`              | Float | Porcentaje de tokens que se enmascararán durante el entrenamiento. |
| `save_steps`       | Int   | Número de pasos de entrenamiento por cada guardado de checkpoint.  |
| `save_total_limit` | Int   | Límite máximo de checkpoints que se conservarán.                   |
| `logging_steps`    | Int   | Número de pasos de entrenamiento por cada loggeo de métricas.      |
| `eval_steps`       | Int   | Número de pasos de entrenamiento por cada evaluación del modelo.   |

## Entrenamiento local

Para realizar un entrenamiento local, se deberán instalar las dependencias del fichero `requirements.txt` y ejecutar el script `src/main.py`.

Se recomienda establecer las variable de entorno `WANDB_API_KEY` para reportar a Weights & Biases, y `PYTHONPATH` a la raíz del proyecto.

Se debe incluir en la variable de entorno `MAIN_CONFIG` la ruta a la configuración principal para el entrenamiento.

Ejemplo en Windows:

> `$env:MAIN_CONFIG="config/main_config.yaml"; $env:PYTHONPATH="."; & python .\src\main.py`

Se recomienda utilizar la versión de Python 3.10.

## Entrenamiento desde contenedor Docker

Para ejecutar la imagen Docker creada con anterioridad en modo de entrenamiento, se puede utilizar el siguiente comando:

> `docker run -e MAIN_CONFIG="config/<config_file>" -e MODE="train" bgc-infiller`

Se recomienda establecer la variable de entorno `WANDB_API_KEY` para reportar a Weights & Biases.

# Instrucciones de despliegue

## Despliegue local

Para realizar un entrenamiento local, se deberán instalar las dependencias del fichero `space/requirements.txt` y ejecutar el script `space/app.py`.

Ejemplo en Windows:

> `python .\space\app.py`

El servidor será accesible desde la ruta:

> `http://localhost:7860`

## Despliegue desde contenedor Docker

Para ejecutar la imagen Docker creada con anterioridad en modo de despliegue, se puede utilizar el siguiente comando, sustituyendo `[PORT]` por el puerto de la máquina en donde se desee escuchar:

> `docker run -p [PORT]:7860 -e MAIN_CONFIG="config/<config_file>" -e MODE="app" bgc-infiller`

La definición de variable de entorno `MODE="app"` es opcional en este caso.

El servidor será accesible desde la ruta:

> `http://<server-ip>:[PORT]`

## Acceso online en HuggingFace Spaces

Se puede acceder al despliegue online desde el siguiente enlace:

> https://huggingface.co/spaces/lm-jim/bgc-infiller

# Enlaces de interés

| Resource          | Link                                                                    |
| :---------------- | :---------------------------------------------------------------------- |
| GitHub Repository | https://github.com/lm-jim/bgc-infiller                                  |
| Weights & Biases  | https://wandb.ai/lm-jim-universidad-polit-cnica-de-madrid/bgc-infiller/ |
| HuggingFace Space | https://huggingface.co/spaces/lm-jim/bgc-infiller/                      |
| HuggingFace Model | https://huggingface.co/lm-jim/bgc-infiller/                             |
