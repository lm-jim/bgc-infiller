FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

EXPOSE 7860

# Por defecto se construirá una imagen para desplegar Gradio
ENV MODE=app
CMD if [ "$MODE" = "train" ]; then \
        python src/main.py; \
    else \
        python space/app.py; \
    fi