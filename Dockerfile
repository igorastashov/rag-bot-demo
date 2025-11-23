FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Устанавливаем системные зависимости (минимально необходимые)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только зависимости и локальные wheel-файлы, чтобы лучше использовать Docker cache
COPY requirements.txt ./requirements.txt
COPY local_packages ./local_packages

# Устанавливаем torch/torchvision из локальных wheel-файлов (Linux, CUDA 12.8)
RUN pip install --no-cache-dir \
    ./local_packages/torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    ./local_packages/torchvision-0.23.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    && pip install --no-cache-dir -r requirements.txt

# Копируем остальной код приложения (включая sample_prj/LightRAG и модели, если есть)
COPY . .

# Папки с данными и моделями — отдельные тома по желанию
VOLUME ["/app/data"]
VOLUME ["/app/models"]

# Порт Streamlit
EXPOSE 8501

# По умолчанию запускаем Streamlit UI
CMD ["streamlit", "run", "ui/app.py"]



