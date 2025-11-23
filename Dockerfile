FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# Базовые переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Устанавливаем Python 3.10, pip и системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    libpq-dev \
    curl \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Обновляем pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Сначала копируем только зависимости и локальные wheel-файлы, чтобы лучше использовать Docker cache
COPY requirements.txt ./requirements.txt
COPY local_packages ./local_packages

# Устанавливаем torch/torchvision из локальных wheel-файлов (Linux, CUDA 12.8),
# затем остальные зависимости из requirements.txt
RUN python -m pip install --no-cache-dir --no-deps \
    ./local_packages/torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    ./local_packages/torchvision-0.23.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    && python -m pip install --no-cache-dir -r requirements.txt

# Копируем остальной код приложения (включая sample_prj/LightRAG и модели, если есть)
COPY . .

# Папки с данными и моделями — отдельные тома по желанию
VOLUME ["/app/data"]
VOLUME ["/app/models"]

# Порт Streamlit
EXPOSE 8501

# По умолчанию запускаем Streamlit UI
CMD ["streamlit", "run", "ui/app.py"]