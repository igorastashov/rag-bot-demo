## Запуск сервисов и приложения (краткий чек‑лист)

Этот файл служит **краткой шпаргалкой** по запуску всех частей системы.  
Подробная настройка инфраструктуры (vLLM, bge‑m3, Chroma, Neo4j) описана в `INFRA_SETUP.md`.

---

## 1. Предусловия

- **LLM‑сервис (Qwen/vLLM)**: поднят и доступен по `LLM_BASE_URL` из `.env`.  
- **Neo4j**: контейнер запущен и доступен по `NEO4J_URI` из `.env`.  
- **Эмбеддер `bge-m3`**: модель лежит по пути `EMBEDDER_MODEL_PATH` из `.env`.  
- **Chroma DB**: достаточно, чтобы существовала (или создавалась) директория `CHROMA_DB_PATH` из `.env`.

Если что‑то из этого не готово — сначала пройти шаги в `INFRA_SETUP.md`.

---

## 2. Локальный запуск (Windows, разработка)

- **Создать и активировать виртуальное окружение** (один раз на проект):

```bash
cd D:/__projects__/drop-rag
python -m venv .venv

# Git Bash
source .venv/Scripts/activate
# или PowerShell
# .\.venv\Scripts\activate
```

- **Установить зависимости** (после активации `.venv`):

```bash
pip install -r requirements.txt
```

- **Создать `.env` в корне** (минимальный пример):

```env
LLM_BASE_URL=http://192.168.52.119:8000/v1
LLM_API_KEY=dummy
LLM_MODEL_NAME=qwen-4b-instruct

EMBEDDER_MODEL_PATH=./models/bge-m3
# сначала можно проверить на CPU:
# EMBEDDER_DEVICE=cpu
# потом, когда убедишься, что torch с CUDA норм, поменять на:
EMBEDDER_DEVICE=cuda:0

CHROMA_DB_PATH=./data/chroma_db

NEO4J_URI=bolt://192.168.52.119:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test1234

PDF_STORAGE_ROOT=./data/pdf_storage
RAG_SCOPE=session

LLM_MAX_OUTPUT_TOKENS=2048

# Выбор бекенда графа для UI:
GRAPH_BACKEND=lightrag      # LightRAG
# GRAPH_BACKEND=simple      # старый режим, если нужно откатиться

# Рабочая директория LightRAG (по умолчанию data/lightrag_storage):
# LIGHTRAG_WORKING_DIR=./data/lightrag_storage

# Язык описаний сущностей/связей в LightRAG:
SUMMARY_LANGUAGE=Russian    # или, если хотите, "Русский"
```

- **Запустить Streamlit‑приложение**:

```bash
(.venv) streamlit run ui/app.py
```

После этого приложение будет доступно по адресу, который покажет Streamlit (обычно `http://localhost:8501`).

---

## 3. Базовый сценарий проверки приложения

- **Новый чат**:
  - открыть страницу;
  - при необходимости нажать кнопку **«Новый чат»** в сайдбаре.
- **Задать вопрос без PDF**:
  - в поле чата написать вопрос (например, "Привет") и проверить, что приходит ответ от Qwen.
- **Загрузить PDF и проиндексировать**:
  - в блоке `1. Загрузка PDF` выбрать один или несколько файлов;
  - нажать кнопку **«Индексировать загруженные PDF»** и дождаться завершения прогресса.
- **Задать вопрос по содержимому PDF**:
  - в чате задать вопрос, который можно ответить на основе загруженного документа;
  - убедиться, что ответ ссылается на содержимое PDF (по логам видно, что происходил поиск в Chroma).

---

## 4. Запуск приложения в Docker

Этот режим нужен, чтобы **на любой машине** можно было:

- собрать Docker‑образ с приложением;
- запустить контейнер;
- передать `.env` и смонтировать тома для данных/моделей;
- получить тот же функционал (чат + RAG + GraphRAG/LightRAG), что и при локальном запуске.

### 4.1. Предусловия

- На машине установлен **Docker** (Docker Desktop / Docker Engine).
- LLM‑сервис (Qwen/vLLM) и Neo4j **подняты где‑то снаружи** (на той же машине или на сервере) и доступны по адресам из `.env`:
  - `LLM_BASE_URL` — URL vLLM;
  - `NEO4J_URI` — URI Neo4j.

Контейнер с приложением будет просто подключаться к этим сервисам по сети.

### 4.2. Сборка Docker‑образа

Из корня проекта:

```bash
cd D:/__projects__/drop-rag

docker build -t drop-rag-app .
```

При сборке:

- в образ копируются:
  - исходники приложения (`app/`, `ui/`, `scripts/`, `sample_prj/LightRAG` и т.д.);
  - `requirements.txt`;
- в образе устанавливаются все Python‑зависимости;
- рабочая директория внутри контейнера — `/app`;
- по умолчанию запускается `streamlit run ui/app.py` на порту `8501`.

> Примечание: папка `models/` и данные (`data/`) по умолчанию находятся внутри образа, но для реальной эксплуатации лучше использовать тома (см. ниже).

### 4.3. Подготовка `.env` для контейнера

Формат `.env` такой же, как для локального запуска (см. раздел 2):

```env
LLM_BASE_URL=http://192.168.52.119:8000/v1
LLM_API_KEY=dummy
LLM_MODEL_NAME=qwen-4b-instruct

EMBEDDER_MODEL_PATH=./models/bge-m3
EMBEDDER_DEVICE=cpu          # или cuda:0, если используете GPU и передаёте её в контейнер

CHROMA_DB_PATH=./data/chroma_db

NEO4J_URI=bolt://192.168.52.119:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test1234

PDF_STORAGE_ROOT=./data/pdf_storage
RAG_SCOPE=session

LLM_MAX_OUTPUT_TOKENS=2048

# Бекенд для графа в UI
GRAPH_BACKEND=lightrag

# Рабочая директория LightRAG
LIGHTRAG_WORKING_DIR=./data/lightrag_storage

# Язык описаний сущностей/связей в LightRAG
SUMMARY_LANGUAGE=Russian
```

При запуске контейнера удобно использовать `--env-file .env`, чтобы не копировать файл внутрь образа.

### 4.4. Томa (volumes) для данных и моделей

Чтобы не терять данные при пересборке образа и удобно работать с файлами, рекомендуется монтировать тома:

- `/app/data` — всё рабочее состояние приложения:
  - `data/chroma_db` — Chroma (векторное хранилище);
  - `data/pdf_storage` — загруженные PDF;
  - `data/lightrag_storage` — внутреннее хранилище LightRAG;
  - `data/graphs` — HTML‑файлы с графами.
- `/app/models` — модели эмбеддера (`bge-m3`).

Пример маппинга томов (Windows PowerShell, путь подставьте свой):

```bash
docker run --rm ^
  -p 8501:8501 ^
  --env-file .env ^
  -v D:/__projects__/drop-rag/data:/app/data ^
  -v D:/__projects__/drop-rag/models:/app/models ^
  drop-rag-app
```

В Linux/macOS формат такой же, но пути будут другими, например:

```bash
docker run --rm \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  drop-rag-app
```

### 4.5. Доступ к LLM и Neo4j из контейнера

Важно, чтобы значения `LLM_BASE_URL` и `NEO4J_URI` указывали на адреса, **доступные из контейнера**:

- если LLM/Neo4j крутятся на той же машине, что и Docker, IP вида `192.168.x.x` обычно работает и с хоста, и из контейнера;
- если LLM/Neo4j запущены в других контейнерах, то лучше использовать `docker network` и имена сервисов (это можно будет добавить в отдельный `docker-compose.yml` позже).

### 4.6. Быстрый чек‑лист для Docker‑режима

1. Собрать образ:
   ```bash
   docker build -t drop-rag-app .
   ```
2. Убедиться, что LLM (vLLM) и Neo4j запущены и доступны по адресам из `.env`.
3. Запустить контейнер с томами и `.env`:
   ```bash
   docker run --rm -p 8501:8501 --env-file .env -v D:/__projects__/drop-rag/data:/app/data -v D:/__projects__/drop-rag/models:/app/models drop-rag-app
   ```
4. Открыть браузер на `http://localhost:8501` и пройти базовый сценарий проверки (как в разделе 3):
   - новый чат;
   - вопрос без PDF;
   - загрузка PDF и индексация;
   - построение графа (RAG‑граф через LightRAG).
