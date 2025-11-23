## Инфраструктура для запуска RAG + GraphRAG сервиса

Этот документ описывает, **какие сервисы должны быть подняты на машине с двумя GPU** (Linux‑сервер) и **как проверить их работу** до запуска фронтенда/бэкенда (`streamlit` + наше приложение).

Рекомендуется выполнять все проверки в отдельном виртуальном окружении Python (`venv` или conda).

---

## 1. LLM‑сервис (Qwen через vLLM)

### 1.1. Запуск vLLM с Qwen

Пример команды (адаптируй пути и модель под свою среду):

```bash
docker run --gpus all -p 8000:8000 -d -v /home/dmd/models/Qwen3-4B-Instruct-2507:/app/model --rm vllm/vllm-openai:latest --model /app/model --served-model-name qwen-4b-instruct --max-model-len 93808 --gpu-memory-utilization 0.90 --dtype float1
```

Убедись, что контейнер запущен:

```bash
docker ps
```

### 1.2.1 Проверка LLM из cmd

```bash
curl http://192.168.52.119:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"qwen-4b-instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}]}"
```

### 1.2.2 Проверка LLM из Python

```bash
source venv/bin/activate
```

Создать файл `check_llm.py`:

```bash
cat <<EOF > check_llm.py
from openai import OpenAI

# Так как мы на той же машине, где Docker, можно использовать localhost или IP
client = OpenAI(
    base_url="http://192.168.52.119:8000/v1",
    api_key="dummy"
)

print("Sending request to Qwen-4B...")

try:
    resp = client.chat.completions.create(
        model="qwen-4b-instruct",
        messages=[{"role": "user", "content": "Say hello!"}],
        max_tokens=50
    )
    print("\n--- SUCCESS ---")
    print("Response:", resp.choices[0].message.content)
except Exception as e:
    print("\n--- ERROR ---")
    print(e)
EOF
```

```bash
python check_llm.py
```

Если есть осмысленный ответ — **LLM‑сервис готов**.

---

## 2. Эмбеддер (bge-m3)

### 2.1. Структура моделей

На сервере должна быть папка с эмбеддером, например:

```text
/home/dmd/models/bge-m3/  # или аналогичный путь
```

### 2.2. Проверка загрузки модели на нужный GPU

Создать `check_embedder.py`:

```bash
cat <<EOF > check_embedder.py
from sentence_transformers import SentenceTransformer
import os

# Путь относительно текущей папки (или полный /home/dmd/models/bge-m3)
model_path = "./models/bge-m3"

print(f"Loading model from: {model_path} ...")

try:
    # Загружаем модель
    model = SentenceTransformer(model_path)
    print("--- SUCCESS: Model loaded ---")

    # Проверка генерации вектора
    test_text = "Hello, RAG world!"
    embedding = model.encode([test_text])
    
    print(f"Input text: '{test_text}'")
    print(f"Embedding shape: {embedding.shape}") 
    # Должно быть (1, 1024)
    
except Exception as e:
    print("\n--- ERROR ---")
    print(f"Could not load model from {model_path}")
    print(e)
EOF
```

Запуск:

```bash
CUDA_VISIBLE_DEVICES=1 python check_embedder.py
```

Увидели `SUCCESS` и форму эмбеддинга (например `(1, 1024)`) — **эмбеддер готов**.

---

## 3. Локальная векторная БД (Chroma)

Chroma работает как библиотека (без отдельного сервера), данные лежат в директории, например:

```text
/home/dmd/rag_app/chroma_db/
```

### 3.1. Проверка Chroma + bge-m3

Создать `check_chroma.py`:

```bash
cat <<EOF > check_chroma.py
import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Путь к модели
model_path = "./models/bge-m3"

print(f"1. Loading Embedder from: {model_path} ...")
# Модель сама подхватит GPU, который мы укажем через переменную окружения
embed_model = SentenceTransformer(model_path)

print("2. Initializing ChromaDB (local mode)...")
# Данные будут лежать в папке ./chroma_db
client = PersistentClient(path="./chroma_db")

# Создаем или берем коллекцию
collection = client.get_or_create_collection("test_collection")

# Тестовые данные
docs = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Qwen is a powerful LLM."
]
ids = ["1", "2", "3"]

print("3. Embedding documents...")
# Генерируем векторы для документов
doc_embeddings = embed_model.encode(docs).tolist()

print("4. Adding to Chroma...")
collection.add(
    ids=ids,
    documents=docs,
    embeddings=doc_embeddings
)

print("5. Querying: 'What is the capital of France?'")
query_text = "What is the capital of France?"
query_emb = embed_model.encode([query_text]).tolist()[0]

results = collection.query(
    query_embeddings=[query_emb],
    n_results=1
)

print("\n--- RESULT ---")
print("Document:", results["documents"][0][0])
# Ожидаем строку про Париж
EOF
```

```bash
CUDA_VISIBLE_DEVICES=1 python check_chroma.py
```

Если вернулась строка про Париж — **Chroma + эмбеддер работают корректно**.

---

## 4. Neo4j (графовая БД)

### 4.1. Запуск Neo4j в Docker

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:latest
```

Проверить, что контейнер запущен:

```bash
docker ps
```

### 4.2. Проверка через браузер

1. Открыть в браузере: `http://192.168.52.119:7474` (IP сервера).
2. Логин: `neo4j`, пароль: `test1234`.
3. Выполнить простой запрос:

```cypher
RETURN 1;
```

Если запрос выполняется — **Neo4j готов**.

### 4.3. Быстрая проверка из Python

```bash
cat <<EOF > check_neo4j.py
from neo4j import GraphDatabase

uri = "bolt://192.168.52.119:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "test1234"))

with driver.session() as session:
    res = session.run("RETURN 1 AS n").single()
    print("Neo4j result:", res["n"])
EOF
```

```bash
python check_neo4j.py
```


---

## 5. Проверка перед запуском фронтенда/бэкенда

Перед тем как запускать наше Streamlit‑приложение:

1. **LLM (vLLM + Qwen)**:
   - контейнер `qwen-vllm` запущен;
   - `check_llm.py` успешно возвращает ответ.
2. **Эмбеддер (bge-m3)**:
   - `CUDA_VISIBLE_DEVICES=1 python check_embedder.py` успешно грузит модель на выбранный GPU и выдаёт эмбеддинг.
3. **Chroma**:
   - `CUDA_VISIBLE_DEVICES=1 check_chroma.py` создаёт коллекцию, добавляет документы и находит строку про Париж.
4. **Neo4j**:
   - контейнер `neo4j` запущен;
   - `python check_neo4j.py`;
   - `RETURN 1` выполняется в браузере или через небольшой Python‑скрипт.

Если все четыре проверки проходят, можно переходить к запуску нашего приложения (уже на этой же машине или с другого хоста, указав в `.env` правильные `LLM_BASE_URL` и `NEO4J_URI`).


