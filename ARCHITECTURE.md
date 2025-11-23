## Overview

Этот репозиторий содержит **новое RAG/GraphRAG приложение**, работающее поверх:
- локального LLM (**Qwen через vLLM**, OpenAI‑совместимый API);
- локального эмбеддера (**bge-m3**, `sentence-transformers`);
- локальной векторной БД (**Chroma**);
- локальной графовой БД (**Neo4j**).

Рядом лежат справочные проекты в `sample_prj/` (исходный `drop-rag`, `rag-qa-self`), но новое приложение от них **не зависит**.

---

## Режимы работы RAG (флаг в .env)

Переключение режима задаётся через переменную окружения `RAG_SCOPE`:

- `RAG_SCOPE=session` — **сессионный режим**:
  - при старте нового чата создаётся своя коллекция в Chroma `session_<uuid>`;
  - при загрузке PDF:
    - файл сохраняется в общее файловое хранилище;
    - текст индексируется **только** в Chroma‑коллекцию `session_<uuid>`;
  - все RAG‑запросы используют только `session_<uuid>`;
  - при новом чате создаётся новый `session_<uuid>`, старый не используется.

- `RAG_SCOPE=global` — **глобальный режим**:
  - используется единая коллекция Chroma `global_docs`, содержащая все документы;
  - при загрузке PDF:
    - файл сохраняется в общее файловое хранилище;
    - текст индексируется **только** в Chroma‑коллекцию `global_docs`;
  - все RAG‑запросы используют `global_docs`.

Во всех режимах **сырые PDF** лежат в одном дереве папок (см. ниже), чтобы можно было переиндексировать оффлайн.

---

## Структура данных и каталогов

Планируемая структура:

- `models/`
  - `bge-m3/` — локальная модель эмбеддинга.

- `data/`
  - `pdf_storage/`
    - `global/` — все загруженные PDF (архив);
    - `session_<uuid>/` — копии/ссылки PDF, которые использовались в конкретной сессии.
  - `chroma_db/` — данные PersistentClient Chroma.

---

## Основные модули Python

Пакет приложения: `app/`

- `app/config.py`
  - Читает `.env` (через `python-dotenv`) и предоставляет объект настроек:
    - LLM: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`;
    - эмбеддер: `EMBEDDER_MODEL_PATH`, `EMBEDDER_DEVICE`;
    - Chroma: `CHROMA_DB_PATH`;
    - Neo4j: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`;
    - RAG: `RAG_SCOPE`;
    - хранилище PDF: `PDF_STORAGE_ROOT`.

- `app/llm_client.py`
  - Обёртка над OpenAI‑совместимым API vLLM:
    - инициализация `OpenAI(base_url=..., api_key=...)`;
    - метод `chat(messages: list[dict], max_tokens: int = 512) -> str`.

- `app/embedder.py`
  - Обёртка над `SentenceTransformer`:
    - загрузка модели `bge-m3` с нужного пути и устройства;
    - `encode(texts: list[str]) -> list[list[float]]`.

- `app/vector_store.py`
  - Инициализирует `chromadb.PersistentClient(CHROMA_DB_PATH)`;
  - управляет коллекциями:
    - `global_docs` (глобальный режим),
    - `session_<uuid>` (сессионный режим);
  - методы:
    - `get_collection_for_session(session_id: str)`;
    - `add_documents(texts, metadatas, session_id)` — с учётом `RAG_SCOPE`;
    - `search(query_text, session_id, k)` — с учётом `RAG_SCOPE`.

- `app/pdf_ingestion.py`
  - Сохранение загруженных PDF в:
    - `pdf_storage/global/` (архив);
    - `pdf_storage/session_<uuid>/` (логическая привязка к сессии);
  - извлечение текста из PDF;
  - чанкинг текста;
  - вызов `vector_store.add_documents(...)`.

- `app/session_manager.py`
  - Создание нового `session_id` (uuid4);
  - хранение:
    - истории сообщений (user/assistant);
    - списка PDF, привязанных к сессии.

- `app/rag_pipeline.py`
  - На вход: `session_id`, текущий вопрос, история;
  - шаги:
    - history‑aware подготовка запроса (включение части истории);
    - поиск релевантных чанков через `vector_store.search(...)`;
    - вызов `llm_client.chat(...)` с:
      - системным промптом;
      - историей;
      - retrieved‑контекстом.

- `app/graph_store.py`
  - Подключение к Neo4j;
  - построение/обновление подграфа для `session_id`:
    - при `RAG_SCOPE=session`: PDF + диалог только этой сессии;
    - при `RAG_SCOPE=global`: все PDF из `global_docs` + диалог сессии;
  - генерация сущностей/связей через LLM (Qwen);
  - функции:
    - `build_graph_for_session(session_id)` → (summary_text, visualization_html).

---

## Streamlit UI

Планируемый `ui/app.py`:

- Кнопка **«Новый чат»**:
  - создаёт новый `session_id`;
  - очищает историю сообщений;
  - (в session‑режиме) создаёт новую коллекцию Chroma `session_<id>`.

- Блок **чата**:
  - показывает историю;
  - поле ввода → вызывает `rag_pipeline` → добавляет ответ в историю.

- Блок **загрузки PDF**:
  - `file_uploader` (accept_multiple_files=True);
  - при загрузке → `pdf_ingestion`.

- Кнопка **«Показать/обновить граф»**:
  - вызывает `graph_store.build_graph_for_session(session_id)`;
  - результат отображается как сообщение ассистента:
    - сверху текст;
    - снизу интерактивный граф (HTML) в том же сообщении.


