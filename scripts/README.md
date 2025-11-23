## Скрипты для тестирования и диагностики

Эта папка содержит небольшие утилиты, которые помогают **проверить и отладить отдельные части системы** (ингестию PDF, работу RAG и т.д.) без запуска полного Streamlit‑приложения.

Все команды ниже предполагают, что вы находитесь в корне проекта (`D:/__projects__/drop-rag`) и активировали виртуальное окружение:

```bash
source .venv/Scripts/activate   # Git Bash
# или
.\.venv\Scripts\activate        # PowerShell / cmd
```

---

## 1. Тест индексации PDF (`test_pdf_ingestion.py`)

Проверяет, что:
- PDF читается без ошибок;
- текст успешно извлекается и режется на чанки;
- чанки добавляются в Chroma для указанной сессии.

Запуск:

```bash
python scripts/test_pdf_ingestion.py "data/pdfs/Асташов Игорь.pdf"
```

Можно передать второй аргумент — `session_id` (иначе используется `test_session`):

```bash
python scripts/test_pdf_ingestion.py "data/pdfs/Асташов Игорь.pdf" my_debug_session
```

Скрипт выведет в терминал:
- путь к файлу;
- статистику по страницам, чанкам и количеству символов;
- лог‑сообщения о сохранении PDF и добавлении чанков в векторную БД.
Если файл не читается как PDF (ошибка `No /Root object!` и т.п.), он будет отмечен полем `error` и пропущен.

---

## 2. Тест построения графа знаний (`test_graph_from_pdfs.py`)

Строит граф знаний для одной временной сессии на основе указанных PDF (и, опционально, текстового сообщения‑истории), записывает его в Neo4j и сохраняет HTML‑визуализацию на диск.

Примеры запуска:

```bash
python scripts/test_graph_from_pdfs.py "data/pdfs/Асташов Игорь.pdf"

python scripts/test_graph_from_pdfs.py "data/pdfs/Асташов Игорь.pdf" "data/pdfs/Astashov_HSE (2).pdf"

python scripts/test_graph_from_pdfs.py "data/pdfs/Асташов Игорь.pdf" --message "Игорь учится в университете ВШЭ."
```

Скрипт:
- создаёт новую `session_id`;
- индексацирует указанные PDF так же, как это делает Streamlit;
- достаёт все чанки из Chroma;
- вызывает `build_graph_for_session` (LLM → JSON‑граф → Neo4j → `pyvis` HTML);
- печатает summary и путь к файлу `data/graphs/graph_<session_id>.html`, который можно открыть в браузере.
