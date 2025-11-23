import os
import sys
import logging

import streamlit as st

# Ensure project root is on sys.path so that 'app' package can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Basic logging configuration so everything пишется в терминал
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from app.config import get_settings
from app.pdf_ingestion import ingest_uploaded_pdfs
from app.rag_pipeline import answer_question
from app.session_manager import create_session


settings = get_settings()

st.set_page_config(page_title="RAG + GraphRAG Demo", layout="wide")
st.title("RAG + GraphRAG Demo")


if "session_id" not in st.session_state:
    session = create_session()
    st.session_state["session_id"] = session.session_id

session_id: str = st.session_state["session_id"]

st.sidebar.header("Settings")
st.sidebar.write(f"RAG scope: **{settings.rag_scope}**")
st.sidebar.write(f"Session ID: `{session_id}`")


if st.sidebar.button("🔄 Новый чат"):
    session = create_session()
    st.session_state["session_id"] = session.session_id
    st.experimental_rerun()


st.subheader("1. Загрузка PDF")
uploaded_files = st.file_uploader(
    "Загрузите один или несколько PDF файлов",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Индексировать загруженные PDF"):
        total = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_stats = []

        logger.info("Start indexing %d uploaded PDF(s) for session %s", total, session_id)

        for idx, f in enumerate(uploaded_files, start=1):
            status_text.write(f"Обработка файла {idx}/{total}: **{f.name}**")
            logger.info("Processing file %d/%d: %s", idx, total, f.name)

            raw_bytes = f.read()
            file_stats = ingest_uploaded_pdfs([(raw_bytes, f.name)], session_id=session_id)
            all_stats.extend(file_stats)

            progress_bar.progress(idx / total)

        status_text.write("Индексация завершена.")
        st.success(f"PDF успешно проиндексированы для текущей сессии. Всего файлов: {total}")
        logger.info("Finished indexing %d PDF(s) for session %s", total, session_id)

        if all_stats:
            st.markdown("**Сводка по индексации:**")
            # Показываем таблицу: имя файла, страницы, чанки, символы
            st.table(all_stats)


st.subheader("2. Чат")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_input := st.chat_input("Задайте вопрос..."):
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Генерирую ответ..."):
            answer = answer_question(session_id=session_id, question=user_input)
            st.markdown(answer)
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": answer}
    )


st.markdown("---")
st.subheader("3. Граф знаний (Neo4j)")
st.markdown(
    "Пока реализовано только обновление графа в Neo4j без визуализации в UI. "
    "Следующим шагом добавим вывод интерактивного графа прямо под этим блоком."
)


