import os
import sys
import logging

import streamlit as st
import streamlit.components.v1 as components

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

import time

from app.config import get_settings
from app.pdf_ingestion import ingest_uploaded_pdfs
from app.rag_pipeline import answer_question
from app.session_manager import create_session, get_session
from app.vector_store import get_collection
from app.graph_store import build_graph_for_session
from app.lightrag_graph import (
    start_graph_build_async,
    get_graph_build_status,
    is_graph_building,
)


settings = get_settings()
USE_LIGHTRAG_GRAPH = os.getenv("GRAPH_BACKEND", "simple").lower() == "lightrag"

st.set_page_config(page_title="Анализ документов. Выявление сущностей и свзяей.", layout="wide")
st.title("Анализ документов. Выявление сущностей и свзяей.")


if "session_id" not in st.session_state:
    session = create_session()
    st.session_state["session_id"] = session.session_id

session_id: str = st.session_state["session_id"]

st.sidebar.header("Settings")
st.sidebar.write(f"RAG scope: **{settings.rag_scope}**")
st.sidebar.write(f"Session ID: `{session_id}`")
st.sidebar.write(f"Graph backend: **{'RAG' if USE_LIGHTRAG_GRAPH else 'Simple JSON'}**")


if st.sidebar.button("🔄 Новый чат"):
    # Создаём новую серверную сессию (SessionState) и полностью очищаем
    # Streamlit-состояние, чтобы сбросить историю чата и прочие флаги.
    session = create_session()
    st.session_state.clear()
    st.session_state["session_id"] = session.session_id
    st.rerun()


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
            # Если какие‑то файлы не удалось прочитать как PDF, отображаем предупреждение.
            if any("error" in s for s in all_stats):
                st.warning(
                    "Некоторые файлы не удалось прочитать как PDF. "
                    "Проверьте столбец 'error' в таблице и формат исходных файлов."
                )


st.subheader("2. Чат")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Если это сообщение содержит граф, визуализируем его
        graph_html = msg.get("graph_html")
        if graph_html:
            components.html(graph_html, height=600)


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
st.subheader("3. Граф знаний")

# ─────────────────────────────────────────────────────────────────────────────
# Неблокирующее построение графа для LightRAG
# ─────────────────────────────────────────────────────────────────────────────

if USE_LIGHTRAG_GRAPH:
    # Проверяем, есть ли готовый результат от фонового построения
    is_done, summary_text, graph_html = get_graph_build_status(session_id)

    if is_done and summary_text:
        # Результат готов — показываем граф
        with st.chat_message("assistant"):
            st.markdown(summary_text)
            if graph_html:
                components.html(graph_html, height=600)

        # Сохраняем в историю чата
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": summary_text,
                "graph_html": graph_html,
            }
        )

    # Проверяем, выполняется ли сейчас построение
    graph_is_building = is_graph_building(session_id)

    if graph_is_building:
        st.info("⏳ Граф знаний строится в фоне... Можете продолжать работу с чатом.")
        # Автоматически обновляем страницу каждые 3 секунды для проверки статуса
        time.sleep(3)
        st.rerun()

    # Кнопка неактивна, пока граф строится
    if st.button(
        "Построить/обновить граф знаний по текущей сессии",
        disabled=graph_is_building,
    ):
        session_state = get_session(session_id)

        if not session_state.messages and not session_state.attached_pdfs:
            st.warning(
                "Недостаточно данных для построения графа. "
                "Добавьте сообщения в чат или загрузите документы."
            )
        else:
            # Запускаем построение в фоне
            if start_graph_build_async(session_id):
                st.info("🚀 Построение графа запущено в фоне...")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Граф уже строится. Пожалуйста, подождите.")

else:
    # ─────────────────────────────────────────────────────────────────────────
    # Базовый режим (Simple JSON) — синхронное построение
    # ─────────────────────────────────────────────────────────────────────────

    if st.button("Построить/обновить граф знаний по текущей сессии"):
        session_state = get_session(session_id)
        summary_text: str = ""
        graph_html = None

        # Получаем все документы (чанки) из коллекции для текущей сессии
        collection = get_collection(session_id)
        data = collection.get(include=["documents"])
        pdf_chunks = data.get("documents") or []

        if not session_state.messages and not pdf_chunks:
            st.warning(
                "Недостаточно данных для построения графа. "
                "Добавьте сообщения в чат или загрузите документы."
            )
            summary_text, graph_html = "", None
        else:
            with st.spinner("Строю граф знаний на основе диалога и документов..."):
                summary_text, graph_html = build_graph_for_session(
                    session_id=session_id,
                    pdf_chunks=pdf_chunks,
                )

        if summary_text:
            # Показываем как отдельный ответ ассистента
            with st.chat_message("assistant"):
                st.markdown(summary_text)
                if graph_html:
                    components.html(graph_html, height=600)

            # Сохраняем в историю чата
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": summary_text,
                    "graph_html": graph_html,
                }
            )


