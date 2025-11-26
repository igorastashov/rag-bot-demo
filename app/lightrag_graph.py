from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import get_settings
from .session_manager import get_session
from .pdf_ingestion import _extract_text_from_pdf_bytes  # type: ignore
from .graph_store import _build_pyvis_html, _session_history_to_text  # type: ignore
from .embedder import encode_texts
from .llm_client import chat as app_llm_chat


logger = logging.getLogger(__name__)
_settings = get_settings()

# Глобальная блокировка, чтобы не запускать два построения графа параллельно
_build_lock = threading.Lock()

# Хранилище для результатов фоновых задач построения графа
_graph_build_tasks: Dict[str, Dict[str, Any]] = {}

# Подключаем локальную копию LightRAG из sample_prj/LightRAG
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
LIGHTRAG_ROOT = PROJECT_ROOT / "sample_prj" / "LightRAG"

if LIGHTRAG_ROOT.is_dir() and str(LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTRAG_ROOT))

try:
    from lightrag import LightRAG  # type: ignore
    from lightrag.utils import EmbeddingFunc, setup_logger as setup_lightrag_logger  # type: ignore
except ImportError as e:  # pragma: no cover - защитный код
    LightRAG = None  # type: ignore
    EmbeddingFunc = None  # type: ignore
    setup_lightrag_logger = None  # type: ignore
    logger.warning("LightRAG is not available: %s", e)


def _detect_embedding_dim() -> int:
    """
    Определить размерность эмбеддинга, используя текущий bge-m3 через app.embedder.
    """
    probe = encode_texts(["__lightrag_dim_probe__"])
    if not probe or not probe[0]:
        raise RuntimeError("Failed to detect embedding dimension from encode_texts().")
    return len(probe[0])


async def _embedding_func(texts: List[str]):
    """
    Async-обёртка над app.embedder.encode_texts для LightRAG.
    Возвращает numpy‑массив нужной формы.
    """
    import numpy as np

    if not texts:
        return np.empty((0, 0), dtype="float32")

    vectors = await asyncio.to_thread(encode_texts, texts)
    return np.asarray(vectors, dtype="float32")


async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: Optional[List[dict]] = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """
    Async-обёртка над app.llm_client.chat для LightRAG (OpenAI‑совместимый интерфейс).
    """
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    return await asyncio.to_thread(app_llm_chat, messages)


async def _build_lightrag_graph_for_session_async(
    session_id: str,
) -> Tuple[str, Optional[str]]:
    """
    Асинхронное построение графа знаний через LightRAG для указанной сессии.

    Возвращает (summary_text, graph_html), где graph_html — уже готовый HTML для встраивания в UI.
    """
    if LightRAG is None or EmbeddingFunc is None or setup_lightrag_logger is None:
        return (
            "LightRAG недоступен (пакет lightrag не найден). "
            "Проверьте, что папка sample_prj/LightRAG присутствует и импортируется.",
            None,
        )

    session_state = get_session(session_id)
    pdf_paths = [Path(p) for p in session_state.attached_pdfs if Path(p).is_file()]
    history_text = _session_history_to_text(session_id).strip()

    if not pdf_paths and not history_text:
        return (
            "Недостаточно данных для построения графа: нет ни диалога, ни загруженных PDF.",
            None,
        )

    # Настраиваем логгер LightRAG
    setup_lightrag_logger("lightrag", level="INFO")

    # Рабочая директория LightRAG из конфига
    working_dir = _settings.lightrag_working_dir

    logger.info("Using LightRAG working_dir=%s", working_dir)

    # Workspace привязываем к session_id, чтобы логически изолировать данные
    workspace = f"session_{session_id}"
    logger.info("Using LightRAG workspace=%s", workspace)

    embedding_dim = _detect_embedding_dim()
    logger.info("Detected embedding_dim=%d for LightRAG", embedding_dim)

    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        graph_storage="Neo4JStorage",
        llm_model_func=_llm_model_func,
        llm_model_name=_settings.llm_model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            func=_embedding_func,
        ),
    )

    await rag.initialize_storages()

    try:
        # 1) Вставляем все PDF в LightRAG (полный текст, без повторного чанкинга Chroma)
        for pdf_path in pdf_paths:
            logger.info("Reading PDF for LightRAG: %s", pdf_path)
            raw_bytes = pdf_path.read_bytes()
            full_text, page_count = _extract_text_from_pdf_bytes(raw_bytes)

            if not full_text.strip():
                logger.warning(
                    "No text extracted from PDF '%s'. Skipping for LightRAG.",
                    pdf_path,
                )
                continue

            logger.info(
                "Inserting PDF into LightRAG: '%s' (pages=%d, chars=%d)",
                pdf_path.name,
                page_count,
                len(full_text),
            )

            await rag.ainsert(
                [full_text],
                ids=[pdf_path.name],
                file_paths=[str(pdf_path)],
            )

        # 2) Добавляем историю диалога как отдельный документ (если есть сообщения)
        if history_text:
            chat_doc_id = f"chat_{session_id}"
            logger.info(
                "Refreshing chat history in LightRAG for session %s (chars=%d)",
                session_id,
                len(history_text),
            )
            # Сначала аккуратно удаляем предыдущую версию чат-документа (если была),
            # чтобы новые факты из диалога (например, новые сущности и связи)
            # пересчитались и попали в граф.
            try:
                await rag.adelete_by_doc_id(chat_doc_id, delete_llm_cache=False)
            except Exception as e:  # защитный fallback, не роняем весь пайплайн
                logger.warning(
                    "Failed to delete previous chat document %s from LightRAG: %s",
                    chat_doc_id,
                    e,
                )

            await rag.ainsert(
                [history_text],
                ids=[chat_doc_id],
                file_paths=[f"chat://{session_id}"],
            )

        # 3) Собираем узлы и рёбра из графа LightRAG (оптимизированно)
        graph = rag.chunk_entity_relation_graph

        # Получаем все рёбра одним запросом к Neo4j (вместо O(N²) итераций)
        all_edges_data = await graph.get_all_edges()
        logger.info("LightRAG graph total edges: %d", len(all_edges_data))

        # Собираем уникальные узлы из рёбер + все метки
        all_entities = await graph.get_all_labels()
        logger.info("LightRAG graph total entities: %d", len(all_entities))

        # Лимиты из конфига для визуализации
        max_nodes = _settings.lightrag_graph_max_nodes
        max_edges = _settings.lightrag_graph_max_edges

        # Если узлов слишком много — берём только самые связанные
        if len(all_entities) > max_nodes:
            # Считаем степень каждого узла (количество связей)
            node_degree: dict[str, int] = {}
            for edge_data in all_edges_data:
                src = edge_data.get("source")
                tgt = edge_data.get("target")
                if src:
                    node_degree[src] = node_degree.get(src, 0) + 1
                if tgt:
                    node_degree[tgt] = node_degree.get(tgt, 0) + 1

            # Сортируем по степени (убывание) и берём топ-N
            sorted_entities = sorted(
                all_entities,
                key=lambda e: node_degree.get(e, 0),
                reverse=True,
            )
            selected_entities = set(sorted_entities[:max_nodes])
            logger.info(
                "Limiting graph to top %d nodes (by degree) out of %d",
                max_nodes,
                len(all_entities),
            )
        else:
            selected_entities = set(all_entities)

        # Собираем узлы
        nodes: List[dict] = []
        for entity_name in selected_entities:
            node_data = await graph.get_node(entity_name) or {}
            label = node_data.get("description") or entity_name
            nodes.append({"id": entity_name, "label": label})

        # Собираем рёбра (только между выбранными узлами)
        edges: List[dict] = []
        for edge_data in all_edges_data:
            src = edge_data.get("source")
            tgt = edge_data.get("target")
            if not src or not tgt:
                continue
            # Пропускаем рёбра, если узлы не в выборке
            if src not in selected_entities or tgt not in selected_entities:
                continue
            rel_type = edge_data.get("description") or edge_data.get("keywords") or ""
            edges.append({
                "source": src,
                "target": tgt,
                "type": rel_type,
            })
            # Лимит на количество рёбер
            if len(edges) >= max_edges:
                logger.info(
                    "Limiting graph to %d edges out of %d",
                    max_edges,
                    len(all_edges_data),
                )
                break

        logger.info(
            "LightRAG graph for visualization: %d node(s), %d edge(s)",
            len(nodes),
            len(edges),
        )

        if not nodes:
            return (
                "LightRAG не смог выделить сущности из данных текущей сессии "
                "(граф пустой). Проверьте содержимое диалога и документов.",
                None,
            )

        # 3) Генерируем HTML через уже существующий pyvis‑билдер
        graph_html = _build_pyvis_html(nodes, edges)

        # Формируем summary с информацией о лимитах
        total_entities = len(all_entities)
        total_edges = len(all_edges_data)
        if len(nodes) < total_entities or len(edges) < total_edges:
            summary = (
                f"Граф знаний построен.\n"
                f"Всего сущностей: {total_entities}, связей: {total_edges}.\n"
                f"Отображено: {len(nodes)} сущностей, {len(edges)} связей "
                f"(лимит: {max_nodes}/{max_edges})."
            )
        else:
            summary = (
                "Граф знаний построен.\n"
                f"Сущностей: {len(nodes)}, связей: {len(edges)}."
            )
        return summary, graph_html

    finally:
        await rag.finalize_storages()


def build_lightrag_graph_for_session(
    session_id: str,
) -> Tuple[str, Optional[str]]:
    """
    Синхронный фасад для использования в Streamlit UI.

    Запускает асинхронную логику в отдельном потоке, чтобы избежать
    конфликтов с уже работающим event loop (например, внутри Streamlit).

    Использует глобальную блокировку _build_lock, чтобы предотвратить
    параллельные запуски построения графа (например, при повторном клике).
    """
    # Пытаемся захватить блокировку без ожидания
    if not _build_lock.acquire(blocking=False):
        logger.warning(
            "Graph build already in progress for another request, skipping."
        )
        return (
            "⏳ Граф уже строится. Пожалуйста, дождитесь завершения предыдущего запроса.",
            None,
        )

    try:
        result: List[Tuple[str, Optional[str]]] = []
        error: List[BaseException] = []

        def _worker() -> None:
            try:
                res = asyncio.run(_build_lightrag_graph_for_session_async(session_id))
                result.append(res)
            except BaseException as e:  # pragma: no cover - пробрасываем наверх
                error.append(e)

        thread = threading.Thread(
            target=_worker,
            name=f"lightrag-graph-{session_id}",
            daemon=True,
        )
        thread.start()
        thread.join()

        if error:
            raise error[0]
        if not result:
            # Защитный fallback, не должен срабатывать в нормальном сценарии
            return (
                "Не удалось построить граф (внутренняя ошибка при построении графа).",
                None,
            )
        return result[0]
    finally:
        _build_lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# Неблокирующий API для Streamlit (фоновое построение графа)
# ─────────────────────────────────────────────────────────────────────────────


def start_graph_build_async(session_id: str) -> bool:
    """
    Запускает построение графа в фоновом потоке (неблокирующий вызов).

    Возвращает True, если задача успешно запущена.
    Возвращает False, если построение уже выполняется для этой сессии.
    """
    # Проверяем, не запущена ли уже задача для этой сессии
    if session_id in _graph_build_tasks:
        task = _graph_build_tasks[session_id]
        thread = task.get("thread")
        if thread and thread.is_alive():
            logger.warning(
                "Graph build already running for session %s, skipping.",
                session_id,
            )
            return False

    # Пытаемся захватить глобальную блокировку
    if not _build_lock.acquire(blocking=False):
        logger.warning(
            "Graph build lock is held by another session, cannot start for %s.",
            session_id,
        )
        return False

    def _worker() -> None:
        try:
            res = asyncio.run(_build_lightrag_graph_for_session_async(session_id))
            _graph_build_tasks[session_id] = {
                "result": res,
                "error": None,
                "done": True,
                "thread": None,
            }
            logger.info("Graph build completed for session %s", session_id)
        except Exception as e:
            logger.exception("Graph build failed for session %s: %s", session_id, e)
            _graph_build_tasks[session_id] = {
                "result": None,
                "error": str(e),
                "done": True,
                "thread": None,
            }
        finally:
            _build_lock.release()

    thread = threading.Thread(
        target=_worker,
        name=f"lightrag-graph-async-{session_id}",
        daemon=True,
    )

    _graph_build_tasks[session_id] = {
        "thread": thread,
        "done": False,
        "result": None,
        "error": None,
    }

    thread.start()
    logger.info("Started async graph build for session %s", session_id)
    return True


def get_graph_build_status(session_id: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Проверяет статус построения графа для указанной сессии.

    Возвращает кортеж (is_done, summary_text, graph_html):
      - is_done=False: задача ещё выполняется (или не запущена)
      - is_done=True: задача завершена, результат в summary_text и graph_html
        (если была ошибка, summary_text содержит сообщение об ошибке, graph_html=None)

    После получения результата (is_done=True) задача удаляется из хранилища.
    """
    if session_id not in _graph_build_tasks:
        return (False, None, None)

    task = _graph_build_tasks[session_id]

    if not task.get("done"):
        # Задача ещё выполняется
        return (False, None, None)

    # Задача завершена — забираем результат и очищаем
    result = task.get("result")
    error = task.get("error")
    del _graph_build_tasks[session_id]

    if error:
        return (True, f"❌ Ошибка при построении графа: {error}", None)

    if result:
        summary_text, graph_html = result
        return (True, summary_text, graph_html)

    return (True, "Не удалось построить граф (неизвестная ошибка).", None)


def is_graph_building(session_id: str) -> bool:
    """
    Проверяет, выполняется ли сейчас построение графа для указанной сессии.
    """
    if session_id not in _graph_build_tasks:
        return False
    task = _graph_build_tasks[session_id]
    return not task.get("done", True)


__all__ = [
    "build_lightrag_graph_for_session",
    "start_graph_build_async",
    "get_graph_build_status",
    "is_graph_building",
]



