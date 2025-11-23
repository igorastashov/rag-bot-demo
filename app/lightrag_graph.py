from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .config import get_settings
from .session_manager import get_session
from .pdf_ingestion import _extract_text_from_pdf_bytes  # type: ignore
from .graph_store import _build_pyvis_html, _session_history_to_text  # type: ignore
from .embedder import encode_texts
from .llm_client import chat as app_llm_chat


logger = logging.getLogger(__name__)
_settings = get_settings()

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


# Отдельный event loop для всех операций LightRAG в рамках этого процесса.
_lightrag_loop: asyncio.AbstractEventLoop | None = None


def _get_lightrag_loop() -> asyncio.AbstractEventLoop:
    """
    Возвращает (и при необходимости создаёт) единый event loop для RAG.

    Это устраняет проблему RuntimeError: 'Lock ... is bound to a different event loop',
    так как все async-операции RAG выполняются в одном и том же loop.
    """
    global _lightrag_loop
    if _lightrag_loop is None or _lightrag_loop.is_closed():
        _lightrag_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_lightrag_loop)
    return _lightrag_loop


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

    # Определяем рабочую директорию LightRAG (можно переопределить через LIGHTRAG_WORKING_DIR)
    default_working_dir = PROJECT_ROOT / "data" / "lightrag_storage"
    working_dir_env = os.getenv("LIGHTRAG_WORKING_DIR")
    working_dir = (
        Path(working_dir_env).resolve() if working_dir_env else default_working_dir
    )

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
            logger.info(
                "Inserting chat history into LightRAG for session %s (chars=%d)",
                session_id,
                len(history_text),
            )
            # Документ с фиксированным ID; при повторных вызовах LightRAG сам
            # пропустит дубликаты. Это даёт простой способ включить диалог
            # в граф без сложной логики обновления.
            await rag.ainsert(
                [history_text],
                ids=[f"chat_{session_id}"],
                file_paths=[f"chat://{session_id}"],
            )

        # 3) Собираем узлы и рёбра из графа LightRAG
        nodes: List[dict] = []
        edges: List[dict] = []

        graph = rag.chunk_entity_relation_graph
        all_entities = await graph.get_all_labels()
        logger.info("LightRAG graph entities count: %d", len(all_entities))

        for entity_name in all_entities:
            node_data = await graph.get_node(entity_name) or {}
            label = node_data.get("description") or entity_name
            nodes.append({"id": entity_name, "label": label})

        for src in all_entities:
            for tgt in all_entities:
                if src == tgt:
                    continue
                if not await graph.has_edge(src, tgt):
                    continue
                edge_data = await graph.get_edge(src, tgt) or {}
                rel_type = edge_data.get("description") or edge_data.get("keywords") or ""
                edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                    }
                )

        logger.info(
            "LightRAG graph built: %d node(s), %d edge(s)", len(nodes), len(edges)
        )

        if not nodes:
            return (
                "LightRAG не смог выделить сущности из данных текущей сессии "
                "(граф пустой). Проверьте содержимое диалога и документов.",
                None,
            )

        # 3) Генерируем HTML через уже существующий pyvis‑билдер
        graph_html = _build_pyvis_html(nodes, edges)

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

    Вызывает асинхронную логику в рамках единого event loop,
    чтобы избежать конфликтов asyncio.Lock между разными loop.
    """
    loop = _get_lightrag_loop()
    return loop.run_until_complete(_build_lightrag_graph_for_session_async(session_id))


__all__ = ["build_lightrag_graph_for_session"]



