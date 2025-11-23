import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple


# Ensure project root and local LightRAG repo are on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
LIGHTRAG_ROOT = PROJECT_ROOT / "sample_prj" / "LightRAG"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if LIGHTRAG_ROOT.is_dir() and str(LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTRAG_ROOT))


from app.config import get_settings  # noqa: E402
from app.pdf_ingestion import _extract_text_from_pdf_bytes  # type: ignore  # noqa: E402
from app.graph_store import _build_pyvis_html  # type: ignore  # noqa: E402
from app.embedder import encode_texts  # noqa: E402
from lightrag import LightRAG  # type: ignore  # noqa: E402
from lightrag.utils import EmbeddingFunc, setup_logger as setup_lightrag_logger  # type: ignore  # noqa: E402
from app.llm_client import chat as app_llm_chat  # noqa: E402


logger = logging.getLogger(__name__)


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
    Возвращает numpy-массив нужной формы.
    """
    import numpy as np

    if not texts:
        return np.empty((0, 0), dtype="float32")

    # encode_texts синхронная — выполняем в отдельном потоке, чтобы не блокировать event loop
    vectors = await asyncio.to_thread(encode_texts, texts)
    return np.asarray(vectors, dtype="float32")


async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: List[dict] | None = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """
    Async-обёртка над app.llm_client.chat для LightRAG (OpenAI-совместный интерфейс).
    """
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        # LightRAG передаёт сюда список {"role": ..., "content": ...}
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # app_llm_chat синхронная — выносим в отдельный поток
    return await asyncio.to_thread(app_llm_chat, messages)


async def build_lightrag_graph_from_pdfs(
    pdf_paths: List[Path],
    working_dir: Path,
    workspace: str,
) -> Tuple[str, Path | None]:
    """
    Построить граф знаний через LightRAG на основе списка PDF и сохранить HTML‑визуализацию.

    Возвращает:
      - summary_text: краткое текстовое описание результата;
      - html_path: путь к сгенерированному HTML‑файлу (или None, если не удалось).
    """
    settings = get_settings()

    logger.info("Using LightRAG working_dir=%s", working_dir)
    logger.info("Using LightRAG workspace=%s", workspace)

    # Настраиваем логгер LightRAG one-time
    setup_lightrag_logger("lightrag", level="INFO")

    # Определяем размерность эмбеддинга из текущего bge-m3
    embedding_dim = _detect_embedding_dim()
    logger.info("Detected embedding_dim=%d for LightRAG", embedding_dim)

    # Создаём инстанс LightRAG, используя наш LLM и эмбеддер
    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        graph_storage="Neo4JStorage",  # используем ту же Neo4j, что и остальной проект
        llm_model_func=_llm_model_func,
        llm_model_name=settings.llm_model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            func=_embedding_func,
        ),
    )

    # Явная инициализация стораджей
    await rag.initialize_storages()

    try:
        # 1) Читаем и вставляем все PDF в LightRAG
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

            # Вставляем полный текст; LightRAG сам выполнит токен‑чэнкинг и извлечёт KG.
            await rag.ainsert(
                [full_text],
                ids=[pdf_path.name],
                file_paths=[str(pdf_path)],
            )

        # 2) Строим список узлов и рёбер из внутреннего графа LightRAG
        nodes: List[dict] = []
        edges: List[dict] = []

        graph = rag.chunk_entity_relation_graph
        all_entities = await graph.get_all_labels()
        logger.info("LightRAG graph entities count: %d", len(all_entities))

        # Узлы
        for entity_name in all_entities:
            node_data = await graph.get_node(entity_name) or {}
            label = node_data.get("description") or entity_name
            nodes.append(
                {
                    "id": entity_name,
                    "label": label,
                }
            )

        # Рёбра (как в aexport_data: перебор всех пар с has_edge)
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
            summary = (
                "LightRAG не смог выделить сущности из указанных PDF "
                "(граф пустой). Проверьте содержимое документов."
            )
            return summary, None

        # 3) Генерируем HTML через уже существующий pyvis‑билдер
        html = _build_pyvis_html(nodes, edges)
        graphs_dir = PROJECT_ROOT / "data" / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        html_path = graphs_dir / f"lightrag_graph_{workspace}.html"
        html_path.write_text(html, encoding="utf-8")

        summary = (
            "Граф знаний построен.\n"
            f"Сущностей: {len(nodes)}, связей: {len(edges)}.\n"
            f"HTML‑визуализация: {html_path}"
        )
        return summary, html_path

    finally:
        # Аккуратно закрываем стораджи LightRAG
        await rag.finalize_storages()


def _parse_args() -> Tuple[List[Path], str]:
    """
    Разбор аргументов командной строки:
      python scripts/test_lightrag_graph_from_pdfs.py <pdf1> [<pdf2> ...]
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/test_lightrag_graph_from_pdfs.py <pdf1> [<pdf2> ...]"
        )
        sys.exit(1)

    pdf_paths: List[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.is_file():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)
        pdf_paths.append(p)

    # workspace можно сделать произвольным, но для наглядности используем имя файла/папку
    workspace = "lightrag_test_session"
    return pdf_paths, workspace


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    pdf_paths, workspace = _parse_args()

    settings = get_settings()
    logger = logging.getLogger(__name__)
    logger.info("PDF_STORAGE_ROOT=%s", settings.pdf_storage_root)
    logger.info("CHROMA_DB_PATH=%s", settings.chroma_db_path)
    logger.info("NEO4J_URI=%s", settings.neo4j_uri)

    default_working_dir = PROJECT_ROOT / "data" / "lightrag_storage"
    working_dir_env = os.getenv("LIGHTRAG_WORKING_DIR")
    working_dir = (
        Path(working_dir_env).resolve() if working_dir_env else default_working_dir
    )

    summary_text: str
    html_path: Path | None

    # Запускаем основной асинхронный пайплайн LightRAG
    summary_text, html_path = asyncio.run(
        build_lightrag_graph_from_pdfs(pdf_paths, working_dir, workspace)
    )

    print("\n=== LightRAG Graph build finished ===")
    print("Workspace:", workspace)
    print("Summary:")
    print(summary_text)
    if html_path is not None:
        print(f"\nInteractive graph HTML saved to: {html_path}")
        print("You can open it in a browser to inspect the graph.")
    else:
        print("\nNo graph HTML generated (graph is empty or build failed).")


if __name__ == "__main__":
    main()



