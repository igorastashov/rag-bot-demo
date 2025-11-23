import logging
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path so that 'app' package can be imported
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.session_manager import create_session, append_message
from app.pdf_ingestion import ingest_uploaded_pdfs
from app.vector_store import get_collection
from app.graph_store import build_graph_for_session


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/test_graph_from_pdfs.py <pdf1> [<pdf2> ...] "
            "[--message 'optional history text']"
        )
        sys.exit(1)

    # Разбираем аргументы: список PDF и опциональное сообщение истории.
    pdf_paths: List[Path] = []
    history_text = ""
    args_iter = iter(sys.argv[1:])
    for arg in args_iter:
        if arg == "--message":
            try:
                history_text = next(args_iter)
            except StopIteration:
                print("ERROR: --message flag provided but no text given.")
                sys.exit(1)
            break
        else:
            pdf_paths.append(Path(arg))

    if not pdf_paths:
        print("ERROR: At least one PDF path must be provided.")
        sys.exit(1)

    for p in pdf_paths:
        if not p.is_file():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    settings = get_settings()
    logger.info("PDF_STORAGE_ROOT=%s", settings.pdf_storage_root)
    logger.info("CHROMA_DB_PATH=%s", settings.chroma_db_path)
    logger.info("NEO4J_URI=%s", settings.neo4j_uri)

    # Создаём сессию и опциональное сообщение-историю
    session_state = create_session()
    session_id = session_state.session_id
    logger.info("Created session_id=%s", session_id)

    if history_text:
        append_message(session_id, "user", history_text)
        logger.info("Added history message: %s", history_text)

    # Индексируем все указанные PDF
    for pdf_path in pdf_paths:
        logger.info("Reading PDF: %s", pdf_path)
        raw_bytes = pdf_path.read_bytes()
        ingest_uploaded_pdfs([(raw_bytes, os.path.basename(pdf_path))], session_id)

    # Забираем все документы (чанки) из коллекции
    collection = get_collection(session_id)
    data = collection.get(include=["documents"])
    pdf_chunks = data.get("documents") or []
    logger.info("Retrieved %d document chunk(s) from vector store", len(pdf_chunks))

    # Строим граф
    logger.info("Building graph for session %s ...", session_id)
    summary_text, graph_html = build_graph_for_session(session_id, pdf_chunks)

    print("\n=== Graph build finished ===")
    print("Session ID:", session_id)
    print("Summary:")
    print(summary_text)

    # Сохраняем HTML в файл, если он есть
    if graph_html:
        out_dir = Path("data/graphs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"graph_{session_id}.html"
        out_file.write_text(graph_html, encoding="utf-8")
        print(f"\nInteractive graph HTML saved to: {out_file}")
        print("You can open it in a browser to inspect the graph.")
    else:
        print("\nNo graph HTML generated (graph_html is None).")


if __name__ == "__main__":
    main()


