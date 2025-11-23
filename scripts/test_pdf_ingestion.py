import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so that 'app' package can be imported
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.pdf_ingestion import ingest_uploaded_pdfs


def main() -> None:
    # Максимально подробный лог, принудительно переопределяем конфиг, если он уже был.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    print("=== test_pdf_ingestion.py: START ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"sys.path[0:3]: {sys.path[0:3]}")
    print(f"Command-line args: {sys.argv}")

    if len(sys.argv) < 2:
        print("Usage: python scripts/test_pdf_ingestion.py <path_to_pdf> [session_id]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.is_file():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    session_id = sys.argv[2] if len(sys.argv) > 2 else "test_session"

    # Ensure settings are loaded and directories created
    print("Loading settings via get_settings()...")
    settings = get_settings()
    logger.info("Using PDF_STORAGE_ROOT=%s", settings.pdf_storage_root)
    logger.info("Using CHROMA_DB_PATH=%s", settings.chroma_db_path)
    print(f"PDF_STORAGE_ROOT={settings.pdf_storage_root}")
    print(f"CHROMA_DB_PATH={settings.chroma_db_path}")

    logger.info("Reading PDF file: %s", pdf_path)
    print(f"Reading PDF bytes from: {pdf_path}")
    raw_bytes = pdf_path.read_bytes()

    logger.info("Calling ingest_uploaded_pdfs for session '%s'...", session_id)
    print("Calling ingest_uploaded_pdfs ...")
    try:
        stats = ingest_uploaded_pdfs(
            [(raw_bytes, os.path.basename(pdf_path))],
            session_id=session_id,
        )
    except Exception as e:
        import traceback

        print("\n--- EXCEPTION DURING INGESTION ---")
        print(e)
        traceback.print_exc()
        sys.exit(1)

    logger.info("Ingestion stats: %s", stats)
    print("\n=== Ingestion finished ===")
    print(f"Session ID: {session_id}")
    print(f"PDF file:   {pdf_path}")
    print("Stats:")
    for item in stats:
        print(
            f"  - file_name={item['file_name']}, "
            f"pages={item['num_pages']}, "
            f"chunks={item['num_chunks']}, "
            f"chars={item['total_chars']}"
        )


if __name__ == "__main__":
    main()


