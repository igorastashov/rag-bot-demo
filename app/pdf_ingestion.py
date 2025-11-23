from __future__ import annotations

import io
import os
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import pdfplumber

from .config import get_settings
from .session_manager import attach_pdf
from .vector_store import add_documents


logger = logging.getLogger(__name__)
_settings = get_settings()


def _ensure_session_folder(session_id: str) -> Path:
    session_folder = _settings.pdf_storage_root / f"session_{session_id}"
    session_folder.mkdir(parents=True, exist_ok=True)
    return session_folder


def _save_pdf_file(raw_bytes: bytes, original_name: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = os.path.basename(original_name) or "document.pdf"
    target_path = target_dir / safe_name

    # Avoid overwriting by appending a counter if needed
    counter = 1
    base, ext = os.path.splitext(safe_name)
    while target_path.exists():
        target_path = target_dir / f"{base}_{counter}{ext}"
        counter += 1

    with open(target_path, "wb") as f:
        f.write(raw_bytes)

    return target_path


def _extract_text_from_pdf_bytes(raw_bytes: bytes) -> Tuple[str, int]:
    """
    Extract plain text from a PDF file given as bytes.
    """
    text_chunks: List[str] = []
    logger.info("Opening PDF with pdfplumber to extract text...")
    page_count = 0
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            page_count = len(pdf.pages)
            logger.info("PDF opened successfully, pages=%d", page_count)
            for i, page in enumerate(pdf.pages, start=1):
                logger.debug("Extracting text from page %d...", i)
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text)
                else:
                    logger.debug("Page %d has no extractable text.", i)
    except Exception as e:
        # Не падаем, а логируем и возвращаем пустой текст,
        # чтобы вызывающая сторона могла корректно обработать.
        logger.exception("Error while extracting text from PDF: %s", e)
        return "", 0

    logger.info(
        "Finished extracting text from PDF (pages=%d, non-empty pages=%d)",
        page_count,
        len(text_chunks),
    )
    return "\n\n".join(text_chunks), page_count


def _simple_chunk_text(
    text: str,
    max_chars: int = 2000,
    overlap: int = 200,
) -> List[str]:
    """
    Simple character-based chunking for initial implementation.
    """
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Если дошли до конца текста, выходим, чтобы не попасть в бесконечный цикл.
        if end >= n:
            break
        # Смещаемся с перекрытием.
        new_start = end - overlap
        # Защита от зацикливания на коротком хвосте.
        if new_start <= start:
            break
        start = new_start
    return chunks


def ingest_uploaded_pdfs(
    files: Iterable[Tuple[bytes, str]],
    session_id: str,
) -> List[Dict[str, int | str]]:
    """
    Ingest uploaded PDFs for a given session.

    Parameters
    ----------
    files : iterable of (raw_bytes, original_name)
        Raw bytes and original filenames of uploaded PDFs.
    session_id : str
        Current chat session identifier.
    """
    # files может быть итератором, поэтому приводим к списку один раз
    files_list = list(files)
    logger.info("Starting ingestion for session %s: %d file(s)", session_id, len(files_list))

    global_folder = _settings.pdf_storage_root / "global"
    session_folder = _ensure_session_folder(session_id)

    texts: List[str] = []
    metadatas: List[dict] = []
    stats: List[Dict[str, int | str]] = []

    for raw_bytes, original_name in files_list:
        logger.info("Ingesting PDF '%s' for session %s", original_name, session_id)
        # Save into global archive
        global_path = _save_pdf_file(raw_bytes, original_name, global_folder)
        # Save into session-specific folder for traceability
        session_path = _save_pdf_file(raw_bytes, original_name, session_folder)
        logger.info("Saved PDF '%s' to %s (global) and %s (session)", original_name, global_path, session_path)

        # Link PDF to session state
        attach_pdf(session_id, str(session_path))

        # Extract and chunk text
        full_text, page_count = _extract_text_from_pdf_bytes(raw_bytes)
        if not full_text.strip():
            logger.warning(
                "No text extracted from PDF '%s' (session=%s). Skipping.",
                original_name,
                session_id,
            )
            stats.append(
                {
                    "file_name": original_name,
                    "num_pages": page_count,
                    "num_chunks": 0,
                    "total_chars": 0,
                    "error": "no_text_extracted",
                }
            )
            continue

        chunks = _simple_chunk_text(full_text)
        num_chunks = len(chunks)
        total_chars = len(full_text)

        logger.info(
            "Extracted %d characters and %d chunk(s) from '%s'",
            total_chars,
            num_chunks,
            original_name,
        )

        stats.append(
            {
                "file_name": original_name,
                "num_pages": page_count,
                "num_chunks": num_chunks,
                "total_chars": total_chars,
            }
        )
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source_path": str(session_path),
                    "original_name": original_name,
                    "chunk_index": idx,
                    "session_id": session_id,
                }
            )

    if texts:
        logger.info(
            "Adding %d text chunk(s) to vector store for session %s",
            len(texts),
            session_id,
        )
        add_documents(texts, metadatas=metadatas, session_id=session_id)
        logger.info("Ingestion completed for session %s", session_id)
    else:
        logger.info("No text extracted from uploaded PDFs for session %s", session_id)

    return stats


__all__ = ["ingest_uploaded_pdfs"]


