import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api import ClientAPI

from .config import get_settings
from .embedder import encode_texts


logger = logging.getLogger(__name__)
_settings = get_settings()
_client: ClientAPI | None = None


def _get_client() -> ClientAPI:
    global _client
    if _client is None:
        logger.info("Initialising Chroma PersistentClient at %s", _settings.chroma_db_path)
        _client = chromadb.PersistentClient(path=str(_settings.chroma_db_path))
    return _client


def _collection_name_for_session(session_id: Optional[str]) -> str:
    if _settings.is_global_scope:
        return "global_docs"
    # session scope by default
    if not session_id:
        return "session_default"
    return f"session_{session_id}"


def get_collection(session_id: Optional[str]) -> Any:
    """
    Get or create the Chroma collection for given session / scope.
    """
    client = _get_client()
    name = _collection_name_for_session(session_id)
    logger.info("Using Chroma collection '%s' for session_id=%s", name, session_id)
    return client.get_or_create_collection(name=name)


def add_documents(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    Add documents to the appropriate Chroma collection with precomputed embeddings.
    """
    if not texts:
        logger.info("add_documents called with empty texts list; nothing to do.")
        return

    if metadatas is None:
        metadatas = [{} for _ in texts]
    elif len(metadatas) != len(texts):
        raise ValueError("Length of metadatas must match length of texts.")

    collection = get_collection(session_id)

    logger.info(
        "Encoding %d text(s) for collection '%s'",
        len(texts),
        collection.name,
    )
    embeddings = encode_texts(texts)
    ids = [str(uuid.uuid4()) for _ in texts]

    logger.info(
        "Adding %d embedding(s) to collection '%s'",
        len(ids),
        collection.name,
    )
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def search(
    query_text: str,
    session_id: Optional[str],
    k: int = 5,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Perform a similarity search in the appropriate Chroma collection.

    Returns a tuple (documents, metadatas).
    """
    if not query_text:
        logger.info("Empty query text received; returning no results.")
        return [], []

    collection = get_collection(session_id)
    logger.info(
        "Searching in collection '%s' for query='%s' (top_k=%d)",
        collection.name,
        query_text,
        k,
    )
    query_embedding = encode_texts([query_text])[0]

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    # Chroma returns lists per query; we only send one query
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    logger.info(
        "Search returned %d document(s) for query='%s' in collection '%s'",
        len(docs),
        query_text,
        collection.name,
    )
    return docs, metas


__all__ = ["get_collection", "add_documents", "search"]


