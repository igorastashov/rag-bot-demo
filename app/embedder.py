from typing import List

from sentence_transformers import SentenceTransformer

from .config import get_settings


_settings = get_settings()
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            str(_settings.embedder_model_path),
            device=_settings.embedder_device,
        )
    return _model


def encode_texts(texts: List[str]) -> List[List[float]]:
    """
    Encode a list of texts into embedding vectors using bge-m3.
    """
    if not texts:
        return []
    model = _get_model()
    # convert_to_numpy=False to keep it lightweight and JSON-serialisable after .tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


__all__ = ["encode_texts"]


