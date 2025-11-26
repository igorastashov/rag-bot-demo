import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from a .env file if present.
load_dotenv()


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    llm_base_url: str
    llm_api_key: str
    llm_model_name: str
    llm_max_output_tokens: int

    embedder_model_path: Path
    embedder_device: str

    chroma_db_path: Path

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str

    pdf_storage_root: Path

    rag_scope: str  # "session" or "global"

    # LightRAG settings
    lightrag_working_dir: Path
    lightrag_graph_max_nodes: int  # Максимальное кол-во узлов для визуализации
    lightrag_graph_max_edges: int  # Максимальное кол-во рёбер для визуализации

    @property
    def is_session_scope(self) -> bool:
        return self.rag_scope.lower() == "session"

    @property
    def is_global_scope(self) -> bool:
        return self.rag_scope.lower() == "global"


_settings: Settings | None = None


def _ensure_directories(settings: Settings) -> None:
    """
    Ensure that important directories exist (created if missing).
    """
    settings.chroma_db_path.mkdir(parents=True, exist_ok=True)
    settings.pdf_storage_root.mkdir(parents=True, exist_ok=True)
    settings.lightrag_working_dir.mkdir(parents=True, exist_ok=True)
    # Subfolders for clarity; actual usage depends on RAG scope.
    (settings.pdf_storage_root / "global").mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """
    Return singleton Settings instance populated from environment variables.

    Environment variables (with reasonable defaults for local dev):
      - LLM_BASE_URL
      - LLM_API_KEY
      - LLM_MODEL_NAME
      - EMBEDDER_MODEL_PATH
      - EMBEDDER_DEVICE
      - CHROMA_DB_PATH
      - NEO4J_URI
      - NEO4J_USERNAME
      - NEO4J_PASSWORD
      - PDF_STORAGE_ROOT
      - RAG_SCOPE
    """
    global _settings
    if _settings is not None:
        return _settings

    llm_base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    llm_api_key = os.getenv("LLM_API_KEY", "dummy")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "qwen-4b-instruct")
    llm_max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512"))

    embedder_model_path = Path(
        os.getenv("EMBEDDER_MODEL_PATH", "./models/bge-m3")
    ).resolve()
    embedder_device = os.getenv("EMBEDDER_DEVICE", "cpu")

    chroma_db_path = Path(
        os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    ).resolve()

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    pdf_storage_root = Path(
        os.getenv("PDF_STORAGE_ROOT", "./data/pdf_storage")
    ).resolve()

    rag_scope = os.getenv("RAG_SCOPE", "session")

    # LightRAG settings
    lightrag_working_dir = Path(
        os.getenv("LIGHTRAG_WORKING_DIR", "./data/lightrag_storage")
    ).resolve()
    lightrag_graph_max_nodes = int(os.getenv("LIGHTRAG_GRAPH_MAX_NODES", "300"))
    lightrag_graph_max_edges = int(os.getenv("LIGHTRAG_GRAPH_MAX_EDGES", "500"))

    _settings = Settings(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model_name=llm_model_name,
        llm_max_output_tokens=llm_max_output_tokens,
        embedder_model_path=embedder_model_path,
        embedder_device=embedder_device,
        chroma_db_path=chroma_db_path,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        pdf_storage_root=pdf_storage_root,
        rag_scope=rag_scope,
        lightrag_working_dir=lightrag_working_dir,
        lightrag_graph_max_nodes=lightrag_graph_max_nodes,
        lightrag_graph_max_edges=lightrag_graph_max_edges,
    )

    _ensure_directories(_settings)
    return _settings


__all__ = ["Settings", "get_settings"]


