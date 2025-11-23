from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

from neo4j import GraphDatabase, Session
from pyvis.network import Network

from .config import get_settings
from .llm_client import chat as llm_chat
from .session_manager import get_session


_settings = get_settings()


@dataclass
class GraphConfig:
    """
    Simple holder for graph-related configuration.
    """

    uri: str
    username: str
    password: str


_graph_config = GraphConfig(
    uri=_settings.neo4j_uri,
    username=_settings.neo4j_username,
    password=_settings.neo4j_password,
)


def _get_driver():
    return GraphDatabase.driver(
        _graph_config.uri,
        auth=(_graph_config.username, _graph_config.password),
    )


GRAPH_SYSTEM_PROMPT = (
    "You are an assistant that extracts a simple knowledge graph from text.\n"
    "Given the conversation history and text chunks from PDFs, you should "
    "identify important entities and relationships between them.\n\n"
    "Return the result as JSON with two arrays: 'nodes' and 'edges'.\n"
    "Each node: {\"id\": string, \"label\": string}.\n"
    "Each edge: {\"source\": string, \"target\": string, \"type\": string}.\n"
)


def _build_graph_prompt(
    history_text: str,
    pdf_text: str,
) -> str:
    return (
        GRAPH_SYSTEM_PROMPT
        + "\n\nConversation history:\n"
        + history_text
        + "\n\nPDF content (concatenated chunks):\n"
        + pdf_text
    )


def _session_history_to_text(session_id: str) -> str:
    session = get_session(session_id)
    lines: List[str] = []
    for m in session.messages:
        lines.append(f"{m.role.upper()}: {m.content}")
    return "\n".join(lines)


def _combine_pdf_texts(text_chunks: Iterable[str]) -> str:
    return "\n\n---\n\n".join(text_chunks)


def _upsert_graph(session: Session, session_id: str, nodes, edges) -> None:
    """
    Very simple graph upsert: delete previous subgraph for session_id, insert new.
    """
    # Remove previous graph for this session
    session.run(
        """
        MATCH (n {session_id: $sid})-[r]-()
        DETACH DELETE n
        """,
        sid=session_id,
    )

    # Create nodes
    for node in nodes:
        session.run(
            """
            MERGE (n:Entity {id: $id, session_id: $sid})
            SET n.label = $label
            """,
            id=node.get("id"),
            label=node.get("label"),
            sid=session_id,
        )

    # Create edges
    for edge in edges:
        session.run(
            """
            MATCH (s:Entity {id: $source, session_id: $sid})
            MATCH (t:Entity {id: $target, session_id: $sid})
            MERGE (s)-[r:RELATION {type: $type}]->(t)
            """,
            source=edge.get("source"),
            target=edge.get("target"),
            type=edge.get("type"),
            sid=session_id,
        )


def _extract_json_block(raw: str) -> str:
    """
    Try to robustly extract a JSON object from an LLM response.

    Удаляет возможные Markdown-кодовые блоки и берёт подстроку
    от первого '{' до последней '}'.
    """
    text = raw.strip()

    # Удаляем обёртку ```json ... ``` или ``` ... ``` если есть
    if text.startswith("```"):
        # Снимаем первые три бэктика и возможный язык (json)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # Обрезаем на завершающих ```
        end_fence = text.rfind("```")
        if end_fence != -1:
            text = text[:end_fence]
        text = text.strip()

    # Берём подстроку между первым '{' и последней '}'
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # Если не получилось, пусть парсер выше обработает как есть
        return raw

    return text[start : end + 1]


def _build_pyvis_html(nodes, edges) -> str:
    """
    Build an interactive HTML graph using pyvis from nodes and edges.
    """
    net = Network(height="600px", width="100%", directed=True)
    net.barnes_hut()

    for node in nodes:
        node_id = node.get("id")
        label = node.get("label", node_id)
        title = label
        net.add_node(node_id, label=label, title=title)

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        etype = edge.get("type", "")
        net.add_edge(source, target, label=etype, title=etype)

    # Возвращаем HTML как строку, без сохранения на диск.
    return net.generate_html()


def build_graph_for_session(
    session_id: str,
    pdf_chunks: Iterable[str],
) -> Tuple[str, Optional[str]]:
    """
    Build or update a simple knowledge graph in Neo4j for the given session.

    Returns a pair (summary_text, graph_html), где:
      - summary_text — короткий текст для показа в чате,
      - graph_html — HTML‑код интерактивного графа (для st.components.html)
    """
    history_text = _session_history_to_text(session_id)
    pdf_text = _combine_pdf_texts(pdf_chunks)

    prompt = _build_graph_prompt(history_text, pdf_text)

    # Ask LLM to extract a graph in JSON form.
    # The client of this function is responsible for handling JSON errors
    # gracefully in production code.
    extraction = llm_chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )

    # We expect extraction to contain JSON; пытаемся аккуратно достать его.
    import json

    try:
        json_block = _extract_json_block(extraction)
        data = json.loads(json_block)
    except json.JSONDecodeError:
        # Fallback: do not modify graph, just return text.
        return (
            "Не удалось корректно разобрать граф из ответа LLM. "
            "Вот ответ модели:\n\n"
            f"{extraction}",
            None,
        )

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    with _get_driver() as driver:
        with driver.session() as session:
            _upsert_graph(session, session_id, nodes, edges)

    # Пытаемся построить интерактивный граф для отображения в UI.
    graph_html: Optional[str] = None
    try:
        graph_html = _build_pyvis_html(nodes, edges)
    except Exception:
        # На случай ошибок визуализации не ломаем основной сценарий.
        graph_html = None

    summary = "Граф знаний для текущей сессии успешно обновлён в Neo4j."
    return summary, graph_html


__all__ = ["build_graph_for_session"]


