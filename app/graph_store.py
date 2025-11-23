from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import logging

from neo4j import GraphDatabase, Session
from pyvis.network import Network

from .config import get_settings
from .llm_client import chat as llm_chat
from .session_manager import get_session


logger = logging.getLogger(__name__)
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
    "Return the result as a single valid JSON object with two arrays: "
    "'nodes' and 'edges'.\n"
    "Each node: {\"id\": string, \"label\": string}.\n"
    "Each edge: {\"source\": string, \"target\": string, \"type\": string}.\n"
    "Do not output any text before or after the JSON object. "
    "Use only double quotes for strings and do not use comments or trailing "
    "commas.\n"
    "Limit the size of the graph to at most 30 nodes and 60 edges, choosing "
    "the most important entities and relationships.\n"
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


def _try_parse_json_with_trimming(raw: str):
    """
    Попробовать распарсить JSON, по необходимости обрезая хвост.

    Это хак для случаев, когда LLM обрывает вывод внутри массива,
    оставляя "хвост" без закрывающих скобок/кавычек.
    """
    import json

    # Сначала пробуем как есть
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Initial JSON parse failed: %s", e)

    last_error: Optional[json.JSONDecodeError] = None

    # Пробуем обрезать хвост по нескольким шагам
    for cut in range(1, 200):
        trimmed = raw[:-cut].rstrip()
        # Обрезаем до последней закрывающей скобки или фигурной скобки
        last_brace = max(trimmed.rfind("}"), trimmed.rfind("]"))
        if last_brace == -1:
            break
        candidate = trimmed[: last_brace + 1]
        try:
            data = json.loads(candidate)
            logger.info("JSON successfully parsed after trimming %d chars", cut)
            return data
        except json.JSONDecodeError as e:
            last_error = e
            continue

    # Если ничего не вышло — пробрасываем последнее исключение дальше
    if last_error is not None:
        raise last_error
    # На всякий случай, если ошибок не было (что странно), пробрасываем общее
    raise json.JSONDecodeError("Unable to parse JSON after trimming", raw, 0)


def _iter_json_objects(text: str):
    """
    Iterate over top-level JSON object substrings in given text by tracking
    curly brace balance. Used as a fallback when the overall JSON response
    is damaged, but individual objects are still valid.
    """
    depth = 0
    start: Optional[int] = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]
                    start = None


def _salvage_nodes_edges_from_partial(raw: str):
    """
    Fallback‑разбор, если общий JSON сломан (например, обрезан по длине),
    но отдельные объекты всё ещё корректны.

    Стратегия:
      - находим все подстроки вида {...} с помощью балансировки скобок;
      - пытаемся распарсить каждую как отдельный JSON‑объект;
      - объекты с ключами ('id', 'label') считаем узлами;
      - объекты с ключами ('source', 'target') считаем рёбрами.
    """
    import json

    nodes = []
    edges = []
    seen_node_ids = set()

    for obj_str in _iter_json_objects(raw):
        try:
            obj = json.loads(obj_str)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        # Пропускаем корневой объект вида {"nodes": [...], "edges": [...]}
        if "nodes" in obj or "edges" in obj:
            continue

        if "source" in obj and "target" in obj:
            edges.append(obj)
        elif "id" in obj and "label" in obj:
            node_id = obj.get("id")
            if node_id not in seen_node_ids:
                nodes.append(obj)
                seen_node_ids.add(node_id)

    return nodes, edges


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

    json_block: Optional[str] = None
    try:
        json_block = _extract_json_block(extraction)
        data = _try_parse_json_with_trimming(json_block)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse graph JSON even after trimming: %s", e)
        # Дополнительный fallback: пытаемся вытащить отдельные узлы/рёбра
        # даже из частично повреждённого JSON.
        salvage_source = json_block if json_block is not None else extraction
        nodes, edges = _salvage_nodes_edges_from_partial(salvage_source)

        if not nodes and not edges:
            # Совсем ничего не удалось восстановить — сохраняем старое поведение.
            return (
                "Не удалось корректно разобрать граф из ответа LLM. "
                "Вот ответ модели:\n\n"
                f"{extraction}",
                None,
            )

        logger.warning(
            "Graph JSON is damaged; using salvaged data: %d node(s), %d edge(s)",
            len(nodes),
            len(edges),
        )

        # Строим граф по частично восстановленным данным.
        with _get_driver() as driver:
            with driver.session() as session:
                _upsert_graph(session, session_id, nodes, edges)

        graph_html: Optional[str] = None
        try:
            graph_html = _build_pyvis_html(nodes, edges)
        except Exception:
            graph_html = None

        summary = (
            "Ответ LLM содержал некорректный JSON, но удалось частично "
            "восстановить граф знаний в Neo4j "
            f"(узлов: {len(nodes)}, рёбер: {len(edges)})."
        )
        return summary, graph_html

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


