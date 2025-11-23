from typing import List
import logging

from .config import get_settings
from .llm_client import chat as llm_chat
from .session_manager import ChatMessage, get_session, append_message
from .vector_store import search as vector_search


logger = logging.getLogger(__name__)
_settings = get_settings()


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers user questions based on the "
    "conversation history and optional retrieved context from user-provided PDFs. "
    "If the retrieved context is relevant, ground your answer in it; "
    "if not, answer to the best of your knowledge and say when information "
    "is not available in the documents."
)


def _build_messages(
    session_messages: List[ChatMessage],
    user_question: str,
    retrieved_context: str | None,
) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include prior conversation as chat history (light history-aware RAG)
    for msg in session_messages:
        if msg.role not in ("user", "assistant"):
            continue
        messages.append({"role": msg.role, "content": msg.content})

    if retrieved_context:
        context_block = (
            "Here is additional context retrieved from the user's documents:\n\n"
            f"{retrieved_context}\n\n"
            "Use this context when answering if it is relevant."
        )
        messages.append({"role": "system", "content": context_block})

    messages.append({"role": "user", "content": user_question})
    return messages


def answer_question(session_id: str, question: str, k: int = 5) -> str:
    """
    Run a history-aware RAG pipeline for the given session and user question.
    """
    logger.info("answer_question called for session %s with question: %s", session_id, question)
    session = get_session(session_id)

    # 1) RAG retrieval from appropriate Chroma collection
    docs, metas = vector_search(question, session_id=session_id, k=k)
    logger.info(
        "Retrieved %d document(s) from vector store for session %s",
        len(docs),
        session_id,
    )
    retrieved_context = "\n\n---\n\n".join(docs) if docs else None

    # 2) Build messages with history + retrieved context
    messages = _build_messages(session.messages, question, retrieved_context)

    # 3) Call LLM
    answer = llm_chat(messages)
    logger.info("LLM answered for session %s (response length=%d chars)", session_id, len(answer))

    # 4) Update session history
    append_message(session_id, "user", question)
    append_message(session_id, "assistant", answer)

    return answer


__all__ = ["answer_question"]


