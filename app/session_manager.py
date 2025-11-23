import uuid
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class SessionState:
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    attached_pdfs: List[str] = field(default_factory=list)  # file paths


_sessions: Dict[str, SessionState] = {}


def create_session() -> SessionState:
    session_id = str(uuid.uuid4())
    state = SessionState(session_id=session_id)
    _sessions[session_id] = state
    return state


def get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState(session_id=session_id)
    return _sessions[session_id]


def append_message(session_id: str, role: str, content: str) -> None:
    state = get_session(session_id)
    state.messages.append(ChatMessage(role=role, content=content))


def attach_pdf(session_id: str, file_path: str) -> None:
    state = get_session(session_id)
    state.attached_pdfs.append(file_path)


__all__ = [
    "ChatMessage",
    "SessionState",
    "create_session",
    "get_session",
    "append_message",
    "attach_pdf",
]


