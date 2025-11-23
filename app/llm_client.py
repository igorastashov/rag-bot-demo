from typing import Iterable
import logging

from openai import OpenAI

from .config import get_settings


logger = logging.getLogger(__name__)
_settings = get_settings()
_client = OpenAI(
    base_url=_settings.llm_base_url,
    api_key=_settings.llm_api_key,
)


def chat(
    messages: Iterable[dict],
    max_tokens: int | None = None,
    temperature: float = 0.2,
) -> str:
    """
    Call the local Qwen (vLLM) endpoint using OpenAI-compatible API.

    Parameters
    ----------
    messages : iterable of dict
        List of messages in OpenAI format, e.g.
        [{"role": "user", "content": "Hello"}]
    max_tokens : int | None
        Maximum number of tokens to generate. If None, the value from
        settings.llm_max_output_tokens is used.
    temperature : float
        Sampling temperature.
    """
    messages_list = list(messages)
    # Логируем только роли и первые символы, чтобы не забивать лог
    preview = [
        {"role": m.get("role"), "content": str(m.get("content"))[:80]}
        for m in messages_list
    ]
    logger.info("Sending %d message(s) to LLM: %s", len(messages_list), preview)

    effective_max_tokens = max_tokens or _settings.llm_max_output_tokens
    logger.info(
        "Calling LLM with max_tokens=%d, temperature=%.2f",
        effective_max_tokens,
        temperature,
    )

    response = _client.chat.completions.create(
        model=_settings.llm_model_name,
        messages=messages_list,
        max_tokens=effective_max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    logger.info("Received LLM response (length=%d chars)", len(content))
    return content


__all__ = ["chat"]


