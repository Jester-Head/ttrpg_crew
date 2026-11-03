"""Plain chat completions against the local OpenAI-compatible server."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ttrpg_crew import session_log

_SAMPLING_ATTRS = ("temperature", "top_p", "frequency_penalty", "presence_penalty", "seed")


@dataclass
class ChatResult:
    text: str
    seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _model_id(llm: Any) -> str:
    # CrewAI's LLM carries the litellm routing prefix; the server wants the bare id.
    model = str(getattr(llm, "model", "") or "")
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def _extra_body(llm: Any) -> dict[str, Any] | None:
    extra = (getattr(llm, "additional_params", None) or {}).get("extra_body")
    return extra if isinstance(extra, dict) else None


def chat(
    llm: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    label: str = "chat",
) -> ChatResult:
    """One chat completion using the connection and sampling settings of `llm`.

    `max_tokens` is safe here; it's only the CrewAI agent path that returns
    empty replies when it's set (see models.yaml).
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=getattr(llm, "base_url", None) or "http://localhost:1234/v1",
        api_key=getattr(llm, "api_key", None) or "lm-studio",
        timeout=getattr(llm, "timeout", None) or 300,
    )
    params: dict[str, Any] = {
        attr: value
        for attr in _SAMPLING_ATTRS
        if (value := getattr(llm, attr, None)) is not None
    }
    extra = _extra_body(llm)
    if extra:
        params["extra_body"] = extra
    if max_tokens:
        params["max_tokens"] = max_tokens

    session_log.prompt(label, messages, source="direct", model=_model_id(llm))
    start = time.perf_counter()
    resp = client.chat.completions.create(model=_model_id(llm), messages=messages, **params)
    usage = getattr(resp, "usage", None)
    return ChatResult(
        text=(resp.choices[0].message.content or "").strip(),
        seconds=time.perf_counter() - start,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )
