"""Plain chat completions against the local server, outside the CrewAI loop.

Shared by the direct companion path and the scene tracker. Sampling knobs are
read off the CrewAI LLM object the caller already built (models.yaml `llm:`),
so a role keeps its temperature etc. without a second config path.
"""

from __future__ import annotations

from typing import Any

_SAMPLING_ATTRS = ("temperature", "top_p", "frequency_penalty", "presence_penalty", "timeout")


def chat_completion(llm: Any, messages: list[dict[str, str]], **extra: Any) -> Any:
    """One litellm completion using `llm`'s model, endpoint and sampling."""
    import litellm

    params: dict[str, Any] = {}
    for attr in _SAMPLING_ATTRS:
        value = getattr(llm, attr, None)
        if value is not None:
            params[attr] = value
    params.update(extra)
    return litellm.completion(
        model=getattr(llm, "model", None),
        api_base=getattr(llm, "base_url", None),
        api_key=getattr(llm, "api_key", None),
        messages=messages,
        **params,
    )


def json_completion(
    llm: Any, system: str, user: str, *, schema: dict[str, Any], name: str
) -> str:
    """Ask for JSON matching `schema`; return the raw reply text.

    Tries the server's structured-output mode first (LM Studio and other
    OpenAI-compatible servers constrain generation to the schema, which small
    models need). A server that rejects `response_format`, or replies empty,
    gets one plain retry; the caller parses either way. Returns "" when both
    attempts fail.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }
    for extra in ({"response_format": response_format}, {}):
        try:
            resp = chat_completion(llm, messages, **extra)
            text = (resp.choices[0].message.content or "").strip()
        except Exception:
            continue
        if text:
            return text
    return ""
