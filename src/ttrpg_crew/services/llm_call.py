"""Plain chat completions against the local server, outside the CrewAI loop."""

from __future__ import annotations

from typing import Any

from ttrpg_crew import session_log
from ttrpg_crew.services.llm_client import request_extra_body

_SAMPLING_ATTRS = (
    "temperature", "top_p", "frequency_penalty", "presence_penalty", "timeout", "max_tokens",
)


def chat_completion(llm: Any, messages: list[dict[str, str]], **extra: Any) -> Any:
    """One litellm completion using `llm`'s model, endpoint and sampling."""
    import litellm

    params: dict[str, Any] = {}
    for attr in _SAMPLING_ATTRS:
        value = getattr(llm, attr, None)
        if value is not None:
            params[attr] = value
    # Carries the per-model thinking switch (models.yaml `extra_body`).
    extra_body = request_extra_body(llm)
    if extra_body:
        params["extra_body"] = extra_body
    params.update(extra)
    model = getattr(llm, "model", None)
    if getattr(llm, "provider", None) in ("ollama", "ollama_chat"):
        # CrewAI's native Ollama client stores the bare model name, which
        # LiteLLM can't route — its base_url is Ollama's OpenAI-style /v1.
        model = f"openai/{model}"
    return litellm.completion(
        model=model,
        api_base=getattr(llm, "base_url", None),
        api_key=getattr(llm, "api_key", None),
        messages=messages,
        **params,
    )


def json_completion(
    llm: Any, system: str, user: str, *, schema: dict[str, Any], name: str
) -> str:
    """Ask for JSON matching `schema`; return the raw reply text ("" on failure).

    Tries the server's structured-output mode, then a plain call for servers
    that reject `response_format` or reply empty, then `llm.call` if the plain
    call raised. A timeout ends the attempts. Failures go to the debug log.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }
    raised = False
    for label, mode in (("structured", {"response_format": response_format}), ("plain", {})):
        try:
            resp = chat_completion(llm, messages, **mode)
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
        except Exception as exc:
            raised = True
            session_log.diag(f"[llm] {name}: {label} call failed ({type(exc).__name__}: {exc})")
            # Matched by name: the timeout types live in litellm/openai/httpx,
            # which load lazily.
            if "timeout" in type(exc).__name__.lower():
                return ""
            continue
        raised = False
        usage = getattr(resp, "usage", None)
        session_log.diag(
            f"[tokens] {name} ({label}): prompt={getattr(usage, 'prompt_tokens', None)}  "
            f"completion={getattr(usage, 'completion_tokens', None)}"
        )
        if getattr(choice, "finish_reason", None) == "length":
            session_log.diag(f"[llm] {name}: {label} reply cut off at the max_tokens cap")
        if text:
            return text
    if not raised:
        return ""
    try:
        return str(llm.call(messages)).strip()
    except Exception as exc:
        session_log.diag(f"[llm] {name}: llm.call failed ({type(exc).__name__}: {exc})")
        return ""
