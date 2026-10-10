"""The direct companion path: one plain chat completion per companion turn.

The opt-in alternative (TTRPG_DIRECT_COMPANIONS=1) to the default CrewAI agent
loop. It skips the Agent/Task/ReAct wrapper entirely, so the model spends no
tokens on a discarded "Thought:" preamble — measured ~7x fewer completion
tokens — at some cost in the in-scene interplay the CrewAI path showed in
play-testing.

Everything specific to this path lives here: assembling the system/user
messages from the same persona + task YAML the CrewAI path uses (so an A/B
isolates the wrapper, not the prompt content), and the litellm call itself.
The beat orchestration (who speaks, directives, auditing) stays in
beat_runner, shared by both paths.
"""

from __future__ import annotations

import time
from typing import Any

from ttrpg_crew.prompts.directives import directive
from ttrpg_crew.prompts.prompt_loader import load_task_for_game_state
from ttrpg_crew.services.llm_call import chat_completion


def companion_messages(
    *,
    agent: Any,
    merged_vars: dict[str, Any],
    game_state_name: str,
    tier: str | None,
    said_so_far: list[str],
) -> tuple[str, str, str]:
    """Assemble (system, user, display_name) for one direct companion call.

    `merged_vars` is the speaker's full prompt variables (common vars +
    character vars + beat directives), exactly what the CrewAI path renders
    its task from. The agent's role/goal/backstory become the system message;
    the rendered task plus what's been said this beat become the user message.
    """
    name = str(merged_vars.get("character_name", "the companion"))

    cfg = load_task_for_game_state("companion", merged_vars, game_state_name, tier=tier)

    system = "\n\n".join(
        part
        for part in (
            (agent.role or "").strip(),
            ("Your goal: " + agent.goal.strip()) if (agent.goal or "").strip() else "",
            (agent.backstory or "").strip(),
        )
        if part
    )

    user_parts: list[str] = []
    if said_so_far:
        user_parts.append(directive("said_so_far") + "\n" + "\n".join(said_so_far))
    user_parts.append(cfg["description"].strip())
    if (cfg.get("expected_output") or "").strip():
        user_parts.append("Output: " + cfg["expected_output"].strip())
    user_parts.append(directive("respond_now", name=name))
    return system, "\n\n".join(user_parts), name


def companion_call(
    llm: Any, system: str, user: str, *, prefill: str = "", debug: bool = False
) -> tuple[str, float, Any, Any]:
    """One companion turn as a plain chat completion — no CrewAI/ReAct wrapper.

    `prefill` pre-shapes the reply by seeding the assistant turn (small models
    hold format much better mid-answer). Not every OpenAI-compatible server
    supports a trailing assistant message, so an empty or failed prefilled
    call quietly retries without it.

    Returns (text, seconds, prompt_tokens, completion_tokens). Token counts
    are None when the call path can't report usage.
    """
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    start = time.perf_counter()
    text, prompt_tokens, completion_tokens = "", None, None

    def _call(messages):
        return chat_completion(llm, messages)

    try:
        resp = None
        if prefill:
            try:
                resp = _call(base_messages + [{"role": "assistant", "content": prefill}])
                if not (resp.choices[0].message.content or "").strip():
                    resp = None  # server ignored the prefill; retry plain
            except Exception:
                resp = None
        if resp is None:
            prefill = ""
            resp = _call(base_messages)
        text = (prefill + (resp.choices[0].message.content or "")).strip()
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
    except Exception as exc:  # fall back to the crewai wrapper (text only)
        if debug:
            print(f"[direct] litellm.completion failed ({exc}); using llm.call")
        text = str(llm.call(base_messages)).strip()
    return text, time.perf_counter() - start, prompt_tokens, completion_tokens
