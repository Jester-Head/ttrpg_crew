"""The direct companion path: one plain chat completion per companion turn, built
from the same persona + task YAML as the CrewAI path so an A/B isolates the wrapper.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ttrpg_crew.prompts.directives import directive
from ttrpg_crew.prompts.prompt_loader import load_task_config
from ttrpg_crew.services import llm_client

# Models bold or indent the markers, and sometimes drop the colon (a bare
# "SAY" line), so match them loosely.
_SAY_RE = re.compile(r"^[\s*_]*SAY[\s*_]*(?::[\s*_]*|$\n?)", re.I | re.M)
_NOTE_RE = re.compile(r"^[\s*_]*NOTE[\s*_]*:[\s*_]*", re.I)


def companion_messages(
    *,
    agent: Any,
    merged_vars: dict[str, Any],
    task_key: str,
    tier: str | None,
    said_so_far: list[str],
    think_first: bool = False,
    closing: str = "",
) -> tuple[str, str, str]:
    """Assemble (system, user, display_name) for one direct companion call.

    `merged_vars` is the speaker's full prompt variables (common vars +
    character vars + beat directives), exactly what the CrewAI path renders
    its task from. The agent's role/goal/backstory become the system message;
    the rendered task plus what's been said this beat become the user message.
    """
    name = str(merged_vars.get("character_name", "the companion"))

    cfg = load_task_config("companion_tasks.yaml", merged_vars, tier=tier, task_key=task_key)

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
    if closing:
        user_parts.append(closing)
    user_parts.append(directive("think_first" if think_first else "respond_now", name=name))
    return system, "\n\n".join(user_parts), name


def split_thought(text: str) -> tuple[str, str, bool]:
    """Split a think-first reply into (note, spoken_line, marker_found).

    Without a SAY: marker, a leading NOTE: paragraph is still dropped so the
    private note never reaches the story.
    """
    matches = list(_SAY_RE.finditer(text))
    if matches:
        last = matches[-1]
        note = _NOTE_RE.sub("", text[: last.start()], count=1).strip()
        return note, text[last.end():].strip(), True
    if _NOTE_RE.match(text):
        note, _, rest = text.partition("\n\n")
        if not rest.strip():
            note, _, rest = text.partition("\n")
        return _NOTE_RE.sub("", note, count=1).strip(), rest.strip(), False
    return "", text.strip(), False


def companion_call(
    llm: Any, system: str, user: str, *, label: str = "companion", debug: bool = False
) -> llm_client.ChatResult:
    """One companion turn as a plain chat completion, no CrewAI/ReAct wrapper.

    Falls back to the CrewAI LLM wrapper (text only, no token counts) if the
    direct request fails, so a server quirk can't kill the beat.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        # Stops a runaway early; a 2-5 sentence line plus a think-first note
        # fits well inside this.
        return llm_client.chat(llm, messages, max_tokens=600, label=label)
    except Exception as exc:
        if debug:
            print(f"[direct] chat completion failed ({exc}); using llm.call")
        start = time.perf_counter()
        text = str(llm.call(messages)).strip()
        return llm_client.ChatResult(text=text, seconds=time.perf_counter() - start)
