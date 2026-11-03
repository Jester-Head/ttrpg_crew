"""Which prompt each companion and the GM uses this beat, picked in code from
intent, stance, scene and situation (no LLM call)."""

from __future__ import annotations

from typing import Any

from ttrpg_crew.prompts.prompt_loader import (
    list_task_keys,
    load_game_states_config,
    resolve_task_key,
)


def _if_defined(task_file: str, key: str, fallback: str) -> str:
    # A key missing from the YAML falls back instead of failing the beat, so a
    # situation prompt can be deleted (or not yet written) without code changes.
    return key if key in list_task_keys(task_file) else fallback


def companion_move(
    speaker_id: str,
    intent: Any | None,
    stance: Any | None,
    scene_type: str,
) -> str:
    """Task key for one companion: answer > agree/disagree > the scene's task."""
    scene_key = resolve_task_key("companion", scene_type)
    if intent is not None and intent.addressed_id == speaker_id:
        return _if_defined("companion_tasks.yaml", "answer", scene_key)
    if stance is not None:
        return _if_defined("companion_tasks.yaml", stance.move, scene_key)
    return scene_key


def gm_task_key(scene_type: str, situations: list[str] | tuple[str, ...]) -> str:
    """Task key for the GM — combat/opening from the scene win, otherwise the
    first situation listed under game_states.yaml `gm_situations`."""
    scene_key = resolve_task_key("gm", scene_type)
    if scene_key != "narrative":
        return scene_key
    mapping = load_game_states_config().get("gm_situations") or {}
    for situation, key in mapping.items():
        if situation in situations:
            return _if_defined("gm_tasks.yaml", str(key), scene_key)
    return scene_key
