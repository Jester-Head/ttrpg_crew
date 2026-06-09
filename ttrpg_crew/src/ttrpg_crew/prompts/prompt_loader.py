"""Load and render agent/task prompts from YAML with model-tier overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from jinja2 import Environment, StrictUndefined

AgentRole = Literal["gm", "companion", "manager"]

_PROMPT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

_TASK_FILES: dict[AgentRole, str] = {
    "gm": "gm_tasks.yaml",
    "companion": "companion_tasks.yaml",
    "manager": "manager_task.yaml",
}

_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)

_RESERVED_TASK_DOC_KEYS = frozenset(
    {"_defaults", "model_tiers", "base"}
)


def _read_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"Prompt file not found: {path}"
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), f"Expected mapping in {path}"
    return data


def _render_template(text: str, variables: dict[str, Any]) -> str:
    template = _JINJA_ENV.from_string(text)
    return template.render(**variables).strip()


def _concat_field(*parts: str | None) -> str | None:
    chunks = [str(part).strip() for part in parts if part and str(part).strip()]
    return "\n\n".join(chunks) if chunks else None


def _normalize_task_key(doc: dict[str, Any], task_key: str) -> str:
    if task_key in doc:
        return task_key
    if task_key == "narrative" and "base" in doc:
        return "base"
    return task_key


def _get_tier_field_override(
    doc: dict[str, Any],
    tier: str,
    task_key: str,
    field: str,
) -> str | None:
    tier_block = doc.get("model_tiers", {}).get(tier, {})
    if not isinstance(tier_block, dict):
        return None

    parts: list[str | None] = []

    # Tier-level _defaults come FIRST so per-task text builds on the shared
    # rules (e.g. boundary constraints) rather than replacing them. This mirrors
    # how the top-level _defaults concatenate into base tasks.
    tier_defaults = tier_block.get("_defaults", {})
    if isinstance(tier_defaults, dict) and field in tier_defaults:
        parts.append(tier_defaults[field])

    task_specific = tier_block.get(task_key, {})
    if isinstance(task_specific, dict) and field in task_specific:
        parts.append(task_specific[field])

    # Legacy: a bare field set directly on the tier block (non-dict).
    if not parts and field in tier_block and not isinstance(tier_block[field], dict):
        parts.append(tier_block[field])

    return _concat_field(*parts)


def _build_merged_task_fields(
    doc: dict[str, Any],
    task_key: str,
    tier: str,
) -> dict[str, str | None]:
    resolved_key = _normalize_task_key(doc, task_key)

    assert resolved_key in doc, (
        f"Task key '{task_key}' resolved to '{resolved_key}', "
        f"but it was not found in task YAML "
        f"(available: {list_task_keys_from_doc(doc)})"
    )

    task_block = doc[resolved_key]
    assert isinstance(task_block, dict), (
        f"Task key '{resolved_key}' must be a mapping "
        f"(available: {list_task_keys_from_doc(doc)})"
    )

    defaults = doc.get("_defaults", {})
    assert isinstance(defaults, dict), "_defaults must be a mapping when present"

    merged: dict[str, str | None] = {}
    for field in ("description", "expected_output"):
        base_text = _concat_field(defaults.get(field), task_block.get(field))
        tier_override = _get_tier_field_override(doc, tier, resolved_key, field)

        if tier_override is not None:
            merged[field] = tier_override
        else:
            tier_override_alias = _get_tier_field_override(
                doc, tier, task_key, field
            )
            merged[field] = (
                tier_override_alias if tier_override_alias is not None else base_text
            )

    return merged
    
def list_task_keys(task_file: str) -> list[str]:
    """Return task keys defined in a task YAML (excluding _defaults, model_tiers)."""
    doc = _read_yaml(_PROMPT_CONFIG_DIR / task_file)
    return list_task_keys_from_doc(doc)


def list_task_keys_from_doc(doc: dict[str, Any]) -> list[str]:
    return [
        key
        for key in doc
        if key not in _RESERVED_TASK_DOC_KEYS
        and isinstance(doc[key], dict)
    ]


def load_models_config() -> dict[str, Any]:
    return _read_yaml(_PROMPT_CONFIG_DIR / "models.yaml")


def load_game_states_config() -> dict[str, Any]:
    return _read_yaml(_PROMPT_CONFIG_DIR / "game_states.yaml")


def _parse_param_billions(model_id: str) -> float | None:
    """Best-effort parse of a parameter count (in billions) from a model id.

    Matches things like "7b", "70B", "3.8b", and the "8x7b" MoE form (where it
    takes the per-expert size). Returns the largest plausible match, or None.
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", model_id.lower())
    sizes = [float(m) for m in matches]
    return max(sizes) if sizes else None


def resolve_tier_for_model(model_id: str | None) -> str:
    """Pick a prompt tier for the given model id (see models.yaml tier_selection).

    Substring override wins; else bucket by parsed parameter count; else the
    configured default_tier.
    """
    models = load_models_config()
    default_tier = models.get("default_tier", "large")
    assert isinstance(default_tier, str), "default_tier must be a string"

    if not model_id:
        return default_tier

    selection = models.get("tier_selection", {})
    if not isinstance(selection, dict):
        return default_tier

    lowered = model_id.lower()

    overrides = selection.get("overrides") or {}
    if isinstance(overrides, dict):
        for substring, tier in overrides.items():
            if substring and str(substring).lower() in lowered and isinstance(tier, str):
                return tier

    params = _parse_param_billions(model_id)
    size_tiers = selection.get("size_tiers") or {}
    if params is not None and isinstance(size_tiers, dict):
        for threshold in sorted(size_tiers, key=lambda k: float(k)):
            if params <= float(threshold) and isinstance(size_tiers[threshold], str):
                return size_tiers[threshold]

    return default_tier


def resolve_model_tier(
    *,
    tier: str | None = None,
    llm: str | None = None,
    llm_tier_map: dict[str, str] | None = None,
) -> str:
    """Resolve tier from explicit arg, env, or optional llm→tier map."""
    if tier:
        return tier

    env_tier = os.getenv("TTRPG_CREW_MODEL_TIER")
    if env_tier:
        return env_tier

    if llm and llm_tier_map and llm in llm_tier_map:
        return llm_tier_map[llm]

    models = load_models_config()
    default_tier = models.get("default_tier", "large")
    assert isinstance(default_tier, str), "default_tier must be a string"
    return default_tier


def resolve_task_key(agent_role: AgentRole, game_state: str) -> str:
    """
    Map a runtime game_state string to a task key in the agent's task YAML.

    Examples: game_state \"combat\" -> \"combat\"; unknown -> default_task.
    """
    config = load_game_states_config()
    role_block = config.get(agent_role, {})
    assert isinstance(role_block, dict), f"game_states.yaml missing '{agent_role}'"

    mapped = role_block.get(game_state)
    if isinstance(mapped, str):
        return mapped

    default_task = config.get("default_task", "narrative")
    assert isinstance(default_task, str), "default_task must be a string"
    return default_task


def load_agent_config(
    agent_file: str,
    variables: dict[str, Any],
    *,
    tier: str | None = None,
    llm: str | None = None,
    llm_tier_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build a CrewAI Agent config dict (role, goal, backstory) from YAML.

    agent_file: filename under src/ttrpg_crew/config/, e.g. companion_agent.yaml
    """
    path = _PROMPT_CONFIG_DIR / agent_file
    doc = _read_yaml(path)
    base = doc.get("base", {})
    assert isinstance(base, dict), "agent yaml must define base:"

    resolved_tier = resolve_model_tier(
        tier=tier, llm=llm, llm_tier_map=llm_tier_map
    )

    merged = dict(base)
    tier_block = doc.get("model_tiers", {}).get(resolved_tier, {})
    if isinstance(tier_block, dict):
        for field in ("role", "goal", "backstory"):
            if field in tier_block and tier_block[field] is not None:
                merged[field] = tier_block[field]

    return {
        key: _render_template(str(merged[key]), variables)
        for key in ("role", "goal", "backstory")
        if key in merged and merged[key]
    }


def load_task_config(
    task_file: str,
    variables: dict[str, Any],
    *,
    tier: str | None = None,
    llm: str | None = None,
    llm_tier_map: dict[str, str] | None = None,
    task_key: str = "narrative",
) -> dict[str, str]:
    """
    Build a CrewAI Task config dict (description, expected_output) from YAML.

    task_file: e.g. gm_tasks.yaml
    task_key: e.g. narrative, combat, opening_scene, session_resume
    """
    path = _PROMPT_CONFIG_DIR / task_file
    doc = _read_yaml(path)
    resolved_tier = resolve_model_tier(
        tier=tier, llm=llm, llm_tier_map=llm_tier_map
    )
    merged_fields = _build_merged_task_fields(doc, task_key, resolved_tier)

    result: dict[str, str] = {}
    for key in ("description", "expected_output"):
        if merged_fields.get(key):
            result[key] = _render_template(str(merged_fields[key]), variables)
    assert result, f"No task fields rendered for {task_file} key={task_key}"
    return result


def load_task_for_game_state(
    agent_role: AgentRole,
    variables: dict[str, Any],
    game_state: str,
    *,
    tier: str | None = None,
    llm: str | None = None,
    llm_tier_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Resolve game_state -> task_key, then load that task from the role's YAML file.

    Example:
        load_task_for_game_state("gm", vars, "combat")
    """
    task_key = resolve_task_key(agent_role, game_state)
    task_file = _TASK_FILES[agent_role]
    return load_task_config(
        task_file,
        variables,
        tier=tier,
        llm=llm,
        llm_tier_map=llm_tier_map,
        task_key=task_key,
    )
