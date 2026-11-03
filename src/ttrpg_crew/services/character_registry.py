from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_character_profiles(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or CONFIG_DIR / "characters.yaml"

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")

    characters = data.get("characters", {})

    if not isinstance(characters, dict):
        raise ValueError(f"Expected 'characters' mapping in {path}")

    return {
        character_id: profile
        for character_id, profile in characters.items()
        if isinstance(profile, dict) and profile.get("enabled", True)
    }


def build_character_prompt_vars(
    *,
    character_id: str,
    character_profile: dict[str, Any],
    common_vars: dict[str, Any],
    character_state: Any | None = None,
) -> dict[str, Any]:
    """
    Merge static character profile, shared prompt variables, and mutable state.

    Static profile:
      name, species, class, persona, speech patterns

    Common vars:
      scene_context, recent_story, player_input, party_members

    Mutable state:
      conditions, recent turns, inventory, wounds, etc.
    """

    state_vars: dict[str, Any] = {}

    if character_state is not None:
        state_vars = {
            "recent_turns": getattr(character_state, "recent_turns", 0),
            "total_turns": getattr(character_state, "total_turns", 0),
            "conditions": getattr(character_state, "conditions", []),
            "unresolved_hooks": getattr(character_state, "unresolved_hooks", []),
        }

    # Prompts (companion_tasks.yaml) reference {{ character_name }}. Resolve it
    # from state, then the static profile, falling back to the id so the
    # StrictUndefined Jinja env never blows up on a missing name.
    character_name = (
        getattr(character_state, "name", None)
        or character_profile.get("name")
        or character_id
    )

    return {
        **common_vars,
        **character_profile,
        **state_vars,
        "character_id": character_id,
        "character_name": character_name,
        # Few-shot voice examples (companion_tasks.yaml injects these). Guaranteed
        # present so the StrictUndefined template never trips on a character that
        # has none. The beat runner re-picks them per beat from example_pool.
        "example_pool": character_profile.get("examples") or [],
        "examples": examples_for(character_profile.get("examples"), "", []),
    }


def examples_for(
    examples: Any,
    move: str,
    situations: list[str] | tuple[str, ...],
    k: int = 2,
) -> list[str]:
    """Up to k voice lines for this moment — the move's lines, else the first
    detected situation that has lines, else `default`.

    `examples` is a character's `examples:` value — a mapping keyed by move /
    situation / default, or a plain list (treated as default). Sampling keeps a
    companion from seeing, and copying, the same lines every beat.
    """
    pools = examples if isinstance(examples, dict) else {"default": examples or []}
    for key in (move, *situations, "default"):
        lines = [str(line) for line in pools.get(key) or []]
        if lines:
            return random.sample(lines, min(k, len(lines)))
    return []


def build_all_character_prompt_vars(
    *,
    character_profiles: dict[str, dict[str, Any]],
    common_vars: dict[str, Any],
    game_state: Any,
) -> dict[str, dict[str, Any]]:
    result = {}

    for character_id, profile in character_profiles.items():
        character_state = game_state.characters.get(character_id)

        result[character_id] = build_character_prompt_vars(
            character_id=character_id,
            character_profile=profile,
            common_vars=common_vars,
            character_state=character_state,
        )

    return result