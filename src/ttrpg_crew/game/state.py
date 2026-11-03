from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CharacterState:
    id: str
    name: str
    recent_turns: int = 0
    total_turns: int = 0
    conditions: list[str] = field(default_factory=list)
    unresolved_hooks: list[str] = field(default_factory=list)


@dataclass
class GameState:
    campaign_premise: str
    location: str
    scene_type: str
    danger: str | None = None
    # Scene memory, kept current by the scene tracker after each beat:
    # party_goal is the immediate objective, active_npcs the named NPCs met
    # ("Name: who they are"), known_facts the durable things established.
    party_goal: str | None = None
    scene_summary: str = ""
    known_facts: list[str] = field(default_factory=list)
    # Next moves offered to the player; shown outside the prose, never in it.
    suggested_actions: list[str] = field(default_factory=list)
    active_npcs: list[str] = field(default_factory=list)
    characters: dict[str, CharacterState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
