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
    party_goal: str | None = None
    active_npcs: list[str] = field(default_factory=list)
    characters: dict[str, CharacterState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_premise": self.campaign_premise,
            "location": self.location,
            "scene_type": self.scene_type,
            "danger": self.danger,
            "party_goal": self.party_goal,
            "active_npcs": self.active_npcs,
            "characters": {
                key: value.__dict__
                for key, value in self.characters.items()
            },
            "metadata": self.metadata,
        }
