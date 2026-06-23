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
    # Daggerheart Hope: a per-character resource the player spends; capped at 6.
    hope: int = 0
    hope_max: int = 6

    def gain_hope(self, amount: int = 1) -> None:
        """Gain Hope, clamped to the cap. A negative amount is ignored."""
        amount = max(0, amount)
        self.hope = min(self.hope + amount, self.hope_max)

    def spend_hope(self, amount: int = 1) -> bool:
        """Spend Hope if affordable; return whether it went through.

        A negative amount is treated as zero, never a backdoor to gain Hope.
        """
        amount = max(0, amount)
        if self.hope < amount:
            return False
        self.hope -= amount
        return True


@dataclass
class GameState:
    campaign_premise: str
    location: str
    scene_type: str
    danger: str | None = None
    party_goal: str | None = None
    # Daggerheart Fear: the GM's single shared pool; capped at 12.
    fear: int = 0
    fear_max: int = 12
    active_npcs: list[str] = field(default_factory=list)
    characters: dict[str, CharacterState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def gain_fear(self, amount: int = 1) -> None:
        """Add Fear to the GM pool, clamped to the cap. A negative amount is ignored."""
        amount = max(0, amount)
        self.fear = min(self.fear + amount, self.fear_max)

    def spend_fear(self, amount: int = 1) -> bool:
        """Spend Fear if affordable; return whether it went through.

        A negative amount is treated as zero, never a backdoor to gain Fear.
        """
        amount = max(0, amount)
        if self.fear < amount:
            return False
        self.fear -= amount
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_premise": self.campaign_premise,
            "location": self.location,
            "scene_type": self.scene_type,
            "danger": self.danger,
            "party_goal": self.party_goal,
            "fear": self.fear,
            "fear_max": self.fear_max,
            "active_npcs": self.active_npcs,
            "characters": {
                key: value.__dict__
                for key, value in self.characters.items()
            },
            "metadata": self.metadata,
        }
