from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


def _item_text(item: object) -> str:
    """Flatten one list entry to text ({"name": .., "desc": ..} -> "name: desc")."""
    if isinstance(item, dict):
        return ": ".join(str(v) for v in item.values() if v)
    return str(item)


class SceneUpdate(BaseModel):
    """What the scene tracker reads out of one finished beat (game/scene_tracker)."""

    location: str = ""
    location_changed: bool = False
    objective: str = ""
    objective_changed: bool = False
    summary: str = ""
    suggested_actions: list[str] = Field(default_factory=list)
    new_facts: list[str] = Field(default_factory=list)
    resolved_facts: list[str] = Field(default_factory=list)
    npcs: list[str] = Field(default_factory=list)

    @field_validator("suggested_actions", "new_facts", "resolved_facts", "npcs", mode="before")
    @classmethod
    def _as_string_list(cls, value):
        # Small models sometimes return one string, null, or objects; keep
        # whatever text is usable rather than rejecting the whole update.
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [_item_text(item) for item in value if item is not None]
        return value

    @field_validator("location", "objective", "summary", mode="before")
    @classmethod
    def _as_string(cls, value):
        return "" if value is None else value


class ModeratorSelection(BaseModel):
    selected_speakers: list[str] = Field(default_factory=list)
    gm_should_frame_first: bool = True
    gm_intent: str = "Frame the next beat naturally."
    reasoning: str = ""
    spotlight_notes: dict = Field(default_factory=dict)
