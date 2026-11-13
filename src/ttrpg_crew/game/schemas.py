from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator


def first_json_object(raw: str) -> dict | None:
    """The first JSON object embedded in `raw`; fences and prose around it are skipped."""
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        start = raw.find("{", start + 1)
    return None


def _item_text(item: object) -> str:
    """Flatten one list entry to text ({"desc": .., "name": ..} -> "name: desc")."""
    if isinstance(item, dict):
        name_first = sorted(item.items(), key=lambda pair: pair[0] != "name")
        return ": ".join(str(value) for _, value in name_first if value)
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
    # How many checkpoints ahead this beat got to: 0 none, 1 the current one,
    # 2 the one after (the player went straight there), and so on.
    checkpoint_reached: int = 0

    # Small models return loose shapes (null, a bare string, objects). The
    # validators below keep whatever is usable, so one odd field doesn't cost
    # the whole update.

    @field_validator("suggested_actions", "new_facts", "resolved_facts", "npcs", mode="before")
    @classmethod
    def _as_string_list(cls, value, info):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            if "name" in value:
                return [_item_text(value)]
            if info.field_name == "npcs":
                return [f"{key}: {text}" for key, text in value.items() if text]
            return [str(text) for text in value.values() if text]
        if isinstance(value, list):
            return [_item_text(item) for item in value if item is not None]
        return value

    @field_validator("checkpoint_reached", mode="before")
    @classmethod
    def _as_checkpoint_number(cls, value):
        if isinstance(value, bool):
            return 1 if value else 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @field_validator("location", "objective", "summary", mode="before")
    @classmethod
    def _as_string(cls, value):
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(_item_text(item) for item in value if item is not None)
        if isinstance(value, dict):
            return _item_text(value)
        return value if isinstance(value, str) else str(value)

    @field_validator("location_changed", "objective_changed", mode="before")
    @classmethod
    def _as_flag(cls, value):
        return False if value is None else value


class ModeratorSelection(BaseModel):
    selected_speakers: list[str] = Field(default_factory=list)
    gm_should_frame_first: bool = True
    gm_intent: str = "Frame the next beat naturally."
    reasoning: str = ""
    spotlight_notes: dict = Field(default_factory=dict)
