"""The adventure frame: a hook, checkpoints the GM steers toward, and which
cast members the GM may use so far."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Checkpoint:
    goal: str
    reached_when: str
    pressure: str = ""
    npcs: list[str] = field(default_factory=list)


@dataclass
class Adventure:
    title: str = ""
    pitch: str = ""
    hook: str = ""
    starting_location: str = ""
    checkpoints: list[Checkpoint] = field(default_factory=list)
    ending: str = ""


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_adventure(frame: dict[str, Any]) -> Adventure:
    """Build the adventure from frame.yaml. Old placeholder frames without
    checkpoints load as an empty adventure (no guidance, full cast)."""
    checkpoints = [
        Checkpoint(
            goal=_text(item.get("goal")),
            reached_when=_text(item.get("reached_when")),
            pressure=_text(item.get("pressure")),
            npcs=[_text(name) for name in item.get("npcs") or [] if _text(name)],
        )
        for item in frame.get("checkpoints") or []
        if isinstance(item, dict) and item.get("goal")
    ]
    return Adventure(
        title=_text(frame.get("title")),
        pitch=_text(frame.get("pitch")),
        hook=_text(frame.get("hook")),
        starting_location=_text(frame.get("starting_location")),
        checkpoints=checkpoints,
        ending=_text(frame.get("ending")),
    )


def guidance(
    adventure: Adventure, index: int, beats_since_progress: int, press_after: int
) -> dict[str, str]:
    """The GM's checkpoint, pressure and ending lines for this beat ("" when not shown)."""
    lines = {"checkpoint": "", "pressure": "", "adventure_ending": ""}
    if not adventure.checkpoints:
        return lines
    if index >= len(adventure.checkpoints):
        lines["adventure_ending"] = adventure.ending
        return lines
    current = adventure.checkpoints[index]
    lines["checkpoint"] = current.goal
    if current.pressure and beats_since_progress >= press_after:
        lines["pressure"] = current.pressure
    return lines


def _named_in(name: str, text: str) -> bool:
    first = name.split()[0] if name.split() else name
    return any(
        part and re.search(rf"\b{re.escape(part)}\b", text, re.I) for part in (name, first)
    )


def visible_cast(
    cast: list[dict[str, Any]],
    adventure: Adventure,
    index: int,
    met: list[str],
) -> list[dict[str, Any]]:
    """Cast members the GM may use: named in the hook, introduced by a reached
    or current checkpoint, or already met in play. Everyone else waits."""
    if not adventure.checkpoints:
        return cast
    introduced = {
        name.casefold()
        for checkpoint in adventure.checkpoints[: index + 1]
        for name in checkpoint.npcs
    }
    met_text = " ".join(met)
    visible = []
    for member in cast:
        name = _text(member.get("name"))
        if not name:
            continue
        if (
            name.casefold() in introduced
            or _named_in(name, adventure.hook)
            or _named_in(name, met_text)
        ):
            visible.append(member)
    return visible
