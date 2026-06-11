"""Deterministic player-intent classification — no LLM call.

The speaker selector already computes who the player addressed and which
situations a beat implies, but only used them to pick speakers. This module
promotes the same signals into a BeatIntent the agents are TOLD about, so a
small model doesn't have to infer "the player asked ME a question" from raw
transcript. Code computes intent; prompts just receive it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ttrpg_crew.game.speaker_policy import detect_situations, find_direct_address

# "The player is asking something" — a question mark, or an asking/telling verb.
_QUESTION_RE = re.compile(r"\?|\b(ask|asks|asking|question|wonder)\b", re.I)


@dataclass
class BeatIntent:
    kind: str  # "open" | "action" | "question" | "direct_question" | "direct_address"
    addressed_id: str | None = None
    addressed_name: str | None = None
    situations: list[str] = field(default_factory=list)


def classify_intent(
    *,
    player_input: str,
    character_profiles: dict[str, dict[str, Any]],
    available_speakers: list[str],
) -> BeatIntent:
    """Classify the player's input for this beat."""
    text = player_input.strip()
    if not text:
        return BeatIntent(kind="open")

    lower = text.lower()
    situations = sorted(detect_situations(lower))
    is_question = bool(_QUESTION_RE.search(text))

    addressed = find_direct_address(character_profiles, available_speakers, lower)
    if addressed:
        cid, name = addressed
        return BeatIntent(
            kind="direct_question" if is_question else "direct_address",
            addressed_id=cid,
            addressed_name=name,
            situations=situations,
        )

    return BeatIntent(
        kind="question" if is_question else "action",
        situations=situations,
    )
