"""Deterministic companion stance assignment — ported from the AutoGen
version's BanterGenerator, the engine behind its party friction.

The idea: companions shouldn't react generically; the ENGINE decides their
conversational move (agree / disagree / none) and the model only performs it.
The player's input implies an approach (aggressive, stealthy, ...); each
companion carries a disposition toward each approach in characters.yaml
(selection.approaches, -2..+2). A strong conflict yields a disagreement with a
reason; strong alignment yields support. No LLM call, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Approach implied by the player's action. Same keyword style as
# speaker_policy._SITUATION_KEYWORDS; tune words freely.
_APPROACH_KEYWORDS: dict[str, tuple[str, ...]] = {
    "aggressive": ("attack", "fight", "force", "threaten", "draw", "strike", "shove", "demand"),
    "diplomatic": ("talk", "negotiate", "persuade", "reason", "offer", "bargain", "calm"),
    "stealthy": ("sneak", "quiet", "hide", "slip", "shadow", "pick", "steal", "creep"),
    "scholarly": ("study", "research", "examine", "read", "decipher", "analyze"),
    "compassionate": ("help", "save", "protect", "mercy", "comfort", "spare", "heal"),
}

# Disposition thresholds: at or below DISAGREE_AT -> pushback; at or
# above AGREE_AT -> support. Between them the companion takes no stance.
_DISAGREE_AT = -1
_AGREE_AT = 2


@dataclass
class Stance:
    move: str  # "agree" | "disagree"
    approach: str
    strength: int  # abs(disposition): 1 = soft, 2 = hard


def detect_approach(text: str) -> str | None:
    """The first approach the text implies, or None."""
    lower = text.lower()
    for approach, keywords in _APPROACH_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return approach
    return None


def assign_stance(profile: dict[str, Any], player_input: str) -> Stance | None:
    """This companion's stance toward the player's current action, if any."""
    approach = detect_approach(player_input)
    if not approach:
        return None
    dispositions = (profile.get("selection") or {}).get("approaches") or {}
    try:
        value = int(dispositions.get(approach, 0))
    except (TypeError, ValueError):
        return None
    if value <= _DISAGREE_AT:
        return Stance(move="disagree", approach=approach, strength=min(abs(value), 2))
    if value >= _AGREE_AT:
        return Stance(move="agree", approach=approach, strength=min(abs(value), 2))
    return None


def stance_directive(stance: Stance | None) -> str:
    """Render a stance as a THIS BEAT directive line.

    States only the MOVE (agree/disagree + strength). HOW it is expressed —
    sharp, warm, cold, mocking — comes from the character's persona and voice
    examples, never from here; tone direction would flatten every companion
    into the same register.
    """
    if stance is None:
        return ""
    if stance.move == "disagree":
        if stance.strength >= 2:
            return (
                f"- You DISAGREE with this {stance.approach} approach, firmly — "
                "expressed the way YOUR character would."
            )
        return (
            f"- You have doubts about this {stance.approach} approach — voice them "
            "the way YOUR character would."
        )
    if stance.strength >= 2:
        return (
            f"- You strongly AGREE with this {stance.approach} approach — back them "
            "up the way YOUR character would."
        )
    return (
        f"- You lean toward agreeing with this {stance.approach} approach — "
        "in your own manner."
    )
