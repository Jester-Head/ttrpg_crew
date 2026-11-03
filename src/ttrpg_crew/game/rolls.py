"""Roll requests from the GM's [ROLL: Skill DC N] tag, resolved by the engine.

The GM only calls for the roll; the engine rolls (docs/roll_resolution_design.md).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_ROLL_TAG_RE = re.compile(
    r"\[ROLL:\s*(?P<skill>[^\]]*?)\s*(?:,\s*)?(?:vs\.?\s*)?DC\s*(?P<dc>\d+)\s*\]",
    re.I,
)


@dataclass
class RollRequest:
    skill: str
    dc: int


@dataclass
class RollResult:
    skill: str
    dc: int
    d20: int
    modifier: int
    total: int
    success: bool

    def describe(self) -> str:
        outcome = "success" if self.success else "failure"
        return f"{self.skill}: rolled {self.total} vs DC {self.dc} — {outcome}"


def parse_roll_request(text: str) -> RollRequest | None:
    match = _ROLL_TAG_RE.search(text)
    if not match:
        return None
    skill = match.group("skill").strip(" ,") or "Check"
    return RollRequest(skill=skill, dc=int(match.group("dc")))


def strip_roll_tag(text: str) -> str:
    return _ROLL_TAG_RE.sub("", text).rstrip()


def resolve_roll(request: RollRequest, rng: random.Random | None = None) -> RollResult:
    """Always the engine's d20, never a number from the player. The modifier is
    0 until characters have D&D sheets."""
    d20 = (rng or random).randint(1, 20)
    modifier = 0
    total = d20 + modifier
    return RollResult(
        skill=request.skill,
        dc=request.dc,
        d20=d20,
        modifier=modifier,
        total=total,
        success=total >= request.dc,
    )
