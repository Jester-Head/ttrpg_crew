"""Deterministic Daggerheart action-roll resolver — ported from the AutoGen
version's combat.DiceRoller.

The engine owns the dice; the GM only narrates the result it is handed (see
docs/roll_resolution_design.md). An action roll is 2d12 "duality dice" — a Hope
die and a Fear die — plus the actor's trait modifier, against a Difficulty. The
pair of dice yields one of five outcome tiers, and each tier moves the Hope/Fear
economy deterministically. No LLM, and seedable so eval runs and bug repros are
reproducible.

Pure on purpose: `resolve_action` returns a `RollResult` carrying the intended
Hope/Fear deltas as data; applying them to game state is the caller's job, so
this module never touches GameState and stays trivially testable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

_DIE_SIDES = 12  # duality dice are d12s
_ADV_DIE_SIDES = 6  # advantage / disadvantage is a d6 swing on the total


class Outcome(Enum):
    """The five Daggerheart action-roll tiers."""

    CRITICAL = "critical"
    SUCCESS_WITH_HOPE = "success_with_hope"
    SUCCESS_WITH_FEAR = "success_with_fear"
    FAILURE_WITH_HOPE = "failure_with_hope"
    FAILURE_WITH_FEAR = "failure_with_fear"

    @property
    def is_success(self) -> bool:
        return self in (
            Outcome.CRITICAL,
            Outcome.SUCCESS_WITH_HOPE,
            Outcome.SUCCESS_WITH_FEAR,
        )

    @property
    def with_hope(self) -> bool:
        # A critical success counts as a roll "with Hope" per the SRD.
        return self in (
            Outcome.CRITICAL,
            Outcome.SUCCESS_WITH_HOPE,
            Outcome.FAILURE_WITH_HOPE,
        )

    @property
    def with_fear(self) -> bool:
        return self in (Outcome.SUCCESS_WITH_FEAR, Outcome.FAILURE_WITH_FEAR)


@dataclass(frozen=True)
class RollResult:
    """Everything the engine and the GM nudge need from one action roll.

    The Hope/Fear deltas are exposed as data (not applied here) so the resolver
    stays pure. `seed` is echoed back for the transparency log.
    """

    hope_die: int
    fear_die: int
    trait_modifier: int
    difficulty: int
    total: int  # hope_die + fear_die + trait_modifier + advantage swing
    outcome: Outcome
    seed: int | None = None

    @property
    def player_hope_delta(self) -> int:
        """Hope the player gains from this roll (SRD: Hope/critical rolls)."""
        return 1 if self.outcome.with_hope else 0

    @property
    def gm_fear_delta(self) -> int:
        """Fear the GM gains from this roll (SRD: Fear rolls)."""
        return 1 if self.outcome.with_fear else 0


def classify(hope_die: int, fear_die: int, total: int, difficulty: int) -> Outcome:
    """The outcome tier for an already-rolled pair and final total. Pure.

    Matching dice are always a critical success regardless of the total; the
    duality comparison (which die is higher) decides Hope vs Fear, and the total
    vs the Difficulty decides success vs failure.
    """
    if hope_die == fear_die:
        return Outcome.CRITICAL
    success = total >= difficulty
    if success:
        return Outcome.SUCCESS_WITH_HOPE if hope_die > fear_die else Outcome.SUCCESS_WITH_FEAR
    return Outcome.FAILURE_WITH_HOPE if hope_die > fear_die else Outcome.FAILURE_WITH_FEAR


def resolve_action(
    trait_modifier: int,
    difficulty: int,
    *,
    advantage: bool = False,
    disadvantage: bool = False,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> RollResult:
    """Roll one Daggerheart action roll and return the full result.

    Pass `seed` (or your own `rng`) for reproducible rolls; the seed is echoed
    back on the result for the log.

    Advantage/disadvantage apply a d6 swing to the total — advantage adds a d6,
    disadvantage subtracts one — and cancel out if both are set, per Daggerheart.
    (The old version used a D&D-style "roll twice, keep the better total" from
    when this ran on 5e rules; that has been dropped.)
    """
    if rng is None:
        rng = random.Random(seed)

    hope_die = rng.randint(1, _DIE_SIDES)
    fear_die = rng.randint(1, _DIE_SIDES)

    swing = 0
    if advantage and not disadvantage:
        swing = rng.randint(1, _ADV_DIE_SIDES)
    elif disadvantage and not advantage:
        swing = -rng.randint(1, _ADV_DIE_SIDES)

    total = hope_die + fear_die + trait_modifier + swing
    outcome = classify(hope_die, fear_die, total, difficulty)
    return RollResult(
        hope_die=hope_die,
        fear_die=fear_die,
        trait_modifier=trait_modifier,
        difficulty=difficulty,
        total=total,
        outcome=outcome,
        seed=seed,
    )
