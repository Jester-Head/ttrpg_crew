"""Tests for the deterministic Daggerheart action-roll resolver."""

from ttrpg_crew.game.resolver import Outcome, classify, resolve_action


# --- classify: the five tiers --------------------------------------------- #

def test_classify_critical_on_doubles_even_below_difficulty():
    # Matching dice are a critical success regardless of the total.
    assert classify(3, 3, total=6, difficulty=20) is Outcome.CRITICAL


def test_classify_success_tiers():
    assert classify(10, 4, total=14, difficulty=12) is Outcome.SUCCESS_WITH_HOPE
    assert classify(4, 10, total=14, difficulty=12) is Outcome.SUCCESS_WITH_FEAR


def test_classify_failure_tiers():
    assert classify(8, 2, total=10, difficulty=12) is Outcome.FAILURE_WITH_HOPE
    assert classify(2, 8, total=10, difficulty=12) is Outcome.FAILURE_WITH_FEAR


# --- resolve_action ------------------------------------------------------- #

def test_total_includes_trait_modifier():
    r = resolve_action(trait_modifier=3, difficulty=10, seed=1)
    assert r.total == r.hope_die + r.fear_die + 3


def test_determinism_same_seed():
    assert resolve_action(2, 12, seed=42) == resolve_action(2, 12, seed=42)


def test_dice_in_range():
    for s in range(200):
        r = resolve_action(0, 12, seed=s)
        assert 1 <= r.hope_die <= 12
        assert 1 <= r.fear_die <= 12


def test_advantage_swing_range():
    # Same seed -> same duality dice; only the d6 swing differs.
    for s in range(100):
        base = resolve_action(0, 12, seed=s)
        adv = resolve_action(0, 12, advantage=True, seed=s)
        dis = resolve_action(0, 12, disadvantage=True, seed=s)
        assert adv.total - base.total in range(1, 7)
        assert base.total - dis.total in range(1, 7)


def test_advantage_and_disadvantage_cancel():
    base = resolve_action(0, 12, seed=7)
    both = resolve_action(0, 12, advantage=True, disadvantage=True, seed=7)
    assert both.total == base.total


# --- the Hope/Fear economy ------------------------------------------------ #

def test_every_outcome_moves_exactly_one_of_hope_or_fear():
    for s in range(300):
        r = resolve_action(0, 12, seed=s)
        assert not (r.outcome.with_hope and r.outcome.with_fear)
        assert r.player_hope_delta + r.gm_fear_delta == 1
        if r.outcome.with_hope:
            assert (r.player_hope_delta, r.gm_fear_delta) == (1, 0)
        else:
            assert (r.player_hope_delta, r.gm_fear_delta) == (0, 1)


def test_critical_is_a_hope_side_success():
    for s in range(1000):
        r = resolve_action(0, 12, seed=s)
        if r.hope_die == r.fear_die:
            assert r.outcome is Outcome.CRITICAL
            assert r.outcome.is_success
            assert r.outcome.with_hope
            assert r.player_hope_delta == 1
            return
    raise AssertionError("expected at least one doubles roll within 1000 seeds")
