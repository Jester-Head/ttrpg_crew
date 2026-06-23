"""Tests for the Hope/Fear economy on the game state objects."""

from ttrpg_crew.game.state import CharacterState, GameState


def _state() -> GameState:
    return GameState(campaign_premise="", location="x", scene_type="narrative")


# --- Hope (per-character, cap 6) ------------------------------------------ #

def test_gain_hope_clamps_to_cap():
    c = CharacterState(id="p", name="P")
    c.gain_hope(10)
    assert c.hope == c.hope_max == 6


def test_spend_hope_succeeds_when_affordable():
    c = CharacterState(id="p", name="P", hope=3)
    assert c.spend_hope(2) is True
    assert c.hope == 1


def test_spend_hope_fails_and_does_not_change_when_short():
    c = CharacterState(id="p", name="P", hope=1)
    assert c.spend_hope(2) is False
    assert c.hope == 1


def test_negative_amounts_never_change_hope():
    c = CharacterState(id="p", name="P", hope=2)
    c.gain_hope(-5)
    assert c.hope == 2  # negative gain is a no-op, not a drain
    c.spend_hope(-5)
    assert c.hope == 2  # negative spend is a no-op, not a backdoor gain


# --- Fear (GM pool, cap 12) ----------------------------------------------- #

def test_gain_fear_clamps_to_cap():
    g = _state()
    g.gain_fear(100)
    assert g.fear == g.fear_max == 12


def test_spend_fear_succeeds_when_affordable():
    g = _state()
    g.gain_fear(3)
    assert g.spend_fear(2) is True
    assert g.fear == 1


def test_spend_fear_fails_and_does_not_change_when_short():
    g = _state()
    g.gain_fear(1)
    assert g.spend_fear(2) is False
    assert g.fear == 1


def test_negative_amounts_never_change_fear():
    g = _state()
    g.gain_fear(3)
    g.gain_fear(-5)
    assert g.fear == 3  # negative gain is a no-op
    g.spend_fear(-5)
    assert g.fear == 3  # negative spend is a no-op, not a backdoor gain


# --- serialization carries the new fields --------------------------------- #

def test_to_dict_includes_fear_and_character_hope():
    g = _state()
    g.gain_fear(4)
    g.characters["p"] = CharacterState(id="p", name="P", hope=2)
    d = g.to_dict()
    assert d["fear"] == 4
    assert d["fear_max"] == 12
    assert d["characters"]["p"]["hope"] == 2
    assert d["characters"]["p"]["hope_max"] == 6
