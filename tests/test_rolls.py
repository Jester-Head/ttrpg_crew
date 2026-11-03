import random

import pytest

from ttrpg_crew.game.rolls import (
    RollRequest,
    parse_roll_request,
    resolve_roll,
    strip_roll_tag,
)
from ttrpg_crew.game.validator import audit_gm


@pytest.mark.parametrize(
    "tag, skill, dc",
    [
        ("[ROLL: Intimidation DC 15]", "Intimidation", 15),
        ("[roll: Sleight of Hand vs DC 12]", "Sleight of Hand", 12),
        ("[ROLL: Investigation vs. DC 17]", "Investigation", 17),
        ("[ROLL: Perception, DC 10]", "Perception", 10),
        ("[ROLL: DC 20]", "Check", 20),
    ],
)
def test_parses_tag_variants(tag, skill, dc):
    request = parse_roll_request(f"He waits for your answer. {tag}")
    assert request == RollRequest(skill=skill, dc=dc)


def test_no_tag_means_no_roll():
    assert parse_roll_request("Roll d20 + Intimidation against DC 15.") is None


def test_strip_keeps_the_prose():
    text = "The clerk swallows. Make it count.\n\n[ROLL: Intimidation DC 15]"
    assert strip_roll_tag(text) == "The clerk swallows. Make it count."


def test_engine_roll_is_seedable_and_has_no_player_input():
    request = RollRequest("Intimidation", 15)
    first = resolve_roll(request, random.Random(7))
    second = resolve_roll(request, random.Random(7))
    assert first == second
    assert 1 <= first.d20 <= 20
    assert first.modifier == 0
    with pytest.raises(TypeError):
        resolve_roll(request, typed=20)  # type: ignore[call-arg]


@pytest.mark.parametrize("d20, success", [(14, False), (15, True), (20, True)])
def test_success_is_total_at_or_above_dc(d20, success):
    class Fixed(random.Random):
        def randint(self, a, b):
            return d20

    result = resolve_roll(RollRequest("Intimidation", 15), Fixed())
    assert result.success is success


def test_describe():
    class Fixed(random.Random):
        def randint(self, a, b):
            return 12

    result = resolve_roll(RollRequest("Intimidation", 15), Fixed())
    assert result.describe() == "Intimidation: rolled 12 vs DC 15 — failure"


def test_roll_call_does_not_need_a_question():
    text = "The clerk's eyes dart to the door. Everything rides on the next breath."
    rules = [v.rule for v in audit_gm(text, require_prompt=False)]
    assert "missing_prompt" not in rules
    assert "missing_prompt" in [v.rule for v in audit_gm(text)]
