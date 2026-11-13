"""Tests for character-name matching."""

from ttrpg_crew.game.names import mentions_name, short_name
from ttrpg_crew.game.speaker_policy import find_direct_address


def test_short_name_skips_titles_and_articles():
    assert short_name("Sirra Nightcall") == "Sirra"
    assert short_name("Sister Maren") == "Maren"
    assert short_name("The Courier") == "Courier"
    assert short_name("Dr. Vane") == "Vane"
    assert short_name("Roland") == "Roland"
    assert short_name("Mother") == "Mother"
    assert short_name("") == ""


def test_mentions_name_by_full_or_short_name():
    assert mentions_name("ask sirra nightcall", "Sirra Nightcall")
    assert mentions_name("Follow Sirra upstairs", "Sirra Nightcall")
    assert mentions_name("follow sirra upstairs", "Sirra Nightcall")
    assert not mentions_name("Follow the siren upstairs", "Sirra Nightcall")
    assert not mentions_name("Ask Bo", "")


def test_match_case_applies_to_the_short_name_only():
    assert not mentions_name("ask what he will do", "Will Harrow", match_case=True)
    assert mentions_name("Ask Will to wait", "Will Harrow", match_case=True)
    assert mentions_name("ask will harrow to wait", "Will Harrow", match_case=True)


def test_direct_address_finds_a_titled_companion_by_their_name():
    profiles = {"maren": {"name": "Sister Maren"}, "sirra": {"name": "Sirra Nightcall"}}
    speakers = list(profiles)
    assert find_direct_address(profiles, speakers, "i ask maren about the vial") == (
        "maren",
        "Sister Maren",
    )
    assert find_direct_address(profiles, speakers, "sirra, watch the door") == (
        "sirra",
        "Sirra Nightcall",
    )
    assert find_direct_address(profiles, speakers, "i ask my sister about the vial") is None
