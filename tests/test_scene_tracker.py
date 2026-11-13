"""Tests for the scene tracker's pure parts: parsing, cleaning, state merge."""

from ttrpg_crew import config_loader
from ttrpg_crew.game.beat import Beat, BeatSegment
from ttrpg_crew.game.scene_tracker import (
    SCENE_UPDATE_SCHEMA,
    apply_scene_update,
    clean_suggestions,
    format_scene_memory,
    parse_scene_update,
    pick_suggestion,
    scene_update_messages,
    tracker_config,
)
from ttrpg_crew.game.schemas import SceneUpdate
from ttrpg_crew.game.state import GameState

_CFG = {
    "enabled": True,
    "suggestions": 3,
    "max_suggestion_chars": 90,
    "max_facts": 3,
    "max_npcs": 2,
    "max_memory_chars": 240,
}
_PARTY = ["Mireille Vane", "Oskar Tull"]


def _state(**kwargs) -> GameState:
    return GameState(campaign_premise="", location="The Cinder Docks", scene_type="narrative", **kwargs)


def _apply(
    state: GameState,
    update: SceneUpdate,
    *,
    tagged=False,
    player_input="",
    party=_PARTY,
    **overrides,
) -> None:
    apply_scene_update(
        state,
        update,
        protagonist_name="Shimoda Rice",
        party_names=party,
        player_input=player_input,
        tagged_location=tagged,
        config={**_CFG, **overrides},
    )


def test_schema_matches_model_fields():
    assert set(SCENE_UPDATE_SCHEMA["properties"]) == set(SceneUpdate.model_fields)
    assert set(SCENE_UPDATE_SCHEMA["required"]) == set(SceneUpdate.model_fields)


def test_parse_tolerates_fences_and_prose():
    raw = 'Here you go:\n```json\n{"location": "Salt Stair", "suggested_actions": ["Run"]}\n```'
    update = parse_scene_update(raw)
    assert update is not None
    assert update.location == "Salt Stair"
    assert update.suggested_actions == ["Run"]


def test_parse_skips_braces_in_the_prose_before_the_object():
    update = parse_scene_update('Sure {like this}: {"location": "Salt Stair"}')
    assert update is not None
    assert update.location == "Salt Stair"


def test_parse_returns_none_for_garbage():
    assert parse_scene_update("") is None
    assert parse_scene_update("no json here") is None
    assert parse_scene_update("{not: valid json}") is None
    assert parse_scene_update('{"unrelated": 1}') is None


def test_parse_coerces_loose_shapes():
    raw = '{"npcs": [{"name": "Vell", "desc": "ferryman"}], "new_facts": "Vell owes a debt", "summary": null}'
    update = parse_scene_update(raw)
    assert update.npcs == ["Vell: ferryman"]
    assert update.new_facts == ["Vell owes a debt"]
    assert update.summary == ""


def test_parse_keeps_the_update_when_one_field_is_off_type():
    raw = (
        '{"location_changed": null, "location": ["Salt Stair"], '
        '"npcs": {"Vell": "ferryman"}, "suggested_actions": ["Hail the barge"]}'
    )
    update = parse_scene_update(raw)
    assert update is not None
    assert update.location_changed is False
    assert update.location == "Salt Stair"
    assert update.npcs == ["Vell: ferryman"]
    assert update.suggested_actions == ["Hail the barge"]


def test_parse_puts_the_name_first_whatever_the_key_order():
    update = parse_scene_update('{"npcs": [{"desc": "ferryman", "name": "Vell"}]}')
    assert update.npcs == ["Vell: ferryman"]


def _clean(values, *, player_input="", limit=3, protagonist="Shimoda Rice"):
    return clean_suggestions(
        values, protagonist_name=protagonist, player_input=player_input, limit=limit
    )


def test_clean_strips_markers_and_quotes():
    assert _clean(['1. Ask Vell who paid him', '- "Check the hull"', "(b) Wait"]) == [
        "Ask Vell who paid him",
        "Check the hull",
        "Wait",
    ]


def test_clean_strips_a_marker_inside_wrapping_quotes():
    assert _clean(['"1. Ask Vell who paid him"']) == ["Ask Vell who paid him"]


def test_clean_keeps_a_quotation_inside_the_suggestion():
    assert _clean(['Ask Vell "who paid you?"', '"Check the hull']) == [
        'Ask Vell "who paid you?"',
        "Check the hull",
    ]


def test_clean_drops_roll_requests_but_keeps_rolling_actions():
    assert _clean(["Roll for Perception", "Make a DC 15 check", "Roll the barrel aside"]) == [
        "Roll the barrel aside"
    ]


def test_clean_drops_roll_requests_in_other_phrasings():
    rolls = ["Make a Perception check", "Roll Stealth", "Roll initiative", "Roll to hit the guard"]
    assert _clean(rolls, limit=9) == []


def test_clean_keeps_actions_that_only_sound_like_rolls_or_placeholders():
    actions = [
        "Join the dice game",
        "Roll a barrel at the guards",
        "Run",
        "Wait... then follow him",
        "Ask about the option to pay later",
    ]
    assert _clean(actions, limit=9) == actions


def test_clean_drops_protagonist_as_other_person():
    assert _clean(["Follow Shimoda upstairs", "Ask Mireille what she saw"]) == [
        "Ask Mireille what she saw"
    ]


def test_clean_does_not_match_a_name_that_is_also_a_common_word():
    assert _clean(["Ask the ferryman who paid him"], protagonist="The Courier") == [
        "Ask the ferryman who paid him"
    ]
    assert _clean(["Ask what he will do next"], protagonist="Will Harrow") == [
        "Ask what he will do next"
    ]
    assert _clean(["Ask Will to wait outside"], protagonist="Will Harrow") == []


def test_clean_drops_repeat_of_action_just_taken_and_duplicates():
    out = _clean(
        ["Search the crates.", "Search the crates", "search the crates", "Hail the barge"],
        player_input="search the crates",
    )
    assert out == ["Hail the barge"]


def test_clean_drops_placeholders_and_overlong_and_caps():
    long = "Walk " + "very " * 30 + "slowly"
    out = _clean(
        ["<action>", "Do something...", "Option 2", long, "A1", "Go", "Bribe the clerk", "Hide", "Leave", "Wait"]
    )
    assert out == ["Bribe the clerk", "Hide", "Leave"]


def test_clean_returns_nothing_at_a_limit_of_zero():
    assert _clean(["Hail the barge"], limit=0) == []


def test_location_moves_only_when_flagged_and_untagged():
    state = _state()
    _apply(state, SceneUpdate(location="Somewhere Vague", location_changed=False))
    assert state.location == "The Cinder Docks"

    _apply(state, SceneUpdate(location="The Salt Stair", location_changed=True), tagged=True)
    assert state.location == "The Cinder Docks"  # the GM's [LOCATION:] tag wins

    _apply(state, SceneUpdate(location="The Salt Stair", location_changed=True))
    assert state.location == "The Salt Stair"


def test_objective_set_when_empty_then_only_on_change():
    state = _state()
    _apply(state, SceneUpdate(objective="Find who sent the vial"))
    assert state.party_goal == "Find who sent the vial"
    _apply(state, SceneUpdate(objective="Something else", objective_changed=False))
    assert state.party_goal == "Find who sent the vial"
    _apply(state, SceneUpdate(objective="Get the vial back", objective_changed=True))
    assert state.party_goal == "Get the vial back"


def test_placeholder_echoes_are_not_stored():
    state = _state()
    _apply(
        state,
        SceneUpdate(
            objective="None yet",
            location="Unknown",
            location_changed=True,
            new_facts=["(none)"],
            npcs=["None yet"],
        ),
    )
    assert state.party_goal is None
    assert state.location == "The Cinder Docks"
    assert state.known_facts == []
    assert state.active_npcs == []


def test_overlong_memory_is_dropped():
    state = _state(scene_summary="Vell is stalling.")
    _apply(
        state,
        SceneUpdate(
            location="L" * 300,
            location_changed=True,
            summary="s" * 300,
            new_facts=["x" * 300, "The gate is shut"],
            npcs=["Ada Kross: " + "y" * 300],
        ),
    )
    assert state.location == "The Cinder Docks"
    assert state.scene_summary == "Vell is stalling."
    assert state.known_facts == ["The gate is shut"]
    assert state.active_npcs == []


def test_facts_merge_resolve_dedupe_and_cap():
    state = _state(known_facts=["Vell owes a debt", "The vial is cracked"])
    _apply(
        state,
        SceneUpdate(
            new_facts=["vell owes a debt", "The clerk is bribed", "The gate is shut", "Oskar lied"],
            resolved_facts=["The vial is cracked"],
        ),
    )
    # resolved fact gone, duplicate ignored, oldest dropped at the cap of 3
    assert state.known_facts == ["The clerk is bribed", "The gate is shut", "Oskar lied"]


def test_npcs_update_and_skip_party():
    state = _state(active_npcs=["Vell: ferryman"])
    _apply(
        state,
        SceneUpdate(npcs=["Vell: ferryman, now hostile", "Mireille Vane: alchemist", "Shimoda: courier"]),
    )
    assert state.active_npcs == ["Vell: ferryman, now hostile"]


def test_npc_sharing_a_title_with_a_party_member_is_kept():
    state = _state()
    _apply(state, SceneUpdate(npcs=["Sister Kell: a nun"]), party=["Sister Maren"])
    assert state.active_npcs == ["Sister Kell: a nun"]


def test_npcs_cap_keeps_newest():
    state = _state(active_npcs=["Vell: ferryman"])
    _apply(state, SceneUpdate(npcs=["Ada Kross: clerk", "Bram: guard"]))
    assert state.active_npcs == ["Ada Kross: clerk", "Bram: guard"]


def test_npcs_cap_keeps_one_updated_this_beat():
    state = _state(active_npcs=["Vell: ferryman", "Ada Kross: clerk"])
    _apply(state, SceneUpdate(npcs=["Vell: ferryman, now hostile", "Bram: guard"]))
    assert state.active_npcs == ["Vell: ferryman, now hostile", "Bram: guard"]


def test_zero_caps_keep_nothing():
    state = _state(known_facts=["Vell owes a debt"], active_npcs=["Vell: ferryman"])
    _apply(
        state,
        SceneUpdate(
            new_facts=["The gate is shut"],
            npcs=["Ada Kross: clerk"],
            suggested_actions=["Hail the barge"],
        ),
        max_facts=0,
        max_npcs=0,
        suggestions=0,
    )
    assert state.known_facts == []
    assert state.active_npcs == []
    assert state.suggested_actions == []


def test_apply_replaces_suggestions_with_cleaned_list():
    state = _state(suggested_actions=["stale"])
    _apply(state, SceneUpdate(suggested_actions=["1. Hail the barge", "Roll for stealth"]))
    assert state.suggested_actions == ["Hail the barge"]


def test_tracker_config_reads_negatives_as_zero_and_blanks_as_default(monkeypatch):
    monkeypatch.setattr(
        config_loader,
        "config_section",
        lambda name: {"suggestions": -2, "max_facts": None, "enabled": "false", "max_npcs": "x"},
    )
    cfg = tracker_config()
    assert cfg["suggestions"] == 0
    assert cfg["max_facts"] == 12
    assert cfg["max_npcs"] == 10
    assert cfg["enabled"] is False


def test_messages_carry_state_and_beat():
    state = _state(party_goal="Find the sender", known_facts=["Vell owes a debt"])
    system, user = scene_update_messages(
        beat_text="Vell spits into the canal.",
        state=state,
        player_input="I ask Vell about the vial",
        protagonist_name="Shimoda Rice",
        party_names=_PARTY,
        suggestions=3,
    )
    assert "JSON" in system
    for expected in ("The Cinder Docks", "Find the sender", "- Vell owes a debt",
                     "Vell spits into the canal.", "I ask Vell about the vial",
                     "Mireille Vane", "3 short things", "under 90 characters"):
        assert expected in user


def test_messages_use_configured_suggestion_length():
    _, user = scene_update_messages(
        beat_text="Vell spits into the canal.",
        state=_state(),
        player_input="",
        protagonist_name="Shimoda Rice",
        party_names=_PARTY,
        suggestions=3,
        max_suggestion_chars=60,
    )
    assert "under 60 characters" in user
    assert "under 90 characters" not in user


def test_messages_ask_for_no_suggestions_at_zero():
    _, user = scene_update_messages(
        beat_text="Vell spits into the canal.",
        state=_state(),
        player_input="",
        protagonist_name="Shimoda Rice",
        party_names=_PARTY,
        suggestions=0,
    )
    assert "suggested_actions: an empty list." in user
    assert "None yet" not in user


def test_scene_memory_empty_until_something_is_known():
    assert format_scene_memory(_state()) == ""
    memory = format_scene_memory(
        _state(party_goal="Find the sender", known_facts=["A", "B"], active_npcs=["Vell: ferryman"])
    )
    assert memory == "Objective: Find the sender\nEstablished facts: A; B\nNamed NPCs met: Vell: ferryman"


def test_pick_suggestion_by_number_or_free_text():
    options = ["Hail the barge", "Hide"]
    assert pick_suggestion("2", options) == "Hide"
    assert pick_suggestion(" 1 ", options) == "Hail the barge"
    assert pick_suggestion("2.", options) == "Hide"
    assert pick_suggestion("I wade in", options) == "I wade in"
    assert pick_suggestion("2 coins", options) == "2 coins"


def test_pick_suggestion_returns_none_for_a_number_with_no_option():
    assert pick_suggestion("3", ["Hail the barge", "Hide"]) is None
    assert pick_suggestion("0", ["Hail the barge", "Hide"]) is None
    assert pick_suggestion("1", []) is None


def test_pick_suggestion_passes_through_digits_int_cannot_parse():
    assert pick_suggestion("²", ["Hail the barge", "Hide"]) == "²"


def test_suggestions_stay_out_of_the_transcript_text():
    beat = Beat(
        segments=[BeatSegment("GM", "GM", "The barge drifts closer.")],
        closing_prompt="What do you do?",
        suggested_actions=["Hail the barge"],
    )
    assert "Hail the barge" not in str(beat)
