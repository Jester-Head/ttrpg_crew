"""Tests for the scene tracker's pure parts: parsing, cleaning, state merge."""

from ttrpg_crew.game.beat import Beat, BeatSegment
from ttrpg_crew.game.scene_tracker import (
    SCENE_UPDATE_SCHEMA,
    apply_scene_update,
    clean_suggestions,
    format_scene_memory,
    parse_scene_update,
    pick_suggestion,
    scene_update_messages,
)
from ttrpg_crew.game.schemas import SceneUpdate
from ttrpg_crew.game.state import GameState

_CFG = {
    "enabled": True,
    "suggestions": 3,
    "max_suggestion_chars": 90,
    "max_facts": 3,
    "max_npcs": 2,
}
_PARTY = ["Mireille Vane", "Oskar Tull"]


def _state(**kwargs) -> GameState:
    return GameState(campaign_premise="", location="The Cinder Docks", scene_type="narrative", **kwargs)


def _apply(state: GameState, update: SceneUpdate, *, tagged=False, player_input="") -> None:
    apply_scene_update(
        state,
        update,
        protagonist_name="Shimoda Rice",
        party_names=_PARTY,
        player_input=player_input,
        tagged_location=tagged,
        config=_CFG,
    )


# --- schema / parsing ------------------------------------------------------ #

def test_schema_matches_model_fields():
    assert set(SCENE_UPDATE_SCHEMA["properties"]) == set(SceneUpdate.model_fields)
    assert set(SCENE_UPDATE_SCHEMA["required"]) == set(SceneUpdate.model_fields)


def test_parse_tolerates_fences_and_prose():
    raw = 'Here you go:\n```json\n{"location": "Salt Stair", "suggested_actions": ["Run"]}\n```'
    update = parse_scene_update(raw)
    assert update is not None
    assert update.location == "Salt Stair"
    assert update.suggested_actions == ["Run"]


def test_parse_returns_none_for_garbage():
    assert parse_scene_update("") is None
    assert parse_scene_update("no json here") is None
    assert parse_scene_update("{not: valid json}") is None


def test_parse_coerces_loose_shapes():
    raw = '{"npcs": [{"name": "Vell", "desc": "ferryman"}], "new_facts": "Vell owes a debt", "summary": null}'
    update = parse_scene_update(raw)
    assert update.npcs == ["Vell: ferryman"]
    assert update.new_facts == ["Vell owes a debt"]
    assert update.summary == ""


# --- suggestion cleaning --------------------------------------------------- #

def _clean(values, *, player_input="", limit=3):
    return clean_suggestions(
        values, protagonist_name="Shimoda Rice", player_input=player_input, limit=limit
    )


def test_clean_strips_markers_and_quotes():
    assert _clean(['1. Ask Vell who paid him', '- "Check the hull"', "(b) Wait"]) == [
        "Ask Vell who paid him",
        "Check the hull",
        "Wait",
    ]


def test_clean_drops_roll_requests_but_keeps_rolling_actions():
    assert _clean(["Roll for Perception", "Make a DC 15 check", "Roll the barrel aside"]) == [
        "Roll the barrel aside"
    ]


def test_clean_drops_protagonist_as_other_person():
    assert _clean(["Follow Shimoda upstairs", "Ask Mireille what she saw"]) == [
        "Ask Mireille what she saw"
    ]


def test_clean_drops_repeat_of_action_just_taken_and_duplicates():
    out = _clean(
        ["Search the crates.", "Search the crates", "search the crates", "Hail the barge"],
        player_input="search the crates",
    )
    assert out == ["Hail the barge"]


def test_clean_drops_placeholders_and_overlong_and_caps():
    long = "Walk " + "very " * 30 + "slowly"
    out = _clean(["<action>", "Do something...", long, "A1", "Go", "Bribe the clerk", "Hide", "Leave", "Wait"])
    assert out == ["Bribe the clerk", "Hide", "Leave"]


# --- applying an update ---------------------------------------------------- #

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


def test_npcs_update_in_place_and_skip_party():
    state = _state(active_npcs=["Vell: ferryman"])
    _apply(
        state,
        SceneUpdate(npcs=["Vell: ferryman, now hostile", "Mireille Vane: alchemist", "Shimoda: courier"]),
    )
    assert state.active_npcs == ["Vell: ferryman, now hostile"]


def test_npcs_cap_keeps_newest():
    state = _state(active_npcs=["Vell: ferryman"])
    _apply(state, SceneUpdate(npcs=["Ada Kross: clerk", "Bram: guard"]))
    assert state.active_npcs == ["Ada Kross: clerk", "Bram: guard"]


def test_apply_replaces_suggestions_with_cleaned_list():
    state = _state(suggested_actions=["stale"])
    _apply(state, SceneUpdate(suggested_actions=["1. Hail the barge", "Roll for stealth"]))
    assert state.suggested_actions == ["Hail the barge"]


# --- prompt + memory + picking --------------------------------------------- #

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
                     "Mireille Vane", "3 short things"):
        assert expected in user


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
    assert pick_suggestion("3", options) == "3"
    assert pick_suggestion("I wade in", options) == "I wade in"
    assert pick_suggestion("1", []) == "1"


def test_suggestions_stay_out_of_the_transcript_text():
    beat = Beat(
        segments=[BeatSegment("GM", "GM", "The barge drifts closer.")],
        closing_prompt="What do you do?",
        suggested_actions=["Hail the barge"],
    )
    assert "Hail the barge" not in str(beat)
