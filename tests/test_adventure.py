from ttrpg_crew.game.adventure import guidance, load_adventure, visible_cast
from ttrpg_crew.game.scene_tracker import parse_scene_update

FRAME = {
    "title": "T",
    "hook": "Mara Venn asks you to hide her brother from his assay.",
    "starting_location": "The Residuum",
    "checkpoints": [
        {"goal": "Learn when the assay is", "reached_when": "the date is known",
         "pressure": "A Census notice arrives", "npcs": ["Odo Pell"]},
        {"goal": "Find a hiding place", "reached_when": "a place is agreed",
         "pressure": "Collectors walk the street", "npcs": ["Sister Hale"]},
    ],
    "ending": "The assay happens or is escaped.",
}
CAST = [
    {"name": "Mara Venn"},
    {"name": "Odo Pell"},
    {"name": "Sister Hale"},
    {"name": "Teodor Quill"},
]


def names(members):
    return [m["name"] for m in members]


def test_frame_loads():
    adventure = load_adventure(FRAME)
    assert adventure.hook.startswith("Mara Venn")
    assert [c.npcs for c in adventure.checkpoints] == [["Odo Pell"], ["Sister Hale"]]


def test_old_frame_falls_back_to_full_cast_and_no_guidance():
    adventure = load_adventure({"opening": {"location": "x"}, "beats": ["a"]})
    assert visible_cast(CAST, adventure, 0, []) == CAST
    assert guidance(adventure, 0, 9, 4) == {"checkpoint": "", "pressure": "", "adventure_ending": ""}


def test_guidance_presses_only_after_stalling():
    adventure = load_adventure(FRAME)
    assert guidance(adventure, 0, 3, 4) == {
        "checkpoint": "Learn when the assay is", "pressure": "", "adventure_ending": ""
    }
    assert guidance(adventure, 0, 4, 4)["pressure"] == "A Census notice arrives"


def test_guidance_shows_only_the_ending_past_the_last_checkpoint():
    adventure = load_adventure(FRAME)
    assert guidance(adventure, 2, 0, 4) == {
        "checkpoint": "", "pressure": "", "adventure_ending": "The assay happens or is escaped."
    }


def test_cast_appears_as_the_story_reaches_them():
    adventure = load_adventure(FRAME)
    assert names(visible_cast(CAST, adventure, 0, [])) == ["Mara Venn", "Odo Pell"]
    assert names(visible_cast(CAST, adventure, 1, [])) == ["Mara Venn", "Odo Pell", "Sister Hale"]


def test_met_npcs_stay_visible_by_first_name():
    adventure = load_adventure(FRAME)
    met = ["Teodor: a nervous clerk"]
    assert "Teodor Quill" in names(visible_cast(CAST, adventure, 0, met))


def test_checkpoint_reached_parses_numbers_and_defaults_to_zero():
    def reached(raw):
        update = parse_scene_update(raw)
        assert update is not None
        return update.checkpoint_reached

    assert reached('{"checkpoint_reached": 2}') == 2
    assert reached('{"checkpoint_reached": true}') == 1
    assert reached('{"checkpoint_reached": "nope"}') == 0
    assert reached('{"summary": "x"}') == 0
