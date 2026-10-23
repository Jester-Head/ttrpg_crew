"""Think-first splitting and the A/B harness's extra output checks."""

from ttrpg_crew.game.validator import (
    acts_as_other,
    assistant_speak,
    format_leak,
    repeats_player_action,
)
from ttrpg_crew.services.direct_companions import split_thought


def test_split_thought_takes_text_after_say():
    note, line, found = split_thought('NOTE: she distrusts him.\nSAY: "Not a chance."')
    assert (note, line, found) == ("she distrusts him.", '"Not a chance."', True)


def test_split_thought_handles_bold_markers():
    note, line, found = split_thought('**NOTE:** wary\n\n**SAY:** I step back.')
    assert (note, line, found) == ("wary", "I step back.", True)


def test_split_thought_drops_note_without_say_marker():
    note, line, found = split_thought('NOTE: annoyed\n\n"Fine. Lead on."')
    assert (note, line, found) == ("annoyed", '"Fine. Lead on."', False)


def test_split_thought_passes_plain_text_through():
    assert split_thought('"Fine."') == ("", '"Fine."', False)


def test_format_leak_catches_react_markers():
    assert format_leak("Thought: I now can give a great answer\nFinal Answer: hi")
    assert format_leak('I nod. "Thoughtful of you."') is None


def test_assistant_speak_catches_phrases_and_markdown():
    assert assistant_speak("I'd be happy to help with the vial.")
    assert assistant_speak("- First, we wait\n- Then we run")
    assert assistant_speak('"Of course. Keep your voice down."') is None


def test_acts_as_other_ignores_quoted_address():
    assert acts_as_other('"Sirra, watch the door."', ("Sirra",)) == []


def test_repeats_player_action_flags_taking_the_same_object():
    line = "I just flip the vial in my hand to check the weight, then pocket it."
    assert repeats_player_action(line, "I take the vial and ask what they want.")


def test_repeats_player_action_flags_undoing_a_hand_back():
    line = "I step closer and pry the vial from her grip."
    assert repeats_player_action(line, "I refuse, hand the vial back, and turn to leave.")


def test_repeats_player_action_ignores_dialogue_and_looking():
    player = "I take the vial."
    assert repeats_player_action('"Take the vial somewhere safe," I say.', player) is None
    assert repeats_player_action("I watch the hand holding the vial.", player) is None


def test_acts_as_other_flags_narrating_another_companion():
    rules = [v.rule for v in acts_as_other("Sirra nods and draws her knife.", ("Sirra",))]
    assert rules == ["acts_as_other"]
