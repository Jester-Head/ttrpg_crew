"""Companions are told the player's actual words, whatever the intent."""

from ttrpg_crew.game.intent import BeatIntent
from ttrpg_crew.services.beat_runner import PromptBackedBeatRunner


def _runner() -> PromptBackedBeatRunner:
    return PromptBackedBeatRunner(
        gm_agent=None,
        companion_agents={},
        companion_prompt_vars={
            "sirra": {
                "character_name": "Sirra",
                "selection": {"expertise": ["negotiation"], "triggers": ["exit"]},
            },
            "bray": {"character_name": "Braynstorm"},
        },
    )


def _directives(intent: BeatIntent, player_input: str) -> str:
    return _runner()._beat_directives(
        "sirra", intent, ["sirra"], player_input=player_input, protagonist_name="Shimoda"
    )


def test_action_quotes_the_players_words():
    text = _directives(BeatIntent(kind="action"), "I hand the vial back and leave.")
    assert '"I hand the vial back and leave."' in text
    assert "Shimoda" in text


def test_question_quotes_the_players_words():
    text = _directives(BeatIntent(kind="question"), "Who sent you?")
    assert '"Who sent you?"' in text


def test_why_you_names_the_matching_expertise_and_trigger():
    text = _directives(BeatIntent(kind="action"), "I offer him a deal and check the exit.")
    assert 'the moment involves a deal, and it may touch on "exit"' in text


def test_no_why_you_line_without_a_match():
    text = _directives(BeatIntent(kind="action"), "I sit down.")
    assert "You're speaking because" not in text


def test_opening_gm_text_gets_the_presence_line():
    from ttrpg_crew.game.beat import Beat

    runner = _runner()
    presence = runner._presence_line("opening")
    segment = runner._build_segment("GM", "Jessa grips your wrist.", Beat(), presence=presence)
    assert segment.text.endswith("With you: Sirra and Braynstorm.")


def test_presence_line_skipped_when_gm_named_everyone_or_mid_scene():
    from ttrpg_crew.game.beat import Beat

    runner = _runner()
    assert runner._presence_line("narrative") == ""
    text = "Sirra waits by the door. Braynstorm is at the bar."
    segment = runner._build_segment("GM", text, Beat(), presence=runner._presence_line("opening"))
    assert segment.text == text


def test_words_addressed_to_another_companion_are_quoted():
    intent = BeatIntent(kind="direct_question", addressed_id="bray", addressed_name="Braynstorm")
    text = _directives(intent, "Braynstorm, are the canals safe?")
    assert "Braynstorm" in text
    assert '"Braynstorm, are the canals safe?"' in text
