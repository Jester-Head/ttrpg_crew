"""Companions staying quiet: when it's offered, and how a silence is settled."""

import random
import re

import pytest

from ttrpg_crew.game.intent import BeatIntent
from ttrpg_crew.game.validator import (
    observes_tag,
    repeats_silent_action,
    silent_action,
    strip_observes,
)
from ttrpg_crew.prompts.prompt_loader import load_task_for_game_state
from ttrpg_crew.services.beat_runner import PromptBackedBeatRunner

_VARS = {
    "party_members": "Shimoda, Sirra",
    "protagonist_name": "Shimoda",
    "character_name": "Braynstorm",
    "examples": [],
    "beat_directives": "",
    "scene_context": "A tavern.",
    "recent_story": "",
    "session_recap": "",
    "combat_state": "",
}
_SILENCE_OFFER = re.compile(r"bracketed action|action in square brackets|\[observes\]", re.I)


@pytest.mark.parametrize("tier", ["large", "medium", "small"])
@pytest.mark.parametrize("state", ["narrative", "opening_scene", "session_resume", "combat"])
def test_silence_offered_only_when_allowed(tier, state):
    def render(may_observe):
        cfg = load_task_for_game_state(
            "companion", {**_VARS, "may_observe": may_observe}, state, tier=tier
        )
        return cfg["description"] + cfg.get("expected_output", "")

    assert _SILENCE_OFFER.search(render(True))
    assert not _SILENCE_OFFER.search(render(False))
    assert "[observes]" not in render(True)


def test_forced_or_addressed_speakers_may_not_observe():
    asked = BeatIntent(kind="direct_question", addressed_id="sirra", addressed_name="Sirra")
    may = PromptBackedBeatRunner._may_observe
    assert may("bray", None, speakers_forced=False)
    assert may("bray", asked, speakers_forced=False)
    assert not may("sirra", asked, speakers_forced=False)
    assert not may("bray", None, speakers_forced=True)


def test_observes_tag_check():
    assert observes_tag("[observes]", may_observe=True) is None
    assert observes_tag("[observes]", may_observe=False)
    assert observes_tag('"Not tonight." [observes]', may_observe=True)
    assert observes_tag("[I observe] He watches the door.", may_observe=True)
    assert observes_tag('"Not tonight," I say.', may_observe=False) is None


def test_silent_action_is_one_short_wordless_bracket():
    assert silent_action("[keeps her eyes on Frost]") == "keeps her eyes on Frost"
    assert silent_action("[observes]") is None
    assert silent_action('[nods] "Fine."') is None
    assert silent_action('[says "no"]') is None
    assert silent_action("[" + "word " * 30 + "]") is None


def test_strip_observes_keeps_the_rest():
    assert strip_observes('"Not tonight."  [observes]') == '"Not tonight."'
    assert strip_observes("[Observes]") == ""


def test_repeats_silent_action():
    assert repeats_silent_action("grips his hilt", ["grips the hilt"])
    assert not repeats_silent_action("looks at the door", ["grips the hilt"])


def _runner(beats=None, recent=None):
    return PromptBackedBeatRunner(
        gm_agent=None,
        companion_agents={},
        companion_prompt_vars={"sirra": {"character_name": "Sirra", "silent_beats": beats or []}},
        recent_silences=recent,
    )


def test_settle_keeps_speech_and_a_fresh_silence():
    runner = _runner()
    assert runner._settle_silence("sirra", '"Go." [observes]') == ('"Go."', False)
    assert runner._settle_silence("sirra", "[watches the door]") == ("[watches the door]", True)
    assert runner.recent_silences["sirra"] == ["watches the door"]


def test_settle_falls_back_to_the_list_then_to_no_line():
    random.seed(0)
    runner = _runner(beats=["folds her arms"], recent={"sirra": ["watches the door"]})
    assert runner._settle_silence("sirra", "[watches the door]") == ("[folds her arms]", True)
    assert runner._settle_silence("sirra", "[observes]") == ("", True)
    assert _runner()._settle_silence("sirra", "[observes]") == ("", True)
