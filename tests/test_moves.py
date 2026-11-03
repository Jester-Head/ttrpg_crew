import pytest

from ttrpg_crew.game import moves
from ttrpg_crew.game.intent import BeatIntent
from ttrpg_crew.game.stance import Stance
from ttrpg_crew.services.character_registry import examples_for

ALL_KEYS = [
    "narrative", "opening_scene", "session_resume", "combat",
    "answer", "agree", "disagree", "social", "investigation", "dilemma",
]


@pytest.fixture
def all_keys(monkeypatch):
    monkeypatch.setattr(moves, "list_task_keys", lambda task_file: ALL_KEYS)


def test_addressed_companion_answers_even_with_a_stance(all_keys):
    intent = BeatIntent(kind="direct_question", addressed_id="sirra")
    stance = Stance(move="disagree", approach="aggressive", strength=2)
    assert moves.companion_move("sirra", intent, stance, "combat") == "answer"


def test_other_companions_take_their_stance(all_keys):
    intent = BeatIntent(kind="direct_question", addressed_id="sirra")
    stance = Stance(move="agree", approach="scholarly", strength=2)
    assert moves.companion_move("braynstorm", intent, stance, "narrative") == "agree"


def test_no_stance_falls_to_the_scene_task(all_keys):
    intent = BeatIntent(kind="action")
    assert moves.companion_move("nizin", intent, None, "combat") == "combat"
    assert moves.companion_move("nizin", intent, None, "opening") == "opening_scene"
    assert moves.companion_move("nizin", intent, None, "narrative") == "narrative"


def test_undefined_move_prompt_falls_back(monkeypatch):
    monkeypatch.setattr(moves, "list_task_keys", lambda task_file: ["narrative", "combat"])
    intent = BeatIntent(kind="direct_question", addressed_id="sirra")
    assert moves.companion_move("sirra", intent, None, "combat") == "combat"


def test_gm_scene_beats_situation(all_keys):
    assert moves.gm_task_key("combat", ["social"]) == "combat"
    assert moves.gm_task_key("opening", ["investigation"]) == "opening_scene"


def test_gm_situation_priority(all_keys):
    assert moves.gm_task_key("narrative", ["investigation", "negotiation"]) == "social"
    assert moves.gm_task_key("narrative", ["investigation", "memory"]) == "investigation"
    assert moves.gm_task_key("narrative", ["moral_dilemma"]) == "dilemma"
    assert moves.gm_task_key("narrative", ["memory", "supernatural"]) == "narrative"
    assert moves.gm_task_key("narrative", []) == "narrative"


def test_gm_undefined_situation_prompt_falls_back(monkeypatch):
    monkeypatch.setattr(moves, "list_task_keys", lambda task_file: ["narrative"])
    assert moves.gm_task_key("narrative", ["social"]) == "narrative"


def test_examples_prefer_move_then_situation_then_default():
    pool = {
        "default": ["d1", "d2"],
        "disagree": ["x1"],
        "investigation": ["i1", "i2", "i3"],
    }
    assert examples_for(pool, "disagree", ["investigation"]) == ["x1"]
    assert set(examples_for(pool, "answer", ["investigation"])) <= {"i1", "i2", "i3"}
    assert sorted(examples_for(pool, "answer", ["memory"])) == ["d1", "d2"]


def test_examples_accept_a_plain_list_and_cap_at_k():
    assert len(examples_for(["a", "b", "c", "d"], "disagree", [], k=2)) == 2
    assert examples_for(None, "answer", []) == []
