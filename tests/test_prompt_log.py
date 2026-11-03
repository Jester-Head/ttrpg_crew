"""The per-call prompt log."""

import json

from ttrpg_crew import session_log


def test_prompt_records_messages_with_tags(tmp_path, monkeypatch):
    for name in ("_clean", "_debug", "_prompts"):
        monkeypatch.setattr(session_log, name, None)
    monkeypatch.setattr(session_log, "_tags", {})
    monkeypatch.setattr(session_log, "_log_prompts", True)
    monkeypatch.setattr(session_log, "_log_dir", tmp_path)
    session_log.set_tags(model="m", arm="direct")

    messages = [{"role": "user", "content": "Hello"}]
    session_log.prompt("sirra", messages, source="direct", model="m")

    (log,) = tmp_path.glob("*.prompts.jsonl")
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["label"] == "sirra"
    assert record["arm"] == "direct"
    assert record["messages"] == messages
    for handle in (session_log._clean, session_log._debug, session_log._prompts):
        handle.close()


def test_prompt_is_a_no_op_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(session_log, "_log_prompts", False)
    monkeypatch.setattr(session_log, "_clean", None)
    session_log.prompt("sirra", [], source="direct")
    assert session_log._clean is None
