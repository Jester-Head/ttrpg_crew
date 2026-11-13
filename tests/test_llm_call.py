"""Tests for json_completion's attempt order and failure handling."""

import sys
from types import SimpleNamespace

import pytest

from ttrpg_crew import session_log
from ttrpg_crew.services import llm_call


class ReadTimeout(Exception):
    pass


def _reply(text, finish_reason="stop"):
    message = SimpleNamespace(content=text)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage
    )


@pytest.fixture
def harness(monkeypatch):
    """Fake litellm fed from a script of replies/exceptions; collects calls and log lines."""
    script: list = []
    calls: list[dict] = []
    log: list[str] = []

    def completion(**kwargs):
        calls.append(kwargs)
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setattr(session_log, "diag", log.append)
    return SimpleNamespace(script=script, calls=calls, log=log)


def _llm(call=None, **attrs):
    return SimpleNamespace(
        model="openai/local", base_url="http://x", api_key="k", temperature=0.3, call=call, **attrs
    )


def _ask(llm):
    return llm_call.json_completion(llm, "sys", "user", schema={"type": "object"}, name="t")


def test_structured_reply_is_returned_with_the_llm_settings(harness):
    harness.script.append(_reply('{"a": 1}'))
    assert _ask(_llm(max_tokens=50)) == '{"a": 1}'
    assert len(harness.calls) == 1
    assert harness.calls[0]["response_format"]["type"] == "json_schema"
    assert harness.calls[0]["max_tokens"] == 50
    assert harness.calls[0]["temperature"] == 0.3
    assert "extra_body" not in harness.calls[0]


def test_the_thinking_switch_is_forwarded(harness):
    harness.script.append(_reply('{"a": 1}'))
    llm = _llm(additional_params={"extra_body": {"reasoning_effort": "none"}})
    assert _ask(llm) == '{"a": 1}'
    assert harness.calls[0]["extra_body"] == {"reasoning_effort": "none"}


def test_rejected_structured_call_retries_plain_and_logs_the_error(harness):
    harness.script += [ValueError("response_format unsupported"), _reply('{"a": 1}')]
    assert _ask(_llm()) == '{"a": 1}'
    assert "response_format" not in harness.calls[1]
    assert any("structured call failed (ValueError" in line for line in harness.log)


def test_empty_structured_reply_retries_plain(harness):
    harness.script += [_reply(""), _reply('{"a": 1}')]
    assert _ask(_llm()) == '{"a": 1}'
    assert len(harness.calls) == 2


def test_timeout_ends_the_attempts(harness):
    harness.script += [ReadTimeout("slow"), _reply('{"a": 1}')]
    assert _ask(_llm(call=lambda messages: "unused")) == ""
    assert len(harness.calls) == 1


def test_falls_back_to_llm_call_when_both_calls_raise(harness):
    harness.script += [ValueError("bad route"), ValueError("bad route")]
    assert _ask(_llm(call=lambda messages: ' {"a": 1} ')) == '{"a": 1}'


def test_returns_empty_and_logs_when_everything_fails(harness):
    def broken(messages):
        raise RuntimeError("down")

    harness.script += [ValueError("bad route"), ValueError("bad route")]
    assert _ask(_llm(call=broken)) == ""
    assert sum("failed" in line for line in harness.log) == 3


def test_two_empty_replies_do_not_reach_llm_call(harness):
    def unexpected(messages):
        raise AssertionError("llm.call should not run")

    harness.script += [_reply(""), _reply("")]
    assert _ask(_llm(call=unexpected)) == ""


def test_truncated_reply_is_logged(harness):
    harness.script.append(_reply('{"a":', finish_reason="length"))
    assert _ask(_llm()) == '{"a":'
    assert any("cut off" in line for line in harness.log)
