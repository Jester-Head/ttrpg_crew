"""Wording for engine-built prompt lines (THIS BEAT directives, stance lines,
direct-path framing).

The engine decides WHICH lines a companion gets (beat_runner / stance.py); the
words themselves are data and live in config/directives.yaml, live-editable
like every other prompt file. The defaults below keep the game working if the
file is missing or a key is absent, same pattern as the auditor thresholds.

Templates use str.format placeholders; each key's available fields are
documented in directives.yaml.
"""

from __future__ import annotations

from ttrpg_crew.config_loader import CONFIG_DIR, load_yaml_config

_PATH = CONFIG_DIR / "directives.yaml"

_DEFAULTS: dict[str, str] = {
    "header": "THIS BEAT:",
    "direct_question": (
        '- The player is asking YOU ({name}) a direct question: "{player_input}". '
        "Answer THAT question, in character."
    ),
    "direct_address": (
        '- The player is speaking directly to YOU ({name}): "{player_input}". '
        "Respond to them."
    ),
    "question": "- The player asked a question; only chime in if you add something.",
    "action": "- The player took an action; react to it, don't narrate it.",
    "situation": "- Situation: {situations}.",
    "also_speaking": "- Also speaking this beat: {others}.",
    "already_said": "- Already said this beat:",
    "anti_echo": (
        "- Do NOT repeat, echo, or paraphrase anyone else's line or mannerisms. "
        "Say only what {name} would say, or [observes]."
    ),
    "stance_disagree_hard": (
        "- You DISAGREE with this {approach} approach, firmly, expressed the "
        "way YOUR character would."
    ),
    "stance_disagree_soft": (
        "- You have doubts about this {approach} approach, voice them the way "
        "YOUR character would."
    ),
    "stance_agree_hard": (
        "- You strongly AGREE with this {approach} approach, back them up the "
        "way YOUR character would."
    ),
    "stance_agree_soft": (
        "- You lean toward agreeing with this {approach} approach, in your own "
        "manner."
    ),
    "said_so_far": "So far this beat:",
    "respond_now": (
        "Respond now as {name}, in character, your actions and words only (or "
        "[observes]). No analysis, reasoning, or planning."
    ),
}


def directive(key: str, **fields: str) -> str:
    """The directive line for `key`, formatted; '' for unknown keys."""
    raw = load_yaml_config(_PATH).get(key)
    template = raw if isinstance(raw, str) and raw.strip() else _DEFAULTS.get(key, "")
    if not template:
        return ""
    try:
        return template.format(**fields)
    except (KeyError, IndexError):  # bad placeholder in the config file
        return _DEFAULTS.get(key, "").format(**fields)
