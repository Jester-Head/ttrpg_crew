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
        '- The player is asking YOU ({name}) a direct question: "{player_input}" '
        "Answer THAT question, in character."
    ),
    "direct_address": (
        '- The player is speaking directly to YOU ({name}): "{player_input}" '
        "Respond to them."
    ),
    "addressed_other": (
        '- {protagonist} is speaking to {other}, not you: "{player_input}" '
        "Don't answer for {other}."
    ),
    "question": '- {protagonist} asked: "{player_input}"',
    "answer_first": '{protagonist} asked you: "{player_input}" Answer that first, in character.',
    "why_you": (
        "- You're speaking because the moment involves {reason}. Say what "
        "{name} would add from that angle."
    ),
    "action": (
        '- {protagonist} just did this: "{player_input}" React to THAT. It\'s '
        "their decision: you can object, but don't undo it, repeat it, or act "
        "as if it didn't happen."
    ),
    "situation": "- Situation: {situations}.",
    "also_speaking": "- Also speaking this beat: {others}.",
    "already_said": "- Already said this beat:",
    "anti_echo": (
        "- Do NOT repeat, echo, or paraphrase anyone else's line or mannerisms. "
        "Say only what {name} would say."
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
    "party_present": "With you: {names}.",
    "said_so_far": "So far this beat:",
    "respond_now": (
        "Respond now as {name}, in character, your actions and words only. No "
        "analysis, reasoning, or planning."
    ),
    "think_first": (
        "First write one line starting with NOTE: followed by a private thought "
        "on how {name} would respond. The note is never shown. Then write SAY: "
        "followed by {name}'s actions and words only."
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
