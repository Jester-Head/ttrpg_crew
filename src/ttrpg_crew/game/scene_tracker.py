"""Scene tracker: read a finished beat into scene memory and next-move suggestions.

Pure functions (prompt in, text out, state updated in place); the LLM call is in flow.py.
"""
# Adapted from tegridydev/dnd-llm-game (IDCDW licence).

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from ttrpg_crew.config_loader import typed_section
from ttrpg_crew.game.names import mentions_name
from ttrpg_crew.game.schemas import SceneUpdate, first_json_object
from ttrpg_crew.game.state import GameState

_DEFAULTS = {
    "enabled": True,
    "suggestions": 3,
    "max_suggestion_chars": 90,
    "max_facts": 12,
    "max_npcs": 10,
    "max_memory_chars": 240,
}

# Mirrors SceneUpdate; a test keeps the two in step.
_STRING_LIST = {"type": "array", "items": {"type": "string"}}
SCENE_UPDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "location_changed": {"type": "boolean"},
        "objective": {"type": "string"},
        "objective_changed": {"type": "boolean"},
        "summary": {"type": "string"},
        "suggested_actions": _STRING_LIST,
        "new_facts": _STRING_LIST,
        "resolved_facts": _STRING_LIST,
        "npcs": _STRING_LIST,
        "checkpoint_reached": {"type": "integer"},
    },
    "required": [
        "location",
        "location_changed",
        "objective",
        "objective_changed",
        "summary",
        "suggested_actions",
        "new_facts",
        "resolved_facts",
        "npcs",
        "checkpoint_reached",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You track scene state for a solo tabletop RPG. Read the latest beat and "
    "return only JSON matching the schema. Do not continue the story."
)


def tracker_config() -> dict[str, Any]:
    """Tracker knobs from models.yaml `scene_tracker:`; negative numbers read as 0."""
    cfg = typed_section("scene_tracker", _DEFAULTS)
    return {
        key: max(0, value) if isinstance(value, int) and not isinstance(value, bool) else value
        for key, value in cfg.items()
    }


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "(none)"


def _suggestion_rule(count: int, max_chars: int, player: str) -> str:
    if count <= 0:
        return "an empty list."
    if count == 3:
        distinct = (
            " Make them distinct: one direct, one social or investigative, one "
            "cautious or unexpected."
        )
    elif count > 1:
        distinct = " Make each a different kind of move."
    else:
        distinct = ""
    things = "thing" if count == 1 else "things"
    return (
        f"{count} short {things} {player} could do next, each under {max_chars} "
        "characters, written as a command in the player's voice (for example "
        '"Ask the ferryman who paid him"). Ground each in what the beat shows.'
        f"{distinct} Never repeat the action just taken, ask for a dice roll, or "
        f"decide how an action turns out. {player} is the player: never suggest "
        f"talking to, following, or helping {player}."
    )


def scene_update_messages(
    *,
    beat_text: str,
    state: GameState,
    player_input: str,
    protagonist_name: str,
    party_names: list[str],
    suggestions: int,
    max_suggestion_chars: int = _DEFAULTS["max_suggestion_chars"],
    checkpoints: list[str] | None = None,
) -> tuple[str, str]:
    """Build (system, user) for the tracker call."""
    player = protagonist_name or "the player"
    action = player_input.strip() or "(opening scene; no action yet)"
    party = ", ".join(n for n in (protagonist_name, *party_names) if n) or "None"
    # The current checkpoint and the ones after it, so a player who goes
    # straight to a later one still moves the story on.
    if checkpoints:
        listed = "\n".join(f"  {n}. {when}" for n, when in enumerate(checkpoints, start=1))
        checkpoint_question = (
            "the number of the furthest of these this beat clearly did, or 0 if "
            f"none:\n{listed}"
        )
    else:
        checkpoint_question = "0."
    user = f"""
Current state (keep it unless the beat explicitly changes it):
Location: {state.location or "(not set)"}
Objective: {state.party_goal or "(not set)"}
Established facts:
{_bullets(state.known_facts)}
Named NPCs:
{_bullets(state.active_npcs)}

{player} (the human player) just did: {action}

Latest beat:
<beat>
{beat_text.strip()}
</beat>

Party (never list these as NPCs): {party}

Return:
- location: a concrete 2-6 word place name. Set location_changed true only if
  the beat moves the party somewhere new; otherwise repeat the current location.
- objective: the party's immediate goal in 4-14 words. Set objective_changed
  true only if the beat changes it; otherwise repeat the current objective. If
  none is set and the beat gives none, return an empty string.
- summary: one plain sentence on where things stand right now.
- suggested_actions: {_suggestion_rule(suggestions, max_suggestion_chars, player)}
- new_facts: only durable facts this beat established (a name, a debt, a
  discovery, a promise). Not mood, weather, or scenery. A line that starts
  with a party member's name is that character talking, and what they claim
  is not a fact unless the narration shows it.
- resolved_facts: existing facts, copied exactly, that this beat made untrue
  or finished.
- npcs: named non-party characters this beat introduced or changed, each as
  "Name: who they are, in a few words".
- checkpoint_reached: {checkpoint_question}
""".strip()
    return SYSTEM_PROMPT, user


def parse_scene_update(raw: str) -> SceneUpdate | None:
    """Parse the tracker's reply; None when it holds no usable update."""
    data = first_json_object(raw or "")
    if data is None or not data.keys() & SceneUpdate.model_fields.keys():
        return None
    try:
        return SceneUpdate.model_validate(data)
    except ValidationError:
        return None


# Leading list markers a model adds anyway: "1.", "2)", "(a)", "-", "*", "•".
_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)]|\(?[A-Da-d][.)])\s*")
_DOUBLE_QUOTES = "\"“”"
_QUOTES = _DOUBLE_QUOTES + "'‘’"

_STATS = (
    "initiative|strength|dexterity|constitution|intelligence|wisdom|charisma|"
    "acrobatics|animal handling|arcana|athletics|deception|history|insight|"
    "intimidation|investigation|medicine|nature|perception|performance|"
    "persuasion|religion|sleight of hand|stealth|survival"
)
# Roll requests only: "Roll the barrel aside" and "Join the dice game" are actions.
_ROLL_RE = re.compile(
    r"\bd20\b|\bDC\s*\d|\bsaving throw\b"
    rf"|\b(?:{_STATS}|skill|ability|attack)\s+(?:check|roll|save)\b"
    rf"|\broll(?:s|ing)?\s+(?:for|to\s+hit|(?:an?\s+|your\s+)?(?:{_STATS}|check|save))\b",
    re.IGNORECASE,
)
# A mid-line ellipsis or the word "option" in a sentence is ordinary prose.
_PLACEHOLDER_RE = re.compile(
    r"^(?:\.\.\.|…)|(?:\.\.\.|…)$|<[^>]*>|^[Oo]ption\b|\b[Oo]ption\s*(?:\d+|[A-D])\b"
)


def _unwrap(text: str) -> str:
    """Strip the list markers and wrapping quotes a model puts around a suggestion."""
    while True:
        inner = _MARKER_RE.sub("", text).strip()
        if len(inner) >= 2 and inner[0] in _QUOTES and inner[-1] in _QUOTES:
            inner = inner[1:-1].strip()
        if inner == text:
            break
        text = inner
    if sum(text.count(quote) for quote in _DOUBLE_QUOTES) == 1:
        text = text.strip(_DOUBLE_QUOTES).strip()
    return text


def clean_suggestions(
    values: list[str],
    *,
    protagonist_name: str,
    player_input: str,
    limit: int,
    max_chars: int = _DEFAULTS["max_suggestion_chars"],
) -> list[str]:
    """Deterministic last pass over the model's suggested actions.

    Drops list markers, wrapping quotes, placeholders, roll requests, the
    action just taken, overlong or near-empty lines, duplicates, and anything
    that names the protagonist as someone else in the room.
    """
    cleaned: list[str] = []
    if limit <= 0:
        return cleaned
    taken = " ".join(player_input.split()).casefold().rstrip(".!")
    seen: set[str] = set()
    for value in values:
        text = _unwrap(" ".join(str(value).split()))
        key = text.casefold().rstrip(".!")
        if (
            len(text) < 3
            or len(text) > max_chars
            or key in seen
            or key == taken
            or _ROLL_RE.search(text)
            or _PLACEHOLDER_RE.search(text)
            or mentions_name(text, protagonist_name, match_case=True)
        ):
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


_PLACEHOLDERS = frozenset({"", "none", "none yet", "unknown", "n/a", "null", "not set", "tbd"})


def _tidy(value: str, max_chars: int) -> str:
    """Whitespace-collapsed text; "" for a placeholder or anything over max_chars."""
    text = " ".join(str(value).split())
    if len(text) > max_chars or text.casefold().strip(" .()-") in _PLACEHOLDERS:
        return ""
    return text


def _newest(items: list[str], count: int) -> list[str]:
    # items[-0:] is the whole list, so a cap of zero needs its own branch.
    return items[-count:] if count > 0 else []


def _npc_name(entry: str) -> str:
    """The name part of "Name: description" (or "Name - ...", "Name (...)")."""
    return re.split(r"\s*[:(—–]\s*|\s+-\s+", entry, maxsplit=1)[0].strip()


def apply_scene_update(
    state: GameState,
    update: SceneUpdate,
    *,
    protagonist_name: str,
    party_names: list[str],
    player_input: str,
    tagged_location: bool,
    config: dict[str, Any] | None = None,
) -> None:
    """Fold one tracker update into the game state, in place.

    A [LOCATION:] tag the GM wrote this beat outranks the tracker's reading,
    and location/objective only move when the tracker flags a change.
    """
    cfg = config or tracker_config()
    cap = cfg["max_memory_chars"]

    location = _tidy(update.location, cap)
    if location and update.location_changed and not tagged_location:
        state.location = location

    objective = _tidy(update.objective, cap)
    if objective and (update.objective_changed or not state.party_goal):
        state.party_goal = objective

    summary = _tidy(update.summary, cap)
    if summary:
        state.scene_summary = summary

    resolved = {" ".join(fact.split()).casefold() for fact in update.resolved_facts}
    facts = [fact for fact in state.known_facts if fact.casefold() not in resolved]
    known = {fact.casefold() for fact in facts}
    for fact in update.new_facts:
        fact = _tidy(fact, cap)
        if fact and fact.casefold() not in known:
            facts.append(fact)
            known.add(fact.casefold())
    state.known_facts = _newest(facts, cfg["max_facts"])

    party = [name for name in (protagonist_name, *party_names) if name]
    npcs = list(state.active_npcs)
    for entry in update.npcs:
        entry = _tidy(entry, cap)
        name = _npc_name(entry)
        if not name or any(mentions_name(name, member) for member in party):
            continue
        # Re-appending keeps the list in last-mentioned order, so the cap
        # drops whoever has gone longest without one.
        npcs = [kept for kept in npcs if _npc_name(kept).casefold() != name.casefold()]
        npcs.append(entry)
    state.active_npcs = _newest(npcs, cfg["max_npcs"])

    state.suggested_actions = clean_suggestions(
        update.suggested_actions,
        protagonist_name=protagonist_name,
        player_input=player_input,
        limit=cfg["suggestions"],
        max_chars=cfg["max_suggestion_chars"],
    )


def format_scene_memory(state: GameState) -> str:
    """The tracked memory as a compact prompt block ("" before anything is known)."""
    lines: list[str] = []
    if state.party_goal:
        lines.append(f"Objective: {state.party_goal}")
    if state.scene_summary:
        lines.append(f"Right now: {state.scene_summary}")
    if state.known_facts:
        lines.append("Established facts: " + "; ".join(state.known_facts))
    if state.active_npcs:
        lines.append("Named NPCs met: " + "; ".join(state.active_npcs))
    return "\n".join(lines)


_CHOICE_RE = re.compile(r"#?\s*(\d{1,3})\s*[.)]?", re.ASCII)


def pick_suggestion(player_input: str, suggestions: list[str]) -> str | None:
    """Resolve what the player typed against this beat's numbered suggestions.

    A bare number picks that suggestion; a number with no matching suggestion
    returns None so the caller can ask again. Anything else passes through.
    """
    text = player_input.strip()
    choice = _CHOICE_RE.fullmatch(text)
    if choice is None:
        return text
    number = int(choice.group(1))
    return suggestions[number - 1] if 1 <= number <= len(suggestions) else None
