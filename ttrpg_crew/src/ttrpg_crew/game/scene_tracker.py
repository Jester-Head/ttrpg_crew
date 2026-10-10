"""Scene tracker: read a finished beat into scene state and next-move options.

After each beat, one small structured-output call reads the narration and
returns where the party is, what they're after, the durable facts and named
NPCs it established, and a few concrete things the player could do next.
Adapted from tegridydev/dnd-llm-game's world-update pass (IDCDW licence).

The suggested actions are the point for play: the player can pick one by
number instead of composing a free-form reply. They live OUTSIDE the prose
(Beat.suggested_actions, shown by the CLI under the hand-back prompt), so the
GM's "no option menus" rule still holds for the narration itself.

Everything here is pure (prompt in, text out, state updated in place); the
LLM call lives in flow.py, so this module tests without a model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from ttrpg_crew.config_loader import config_section
from ttrpg_crew.game.schemas import SceneUpdate
from ttrpg_crew.game.state import GameState

_DEFAULTS = {
    "enabled": True,
    "suggestions": 3,  # how many next moves to ask for (2-4 reads well)
    "max_suggestion_chars": 90,
    "max_facts": 12,  # oldest facts drop off first
    "max_npcs": 10,
}

# Strict JSON schema for the server's structured-output mode. Mirrors
# SceneUpdate (a test keeps the two in step).
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
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You track scene state for a solo tabletop RPG. Read the latest beat and "
    "return only JSON matching the schema. Do not continue the story."
)


def tracker_config() -> dict[str, Any]:
    """Resolve tracker knobs from models.yaml `scene_tracker:`, with defaults."""
    cfg = config_section("scene_tracker")
    out = dict(_DEFAULTS)
    for key, default in _DEFAULTS.items():
        if key in cfg:
            try:
                out[key] = type(default)(cfg[key])
            except (TypeError, ValueError):
                pass
    return out


def _aliases(name: str) -> list[str]:
    """A character's full name and first name (3+ letters), for matching."""
    name = (name or "").strip()
    first = name.split()[0] if name else ""
    return [alias for alias in dict.fromkeys((name, first)) if len(alias) >= 3]


def _mentions(text: str, name: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE) for alias in _aliases(name)
    )


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "None yet"


def scene_update_messages(
    *,
    beat_text: str,
    state: GameState,
    player_input: str,
    protagonist_name: str,
    party_names: list[str],
    suggestions: int,
) -> tuple[str, str]:
    """Build (system, user) for the tracker call."""
    player = protagonist_name or "the player"
    action = player_input.strip() or "(opening scene; no action yet)"
    party = ", ".join(n for n in (protagonist_name, *party_names) if n) or "None"
    user = f"""
Current state (keep it unless the beat explicitly changes it):
Location: {state.location or "Unknown"}
Objective: {state.party_goal or "None yet"}
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
  true only if the beat changes it; otherwise repeat the current objective.
- summary: one plain sentence on where things stand right now.
- suggested_actions: {suggestions} short things {player} could do next, each
  under 90 characters, written as a command in the player's voice (for example
  "Ask the ferryman who paid him"). Ground each in what the beat shows. Make
  them distinct: one direct, one social or investigative, one cautious or
  unexpected. Never repeat the action just taken, ask for a dice roll, or
  decide how an action turns out. {player} is the player: never suggest
  talking to, following, or helping {player}.
- new_facts: only durable facts this beat established (a name, a debt, a
  discovery, a promise). Not mood, weather, or scenery.
- resolved_facts: existing facts, copied exactly, that this beat made untrue
  or finished.
- npcs: named non-party characters this beat introduced or changed, each as
  "Name: who they are, in a few words".
""".strip()
    return SYSTEM_PROMPT, user


def parse_scene_update(raw: str) -> SceneUpdate | None:
    """Parse the tracker's reply; None when there is no usable JSON object.

    Tolerates code fences and prose around the object (the plain-call
    fallback path, when the server ignored structured output).
    """
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return SceneUpdate.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None


# Leading list markers a model adds anyway: "1.", "2)", "(a)", "-", "*", "•".
_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)]|\(?[A-Da-d][.)])\s*")
# Asking for a roll is the engine's call, not a menu item. Narrow on purpose:
# "Roll the barrel aside" is an action; "Roll for Perception" is not.
_ROLL_RE = re.compile(
    r"\b(?:dice|d20|saving throw|skill check|ability check)\b"
    r"|\broll(?:s|ing)?\s+(?:a|an|for|to)\b|\bDC\s*\d",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"\.\.\.|…|<[^>]*>|\boption\b", re.IGNORECASE)


def clean_suggestions(
    values: list[str],
    *,
    protagonist_name: str,
    player_input: str,
    limit: int,
    max_chars: int = 90,
) -> list[str]:
    """Deterministic last pass over the model's suggested actions.

    Drops what a small model gets wrong most often: list markers, quotes,
    placeholders, roll talk, the action just taken, near-empty or overlong
    lines, duplicates, and anything that treats the protagonist as someone
    else in the room.
    """
    taken = " ".join(player_input.split()).casefold()
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _MARKER_RE.sub("", " ".join(str(value).split()))
        text = text.strip(" \"'“”‘’")
        key = text.casefold().rstrip(".!")
        if (
            len(text) < 4
            or len(text) > max_chars
            or key in seen
            or key == taken.rstrip(".!")
            or _ROLL_RE.search(text)
            or _PLACEHOLDER_RE.search(text)
            or _mentions(text, protagonist_name)
        ):
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _npc_key(entry: str) -> str:
    """The name part of "Name: description" (or "Name - ...", "Name (...)")."""
    return re.split(r"\s*[:(—–]\s*|\s+-\s+", entry, maxsplit=1)[0].strip().casefold()


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

    A [LOCATION:] tag the GM wrote this beat outranks the tracker's guess, and
    the tracker only moves location/objective when it flags a real change, so a
    vague reply can't erase what the state already knows.
    """
    cfg = config or tracker_config()

    location = " ".join(update.location.split())
    if location and update.location_changed and not tagged_location:
        state.location = location

    objective = " ".join(update.objective.split())
    if objective and (update.objective_changed or not state.party_goal):
        state.party_goal = objective

    summary = " ".join(update.summary.split())
    if summary:
        state.scene_summary = summary

    resolved = {" ".join(f.split()).casefold() for f in update.resolved_facts}
    facts = [f for f in state.known_facts if f.casefold() not in resolved]
    known = {f.casefold() for f in facts}
    for fact in update.new_facts:
        fact = " ".join(fact.split())
        if fact and fact.casefold() not in known:
            facts.append(fact)
            known.add(fact.casefold())
    state.known_facts = facts[-cfg["max_facts"]:]

    party = [n for n in (protagonist_name, *party_names) if n]
    npcs = list(state.active_npcs)
    for entry in update.npcs:
        entry = " ".join(entry.split())
        key = _npc_key(entry)
        if not key or any(_mentions(key, name) for name in party):
            continue
        existing = next((i for i, e in enumerate(npcs) if _npc_key(e) == key), None)
        if existing is None:
            npcs.append(entry)
        else:
            npcs[existing] = entry
    state.active_npcs = npcs[-cfg["max_npcs"]:]

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


def pick_suggestion(player_input: str, suggestions: list[str]) -> str:
    """Turn a bare number into the matching suggested action; else pass through."""
    text = player_input.strip()
    if text.isdigit() and 1 <= int(text) <= len(suggestions):
        return suggestions[int(text) - 1]
    return text
