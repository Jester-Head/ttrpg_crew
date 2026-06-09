from __future__ import annotations

import re
from typing import Any

from ttrpg_crew.game.schemas import ModeratorSelection

# --------------------------------------------------------------------------- #
# Deterministic speaker selection (ported from the old AutoGen SpotlightManager)
# --------------------------------------------------------------------------- #
# A coded selector — no LLM call. Faster and more consistent than asking a small
# model "who speaks next" every beat. Companion selection metadata lives in
# characters.yaml under `selection:` (talkativeness / expertise / triggers); the
# weights below are tunable.

_W_TALKATIVENESS = 0.30   # base propensity to speak
_W_EXPERTISE = 0.40       # the scene matches what this companion cares about
_W_TRIGGER = 0.30         # a keyword they react to appears in the beat
_W_SILENCE = 0.20         # boost a companion who has been quiet
_RECENT_PENALTY = 0.30    # damp a companion who just spoke
_SELECT_THRESHOLD = 0.45  # minimum score to be selected
_MAX_SPEAKERS = 2         # never more than this per beat

# Scene-type keyword map. Used to detect which "situations" the current beat
# implies, so a companion's `expertise` tags can match. Tailored to Gloamvere.
_SITUATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "combat": ("attack", "fight", "blade", "strike", "enemy", "guard", "ambush", "draw", "shoot", "flee"),
    "social": ("talk", "persuade", "convince", "noble", "reputation", "plead", "lie", "charm", "weep", "comfort", "beg"),
    "investigation": ("search", "examine", "look", "inspect", "study", "clue", "trace", "lock", "pick", "sneak", "hide", "steal", "vault", "quiet"),
    "negotiation": ("deal", "bargain", "price", "pay", "trade", "offer", "debt", "owe", "coin"),
    "moral_dilemma": ("choice", "decide", "spare", "betray", "sacrifice", "refuse", "grief", "mercy", "help", "save", "suffer"),
    "memory": ("memory", "tinctura", "vial", "extract", "refine", "calcine", "fray", "dreg", "quintessa"),
    "supernatural": ("veil", "spirit", "spiritus", "thin spot", "drift", "ghost", "rite"),
}


def _selection_meta(profile: dict[str, Any]) -> dict[str, Any]:
    sel = profile.get("selection") or {}
    return {
        "talkativeness": float(sel.get("talkativeness", 0.5)),
        "expertise": [str(e).lower() for e in (sel.get("expertise") or [])],
        "triggers": [str(t).lower() for t in (sel.get("triggers") or [])],
    }


def _detect_situations(text: str) -> set[str]:
    return {
        situation
        for situation, keywords in _SITUATION_KEYWORDS.items()
        if any(kw in text for kw in keywords)
    }


def _score_companion(meta: dict[str, Any], recent_turns: int, text: str, situations: set[str]) -> float:
    score = meta["talkativeness"] * _W_TALKATIVENESS
    if situations & set(meta["expertise"]):
        score += _W_EXPERTISE
    if any(trigger in text for trigger in meta["triggers"]):
        score += _W_TRIGGER
    if recent_turns <= 0:
        score += _W_SILENCE
    elif recent_turns >= 2:
        score -= _RECENT_PENALTY
    return max(0.0, score)


_OPENING_STATES = {"opening", "opening_scene", "session_start", "session_resume", "recap"}


def select_speakers(
    *,
    character_profiles: dict[str, dict[str, Any]],
    game_state: Any,
    player_input: str,
    recent_story: str,
    available_speakers: list[str],
    scene_type: str,
    human_player_name: str = "Player",
) -> ModeratorSelection:
    """Pick who speaks this beat, deterministically (no LLM call).

    Rules, in order:
      * Opening / resume beats: GM frames alone (0 companions).
      * The player named a companion: that companion answers directly.
      * Otherwise: score every companion (talkativeness + expertise + triggers +
        recency) and take those above threshold, capped at 2. If the player took
        a real action and nobody clears the bar, the single best-fit reacts.
    """
    if scene_type in _OPENING_STATES:
        return ModeratorSelection(
            selected_speakers=[],
            gm_should_frame_first=True,
            gm_intent="Set the scene and let the player act first.",
        )

    player_lower = player_input.lower()

    # Direct address — the player named a companion (by first name).
    for cid in available_speakers:
        name = str(character_profiles.get(cid, {}).get("name") or cid)
        first = name.split()[0].lower()
        if first and re.search(rf"\b{re.escape(first)}\b", player_lower):
            return ModeratorSelection(
                selected_speakers=[cid],
                gm_should_frame_first=False,
                gm_intent=f"{name} answers the player directly.",
            )

    # Weight the player's current action heavily; trail off into recent story.
    text = f"{player_lower}\n{recent_story[-600:].lower()}"
    situations = _detect_situations(text)

    scored: list[tuple[str, float]] = []
    for cid in available_speakers:
        meta = _selection_meta(character_profiles.get(cid, {}))
        state = game_state.characters.get(cid)
        recent_turns = getattr(state, "recent_turns", 0) if state else 0
        scored.append((cid, _score_companion(meta, recent_turns, text, situations)))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    chosen = [cid for cid, score in scored if score >= _SELECT_THRESHOLD][:_MAX_SPEAKERS]

    # On a substantive player action, make sure at least one companion reacts.
    if not chosen and player_input.strip() and scored:
        chosen = [scored[0][0]]

    return ModeratorSelection(
        selected_speakers=chosen,
        gm_should_frame_first=True,
        gm_intent="React to the player's action and move the scene forward.",
    )


def calculate_soft_speaker_weights(game_state, transcript: str, available_speakers: list[str]) -> dict:
    """
    Soft guardrails for the hidden LLM moderator.

    These scores are advisory, not hard rules.
    The moderator can override them when fiction, pacing, or drama calls for it.
    """

    weights = {}

    for speaker_id in available_speakers:
        character = game_state.characters[speaker_id]

        spotlight_need = max(0.0, 1.0 - character.recent_turns * 0.25)
        recent_dominance_penalty = min(1.0, character.recent_turns * 0.30)

        # TODO: Replace with actual scene relevance logic.
        scene_relevance = 0.50

        overall_score = (
            spotlight_need * 0.35
            + scene_relevance * 0.45
            - recent_dominance_penalty * 0.20
        )

        weights[speaker_id] = {
            "overall_score": round(overall_score, 3),
            "spotlight_need": round(spotlight_need, 3),
            "scene_relevance": round(scene_relevance, 3),
            "recent_dominance_penalty": round(recent_dominance_penalty, 3),
            "recent_turns": character.recent_turns,
            "total_turns": character.total_turns,
            "reasons": [
                f"Recent turns: {character.recent_turns}",
                f"Total turns: {character.total_turns}",
            ],
        }

    return weights
