"""Rule-based output validator (auditor layer 1).

Cheap, instant, no LLM. Catches the structural mistakes small models make most
often — the GM puppeteering the player, option menus, companions leaking
mechanics — by inspection. These are HEURISTICS: advisory warnings, not hard
truths. They feed two things:

  * immediate visibility (printed as [audit] lines after each beat), and
  * a labeled dataset for the eventual fine-tune / LLM-editor layer.

Severity is intentionally just "warning" for now; nothing is blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ttrpg_crew.config_loader import config_section

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Auditor strictness — defaults here, overridable live via models.yaml `audit:`.
_DEFAULT_AUDIT = {
    "companion_bloat_words": 35,
    "em_dash_max": 3,
    "cluster_min": 2,
    "density_min": 3,
}


def _audit_cfg() -> dict[str, int]:
    """Resolve audit thresholds from models.yaml, with code defaults."""
    cfg = config_section("audit")
    out = dict(_DEFAULT_AUDIT)
    for key in out:
        if key in cfg:
            try:
                out[key] = int(cfg[key])
            except (TypeError, ValueError):
                pass
    return out


def _config_terms(filename: str) -> list[str]:
    """Read a curated word-list file (one term per line; '#' comments ignored).

    Loaded fresh each call so edits apply on the next beat without a restart.
    """
    path = _CONFIG_DIR / filename
    if not path.is_file():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


@dataclass
class Violation:
    rule: str
    detail: str
    suggestion: str = ""  # how to fix it, fed to the auto-retry


# Verbs that imply the player's own VOLITION (chosen action/decision/speech).
# The GM may frame the scene — including where the character is (stand, sit,
# are) — and narrate what the character perceives (see, hear, notice), since
# presenting the world is the GM's job. What the GM may not do is decide what
# the character *does*, *says*, or *wants*. Stative positioning and perception
# verbs are deliberately excluded.
_AGENCY_VERBS = (
    r"(?:feels?|felt|thinks?|thought|knows?|knew|remembers?|realizes?|decides?|"
    r"wants?|wishes?|grabs?|draws?|drew|swings?|strikes?|runs?|ran|walks?|steps?|"
    r"reaches?|says?|said|repl(?:y|ies)|shouts?|whispers?|charges?|lunges?|"
    r"attacks?|dodges?|chooses?|refuses?|agrees?)"
)

# "you <verb>" — narrating the player's chosen action, speech, or decision.
_GM_PUPPET_VERB_RE = re.compile(r"\byou\s+(?:\w+\s+){0,1}" + _AGENCY_VERBS + r"\b", re.I)
# "your <(adj )*inner-life noun>" — narrating the player's thoughts or emotions.
# Physical/body nouns are excluded: "blood on your hands" is world description.
_GM_PUPPET_POSS_RE = re.compile(
    r"\byour\s+(?:\w+\s+){0,2}"
    r"(?:mind|memor(?:y|ies)|thoughts?|feelings?|resolve|courage|will|"
    r"conscience|soul|guilt|fear|dread)\b",
    re.I,
)
# Closing "what do you do?" is allowed; don't let it trip the verb rule.
_PROMPT_RE = re.compile(r"what\s+do\s+you\s+do\s*\??", re.I)

# (A)/(B) or "A)" / "1." choice menus.
_OPTION_RE = re.compile(r"\(\s*[A-Da-d]\s*\)|(?:^|\n)\s*[A-D1-4][.)]\s+\S", re.M)

# Companion mechanics talk (specific tokens to avoid flagging "I hope/I fear").
_MECH_RE = re.compile(
    r"\b\d+d\d+\b|\bDC\s*\d+|\broll(?:s|ed|ing)?\b|\b(?:hope|fear)\s+die\b|"
    r"\bspend(?:s|ing)?\s+(?:a\s+)?(?:hope|fear|stress)\b|"
    r"\bmark(?:s|ing)?\s+(?:a\s+)?stress\b",
    re.I,
)


def _snippet(match: re.Match) -> str:
    text = match.string
    start = max(0, match.start() - 20)
    end = min(len(text), match.end() + 20)
    return "…" + text[start:end].strip() + "…"


# Verbs that mean a companion is speaking, emoting, or acting — the companion
# agent's job, not the GM's. Static presence ("stands", "waits", "is", "remains")
# is deliberately excluded so the GM can still note that they're there.
_COMPANION_VOICE_VERBS = (
    r"(?:says?|said|murmurs?|murmured|whispers?|whispered|mutters?|muttered|"
    r"asks?|asked|repl(?:y|ies|ied)|answers?|answered|adds?|added|growls?|"
    r"sighs?|hisses?|snaps?|calls?|shouts?|leans?|nods?|grins?|smiles?|"
    r"warns?|offers?|insists?|scoffs?|watches?|shifts?|considers?|studies?|"
    r"glances?|darts?|hardens?|tenses?|stiffens?|frowns?|winces?|tilts?|"
    r"regards?|flinches?|bristles?|eyes)"
)
# GM describing a companion's body/expression ("Mireille's hand trembles").
_COMPANION_BODY_NOUNS = (
    r"(?:face|eyes|gaze|voice|hands?|expression|jaw|shoulders?|brow|lips|"
    r"fingers?|head|mouth|features?)"
)


def audit_gm(
    text: str,
    *,
    human_player_name: str = "Player",
    companions: tuple[str, ...] = (),
) -> list[Violation]:
    violations: list[Violation] = []

    # GM voicing/acting ANY companion (companions speak as their own agents).
    # Collect every offending name so the editor can fix them all in one pass.
    voiced = []
    for name in companions:
        name = (name or "").strip()
        if not name:
            continue
        n = re.escape(name)
        if re.search(rf"\b{n}\b\s+(?:\w+\s+){{0,3}}{_COMPANION_VOICE_VERBS}\b", text, re.I) or \
           re.search(rf"\b{n}['’]s\s+(?:\w+\s+){{0,2}}{_COMPANION_BODY_NOUNS}\b", text, re.I) or \
           re.search(rf"\b{n}\b[^.!?\n]{{0,40}}[\"“]", text):
            voiced.append(name)
    if voiced:
        violations.append(
            Violation(
                "gm_speaks_for_companion",
                f"voices/acts the companion(s): {', '.join(voiced)}",
                suggestion=(
                    "never speak or act for companions; note their presence at "
                    "most and let them respond as their own characters"
                ),
            )
        )

    # Blank out the closing prompt so "what do you do" can't match agency verbs.
    scannable = _PROMPT_RE.sub(" ", text)

    m = _GM_PUPPET_VERB_RE.search(scannable) or _GM_PUPPET_POSS_RE.search(scannable)
    if m:
        violations.append(
            Violation(
                "player_puppeteering",
                f"narrates the player: {_snippet(m)}",
                suggestion=(
                    "describe only the world and NPCs reacting; cut any line that "
                    "decides the player's action, speech, thought, or feeling"
                ),
            )
        )

    m = _OPTION_RE.search(text)
    if m:
        violations.append(
            Violation(
                "option_menu",
                f"lists explicit choices: {_snippet(m)}",
                suggestion="remove the labeled options; imply the paths through the fiction",
            )
        )

    if "?" not in text[-120:]:
        violations.append(
            Violation(
                "missing_prompt",
                "does not end by handing the turn back (no '?')",
                suggestion="end on the present moment and hand the turn back (e.g. 'What do you do?')",
            )
        )

    return violations


def audit_companion(
    text: str,
    *,
    human_player_name: str = "Player",
) -> list[Violation]:
    violations: list[Violation] = []

    m = _MECH_RE.search(text)
    if m:
        violations.append(
            Violation(
                "mechanics_leak",
                f"references mechanics: {_snippet(m)}",
                suggestion="describe the intent in fiction only; drop dice, rolls, DCs, Hope/Fear/Stress",
            )
        )

    # Companion narrating the human player in third person (dialogue is fine).
    player_narrate = re.compile(re.escape(human_player_name) + r"\s+" + _AGENCY_VERBS, re.I)
    m = player_narrate.search(text)
    if m:
        violations.append(
            Violation(
                "speaks_for_player",
                f"narrates the player: {_snippet(m)}",
                suggestion="react to the player instead; never describe what they do or say",
            )
        )

    # Self-narration bloat: a long turn with no actual spoken line. Catches the
    # "described my voice but never said anything" case. [observes] is exempt;
    # short action-only beats are fine; quoted dialogue of any length is fine.
    stripped = text.strip()
    has_dialogue = any(q in text for q in ('"', "“", "”"))
    word_count = len(stripped.split())
    if stripped and not stripped.lower().startswith("[observes") \
            and not has_dialogue and word_count > _audit_cfg()["companion_bloat_words"]:
        violations.append(
            Violation(
                "companion_bloat",
                f"{word_count} words of self-narration with no spoken line",
                suggestion=(
                    "replace with one short spoken line in their voice; drop the "
                    "self-description. e.g. turn '(I lean in, my voice dropping to "
                    "a hum)' into '\"Easy now. We've carried worse than this.\"'"
                ),
            )
        )

    return violations


# --------------------------------------------------------------------------- #
# AI-pattern style checks (apply to GM and companions alike)
# --------------------------------------------------------------------------- #
# The AI-tell word lists live in editable config files, NOT here — see
# config/cliche_names.txt and config/cliche_diction.txt. The detection *logic*
# (token/grammar regexes) stays in code; only the curated vocabulary is data.
_NAME_TOKEN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
_EM_DASH = "—"


def _cliche_names() -> set[str]:
    return {t.lower() for t in _config_terms("cliche_names.txt")}


def _word_list_re(filename: str) -> re.Pattern | None:
    terms = _config_terms(filename)
    if not terms:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)


def _cliche_diction_re() -> re.Pattern | None:
    return _word_list_re("cliche_diction.txt")


# Structural AI tells (logic, not vocabulary). Calibrated for fiction narration.
_REVERSAL_RES = (
    re.compile(r"\bnot (?:just|only|merely)\b[^.?!\n]{0,50}?\bbut\b", re.I),
    re.compile(r"\b(?:it'?s not|it is not|isn'?t)\b[^.?!\n]{2,50}?,\s*it'?s\b", re.I),
)
_COPULA_RE = re.compile(r"\b(serves as|stands as a testament|boasts)\b", re.I)
_RHETORICAL_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:but what\b|so why\b|what does (?:this|that) mean|why does (?:this|that) matter)",
    re.I,
)
_CALIBRATION_RE = re.compile(r"\b(it'?s worth noting|interestingly|notably)\b", re.I)


def audit_style(text: str) -> list[Violation]:
    """Flag AI tells in generated text: cliché names, purple diction, em-dashes."""
    violations: list[Violation] = []
    ac = _audit_cfg()

    cliche_names = _cliche_names()
    for match in _NAME_TOKEN_RE.finditer(text):
        if match.group(1).lower() in cliche_names:
            violations.append(
                Violation(
                    "cliche_name",
                    f"AI-default name '{match.group(1)}'",
                    suggestion=(
                        "rename this NPC using a name from the campaign name bank "
                        "listed in your prompt (matched to their ancestry)"
                    ),
                )
            )
            break

    diction_re = _cliche_diction_re()
    m = diction_re.search(text) if diction_re else None
    if m:
        violations.append(
            Violation(
                "cliche_diction",
                f"AI-cliché wording: {_snippet(m)}",
                suggestion=(
                    "drop the formulaic 'a ___ of' template; use one concrete, "
                    "grounded image in the campaign's gothic register instead"
                ),
            )
        )

    # Tier 2: fine alone, suspicious in clusters — flag 2+ distinct in one output.
    tier2_re = _word_list_re("cliche_tier2.txt")
    if tier2_re:
        hits = sorted({m.group(1).lower() for m in tier2_re.finditer(text)})
        if len(hits) >= ac["cluster_min"]:
            violations.append(
                Violation(
                    "cliche_cluster",
                    f"AI-register word cluster: {', '.join(hits)}",
                    suggestion="these stack into AI voice; replace the weakest with plain words or a concrete detail",
                )
            )

    # Tier 3: common words flagged only at saturation (vague praise, no specifics).
    tier3_re = _word_list_re("cliche_tier3.txt")
    if tier3_re:
        count = len(tier3_re.findall(text))
        if count >= ac["density_min"]:
            violations.append(
                Violation(
                    "cliche_density",
                    f"{count} vague-intensifier words in one beat",
                    suggestion="replace some with specifics — a number, a name, or a concrete image",
                )
            )

    for reversal_re in _REVERSAL_RES:
        m = reversal_re.search(text)
        if m:
            violations.append(
                Violation(
                    "reversal_cliche",
                    f"'not X, but Y' construction: {_snippet(m)}",
                    suggestion="state it directly as one positive claim",
                )
            )
            break

    m = _COPULA_RE.search(text)
    if m:
        violations.append(
            Violation(
                "copula_avoidance",
                f"press-release phrasing: {_snippet(m)}",
                suggestion="use 'is'/'has' or a concrete verb",
            )
        )

    m = _RHETORICAL_RE.search(text)
    if m:
        violations.append(
            Violation(
                "rhetorical_opener",
                f"rhetorical-question filler: {_snippet(m)}",
                suggestion="cut the question; state the point or its consequence",
            )
        )

    m = _CALIBRATION_RE.search(text)
    if m:
        violations.append(
            Violation(
                "calibration_filler",
                f"telling the reader how to feel: {_snippet(m)}",
                suggestion="cut it; let the fact land on its own",
            )
        )

    em = text.count(_EM_DASH)
    if em >= ac["em_dash_max"]:
        violations.append(
            Violation(
                "em_dash_overuse",
                f"{em} em-dashes in one beat",
                suggestion="use commas, colons, or full stops instead of em-dashes",
            )
        )

    # Register break: modern/casual phrasing that breaks the period tone and
    # reads as the player stepping out of character. One is enough to flag.
    register_re = _word_list_re("register_breaks.txt")
    if register_re:
        m = register_re.search(text)
        if m:
            violations.append(
                Violation(
                    "register_break",
                    f"too-casual/modern phrasing breaks the period tone: '{m.group(1)}'",
                    suggestion=(
                        "reword in the campaign's register; this reads as out-of-"
                        "character or anachronistic for the setting"
                    ),
                )
            )

    return violations


def audit(
    label: str,
    text: str,
    *,
    human_player_name: str = "Player",
    companions: tuple[str, ...] = (),
) -> list[Violation]:
    """Dispatch: GM vs companion rules, plus shared AI-style checks."""
    if label == "GM":
        base = audit_gm(text, human_player_name=human_player_name, companions=companions)
    else:
        base = audit_companion(text, human_player_name=human_player_name)
    return base + audit_style(text)
