"""Match a character's name in free text."""

from __future__ import annotations

import re

_TITLES = frozenset(
    {
        "the", "a", "an", "old", "young",
        "sir", "lady", "lord", "dame", "master", "mistress",
        "mr", "mrs", "ms", "miss", "dr", "doctor", "captain",
        "sister", "brother", "father", "mother",
    }
)


def short_name(name: str) -> str:
    """The one word a character goes by: the first that isn't a title or article."""
    tokens = [token.strip(".,") for token in (name or "").split()]
    tokens = [token for token in tokens if token]
    for token in tokens:
        if token.casefold() not in _TITLES:
            return token
    return tokens[0] if tokens else ""


def mentions_name(text: str, name: str, *, match_case: bool = False) -> bool:
    """Whether `text` names the character, by full name or short name.

    `match_case` applies to the short name only. Prose needs it: a short name
    like "Will" or "Mark" is also an ordinary lowercase word.
    """
    name = " ".join((name or "").split())
    if not name:
        return False
    if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
        return True
    short = short_name(name)
    if len(short) < 3:
        return False
    flags = 0 if match_case else re.IGNORECASE
    return re.search(rf"\b{re.escape(short)}\b", text, flags) is not None
