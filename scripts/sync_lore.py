"""Copy the core Kavath lorebook files into knowledge/lore_private/ for retrieval,
stripping design-tracking text (status lines, open-question sections, status tags).

Usage (from the ttrpg_crew project dir):
    python scripts/sync_lore.py path/to/kavath_lorebook
    python scripts/sync_lore.py                       # path from KAVATH_LOREBOOK
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# source path inside the lorebook -> file name in lore_private/
CORE_FILES = {
    "00_history_scaffold.md": "00_history_scaffold.md",
    "design/01_fact_sheet.md": "01_fact_sheet.md",
    "03_memory_currency.md": "03_memory_currency.md",
    "05_geography.md": "05_geography.md",
    "07_factions.md": "07_factions.md",
}

_PROJECT = Path(__file__).resolve().parents[1]
_DEST = _PROJECT / "knowledge" / "lore_private"

_STATUS_LINE_RE = re.compile(r"^\*\*(?:Status|Purpose):\*\*.*\n?", re.M)
# The fact sheet opens with a note to the team about its tags, up to the first rule.
_TEAM_PREAMBLE_RE = re.compile(r"\A(# [^\n]*\n).*?\n---\n", re.S)
_TAG_RE = re.compile(r"\s*`(?:settled|leaning|open)`")
_HEADING_RE = re.compile(r"^(#+)\s*\**(.*?)\**\s*$")


def strip_meta(text: str) -> str:
    """Drop status lines, status tags, "Open questions" sections and the
    trailing see-also footer, which are team bookkeeping rather than world facts."""
    text = _STATUS_LINE_RE.sub("", text)
    if "For the team." in text.split("\n---\n", 1)[0]:
        text = _TEAM_PREAMBLE_RE.sub(r"\1", text, count=1)
    text = _TAG_RE.sub("", text)
    kept: list[str] = []
    skip_level = 0
    for line in text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_level and level <= skip_level:
                skip_level = 0
            if not skip_level and heading.group(2).strip().lower() == "open questions":
                skip_level = level
                continue
        if skip_level or line.startswith("*Not on this page"):
            continue
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def main() -> None:
    location = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAVATH_LOREBOOK", "")
    if not location:
        sys.exit("Pass the lorebook's path as an argument or set KAVATH_LOREBOOK.")
    source = Path(location)
    if not source.is_dir():
        sys.exit(f"Lorebook not found at {source}.")
    _DEST.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in CORE_FILES.items():
        src = source / src_name
        if not src.is_file():
            print(f"[lore] missing {src_name}, skipped")
            continue
        (_DEST / dest_name).write_text(strip_meta(src.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"[lore] {src_name} -> lore_private/{dest_name}")


if __name__ == "__main__":
    main()
