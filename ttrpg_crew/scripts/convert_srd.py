"""Convert the D&D SRD PDF into the Markdown rules corpus in knowledge/rules/srd/.

Download the PDF from https://www.dndbeyond.com/srd, then run (from the
ttrpg_crew project dir):

    pip install -e ".[srd]"          # pymupdf, only needed for this script
    python scripts/convert_srd.py path/to/SRD_CC_v5.2.1.pdf

It writes one file per chapter and one per class (class_wizard.md, ...),
removes generated files the new SRD no longer has, and leaves the folder's
README.md alone. The game re-indexes the changed rules on its next start. If
the SRD version changed, update the attribution in README.md and NOTICE.md.

Off-the-shelf PDF-to-Markdown tools scramble this PDF's two-column layout
(tables from neighbouring columns interleave), hence the custom pass. Layout
facts it relies on, measured from SRD 5.2.1: 594x783 pages, two columns split
at x~297, running footer at y>=740. Fonts: chapter 26pt, section 18pt, sub
14/14.8pt, feature 12pt GillSans-SemiBold, stat-block parts 12pt GillSans
regular, table captions 10.5pt SemiBold, table cells GillSans 9.2-9.5pt,
sidebars GillSans 9pt, body Cambria 10 / Optima 9-9.5 (stat blocks), monster
ability rows GillSans 9.8 + SC700. A future SRD release with a new layout will
need these re-measured; diff the output against the old files before committing.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

GUTTER = 297.0
FOOTER_Y = 740.0
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
HEADING_MARKS = {"chapter": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####"}
ABIL_RE = re.compile(r"(Str|Dex|Con|Int|Wis|Cha)\s+(\d+)\s+([+−-]\d+)\s+([+−-]\d+)")


@dataclass
class Span:
    text: str
    font: str
    size: float
    x0: float = 0.0
    x1: float = 0.0

    @property
    def bold(self) -> bool:
        return "Bold" in self.font

    @property
    def italic(self) -> bool:
        return "Italic" in self.font


@dataclass
class Line:
    spans: list[Span]
    x0: float
    x1: float
    base: float  # baseline y
    kind: str = "body"
    band: int = -1  # id of the full-width table this cell belongs to

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def side(self) -> str:
        return "L" if self.x0 < GUTTER else "R"

    @property
    def crosses(self) -> bool:
        return self.x0 < GUTTER - 7 and self.x1 > GUTTER + 13


def clean(t: str) -> str:
    for k, v in LIGATURES.items():
        t = t.replace(k, v)
    return t.replace("\xa0", " ").replace("\t", " ")


def classify(ln: Line) -> str:
    sp = [s for s in ln.spans if s.text.strip()]
    if not sp:
        return "skip"
    f, size = sp[0].font, max(s.size for s in sp)
    gill = f.startswith("GillSans")
    if size >= 22:
        return "chapter"
    if 17 <= size < 20:
        return "h2"
    if 13.5 <= size < 16:
        return "h3"
    if "Bold-SC700" in f and size >= 10.5:
        return "sidebar_title"
    if any("SC700" in s.font for s in sp) or (gill and 9.7 <= size <= 9.9):
        return "ability"
    if gill and size <= 6.5:
        return "skip"  # MOD / SAVE column labels in stat blocks
    if gill and 11.5 <= size < 13:
        return "h4" if "SemiBold" in f else "h5"
    if gill and 10.2 <= size < 11:
        return "caption"
    if gill and 9.9 < size < 10.2:
        return "callout"
    if gill and 9.1 <= size <= 9.6:
        return "cell"
    if gill and size < 9.1:
        return "sidebar"
    return "body"


def page_lines(page) -> list[Line]:
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES
    out: list[Line] = []
    for b in page.get_text("dict", flags=flags)["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b["lines"]:
            if abs(ln["dir"][0]) < 0.99:
                continue  # rotated decorative text
            # Some PDF "lines" hold a whole table column: split by baseline.
            rows: list[tuple[float, list]] = []
            for s in ln["spans"]:
                y = s["origin"][1]
                row = next((r for r in rows if abs(r[0] - y) <= 1.5), None)
                if row is None:
                    rows.append((y, [s]))
                else:
                    row[1].append(s)
            for base, raw in rows:
                spans = [Span(clean(s["text"]), s["font"], s["size"], s["bbox"][0], s["bbox"][2]) for s in raw]
                if not "".join(s.text for s in spans).strip() or base >= FOOTER_Y:
                    continue
                real = [s for s in spans if s.text.strip()]
                line = Line(spans, real[0].x0, real[-1].x1, base)
                line.kind = classify(line)
                if line.kind != "skip":
                    out.append(line)
    return out


# --------------------------------------------------------------------------- #
# Reading order: full-width bands split the page; columns read L then R.
# --------------------------------------------------------------------------- #
@dataclass
class Band:
    top: float
    bottom: float
    lines: list[Line] = field(default_factory=list)


def cluster(lines: list[Line], gap: float = 19.0) -> list[list[Line]]:
    groups: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: l.base):
        if groups and ln.base - groups[-1][-1].base <= gap:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    return groups


def order_page(lines: list[Line]) -> list[Line]:
    bands: list[Band] = []
    claimed: set[int] = set()

    # Full-width tables: a cell cluster where some cell crosses the gutter, or
    # a left and a right cluster that start and end together (one table split
    # by the gutter, e.g. a class table's spell-slot columns).
    cells = [l for l in lines if l.kind == "cell"]
    wide: list[list[Line]] = [g for g in cluster(cells) if any(c.crosses for c in g)]
    left = cluster([c for c in cells if c.side == "L" and not c.crosses])
    right = [c for c in cells if c.side == "R"]
    def header(group: list[Line]) -> list[Line]:
        top = min(c.base for c in group)
        return [c for c in group if abs(c.base - top) <= 3]

    def header_text(group: list[Line]) -> list[str]:
        return sorted(c.text.strip().lower() for c in header(group))

    def bold_header(group: list[Line]) -> bool:
        return all(any(s.bold for s in c.spans if s.text.strip()) for c in header(group))

    for lg in left:
        rg = [c for c in right if lg[0].base - 15 <= c.base <= lg[-1].base + 3]
        if not (len(lg) > 4 and len(rg) > 4):
            continue
        if min(c.base for c in rg) > lg[0].base + 15:
            continue  # halves must start together
        if not any(-26 <= c.base - lg[-1].base <= 3 for c in rg):
            continue  # ... and end together
        if header_text(lg) == header_text(rg):
            continue  # a long list continued into the right column, not one wide table
        if not (bold_header(lg) and bold_header(rg)):
            continue  # both halves of a real wide table open with a bold header row
        wide.append(sorted(lg + rg, key=lambda l: l.base))
    seen: set[int] = set()
    for group in sorted(wide, key=lambda g: g[0].base):
        if id(group[0]) in seen or any(id(c) in claimed for c in group):
            continue
        seen.add(id(group[0]))
        top, bottom = group[0].base, group[-1].base
        # A real full-width table has nothing else beside it.
        if any(l.kind not in ("cell", "caption") and top - 1 <= l.base <= bottom + 1 for l in lines):
            continue
        members = [l for l in cells if top - 1 <= l.base <= bottom + 1]
        for l in lines:  # absorb a caption sitting just above the table
            if l.kind == "caption" and 0 < top - l.base <= 28:
                members.append(l)
                top = min(top, l.base)
        bands.append(Band(top, bottom, members))
        claimed.update(id(m) for m in members)
        for m in members:
            if m.kind == "cell":
                m.band = id(group[0])

    # Any other gutter-crossing line is a band of its own. (Chapter titles sit
    # in the left column only; the right column often runs from the page top.)
    for l in lines:
        if id(l) not in claimed and l.crosses and l.kind != "cell":
            bands.append(Band(l.base, l.base, [l]))
            claimed.add(id(l))

    bands.sort(key=lambda b: b.top)
    rest = [l for l in lines if id(l) not in claimed]
    ordered: list[Line] = []
    prev = -1e9
    for band in bands + [Band(1e9, 1e9)]:
        seg = [l for l in rest if prev < l.base < band.top]
        for side in ("L", "R"):
            ordered += sorted((l for l in seg if l.side == side), key=lambda l: (l.base, l.x0))
        ordered += sorted(band.lines, key=lambda l: (l.base, l.x0))
        # Safety net: never drop a line that sits inside a band's height.
        ordered += sorted((l for l in rest if band.top <= l.base <= band.bottom), key=lambda l: (l.side, l.base))
        prev = band.bottom
    assert len(ordered) == len(lines), (len(ordered), len(lines))
    return ordered


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def styled(spans: list[Span]) -> str:
    """Render spans with **bold** / _italic_ markers, merging same-style runs."""
    runs: list[list] = []
    for s in spans:
        style = "b" if s.bold else ("i" if s.italic else "")
        if runs and runs[-1][0] == style:
            runs[-1][1] += s.text
        else:
            runs.append([style, s.text])
    out = []
    for style, text in runs:
        core = text.strip()
        if not core or not style:
            out.append(text)
            continue
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        mark = "**" if style == "b" else "_"
        out.append(f"{lead}{mark}{core}{mark}{trail}")
    return "".join(out)


def join_text(a: str, b: str) -> str:
    a = a.rstrip()
    b = b.lstrip()
    if not a:
        return b
    # Re-join a word hyphenated across lines, including across a style seam.
    for mark in ("**", "_"):
        if re.search(r"[A-Za-z]-" + re.escape(mark) + "$", a) and re.match(re.escape(mark) + r"[a-z]", b):
            return a[: -len(mark) - 1] + b[len(mark):]
    if re.search(r"[A-Za-z]-$", a) and re.match(r"(\*\*|_)?[a-z]", b):
        if re.search(r"\S+-\S+-$", a):  # compound like "10-foot-" keeps its hyphen
            return a + b
        return a[:-1] + b
    # A bold or italic run that continues onto the next line.
    for mark in ("**", "_"):
        if a.endswith(mark) and b.startswith(mark):
            return a[: -len(mark)] + " " + b[len(mark):]
    if a.endswith(("—", "–", "/")):
        return a + b
    return a + " " + b


def tidy(text: str) -> str:
    text = re.sub(r"[ ]{2,}", " ", text).strip()
    return re.sub(r"\*\*(\s*)\*\*", r"\1", text)


def split_cells(ln: Line) -> list[Line]:
    """Split one PDF line into cells wherever its spans have a wide gap."""
    parts: list[list[Span]] = [[]]
    for s in ln.spans:
        prev = next((p for p in reversed(parts[-1]) if p.text.strip()), None)
        if prev is not None and s.text.strip() and s.x0 - prev.x1 > 6:
            parts.append([])
        parts[-1].append(s)
    out = []
    for p in parts:
        real = [s for s in p if s.text.strip()]
        if real:
            out.append(Line(p, real[0].x0, real[-1].x1, ln.base, ln.kind, ln.band))
    return out


def render_table(rows_lines: list[Line]) -> list[str]:
    rows: list[list[Line]] = []
    exploded = [c for ln in rows_lines for c in split_cells(ln)]
    for ln in sorted(exploded, key=lambda l: (l.base, l.x0)):
        if rows and abs(ln.base - rows[-1][0].base) <= 2.5:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    widest = sorted(max(rows, key=len), key=lambda l: l.x0)
    anchors = [c.x0 for c in widest]
    centers = [(c.x0 + c.x1) / 2 for c in widest]

    def col_of(c: Line) -> int:
        # Left-aligned cells match an anchor; centered headers go by center.
        near = [i for i in range(len(anchors)) if abs(c.x0 - anchors[i]) <= 8]
        if near:
            return near[0]
        mid = (c.x0 + c.x1) / 2
        return min(range(len(centers)), key=lambda i: abs(centers[i] - mid))

    def label_row(row: list[Line]) -> bool:
        """A row opening with a bold 'Label:' (spell header lines) never wraps."""
        first = next((s for s in sorted(row, key=lambda l: l.x0)[0].spans if s.text.strip()), None)
        return first is not None and first.bold and first.text.strip().endswith(":")

    grid: list[list[str]] = []
    last_base = -1e9
    for row in rows:
        cells = [""] * len(anchors)
        for c in sorted(row, key=lambda l: l.x0):
            i = col_of(c)
            cells[i] = join_text(cells[i], styled(c.spans).strip())
        base = row[0].base
        # Wrapped cell text: a close row that fills fewer cells than the one
        # above, or follows a header line with an empty first cell. Tight
        # lists (spell lists, d100 tables) fill every row and never merge.
        filled = sum(1 for c in cells if c)
        prev_filled = sum(1 for c in grid[-1] if c) if grid else 0
        wrapped = grid and (filled < prev_filled or not grid[-1][0])
        if grid and base - last_base < 14 and wrapped and not label_row(row):
            grid[-1] = [join_text(p, n) if p and n else (p or n) for p, n in zip(grid[-1], cells)]
        else:
            grid.append(cells)
        last_base = base
    if len(anchors) == 1:
        return [tidy(r[0]) for r in grid]
    grid = [[tidy(c).replace("**", "").replace("|", "\\|") for c in r] for r in grid]
    # "——Spell Slots per Spell Level—— 1 | 2 | 3 ..." -> "Slots L1 | Slots L2 | ..."
    head = grid[0]
    for j, h in enumerate(head):
        if "Spell Slots per Spell Level" in h:
            head[j] = "Slots L" + re.sub(r".*Level\W*", "", h).strip()
            for k in range(j + 1, len(head)):
                if head[k].isdigit():
                    head[k] = "Slots L" + head[k]
    out = ["| " + " | ".join(grid[0]) + " |", "|" + "---|" * len(anchors)]
    out += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return out


def render_abilities(group: list[Line]) -> str:
    rows: dict[float, list[Line]] = {}
    for g in group:
        key = next((b for b in rows if abs(b - g.base) <= 2.5), g.base)
        rows.setdefault(key, []).append(g)
    parts = []
    for key in sorted(rows):
        txt = " ".join(re.sub(r"\s+", " ", c.text).strip() for c in sorted(rows[key], key=lambda l: l.x0))
        for m in ABIL_RE.finditer(txt):
            parts.append(f"{m[1]} {m[2]} ({m[3]}, save {m[4]})")
    if parts:
        return "**Abilities:** " + "; ".join(parts)
    return " ".join(re.sub(r"\s+", " ", g.text).strip() for g in group)


def line_flags(ln: Line, prev: Line | None) -> tuple[bool, bool, bool, bool]:
    """(starts_tab, bullet, numbered, runin) for a prose line."""
    raw = ln.text
    first = next((s for s in ln.spans if s.text.strip()), ln.spans[0])
    bullet = raw.lstrip().startswith("•")
    starts_tab = not bullet and raw[:1] == " "  # the PDF indents new paragraphs with a tab
    numbered = re.match(r"\s*\d+:\s", raw) is not None
    prev_bold_end = False
    if prev is not None:
        tail = [s for s in prev.spans if s.text.strip()]
        prev_bold_end = bool(tail) and tail[-1].bold
    runin = first.bold and not prev_bold_end
    return starts_tab, bullet, numbered, runin


def prose_text(ln: Line, bullet: bool, numbered: bool) -> str:
    text = styled(ln.spans)
    if bullet:
        text = "- " + re.sub(r"^\s*(\*\*|_)?•\s*", lambda m: m[1] or "", text).lstrip()
    elif numbered:
        text = re.sub(r"^\s*(\*\*)?(\d+):\s+", lambda m: f"{m[2]}. {m[1] or ''}", text)
    return text


def mid_sentence(para: str | None) -> bool:
    """True when an open paragraph stops without closing punctuation, i.e. the
    PDF dropped a table into the middle of a sentence that resumes after it."""
    if not para:
        return False
    tail = para.rstrip().rstrip("*_").rstrip()
    return not tail.endswith((".", ":", "!", "?", ")", "”", "’", '"'))


def render(lines: list[Line]) -> str:
    blocks: list[str] = []
    state = {"para": None, "side": None}
    last_body: Line | None = None
    side_last: Line | None = None
    # Sidebars, and tables dropped into the middle of a sentence, wait until the
    # paragraph they interrupt ends, so a rule never splits across retrieval chunks.
    pending: list[str] = []

    def flush_side():
        if state["side"] and state["side"].strip():
            pending.append("> " + tidy(state["side"]))
        state["side"] = None

    def flush():
        if state["para"] and state["para"].strip():
            blocks.append(tidy(state["para"]))
        state["para"] = None
        blocks.extend(pending)
        pending.clear()

    i = 0
    while i < len(lines):
        ln = lines[i]
        k = ln.kind
        if k not in ("sidebar", "sidebar_title"):
            flush_side()
            side_last = None

        if k in HEADING_MARKS:
            flush()
            text = ln.text.strip()
            size = max(s.size for s in ln.spans)
            while i + 1 < len(lines) and lines[i + 1].kind == k and 0 < lines[i + 1].base - ln.base < max(26, size * 1.5):
                i += 1  # heading wrapped onto a second line
                ln = lines[i]
                text = join_text(text, ln.text.strip())
            blocks.append(HEADING_MARKS[k] + " " + re.sub(r"\s+", " ", text))
            last_body = None
        elif k == "caption":
            if mid_sentence(state["para"]):
                pending.append("**" + ln.text.strip() + "**")  # keep the sentence whole
            else:
                flush()
                blocks.append("**" + ln.text.strip() + "**")
                last_body = None
        elif k == "callout":  # "Base AC = 10 + ...", species "Size:" lines
            prev = lines[i - 1] if i else None
            wraps = (prev is not None and prev.kind == "callout" and prev.side == ln.side
                     and 0 < ln.base - prev.base < 14 and not ln.spans[0].bold and blocks)
            if wraps:
                blocks[-1] = tidy(join_text(blocks[-1], styled(ln.spans)))
            else:
                flush()
                blocks.append(tidy(styled(ln.spans)))
            last_body = None
        elif k == "cell":
            group = [ln]
            while i + 1 < len(lines) and lines[i + 1].kind == "cell":
                nx = lines[i + 1]
                if ln.band != -1 or nx.band != -1:
                    if nx.band != ln.band:
                        break
                elif nx.side != group[-1].side or not 0 <= nx.base - group[-1].base < 22:
                    break
                i += 1
                group.append(nx)
            table = "\n".join(render_table(group))
            if mid_sentence(state["para"]):
                pending.append(table)  # the paragraph may continue after it
            else:
                flush()
                blocks.append(table)
                last_body = None
        elif k == "ability":
            flush()
            group = [ln]
            while i + 1 < len(lines) and lines[i + 1].kind == "ability":
                i += 1
                group.append(lines[i])
            blocks.append(render_abilities(group))
            last_body = None
        elif k == "sidebar_title":
            flush_side()
            pending.append("> **" + re.sub(r"\s+", " ", ln.text).strip() + "**")
            side_last = None
            if state["para"] is None:
                flush()
        elif k == "sidebar":
            starts_tab, bullet, numbered, runin = line_flags(ln, side_last)
            text = prose_text(ln, bullet, numbered)
            if state["side"] is None or starts_tab or bullet or numbered or runin:
                flush_side()
                state["side"] = text.strip()
            else:
                state["side"] = join_text(state["side"], text)
            side_last = ln
        else:  # main body prose
            starts_tab, bullet, numbered, runin = line_flags(ln, last_body)
            same_col = last_body is not None and last_body.side == ln.side and ln.base > last_body.base
            big_gap = same_col and ln.base - last_body.base > 16.5
            text = prose_text(ln, bullet, numbered)
            # Same baseline in the same column = one visual row (term + meaning).
            same_row = last_body is not None and last_body.side == ln.side and abs(ln.base - last_body.base) < 1.5
            # An all-italic line (magic item type/rarity) stands apart from the text after it.
            prev_real = [s for s in last_body.spans if s.text.strip()] if last_body else []
            cur_first = next((s for s in ln.spans if s.text.strip()), None)
            after_italic = bool(prev_real) and all(s.italic and not s.bold for s in prev_real) \
                and cur_first is not None and not cur_first.italic
            if state["para"] is None or (not same_row and (starts_tab or bullet or numbered or runin or big_gap or after_italic)):
                flush()
                state["para"] = text.strip()
            else:
                state["para"] = join_text(state["para"], text)
            last_body = ln
        i += 1
    flush_side()
    flush()

    # The PDF numbers some lists continuously across a chapter; restart each.
    n, prev_numbered = 0, False
    for j, b in enumerate(blocks):
        m = re.match(r"(\d+)\. ", b)
        if m:
            n = n + 1 if prev_numbered else 1
            blocks[j] = f"{n}. " + b[m.end():]
            prev_numbered = True
        elif not b.startswith("- "):  # bullets nested under a step don't end the list
            prev_numbered = False
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Output: one file per chapter, one per class
# --------------------------------------------------------------------------- #
def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def split_chapters(md: str) -> dict[str, str]:
    """{filename: markdown}, one per chapter and one per class."""
    files: dict[str, str] = {}
    for chapter in re.split(r"(?m)^(?=# )", md):
        if not chapter.strip():
            continue
        title = chapter.splitlines()[0][2:].strip()
        if title == "Classes":
            # Keep "# Classes" atop each class file so retrieval trails read
            # "Classes ▸ Wizard ▸ Level 2: Scholar".
            for part in re.split(r"(?m)^(?=## )", chapter)[1:]:
                name = part.splitlines()[0][3:].strip()
                files[f"class_{slug(name)}.md"] = "# Classes\n\n" + part.strip() + "\n"
            continue
        files[f"{slug(title)}.md"] = chapter.strip() + "\n"
    return files


def write_chapters(files: dict[str, str], out_dir: Path) -> list[Path]:
    """Replace the generated corpus. Every .md but README.md is removed first:
    retrieval indexes the whole folder, so a chapter a new SRD drops or renames
    must not linger as stale rules."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in sorted(out_dir.glob("*.md")):
        if old.name != "README.md" and old.name not in files:
            print(f"removed stale {old.name}")
            old.unlink()
    written = []
    for name, text in files.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    default_out = Path(__file__).resolve().parent.parent / "knowledge" / "rules" / "srd"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", type=Path, help="the SRD PDF from dndbeyond.com/srd")
    ap.add_argument("--out", type=Path, default=default_out, help=f"output folder (default: {default_out})")
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    # Start at the first chapter bookmark ("Playing the Game"), skipping the
    # legal page and table of contents, which also use chapter-size titles.
    chapters = [page for level, _title, page in doc.get_toc() if level == 2]
    if not chapters:
        raise SystemExit("no chapter bookmarks found; is this the SRD PDF?")
    lines: list[Line] = []
    for pno in range(chapters[0] - 1, doc.page_count):
        lines += order_page(page_lines(doc[pno]))

    files = split_chapters(render(lines))  # convert fully before touching the corpus
    for path in write_chapters(files, args.out):
        print(f"{path.name:28s} {path.stat().st_size:>9,d} bytes")


if __name__ == "__main__":
    main()
