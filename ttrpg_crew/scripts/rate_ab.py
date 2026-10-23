"""Rate the blind pairs from scripts/blind_ab.py, then unblind with --report.

Usage (from the ttrpg_crew project dir):
    python scripts/rate_ab.py logs/ab_<timestamp>            # rate (resumes where you stopped)
    python scripts/rate_ab.py logs/ab_<timestamp> --report   # unblind once everything is rated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

TAGS = {
    "i": "identity",
    "r": "scene roster",
    "p": "interplay",
    "a": "assistant speak",
    "v": "voice",
    "o": "other",
}
_WIDTH = 100


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(para, _WIDTH, initial_indent=indent, subsequent_indent=indent)
        for para in text.splitlines() if para.strip()
    )


def _show(pair: dict, n: int, total: int) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print(f"Pair {n}/{total}    model: {pair['model']}\n")
    if pair["context"].strip():
        print("EARLIER")
        print(_wrap(pair["context"]) + "\n")
    print("PLAYER")
    print(_wrap(pair["player_input"] or "(opening)") + "\n")
    if pair["gm"].strip():
        print("GM")
        print(_wrap(pair["gm"]) + "\n")
    for side in ("left", "right"):
        print("=" * _WIDTH)
        print(f"[{1 if side == 'left' else 2}] {side.upper()}")
        for line in pair[side]:
            print(_wrap(line))
    print("=" * _WIDTH)


def rate(run_dir: Path) -> None:
    pairs = _load_jsonl(run_dir / "blind" / "pairs.jsonl")
    ratings_path = run_dir / "blind" / "ratings.jsonl"
    done = {r["pair_id"] for r in _load_jsonl(ratings_path)}
    todo = [p for p in pairs if p["pair_id"] not in done]
    if not todo:
        print("Everything is rated. Run again with --report to unblind.")
        return

    tag_help = "  ".join(f"{k}={v}" for k, v in TAGS.items())
    with ratings_path.open("a", encoding="utf-8") as fh:
        for pair in todo:
            _show(pair, len(done) + 1, len(pairs))
            choice = ""
            while choice not in ("1", "2", "t", "s", "q"):
                choice = input("Better: [1] left  [2] right  [t] tie  [s] skip  [q] quit > ").strip().lower()
            if choice == "q":
                break
            if choice == "s":
                continue
            raw_tags = input(f"Why (letters, Enter to skip) {tag_help} > ").strip().lower()
            note = input("Note (Enter to skip) > ").strip()
            fh.write(json.dumps({
                "pair_id": pair["pair_id"],
                "choice": {"1": "left", "2": "right", "t": "tie"}[choice],
                "tags": sorted({TAGS[c] for c in raw_tags if c in TAGS}),
                "note": note,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(pair["pair_id"])
    print(f"\n{len(done)}/{len(pairs)} rated.")


def _preferred_arm(rating: dict, pair_key: dict) -> str:
    return "tie" if rating["choice"] == "tie" else pair_key[rating["choice"]]


def report(run_dir: Path, force: bool) -> None:
    pairs = _load_jsonl(run_dir / "blind" / "pairs.jsonl")
    ratings = {r["pair_id"]: r for r in _load_jsonl(run_dir / "blind" / "ratings.jsonl")}
    if len(ratings) < len(pairs) and not force:
        sys.exit(f"Only {len(ratings)}/{len(pairs)} pairs rated. Finish rating, or pass --force.")
    key = json.loads((run_dir / "private" / "key.json").read_text(encoding="utf-8"))
    results = _load_jsonl(run_dir / "private" / "results.jsonl")

    lines = [f"# Blind A/B report: {run_dir.name}", ""]

    by_model: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    loss_tags: dict[str, Counter] = defaultdict(Counter)
    consistent = repeats = 0
    for pair_id, rating in ratings.items():
        k = key[pair_id]
        if "repeat_of" in k:
            original = ratings.get(k["repeat_of"])
            if original:
                repeats += 1
                if _preferred_arm(rating, k) == _preferred_arm(original, key[k["repeat_of"]]):
                    consistent += 1
            continue
        model = k["item"].split("|")[0]
        left, right = k["left"], k["right"]
        if rating["choice"] == "tie":
            by_model[model][left]["tie"] += 1
            by_model[model][right]["tie"] += 1
        else:
            winner, loser = (left, right) if rating["choice"] == "left" else (right, left)
            by_model[model][winner]["win"] += 1
            by_model[model][loser]["loss"] += 1
            loss_tags[loser].update(rating["tags"])

    lines += ["## Preferences", "", "| model | arm | wins | ties | losses | win rate (ties = half) |", "|---|---|---|---|---|---|"]
    for model, arms in sorted(by_model.items()):
        for arm, c in sorted(arms.items(), key=lambda kv: -(kv[1]["win"] + kv[1]["tie"] / 2)):
            n = c["win"] + c["tie"] + c["loss"]
            score = (c["win"] + c["tie"] / 2) / n if n else 0
            lines.append(f"| {model} | {arm} | {c['win']} | {c['tie']} | {c['loss']} | {score:.0%} |")
    if repeats:
        lines += ["", f"Repeated pairs judged the same way: {consistent}/{repeats}."]

    lines += ["", "## Why the losing side lost", ""]
    for arm, tags in sorted(loss_tags.items()):
        lines.append(f"- {arm}: " + (", ".join(f"{t} ×{n}" for t, n in tags.most_common()) or "no tags"))

    lines += ["", "## Automated checks (per companion line)", "",
              "| model | arm | lines | flagged | top rules | avg beat time | missing SAY: |", "|---|---|---|---|---|---|---|"]
    stats: dict[tuple[str, str], dict] = defaultdict(lambda: {"lines": 0, "flagged": 0, "rules": Counter(), "secs": [], "missing": 0})
    for item in results:
        for arm, data in item["arms"].items():
            s = stats[(item["model"], arm)]
            if data["segments"]:
                s["secs"].append(data["seconds"])
            s["missing"] += len(data.get("missing_marker", []))
            for seg in data["segments"]:
                s["lines"] += 1
                s["flagged"] += bool(seg["checks"])
                s["rules"].update(seg["checks"])
    for (model, arm), s in sorted(stats.items()):
        top = ", ".join(f"{r} ×{n}" for r, n in s["rules"].most_common(3)) or "none"
        avg = sum(s["secs"]) / len(s["secs"]) if s["secs"] else 0
        lines.append(f"| {model} | {arm} | {s['lines']} | {s['flagged']} | {top} | {avg:.0f}s | {s['missing']} |")

    out = run_dir / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--report", action="store_true", help="unblind and summarize")
    parser.add_argument("--force", action="store_true", help="report before every pair is rated")
    args = parser.parse_args()
    if args.report:
        report(args.run_dir, args.force)
    else:
        rate(args.run_dir)


if __name__ == "__main__":
    main()
