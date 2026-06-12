"""Run the fixed scenario against a list of models and write a comparison report.

For each model in models.yaml `evaluation:` (or CLI args):
  1. swap it in via the lms CLI (unload everything, load model + embedder),
  2. run scripts/ab_beats.py in a fresh process (auto-detect picks the model),
  3. parse the session .debug.log for the objective evidence: beat timings,
     auditor violations by rule, editor fix rate, token usage.

The report (logs/model_eval_<timestamp>.md) holds a comparison table, each
run's full transcript, and an empty "Verdict" section per model — the numbers
are objective, but whether the prose FITS the campaign is the user's call.

Usage (from the ttrpg_crew project dir):
    python scripts/eval_models.py                 # models from models.yaml
    python scripts/eval_models.py model-a model-b # explicit list
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ttrpg_crew.config_loader import config_section
from ttrpg_crew.services.knowledge import data_root

_AUDIT_RE = re.compile(r"^\[audit\] (\S+): (\w+)")
_FIXED_RE = re.compile(r"editor fixed (\d+)/(\d+)")
_BEAT_RE = re.compile(r"^\[timing\] moderator=\S+\s+beat=([\d.]+)s")


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def swap_model(cfg: dict, model: str) -> None:
    lms = str(cfg.get("lms_path", "lms"))
    ctx = str(cfg.get("context_length", 16384))
    print(f"[eval] loading {model} (this can take a minute)...", flush=True)
    _run([lms, "unload", "--all"])
    load = _run([lms, "load", model, "--gpu", "max", "-c", ctx, "-y"])
    if load.returncode != 0:
        raise RuntimeError(f"lms load failed for {model}: {load.stderr.strip()[:300]}")
    embedder = str(cfg.get("embedding_model", "")).strip()
    if embedder:
        _run([lms, "load", embedder, "-y"])


def run_scenario() -> tuple[Path, Path]:
    """Run ab_beats.py in a fresh process; return (transcript, debug_log) paths."""
    started = time.time()
    script = Path(__file__).resolve().parent / "ab_beats.py"
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=3600
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scenario failed: {proc.stderr.strip()[-500:]}")
    logs = data_root() / "logs"
    candidates = [
        p for p in logs.glob("session_*.md")
        if not p.name.endswith(".debug.log") and p.stat().st_mtime >= started - 1
    ]
    if not candidates:
        raise RuntimeError("scenario produced no session log")
    transcript = max(candidates, key=lambda p: p.stat().st_mtime)
    return transcript, transcript.parent / (transcript.stem + ".debug.log")


def parse_debug(debug_log: Path) -> dict:
    """Objective metrics from a session .debug.log."""
    beats: list[float] = []
    violations: dict[str, int] = {}
    fixed = 0
    for line in debug_log.read_text(encoding="utf-8").splitlines():
        if m := _BEAT_RE.match(line):
            beats.append(float(m.group(1)))
        elif m := _FIXED_RE.search(line):
            fixed += int(m.group(1))
        elif (m := _AUDIT_RE.match(line)) and "editor" not in line:
            rule = m.group(2)
            violations[rule] = violations.get(rule, 0) + 1
    # Every violation is handed to the editor (auto_retry), so the fix rate
    # is fixed / total violations.
    return {
        "beats": beats,
        "avg_beat_s": sum(beats) / len(beats) if beats else 0.0,
        "max_beat_s": max(beats, default=0.0),
        "violations": violations,
        "total_violations": sum(violations.values()),
        "editor_fixed": fixed,
    }


def tier_of(transcript: Path) -> str:
    for line in transcript.read_text(encoding="utf-8").splitlines()[:8]:
        if line.startswith("- tier:"):
            return line.split(":", 1)[1].strip()
    return "?"


def main() -> None:
    cfg = config_section("evaluation")
    models = sys.argv[1:] or [str(m) for m in (cfg.get("models") or [])]
    if not models:
        sys.exit("No models: pass them as args or fill models.yaml evaluation.models")

    results: list[dict] = []
    for model in models:
        try:
            swap_model(cfg, model)
            print(f"[eval] running scenario on {model}...", flush=True)
            transcript, debug_log = run_scenario()
            metrics = parse_debug(debug_log)
            metrics.update(model=model, transcript=transcript, tier=tier_of(transcript))
            results.append(metrics)
            print(
                f"[eval] {model}: avg beat {metrics['avg_beat_s']:.1f}s, "
                f"{metrics['total_violations']} violations, "
                f"editor fixed {metrics['editor_fixed']}/{metrics['total_violations']}",
                flush=True,
            )
        except Exception as exc:
            print(f"[eval] {model} FAILED: {exc}", flush=True)
            results.append({"model": model, "error": str(exc)})

    report = data_root() / "logs" / f"model_eval_{datetime.now():%Y%m%d_%H%M%S}.md"
    lines = [
        f"# Model evaluation — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Objective numbers below; the prose verdicts are yours — read each",
        "transcript and fill in its Verdict section.",
        "",
        "| model | tier | avg beat | max beat | violations | editor fixed | worst rules |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['model']} | — | — | — | FAILED | — | {r['error'][:60]} |")
            continue
        worst = ", ".join(
            f"{rule}×{n}"
            for rule, n in sorted(r["violations"].items(), key=lambda kv: -kv[1])[:3]
        ) or "none"
        lines.append(
            f"| {r['model']} | {r['tier']} | {r['avg_beat_s']:.1f}s | {r['max_beat_s']:.1f}s "
            f"| {r['total_violations']} | {r['editor_fixed']}/{r['total_violations']} | {worst} |"
        )
    for r in results:
        if "error" in r:
            continue
        lines += [
            "",
            f"## {r['model']}",
            "",
            f"Violations by rule: {r['violations'] or 'none'}",
            f"Beat times: {', '.join(f'{b:.1f}s' for b in r['beats'])}",
            "",
            "### Verdict (yours)",
            "",
            "- fit to tone/genre: ",
            "- companion voice: ",
            "- keep / drop: ",
            "",
            "### Transcript",
            "",
            r["transcript"].read_text(encoding="utf-8").strip(),
        ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[eval] report -> {report}", flush=True)


if __name__ == "__main__":
    main()
