"""Blind A/B of companion modes: every arm gets the same GM frames and speakers,
and the outputs become shuffled, unlabeled pairs for scripts/rate_ab.py.

Usage (from the ttrpg_crew project dir):
    python scripts/blind_ab.py model-a model-b --runs 3
    python scripts/blind_ab.py --no-load --runs 1     # use the model already loaded

Writes logs/ab_<timestamp>/:
    blind/pairs.jsonl    what you rate (no arm names)
    private/key.json     which arm is on which side; don't open until rated
    private/results.jsonl  raw outputs, notes, checks and timings per arm
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ttrpg_crew import session_log
from ttrpg_crew.game.schemas import ModeratorSelection
from ttrpg_crew.game.validator import (
    acts_as_other,
    assistant_speak,
    audit,
    duplicate_companion_line,
    format_leak,
    repeats_player_action,
)
from ttrpg_crew.services.beat_runner import COMPANION_MODES
from ttrpg_crew.services.knowledge import data_root

_SCENARIO = Path(__file__).resolve().parent / "ab_scenario.yaml"
# Share of pairs shown a second time with sides swapped, to measure how
# consistent the ratings are.
_REPEAT_SHARE = 0.1
_CONTEXT_CHARS = 1200


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:12], 16)


def _build_flow(mode: str):
    from ttrpg_crew.main import _build_flow as build

    flow = build()
    flow.companion_mode = mode
    flow.auto_retry = False
    return flow


def _forced_selection(spec, roster: list[str], rng: random.Random) -> ModeratorSelection | None:
    if spec is None:
        return None
    if isinstance(spec, int):
        ids = rng.sample(roster, min(spec, len(roster)))
    else:
        ids = [str(s) for s in spec if str(s) in roster]
    return ModeratorSelection(
        selected_speakers=ids,
        gm_should_frame_first=True,
        gm_intent="React to the player's action and move the scene forward.",
    )


def _checks(flow, speaker_id: str, text: str, prior: list[str], player_input: str) -> list[str]:
    names = {
        sid: str(p.get("name") or sid) for sid, p in flow.character_profiles.items()
    }
    name = names.get(speaker_id, speaker_id)
    others = tuple(n for sid, n in names.items() if sid != speaker_id)
    violations = audit(
        speaker_id,
        text,
        protagonist_name=flow.protagonist_name,
        companions=tuple(names.values()),
        character_name=name,
    )
    violations += acts_as_other(text, others)
    for check in (
        format_leak(text),
        assistant_speak(text),
        duplicate_companion_line(text, prior),
        repeats_player_action(text, player_input),
    ):
        if check:
            violations.append(check)
    return [v.rule for v in violations]


def run_arm(flow, mode: str, player_input: str, selection, gm_text, seed: int) -> dict:
    random.seed(seed)
    start = time.perf_counter()
    beat = flow.kickoff(player_input=player_input, selection=selection, gm_text=gm_text)
    seconds = time.perf_counter() - start
    runner = flow.last_runner
    segments, prior = [], []
    for seg in beat.segments:
        if seg.is_gm:
            continue
        rules = _checks(flow, seg.speaker_id, seg.text, prior, player_input)
        prior.append(seg.text)
        segments.append(
            {"speaker": seg.speaker_name, "id": seg.speaker_id, "text": seg.text, "checks": rules}
        )
    return {
        "mode": mode,
        "seconds": round(seconds, 1),
        "segments": segments,
        "raw": [list(o) for o in runner.last_outputs if o[0] != "GM"],
        "notes": runner.last_notes,
        "missing_marker": sorted(runner.last_missing_marker),
    }


def run_model(model: str, modes: list[str], runs: int, beats: list[dict]) -> list[dict]:
    items = []
    for run in range(1, runs + 1):
        gm_flow = _build_flow("crewai")
        arm_flows = {mode: _build_flow(mode) for mode in modes}
        roster = list(gm_flow.character_profiles)
        first_name = str(gm_flow.character_profiles[roster[0]].get("name", "")).split()[0]

        for index, spec in enumerate(beats, start=1):
            player_input = str(spec.get("input", "")).replace("{companion}", first_name)
            seed = _seed(model, run, index)
            context = gm_flow.transcript.tail(_CONTEXT_CHARS)
            forced = _forced_selection(spec.get("speakers"), roster, random.Random(seed))

            random.seed(seed)
            gm_beat = gm_flow.kickoff(player_input=player_input, selection=forced, gm_only=True)
            selection = gm_flow.last_selection
            gm_raw = None
            if selection.gm_should_frame_first:
                # "" rather than None so an empty GM frame is still replayed,
                # never regenerated per arm.
                gm_raw = next(
                    (t for label, t in gm_flow.last_runner.last_outputs if label == "GM"), ""
                )
            gm_shown = "\n\n".join(
                part for part in (
                    "\n\n".join(s.text for s in gm_beat.segments if s.is_gm),
                    gm_beat.closing_prompt,
                ) if part
            )

            arms = {}
            for mode in random.Random(seed).sample(modes, len(modes)):
                arms[mode] = run_arm(arm_flows[mode], mode, player_input, selection, gm_raw, seed)
                print(f"[ab] {model} run {run} beat {index} arm done", flush=True)

            items.append({
                "item": f"{model}|{run}|{index}",
                "model": model,
                "run": run,
                "beat": index,
                "context": context,
                "player_input": player_input,
                "gm": gm_shown,
                "speakers": selection.selected_speakers,
                "arms": arms,
            })
    return items


def write_blind(items: list[dict], out: Path) -> int:
    pairs, key = [], {}
    for item in items:
        modes = [m for m, arm in item["arms"].items() if arm["segments"]]
        for a, b in itertools.combinations(sorted(modes), 2):
            left, right = (a, b) if secrets.randbelow(2) else (b, a)
            pair_id = secrets.token_hex(4)
            key[pair_id] = {"item": item["item"], "left": left, "right": right}
            pairs.append({
                "pair_id": pair_id,
                "model": item["model"],
                "context": item["context"],
                "player_input": item["player_input"],
                "gm": item["gm"],
                "left": [f"{s['speaker']}: {s['text']}" for s in item["arms"][left]["segments"]],
                "right": [f"{s['speaker']}: {s['text']}" for s in item["arms"][right]["segments"]],
            })

    repeats = random.Random(secrets.randbits(32)).sample(
        pairs, max(1, round(len(pairs) * _REPEAT_SHARE))
    ) if pairs else []
    for original in repeats:
        pair_id = secrets.token_hex(4)
        k = key[original["pair_id"]]
        key[pair_id] = {**k, "left": k["right"], "right": k["left"], "repeat_of": original["pair_id"]}
        pairs.append({
            **original, "pair_id": pair_id, "left": original["right"], "right": original["left"],
        })

    random.Random(secrets.randbits(32)).shuffle(pairs)
    (out / "blind").mkdir(parents=True, exist_ok=True)
    with (out / "blind" / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    (out / "private" / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="LM Studio model ids (default: loaded model)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modes", default=",".join(COMPANION_MODES))
    parser.add_argument("--no-load", action="store_true", help="use the model already loaded")
    parser.add_argument("--scenario", type=Path, default=_SCENARIO)
    args = parser.parse_args()

    load_dotenv()
    # Debug output and the normal session logs label each arm, so keep the
    # console quiet and send the logs to the private folder.
    os.environ.pop("TTRPG_DEBUG", None)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if len(modes) < 2 or any(m not in COMPANION_MODES for m in modes):
        sys.exit(f"--modes needs two or more of {COMPANION_MODES}")
    beats = (yaml.safe_load(args.scenario.read_text(encoding="utf-8")) or {}).get("beats") or []

    out = data_root() / "logs" / f"ab_{datetime.now():%Y%m%d_%H%M%S}"
    (out / "private").mkdir(parents=True, exist_ok=True)
    session_log.set_log_dir(out / "private")

    from eval_models import swap_model
    from ttrpg_crew.config_loader import config_section
    from ttrpg_crew.main import detect_loaded_model

    models = args.models
    if args.no_load or not models:
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
        models = [os.getenv("LOCAL_LLM_MODEL") or detect_loaded_model(base_url) or "local-model"]

    items: list[dict] = []
    with (out / "private" / "results.jsonl").open("w", encoding="utf-8") as fh:
        for model in models:
            if not args.no_load and args.models:
                swap_model(config_section("evaluation"), model)
            os.environ["LOCAL_LLM_MODEL"] = model
            for item in run_model(model, modes, args.runs, beats):
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                fh.flush()
                items.append(item)

    n_pairs = write_blind(items, out)
    print(f"[ab] {len(items)} beats, {n_pairs} pairs to rate -> {out}")
    print(f"[ab] rate with: python scripts/rate_ab.py \"{out}\"")


if __name__ == "__main__":
    main()
