"""Append-only session logs.

Files per run under logs/, created lazily, so you can read the whole run
afterward instead of scrolling the terminal:

  * session_<timestamp>.md         — the CLEANED output: each beat's narration
                                      only, nothing else. This is the transcript.
  * session_<timestamp>.debug.log  — diagnostics (timing / tokens / audit / raw
                                      per-agent output), kept out of the clean file.
  * session_<timestamp>.prompts.jsonl — the exact messages sent on every model
                                      call, one JSON object per call. Only when
                                      enabled (log_prompts() or TTRPG_LOG_PROMPTS=1).

Logging failures never break a beat.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_clean = None
_debug = None
_prompts = None
_clean_path: Optional[Path] = None
_debug_path: Optional[Path] = None
_beat_n = 0
_meta: dict[str, str] = {}
_tags: dict[str, Any] = {}
_log_dir: Optional[Path] = None
_log_prompts = os.getenv("TTRPG_LOG_PROMPTS", "").strip().lower() in ("1", "true", "yes")
# CrewAI reports its calls from a worker thread.
_prompt_lock = threading.Lock()


def set_log_dir(path: Path) -> None:
    """Write this run's logs to `path` instead of <data root>/logs (or $TTRPG_LOG_DIR).
    Call before the first beat."""
    global _log_dir
    _log_dir = path


def set_meta(**fields: str) -> None:
    """Record run metadata (model, tier, ...) for the log headers.

    Call before the first beat (main.py does, right after model detection) so
    the lines land in the header when the files are lazily created.
    """
    _meta.update({k: str(v) for k, v in fields.items()})


def log_prompts(enabled: bool = True) -> None:
    """Turn the per-call prompt log on or off."""
    global _log_prompts
    _log_prompts = enabled


def set_tags(**fields: Any) -> None:
    """Replace the fields stamped on every prompt record from now on (e.g. the
    A/B harness's model, run, beat and arm)."""
    _tags.clear()
    _tags.update(fields)


def _ensure_open() -> None:
    global _clean, _debug, _prompts, _clean_path, _debug_path
    if _clean is not None:
        return
    try:
        from ttrpg_crew.services.knowledge import data_root

        root = data_root()
    except Exception:
        root = Path.cwd()
    try:
        env_dir = os.getenv("TTRPG_LOG_DIR")
        logs_dir = _log_dir or (Path(env_dir) if env_dir else root / "logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        _clean_path = logs_dir / f"session_{stamp}.md"
        _debug_path = logs_dir / f"session_{stamp}.debug.log"
        _clean = _clean_path.open("a", encoding="utf-8")
        _debug = _debug_path.open("a", encoding="utf-8")
        if _log_prompts:
            _prompts = (logs_dir / f"session_{stamp}.prompts.jsonl").open("a", encoding="utf-8")
        header = f"# Session {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        if _meta:
            header += "".join(f"- {key}: {value}\n" for key, value in _meta.items())
        _clean.write(header)
        _clean.flush()
        _debug.write(header)
        _debug.flush()
        print(f"[log] transcript -> {_clean_path}")
    except Exception:
        _clean = None
        _debug = None
        _prompts = None


def path() -> Optional[Path]:
    """Path to the clean transcript (the .md)."""
    return _clean_path


def _emit(handle, text: str) -> None:
    if handle is None:
        return
    try:
        handle.write(text if text.endswith("\n") else text + "\n")
        handle.flush()
    except Exception:
        pass


def diag(line: str) -> None:
    """Log one diagnostic line to the .debug.log (kept out of the transcript)."""
    _ensure_open()
    _emit(_debug, line)


def prompt(label: str, messages: Any, *, source: str, model: str = "") -> None:
    """Record the exact messages of one model call (no-op unless enabled)."""
    if not _log_prompts:
        return
    _ensure_open()
    if _prompts is None:
        return
    record = {
        "time": f"{datetime.now():%H:%M:%S}",
        "beat": _beat_n + 1,
        **_tags,
        "source": source,
        "label": label,
        "model": model,
        "messages": messages,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except Exception:
        return
    with _prompt_lock:
        _emit(_prompts, line)


def beat(header: str, text: str) -> None:
    """Log one beat's clean narration to the transcript, under a numbered header."""
    global _beat_n
    _ensure_open()
    _beat_n += 1
    _emit(_clean, f"\n## Beat {_beat_n} — {header}\n\n{text.strip()}\n")
