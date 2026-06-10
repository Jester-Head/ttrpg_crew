"""Append-only session logs.

Two files per run under logs/, created lazily, so you can read the whole run
afterward instead of scrolling the terminal:

  * session_<timestamp>.md         — the CLEANED output: each beat's narration
                                      only, nothing else. This is the transcript.
  * session_<timestamp>.debug.log  — diagnostics (timing / tokens / audit / raw
                                      per-agent output), kept out of the clean file.

Logging failures never break a beat.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

_clean = None
_debug = None
_clean_path: Optional[Path] = None
_debug_path: Optional[Path] = None
_beat_n = 0
_meta: dict[str, str] = {}


def set_meta(**fields: str) -> None:
    """Record run metadata (model, tier, ...) for the log headers.

    Call before the first beat (main.py does, right after model detection) so
    the lines land in the header when the files are lazily created.
    """
    _meta.update({k: str(v) for k, v in fields.items()})


def _ensure_open() -> None:
    global _clean, _debug, _clean_path, _debug_path
    if _clean is not None:
        return
    try:
        from ttrpg_crew.services.knowledge import data_root

        root = data_root()
    except Exception:
        root = Path.cwd()
    try:
        logs_dir = root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        _clean_path = logs_dir / f"session_{stamp}.md"
        _debug_path = logs_dir / f"session_{stamp}.debug.log"
        _clean = _clean_path.open("a", encoding="utf-8")
        _debug = _debug_path.open("a", encoding="utf-8")
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


def beat(header: str, text: str) -> None:
    """Log one beat's clean narration to the transcript, under a numbered header."""
    global _beat_n
    _ensure_open()
    _beat_n += 1
    _emit(_clean, f"\n## Beat {_beat_n} — {header}\n\n{text.strip()}\n")
