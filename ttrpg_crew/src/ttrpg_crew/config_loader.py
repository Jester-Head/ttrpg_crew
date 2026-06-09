"""Single source for the engine's tunable knobs (config/models.yaml).

Data = config, logic = code. Every numeric dial the engine exposes — LLM
sampling, speaker selection, auditor strictness, retrieval, per-beat sampling —
lives in models.yaml and is read through here. The file is re-parsed only when
it changes on disk (mtime check), so edits apply on the next beat without a
restart, the same way the cliche word lists do, but without re-reading YAML on
every call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "models.yaml"

_cache: dict[str, Any] = {}
_cache_mtime: float | None = None


def load_models_config() -> dict[str, Any]:
    """Return models.yaml as a dict, re-reading it only when the file changes."""
    global _cache, _cache_mtime
    try:
        mtime = _CONFIG_PATH.stat().st_mtime
    except OSError:
        return _cache
    if mtime != _cache_mtime:
        import yaml

        try:
            data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = None
        _cache = data if isinstance(data, dict) else {}
        _cache_mtime = mtime
    return _cache


def config_section(name: str) -> dict[str, Any]:
    """Return one top-level mapping from models.yaml ({} if absent or malformed)."""
    section = load_models_config().get(name)
    return section if isinstance(section, dict) else {}
