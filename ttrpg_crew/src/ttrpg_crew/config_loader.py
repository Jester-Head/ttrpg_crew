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

CONFIG_DIR = Path(__file__).resolve().parent / "config"
_CONFIG_PATH = CONFIG_DIR / "models.yaml"

# path -> (mtime, parsed dict)
_caches: dict[Path, tuple[float, dict[str, Any]]] = {}


def load_yaml_config(path: Path) -> dict[str, Any]:
    """A YAML config file as a dict, re-parsed only when the file changes.

    Malformed or missing files yield {} so callers fall back to code defaults.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _caches.get(path, (0.0, {}))[1]
    cached = _caches.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        data = None
    parsed = data if isinstance(data, dict) else {}
    _caches[path] = (mtime, parsed)
    return parsed


def load_models_config() -> dict[str, Any]:
    """Return models.yaml as a dict, re-reading it only when the file changes."""
    return load_yaml_config(_CONFIG_PATH)


# path -> (mtime, lines)
_line_caches: dict[Path, tuple[float, list[str]]] = {}


def load_text_lines(path: Path) -> list[str]:
    """Non-empty, non-comment lines of a text file, mtime-cached.

    The shared reader for curated word lists (cliche tiers, register breaks,
    name banks): live-editable like the YAML configs, without re-reading the
    file on every beat. Missing files yield [].
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _line_caches.get(path, (0.0, []))[1]
    cached = _line_caches.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    _line_caches[path] = (mtime, lines)
    return lines


def config_section(name: str) -> dict[str, Any]:
    """Return one top-level mapping from models.yaml ({} if absent or malformed)."""
    section = load_models_config().get(name)
    return section if isinstance(section, dict) else {}
