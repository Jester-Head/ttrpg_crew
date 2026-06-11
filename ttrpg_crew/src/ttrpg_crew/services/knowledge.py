"""Hybrid knowledge access for the TTRPG.

Two halves, by design (see the project's "hybrid" knowledge decision):

1. Direct injection — small, high-value files are read straight into prompt
   variables every beat: the campaign premise and the latest session recap.

2. Retrieval (RAG) — the large reference corpora (lore + SRD rules) are indexed
   once into a persistent chromadb collection, embedded by the LOCAL LM Studio
   embedding model. The GM reaches them through a search tool, so each beat only
   pays a cheap query, never a re-index.

Everything here is local: embeddings go to the LM Studio OpenAI-compatible
endpoint, nothing leaves the machine.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from pathlib import Path
from typing import Any

from ttrpg_crew.config_loader import config_section, load_text_lines, load_yaml_config

def _find_project_root() -> Path:
    """Locate the data root: the folder that holds ``knowledge/``.

    The Python package lives under ``src/ttrpg_crew/``, but the campaign + SRD
    corpora and the persistent vector store sit at the project root beside it.
    Assuming a fixed parent depth breaks the moment the package is imported from
    somewhere else (a non-editable install puts it under ``site-packages``), so
    instead we walk up from this file until we find the folder that actually
    contains ``knowledge/``. ``TTRPG_CREW_ROOT`` overrides this for unusual
    layouts (e.g. data kept apart from the code).
    """
    override = os.getenv("TTRPG_CREW_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "knowledge").is_dir():
            return parent
    # Last resort: the historical src-layout depth (services -> ttrpg_crew -> src -> root).
    return here.parents[3]


_PROJECT_ROOT = _find_project_root()
_KNOWLEDGE_DIR = _PROJECT_ROOT / "knowledge"
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_STORE_DIR = _PROJECT_ROOT / ".knowledge_store"


def data_root() -> Path:
    """The resolved data root (the folder that holds ``knowledge/``)."""
    return _PROJECT_ROOT


def require_data_root() -> None:
    """Fail early, with guidance, if the knowledge corpus is not where we expect.

    A missing corpus is fatal: the game has no premise, lore, or rules without
    it. That is different from a transient embedding-server outage (which the
    caller may catch and continue past with retrieval disabled), so this raises
    a clear, actionable error instead of letting the game start up empty.
    """
    missing = [
        name
        for name, path in (
            ("knowledge/", _KNOWLEDGE_DIR),
            ("knowledge/campaign/", _KNOWLEDGE_DIR / "campaign"),
            ("config/", _CONFIG_DIR),
        )
        if not path.is_dir()
    ]
    if missing:
        raise RuntimeError(
            f"TTRPG data not found under {_PROJECT_ROOT} (missing: {', '.join(missing)}). "
            "The campaign, lore, and SRD files live at the project root. Either run "
            "from a cloned source checkout after `pip install -e .`, or set "
            "TTRPG_CREW_ROOT to the folder that contains `knowledge/`."
        )

# The `retrieval:` section of models.yaml. Frozen at import on purpose: the
# collection/chunking must stay consistent with the on-disk index for a run.
_rc = config_section("retrieval")

# Retrieval knobs — sourced from config/models.yaml (see `retrieval:`), so the
# corpora and chunk sizes are curated in config, not hardcoded here. Name banks
# (generation) and sessions (direct-injected) are intentionally not listed.
_COLLECTION = _rc.get("collection", "ttrpg_knowledge")
_RAG_GLOBS = tuple(_rc.get("sources") or ("lore/*.md", "rules/srd/*.md"))
_CHUNK_CHARS = int(_rc.get("chunk_chars", 1000))
_CHUNK_OVERLAP = int(_rc.get("chunk_overlap", 150))


# --------------------------------------------------------------------------- #
# Direct injection (no embeddings)
# --------------------------------------------------------------------------- #
def load_premise() -> str:
    """Return the campaign premise text, or '' if the file is empty/missing."""
    path = _KNOWLEDGE_DIR / "campaign" / "premise.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_latest_session_recap() -> str:
    """Return the most recent session summary, or '' if there are none."""
    session_dir = _KNOWLEDGE_DIR / "sessions"
    if not session_dir.is_dir():
        return ""
    files = sorted(session_dir.glob("*.md"))
    if not files:
        return ""
    return files[-1].read_text(encoding="utf-8").strip()


_NAME_BANK_DIR = _KNOWLEDGE_DIR / "name_banks"


def load_name_banks() -> dict[str, list[str]]:
    """Load curated names keyed by ancestry (e.g. {'human': [...], 'elf': [...]}).

    Called every beat (name-pool sampling); each bank file is mtime-cached.
    """
    banks: dict[str, list[str]] = {}
    if not _NAME_BANK_DIR.is_dir():
        return banks
    for path in sorted(_NAME_BANK_DIR.glob("*_names.txt")):
        names = load_text_lines(path)
        if names:
            banks[path.stem.replace("_names", "")] = names
    return banks


def _sampling_count(key: str, default: int) -> int:
    """Per-beat injection count from models.yaml `sampling:`, with a default."""
    try:
        return int(config_section("sampling").get(key, default))
    except (TypeError, ValueError):
        return default


def sample_name_pool(per_ancestry: int | None = None) -> str:
    """A fresh per-beat sample of curated NPC names, grouped by ancestry.

    Injected into the GM prompt so new NPCs are named from the campaign's own
    banks instead of generic AI defaults (Elara, Kael, …). Re-sampled each beat
    for variety; names already in the transcript persist on their own.
    """
    if per_ancestry is None:
        per_ancestry = _sampling_count("names_per_ancestry", 2)
    banks = load_name_banks()
    groups = []
    for ancestry, names in banks.items():
        picks = random.sample(names, min(per_ancestry, len(names)))
        groups.append(f"{ancestry.capitalize()}: {', '.join(picks)}")
    return "; ".join(groups)


def sample_style_cues(per_category: int | None = None) -> str:
    """A fresh per-beat sample of register cues (config/style_palette.yaml).

    The palette is groups of interchangeable words/short phrases (synonyms),
    NOT rare words with definitions. Each beat we sample a few groups and offer
    their options ("a / b / c") so the GM has natural phrasings to reach for.
    Returns '' when the palette is empty, so the GM just writes plainly.
    """
    if per_category is None:
        per_category = _sampling_count("style_cues_per_category", 2)
    data = load_yaml_config(_CONFIG_DIR / "style_palette.yaml")
    if not data:
        return ""

    # Each group is a list of synonyms/phrases, or a single bare phrase.
    groups: list[list[str]] = []
    for group in data.get("phrases") or []:
        if isinstance(group, list):
            options = [str(x).strip() for x in group if str(x).strip()]
            if options:
                groups.append(options)
        elif isinstance(group, str) and group.strip():
            groups.append([group.strip()])

    if not groups:
        return ""
    sample = random.sample(groups, min(per_category, len(groups)))
    return "; ".join(" / ".join(options) for options in sample)


def load_campaign_frame() -> dict[str, Any]:
    """Load the campaign frame (knowledge/campaign/frame.yaml), or {} if absent."""
    import yaml

    path = _KNOWLEDGE_DIR / "campaign" / "frame.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_campaign_cast() -> list[dict[str, Any]]:
    """Load the recurring NPC cast (knowledge/campaign/cast.yaml)."""
    import yaml

    path = _KNOWLEDGE_DIR / "campaign" / "cast.yaml"
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cast = data.get("cast") if isinstance(data, dict) else None
    return [m for m in cast if isinstance(m, dict)] if isinstance(cast, list) else []


def format_cast(cast: list[dict[str, Any]]) -> str:
    """Compact one-line-per-NPC brief for the GM prompt (keeps them consistent)."""
    def _flat(value: Any) -> str:
        return " ".join(str(value).split())

    lines = []
    for m in cast:
        name = _flat(m.get("name", "")).strip()
        if not name:
            continue
        faction = _flat(m.get("faction", "")).strip()
        role = _flat(m.get("role", "")).strip()
        goal = _flat(m.get("goal", "")).strip()
        voice = _flat(m.get("voice", "")).strip()
        head = f"{name} ({faction})" if faction else name
        if role:
            head += f" [{role}]"
        tail = []
        if goal:
            tail.append(f"wants {goal}")
        if voice:
            tail.append(f"voice: {voice}")
        line = f"- {head}"
        if tail:
            line += ": " + "; ".join(tail)
        lines.append(line)
    return "\n".join(lines)


def load_player_character() -> dict[str, Any]:
    """Load the protagonist definition (config/player_character.yaml)."""
    import yaml

    path = _CONFIG_DIR / "player_character.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def format_protagonist(pc: dict[str, Any]) -> str:
    """Render the protagonist as a compact one-paragraph brief for prompts."""
    if not pc:
        return "An unnamed newcomer to Gloamvere City."
    name = str(pc.get("name", "")).strip() or "The protagonist"
    tags = [str(pc.get(f, "")).strip() for f in ("pronouns", "ancestry")]
    tags = [t for t in tags if t]
    head = f"{name} ({', '.join(tags)})" if tags else name
    parts = [head]
    for field in ("concept", "stake", "ties"):
        value = str(pc.get(field, "")).strip()
        if value:
            parts.append(f"{field.capitalize()}: {value}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Retrieval (RAG) over lore + rules
# --------------------------------------------------------------------------- #
def _embedding_fn():
    """chromadb embedding function pointed at the LOCAL LM Studio embed model."""
    from chromadb.utils.embedding_functions.openai_embedding_function import (
        OpenAIEmbeddingFunction,
    )

    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    model_name = os.getenv(
        "LOCAL_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"
    )
    api_key = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")

    return OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model_name,
        api_base=base_url,
    )


def _collection():
    """Get-or-create the persistent knowledge collection."""
    import chromadb
    from chromadb.config import Settings

    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(_STORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        _COLLECTION, embedding_function=_embedding_fn()
    )


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading-trail, body) sections.

    The trail is the chain of enclosing headings (e.g. "Old Blood Council ▸
    Patriarch Thorne Blackwater"), so each chunk knows where it lives and never
    fuses content from unrelated sections.
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    trail = ""
    body: list[str] = []

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if not match:
            body.append(line)
            continue
        joined = "\n".join(body).strip()
        if joined:
            sections.append((trail, joined))
        body = []
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        trail = " > ".join(t for _, t in stack)

    joined = "\n".join(body).strip()
    if joined:
        sections.append((trail, joined))
    return sections


def _split_long(body: str) -> list[str]:
    """Greedy paragraph packing for sections longer than the chunk budget."""
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > _CHUNK_CHARS:
            pieces.append(current)
            current = current[-_CHUNK_OVERLAP:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces or [body]


def _iter_doc_chunks(rel: str, text: str):
    """Yield (id, doc_text) heading-aware chunks for one source file.

    Each chunk is self-labeling: "[rel ▸ heading trail]" so retrieved context
    carries its own provenance and section.
    """
    for trail, body in _split_sections(text):
        label = f"{rel} > {trail}" if trail else rel
        for piece in _split_long(body):
            doc = f"[{label}]\n{piece}"
            yield hashlib.sha256(f"{label}::{piece}".encode()).hexdigest(), doc


def _iter_source_files():
    for pattern in _RAG_GLOBS:
        yield from sorted(_KNOWLEDGE_DIR.glob(pattern))


def ensure_indexed(*, verbose: bool = True) -> int:
    """Index lore + rules into the collection if it is empty. Returns chunk count.

    Safe to call every run: if the collection already holds documents, this is a
    no-op (no re-embedding). Embedding errors propagate to the caller, which may
    choose to disable retrieval and continue.
    """
    collection = _collection()
    existing = collection.count()
    if existing > 0:
        if verbose:
            print(f"[knowledge] collection ready ({existing} chunks, cached)")
        return existing

    ids: list[str] = []
    docs: list[str] = []
    for path in _iter_source_files():
        rel = path.relative_to(_KNOWLEDGE_DIR).as_posix()
        text = path.read_text(encoding="utf-8")
        for chunk_id, doc in _iter_doc_chunks(rel, text):
            ids.append(chunk_id)
            docs.append(doc)

    if not docs:
        if verbose:
            print("[knowledge] no source files found to index")
        return 0

    if verbose:
        print(f"[knowledge] indexing {len(docs)} chunks (one-time, embedding locally)...")
    # Batch to keep request sizes sane for the local embedding server.
    for start in range(0, len(docs), 64):
        collection.add(
            ids=ids[start : start + 64],
            documents=docs[start : start + 64],
        )
    if verbose:
        print(f"[knowledge] indexed {len(docs)} chunks")
    return len(docs)


def search_knowledge(query: str, k: int | None = None) -> str:
    """Return the top-k lore/rules chunks for a query, formatted for a prompt."""
    if k is None:
        try:
            k = int(config_section("retrieval").get("top_k", 5))
        except (TypeError, ValueError):
            k = 5
    collection = _collection()
    if collection.count() == 0:
        return "No world knowledge has been indexed."

    result = collection.query(query_texts=[query], n_results=k)
    docs = (result.get("documents") or [[]])[0]
    if not docs:
        return "No relevant world knowledge found."
    return "\n\n---\n\n".join(docs)
