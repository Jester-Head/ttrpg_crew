from __future__ import annotations

import json
import os
import sys
import urllib.request
import warnings

# Cosmetic: CrewAI Task callbacks (our timing/audit hooks) are closures, which
# makes pydantic warn they can't be serialized for checkpointing. We don't use
# checkpointing, so silence this one line to keep play output clean.
warnings.filterwarnings("ignore", message=r".*callbacks cannot be serialized.*")

# Windows consoles default to cp1252, which crashes CrewAI's verbose logger when
# it prints emoji (and mangles em-dashes/smart-quotes from the lore as "?").
# Force UTF-8 so logs and narration render correctly. line_buffering flushes on
# every newline so status/progress lines appear live during the long model
# calls, instead of being block-buffered until the program exits.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", line_buffering=True)
    except (AttributeError, ValueError):
        pass

from crewai import LLM
from dotenv import load_dotenv

from ttrpg_crew import session_log
from ttrpg_crew.config_loader import config_section
from ttrpg_crew.flow import TTRPGFlow
from ttrpg_crew.prompts.prompt_loader import resolve_tier_for_model


def _get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.load(response)
    except Exception:
        return None


def detect_loaded_model(base_url: str) -> str | None:
    """Return the chat model currently RESIDENT in LM Studio.

    LM Studio's OpenAI-compatible /v1/models lists every *downloaded* model
    (JIT loading advertises them all), so it can't tell us what's actually in
    memory — and naming a non-resident model would force a slow cold load.

    The native /api/v0/models endpoint reports a per-model `state`, so we prefer
    the one that is genuinely `loaded`. We fall back to /v1/models only if the
    native endpoint isn't reachable.
    """
    # LM Studio native API lives under /api/v0; derive it from the /v1 base_url.
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    native = _get_json(root + "/api/v0/models")
    if native:
        for entry in native.get("data") or []:
            if not isinstance(entry, dict):
                continue
            # Chat models are type llm/vlm; skip embeddings. Only take loaded.
            if entry.get("state") == "loaded" and entry.get("type") != "embeddings":
                return entry.get("id")

    # Fallback: OpenAI-compatible list (cannot distinguish loaded; skip embedders).
    data = _get_json(base_url.rstrip("/") + "/models")
    if data:
        for entry in data.get("data") or []:
            model_id = entry.get("id") if isinstance(entry, dict) else None
            if model_id and "embed" not in model_id.lower():
                return model_id
    return None


def _env_flag(name: str, *, default: bool) -> bool:
    """Read an on/off env var properly: '0', 'false', 'no', 'off' disable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _build_flow() -> TTRPGFlow:
    """Resolve the local model + prompt tier and construct the flow.

    Config via .env at the project root (all optional):
        LOCAL_LLM_BASE_URL=http://localhost:1234/v1   # LM Studio server URL
        LOCAL_LLM_API_KEY=lm-studio                   # LM Studio ignores the value
        LOCAL_LLM_MODEL=...        # force a model id instead of auto-detecting
        LOCAL_MANAGER_MODEL=...    # faster model for the moderator (speaker pick)
        LOCAL_EDITOR_MODEL=...     # faster model for the AI-ism editor pass
        LOCAL_EDITOR_BASE_URL=...  # server for the editor model (defaults to main)
        LOCAL_EDITOR_API_KEY=...   # api key for the editor server (defaults to main)
        TTRPG_CREW_MODEL_TIER=...  # force large|medium|small instead of auto
        TTRPG_DEBUG=1              # show engine internals (panels, timing, audit)

    By default the loaded model is detected from the server and its size decides
    the prompt tier (see models.yaml: tier_selection).

    Editor model note: the editor only runs when the auditor flags output, so a
    small/fast model is ideal. But on a SINGLE GPU with one LM Studio server,
    naming a different model than the GM forces JIT swap churn (unload GM, load
    editor, swap back) every time it fires — far slower overall. Use
    LOCAL_EDITOR_MODEL only when that model can stay resident: either a second
    endpoint (set LOCAL_EDITOR_BASE_URL) or enough VRAM to hold both at once.
    """
    debug = bool(os.getenv("TTRPG_DEBUG"))

    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")

    # Sampling knobs (temperature/top_p/max_tokens/timeout) come from models.yaml
    # `llm:`. `defaults` is the narration baseline; manager/editor override it.
    llm_cfg = config_section("llm")
    base_params = dict(llm_cfg.get("defaults") or {})

    def _role_params(role: str) -> dict:
        params = dict(base_params)
        params.update(llm_cfg.get(role) or {})
        return params

    def _make_llm(model_id: str, params: dict, *, url: str | None = None, key: str | None = None) -> LLM:
        model = model_id if model_id.startswith("openai/") else f"openai/{model_id}"
        return LLM(model=model, base_url=url or base_url, api_key=key or api_key, **params)

    model_id = os.getenv("LOCAL_LLM_MODEL") or detect_loaded_model(base_url) or "local-model"
    tier = os.getenv("TTRPG_CREW_MODEL_TIER") or resolve_tier_for_model(model_id)
    llm = _make_llm(model_id, base_params)

    # Manager (LLM moderator) and editor get their own LLM with role sampling,
    # even on the GM's model — sampling is per-request, so no model swap. A
    # separate model id / endpoint is only used when its env var is set.
    manager_model_id = os.getenv("LOCAL_MANAGER_MODEL")
    manager_llm = _make_llm(manager_model_id or model_id, _role_params("manager"))

    editor_model_id = os.getenv("LOCAL_EDITOR_MODEL")
    editor_llm = _make_llm(
        editor_model_id or model_id,
        _role_params("editor"),
        url=os.getenv("LOCAL_EDITOR_BASE_URL"),
        key=os.getenv("LOCAL_EDITOR_API_KEY"),
    )

    if debug:
        print(
            f"[ttrpg] model={model_id}  tier={tier}  "
            f"manager={manager_model_id or model_id}  "
            f"editor={editor_model_id or model_id}  base_url={base_url}"
        )

    # Speaker selection is deterministic by default (fast, no per-beat LLM call).
    # Set TTRPG_LLM_MODERATOR=1 to use the LLM manager agent instead.
    use_llm_moderator = bool(os.getenv("TTRPG_LLM_MODERATOR"))

    # Companions run through the CrewAI agent loop by default — in play-testing
    # it produced richer in-scene interplay than the direct path. Set
    # TTRPG_DIRECT_COMPANIONS=1 for direct chat completions instead (no ReAct
    # preamble; measured ~7x fewer completion tokens — the fast option). The GM
    # is on CrewAI either way.
    use_direct_companions = _env_flag("TTRPG_DIRECT_COMPANIONS", default=False)

    if debug:
        print(f"[ttrpg] direct_companions={use_direct_companions}")

    # Stamp run metadata into the session log headers so each transcript says
    # which model/settings produced it (essential when comparing models).
    session_log.set_meta(
        model=model_id,
        tier=tier,
        manager_model=manager_model_id or model_id,
        editor_model=editor_model_id or model_id,
        direct_companions=use_direct_companions,
    )

    return TTRPGFlow(
        llm=llm,
        manager_llm=manager_llm,
        editor_llm=editor_llm,
        tier=tier,
        debug=debug,
        use_llm_moderator=use_llm_moderator,
        use_direct_companions=use_direct_companions,
    )


_QUIT = {"quit", "exit", "q", ":q"}


def run():
    """Play the game: clean narration in, your action out, repeat.

    Only the story is shown. Set TTRPG_DEBUG=1 to see engine internals.
    """
    load_dotenv()

    print("Loading Gloamvere City…  (first load can take a minute)", flush=True)
    flow = _build_flow()

    # Segments arrive one speaker at a time (audited and edited before
    # delivery), so the GM's text shows the moment it's ready instead of
    # after every companion — the same per-segment contract a chat-style
    # GUI will consume ("X is typing…" then the finished message).
    def _print_segment(segment) -> None:
        print(segment.render() + "\n", flush=True)

    player_input = ""
    first = True
    while True:
        # A cue before the (slow) model call so it's clear the GM is working,
        # not waiting on the player.
        cue = "The party gathers" if first else "The world responds"
        print(f"\n{cue}… (a moment on a local model)\n", flush=True)

        beat = flow.kickoff(player_input=player_input, on_segment=_print_segment)
        if beat.closing_prompt:
            print(beat.closing_prompt + "\n", flush=True)

        if first:
            print("(Type your action and press Enter. 'quit' to leave.)", flush=True)
            first = False

        try:
            player_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            player_input = "quit"

        if player_input.lower() in _QUIT:
            print("\nThe gaslight dims. Farewell.")
            break


if __name__ == "__main__":
    run()
