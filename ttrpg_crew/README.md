# ttrpg_crew

A 5E-compatible solo TTRPG engine, set in Gloamvere City (Kavath),
run by CrewAI agents against a local LLM. A Game Master agent narrates, companion
agents play the party, and a deterministic policy decides who speaks each beat.
Prompts adapt to the size of the loaded model, and an auditor plus editor pass
keeps the prose free of common AI writing tells. Everything runs locally; nothing
leaves your machine.

## Prerequisites

- Python 3.11 or 3.12.
- A local OpenAI-compatible LLM server. [LM Studio](https://lmstudio.ai) is the
  tested setup. In it, load two models:
  - a chat model (e.g. a Gemma or Qwen instruct model) for narration, and
  - an embedding model for retrieval. The default expects
    `text-embedding-nomic-embed-text-v1.5`; override with `LOCAL_EMBED_MODEL`.
- Start the server so it listens on `http://localhost:1234` (LM Studio's default).

## Setup

```
git clone https://github.com/Jester-Head/bioluminescent-tapestry.git
cd bioluminescent-tapestry/ttrpg_crew
pip install -e .
```

The `-e` (editable) install is what puts the `ttrpg_crew` package on your path
and keeps the campaign data (under `knowledge/`) where the engine can find it.

Optionally copy `.env.example` to `.env` and adjust it. With no `.env`, the game
talks to LM Studio on its default port and auto-detects the loaded chat model.

## Run

```
run_ttrpg
```

`run_ttrpg` is the installed console script. Equivalently, from this directory:

```
python -m ttrpg_crew.main
```

Type an action at the prompt and press Enter; `quit` to leave. After each beat the
game also lists a few suggested actions: type a number to take one, or ignore them
and write your own. (Turn them off with `scene_tracker: enabled: false` in
`src/ttrpg_crew/config/models.yaml`.) Set `TTRPG_DEBUG=1` to see engine internals
(agent panels, timing, and audit lines).

The first run indexes the lore and rules into a local vector store
(`.knowledge_store/`), which takes a moment; later runs reuse it.

## Third-party content

This project includes the System Reference Document 5.2.1 under the Creative
Commons Attribution 4.0 International License. See [NOTICE.md](NOTICE.md) for
the required attribution and terms. The engine code and the Gloamvere City setting are original to this
project.
