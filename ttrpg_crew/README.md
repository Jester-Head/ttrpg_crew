# ttrpg_crew

A solo TTRPG engine for the Daggerheart system, set in Gloamvere City (Kavath),
run by CrewAI agents against a local LLM (LM Studio or any OpenAI-compatible
server). A Game Master agent narrates, companion agents play the party, and a
deterministic policy decides who speaks each beat. Prompts adapt to the size of
the loaded model, and an auditor plus editor pass keeps the prose free of common
AI writing tells.

## Setup

Copy `.env.example` to `.env` and adjust if needed (with no `.env`, it talks to
LM Studio on its default port and auto-detects the loaded model). Then:

```
python -m ttrpg_crew.main
```

## Third-party content

This project includes the Daggerheart SRD under the Darrington Press Community
Gaming License. See [NOTICE.md](NOTICE.md) for the required attribution and
terms. The engine code and the Gloamvere City setting are original to this
project.
