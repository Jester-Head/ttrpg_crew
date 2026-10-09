from __future__ import annotations

from crewai.tools import tool

from ttrpg_crew.services.knowledge import search_knowledge


@tool("Search World Knowledge")
def search_world_knowledge(query: str) -> str:
    """Look up Gloamvere City lore or D&D rules.

    Use this when you need a specific fact you don't already know — a faction's
    motives, a location, how a rule resolves, an NPC, the memory economy. Pass a
    short natural-language query (e.g. "who runs the Siderian Clade" or "how
    does grappling work"). Returns the most relevant reference excerpts.
    """
    try:
        return search_knowledge(query)  # top_k comes from models.yaml `retrieval:`
    except Exception as exc:  # keep a bad lookup from killing the beat
        return f"World knowledge lookup failed: {exc}"
