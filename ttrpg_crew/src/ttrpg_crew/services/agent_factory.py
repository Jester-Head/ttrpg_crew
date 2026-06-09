from __future__ import annotations

from typing import Any, Literal

from crewai import Agent

from ttrpg_crew.prompts.prompt_loader import load_agent_config


AgentType = Literal["gm", "companion", "manager", "editor"]


_AGENT_FILES: dict[AgentType, str] = {
    "gm": "gm_agent.yaml",
    "companion": "companion_agent.yaml",
    "manager": "manager_agent.yaml",
    "editor": "editor_agent.yaml",
}


def build_agent(
    agent_type: AgentType,
    variables: dict[str, Any],
    *,
    tier: str | None = None,
    llm: str | None = None,
    llm_tier_map: dict[str, str] | None = None,
    verbose: bool = True,
    allow_delegation: bool = False,
    tools: list[Any] | None = None,
) -> Agent:
    agent_file = _AGENT_FILES[agent_type]

    config = load_agent_config(
        agent_file,
        variables,
        tier=tier,
        llm=llm,
        llm_tier_map=llm_tier_map,
    )

    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm,
        verbose=verbose,
        allow_delegation=allow_delegation,
        tools=tools or [],
    )