from __future__ import annotations

import time
from typing import Any

from crewai import Agent, Crew, Process, Task

from ttrpg_crew.game.validator import audit
from ttrpg_crew.prompts.prompt_loader import load_task_config, load_task_for_game_state
from ttrpg_crew.services.agent_factory import build_agent


class PromptBackedBeatRunner:
    def __init__(
        self,
        gm_agent: Agent,
        companion_agents: dict[str, Agent],
        companion_prompt_vars: dict[str, dict[str, Any]],
        *,
        tier: str | None = None,
        llm: str | None = None,
        editor_llm: Any = None,
        llm_tier_map: dict[str, str] | None = None,
        auto_retry: bool = True,
        debug: bool = False,
    ):
        self.gm_agent = gm_agent
        self.companion_agents = companion_agents
        self.companion_prompt_vars = companion_prompt_vars
        self.tier = tier
        self.llm = llm
        # Model the editor pass runs on; defaults to the main narration model.
        self.editor_llm = editor_llm if editor_llm is not None else llm
        self.llm_tier_map = llm_tier_map
        # When True, an output that trips the auditor gets one corrective rewrite.
        self.auto_retry = auto_retry
        # debug=True prints [timing]/[audit] internals; play mode keeps it quiet.
        self.debug = debug

    def run_beat(
        self,
        *,
        game_state_name: str,
        campaign_premise: str,
        protagonist: str = "",
        cast: str = "",
        name_pool: str = "",
        style_cues: str = "",
        scene_context: str,
        recent_story: str,
        player_input: str,
        campaign_notes: str,
        session_recap: str,
        combat_state: str,
        party_members: str,
        human_player_name: str,
        selected_speakers: list[str],
        gm_intent: str,
        gm_should_frame_first: bool = True,
    ):
        tasks: list[Task] = []
        agents: list[Agent] = []

        # Per-task timing + output capture. Tasks run sequentially, so each
        # callback fires in order; we record the gap since the previous task
        # and stash the raw text for the post-beat audit.
        timings: list[tuple[str, float]] = []
        outputs: list[tuple[str, str]] = []
        clock = {"last": 0.0}

        def _timed_cb(label: str):
            def cb(output: Any) -> None:
                now = time.perf_counter()
                timings.append((label, now - clock["last"]))
                clock["last"] = now
                outputs.append((label, getattr(output, "raw", str(output))))

            return cb

        # Display names of the companions who will speak (as their own agents)
        # right after the GM frames — the GM is told to leave them alone.
        companions_speaking = ", ".join(
            str(self.companion_prompt_vars.get(sid, {}).get("character_name", sid))
            for sid in selected_speakers
        )

        common_vars = {
            "campaign_premise": campaign_premise,
            "protagonist": protagonist,
            "cast": cast,
            "name_pool": name_pool,
            "style_cues": style_cues,
            "scene_context": scene_context,
            "recent_story": recent_story,
            "player_input": player_input,
            "campaign_notes": campaign_notes,
            "session_recap": session_recap,
            "combat_state": combat_state,
            "party_members": party_members,
            "human_player_name": human_player_name,
            "selected_speakers": selected_speakers,
            "companions_speaking": companions_speaking,
            "gm_intent": gm_intent,
        }

        if gm_should_frame_first:
            gm_task_config = load_task_for_game_state(
                "gm",
                common_vars,
                game_state_name,
                tier=self.tier,
                llm=self.llm,
                llm_tier_map=self.llm_tier_map,
            )

            gm_task = Task(
                description=gm_task_config["description"],
                expected_output=gm_task_config["expected_output"],
                agent=self.gm_agent,
                callback=_timed_cb("GM"),
            )

            tasks.append(gm_task)
            agents.append(self.gm_agent)

        for speaker_id in selected_speakers:
            companion_agent = self.companion_agents[speaker_id]
            character_vars = self.companion_prompt_vars[speaker_id]

            companion_vars = {
                **common_vars,
                **character_vars,
            }

            companion_task_config = load_task_for_game_state(
                "companion",
                companion_vars,
                game_state_name,
                tier=self.tier,
                llm=self.llm,
                llm_tier_map=self.llm_tier_map,
            )

            companion_task = Task(
                description=companion_task_config["description"],
                expected_output=companion_task_config["expected_output"],
                agent=companion_agent,
                context=tasks.copy(),
                callback=_timed_cb(speaker_id),
            )

            tasks.append(companion_task)
            agents.append(companion_agent)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )

        clock["last"] = time.perf_counter()
        crew.kickoff()

        if self.debug and timings:
            parts = "  ".join(f"{label}={secs:.1f}s" for label, secs in timings)
            print(f"[timing] beat breakdown: {parts}")

        # Auditor: flag rule violations, then (if enabled) hand the offending
        # output to a dedicated editor pass that fixes it minimally. The edit is
        # kept only when it has strictly fewer violations, so it can never make
        # the beat worse.
        companion_names = tuple(
            str(v.get("character_name")).strip()
            for v in self.companion_prompt_vars.values()
            if v.get("character_name")
        )

        final: list[tuple[str, str]] = []
        for task, (label, text) in zip(tasks, outputs):
            violations = audit(
                label,
                text,
                human_player_name=human_player_name,
                companions=companion_names,
            )
            if self.debug:
                for v in violations:
                    print(f"[audit] {label}: {v.rule} — {v.detail}")

            if violations and self.auto_retry:
                edited = self._edit_output(label, text, violations, human_player_name)
                remaining = audit(
                    label, edited, human_player_name=human_player_name,
                    companions=companion_names,
                )
                if len(remaining) < len(violations):
                    text = edited
                    if self.debug:
                        fixed = len(violations) - len(remaining)
                        print(f"[audit] {label}: editor fixed {fixed}/{len(violations)} issue(s)")
                elif self.debug:
                    print(f"[audit] {label}: editor no better, kept original")

            final.append((label, text))

        return self._assemble_beat(final)

    def _edit_output(
        self, label: str, flagged_text: str, violations: list, human_player_name: str
    ) -> str:
        """Run the dedicated editor agent to fix flagged output, minimally."""
        is_gm = label == "GM"
        profile = {} if is_gm else self.companion_prompt_vars.get(label, {})
        character_name = "" if is_gm else profile.get("character_name", label)
        # The companion's few-shot voice lines, so the editor preserves their
        # register while fixing the tell (GM has no per-character voice).
        voice_examples = [] if is_gm else list(profile.get("examples") or [])

        fix_lines = []
        for v in violations:
            line = f"- {v.rule}: {v.detail}"
            if getattr(v, "suggestion", ""):
                line += f" → {v.suggestion}"
            fix_lines.append(line)

        editor_vars = {
            "role_kind": "GM" if is_gm else "companion",
            "character_name": character_name or "the companion",
            "human_player_name": human_player_name,
            "violations": "\n".join(fix_lines),
            "flagged_text": flagged_text,
            "voice_examples": voice_examples,
        }

        editor_agent = build_agent(
            "editor",
            editor_vars,
            tier=self.tier,
            llm=self.editor_llm,
            llm_tier_map=self.llm_tier_map,
            verbose=self.debug,
        )
        task_config = load_task_config(
            "editor_task.yaml",
            editor_vars,
            tier=self.tier,
            llm=self.editor_llm,
            llm_tier_map=self.llm_tier_map,
            task_key="base",
        )
        editor_task = Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=editor_agent,
        )
        crew = Crew(
            agents=[editor_agent],
            tasks=[editor_task],
            process=Process.sequential,
            verbose=False,
        )
        return str(crew.kickoff()).strip()

    def _assemble_beat(self, outputs: list[tuple[str, str]]) -> str:
        """Join the (possibly corrected) per-agent outputs into the beat text."""
        lines: list[str] = []
        for label, text in outputs:
            if label == "GM":
                lines.append(text)
            else:
                name = self.companion_prompt_vars.get(label, {}).get(
                    "character_name", label
                )
                lines.append(f"{name}: {text}")
        return "\n\n".join(lines)