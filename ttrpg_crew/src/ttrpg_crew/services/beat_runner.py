from __future__ import annotations

import re
import time
from typing import Any

from crewai import Agent, Crew, Process, Task

from ttrpg_crew import session_log
from ttrpg_crew.game.intent import BeatIntent
from ttrpg_crew.game.validator import audit, duplicate_companion_line
from ttrpg_crew.prompts.directives import directive
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
        auto_retry: bool = True,
        debug: bool = False,
        direct_companions: bool = False,
    ):
        self.gm_agent = gm_agent
        self.companion_agents = companion_agents
        self.companion_prompt_vars = companion_prompt_vars
        self.tier = tier
        self.llm = llm
        # Model the editor pass runs on; defaults to the main narration model.
        self.editor_llm = editor_llm if editor_llm is not None else llm
        # When True, an output that trips the auditor gets one corrective rewrite.
        self.auto_retry = auto_retry
        # debug=True prints [timing]/[audit] internals; play mode keeps it quiet.
        self.debug = debug
        # When True, companions are generated with a direct chat completion (no
        # CrewAI Agent/ReAct wrapper), which avoids the discarded "Thought:"
        # preamble. The GM still runs through CrewAI. A/B toggle for now.
        self.direct_companions = direct_companions

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
        beat_intent: BeatIntent | None = None,
        stance_hints: dict[str, str] | None = None,
    ):
        stance_hints = stance_hints or {}
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

        # GM always runs through CrewAI for now (keeps its search-tool option at
        # the large tier). Companions either share that crew (default) or, when
        # direct_companions is on, run as plain chat completions below.
        if gm_should_frame_first:
            gm_task_config = load_task_for_game_state(
                "gm",
                common_vars,
                game_state_name,
                tier=self.tier,
            )

            gm_task = Task(
                description=gm_task_config["description"],
                expected_output=gm_task_config["expected_output"],
                agent=self.gm_agent,
                callback=_timed_cb("GM"),
            )

            tasks.append(gm_task)
            agents.append(self.gm_agent)

        if self.direct_companions:
            # GM (if any) runs in its own crew; companions are direct calls.
            if tasks:
                gm_crew = Crew(
                    agents=agents, tasks=tasks, process=Process.sequential, verbose=False
                )
                clock["last"] = time.perf_counter()
                gm_crew.kickoff()
                self._log_crew_tokens(gm_crew, timings, len(tasks))

            # Feed each companion what has been said this beat (GM frame, then
            # prior companions) so they react in context, mirroring the CrewAI
            # path's task `context`.
            gm_text = next((text for label, text in outputs if label == "GM"), "")
            said: list[str] = [f"GM: {gm_text}"] if gm_text else []
            companion_said: list[str] = []
            for speaker_id in selected_speakers:
                speaker_vars = {
                    **common_vars,
                    "party_members": self._party_for(speaker_id, human_player_name),
                    "beat_directives": self._beat_directives(
                        speaker_id, beat_intent, selected_speakers, companion_said,
                        player_input=player_input,
                        stance_hint=stance_hints.get(speaker_id, ""),
                    ),
                }
                system, user, name = self._companion_prompt(
                    speaker_id, speaker_vars, game_state_name, said
                )
                text, secs, prompt_tokens, completion_tokens = self._direct_companion_call(
                    system, user
                )
                timings.append((speaker_id, secs))
                outputs.append((speaker_id, text))
                said.append(f"{name}: {text}")
                companion_said.append(f"{name}: {text}")
                tps = (completion_tokens / secs) if (completion_tokens and secs) else 0.0
                self._diag(
                    f"[direct] {speaker_id}  prompt={prompt_tokens}  "
                    f"completion={completion_tokens}  {secs:.1f}s  {tps:.1f} tok/s"
                )
        else:
            for speaker_id in selected_speakers:
                companion_agent = self.companion_agents[speaker_id]
                companion_vars = {
                    **common_vars,
                    **self.companion_prompt_vars[speaker_id],
                    "party_members": self._party_for(speaker_id, human_player_name),
                    # Tasks are built before any output exists, so this carries
                    # intent + who else speaks + anti-echo (no said-lines yet;
                    # those arrive at runtime via the task context).
                    "beat_directives": self._beat_directives(
                        speaker_id, beat_intent, selected_speakers,
                        player_input=player_input,
                        stance_hint=stance_hints.get(speaker_id, ""),
                    ),
                }

                companion_task_config = load_task_for_game_state(
                    "companion",
                    companion_vars,
                    game_state_name,
                    tier=self.tier,
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
            self._log_crew_tokens(crew, timings, len(tasks))

        if timings:
            parts = "  ".join(f"{label}={secs:.1f}s" for label, secs in timings)
            self._diag(f"[timing] beat breakdown: {parts}")

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
        prior_companion_lines: list[str] = []
        for label, text in outputs:
            self._diag(f"[output] {label}: {text}")
            violations = audit(
                label,
                text,
                human_player_name=human_player_name,
                companions=companion_names,
            )
            if label != "GM":
                dup = duplicate_companion_line(text, prior_companion_lines)
                if dup:
                    violations.append(dup)
            for v in violations:
                self._diag(f"[audit] {label}: {v.rule} — {v.detail}")

            if violations and self.auto_retry:
                edited = self._edit_output(label, text, violations, human_player_name)
                remaining = audit(
                    label, edited, human_player_name=human_player_name,
                    companions=companion_names,
                )
                if len(remaining) < len(violations):
                    text = edited
                    fixed = len(violations) - len(remaining)
                    self._diag(f"[audit] {label}: editor fixed {fixed}/{len(violations)} issue(s)")
                else:
                    self._diag(f"[audit] {label}: editor no better, kept original")

            if label != "GM":
                prior_companion_lines.append(text)
            final.append((label, text))

        result = self._assemble_beat(final)
        session_log.beat(player_input.strip() or "Opening", result)
        return result

    def _diag(self, line: str) -> None:
        """Diagnostic sink: print to console only in debug, but always log to file."""
        if self.debug:
            print(line)
        session_log.diag(line)

    def _log_crew_tokens(self, crew: Crew, timings: list, task_count: int) -> None:
        """Log this crew's token accounting (console in debug, always to file).

        Read the line as: completion tokens far above the words you actually see
        means discarded reasoning (CrewAI's "Thought:" preamble); requests above
        task_count means an agent took extra ReAct loops; low completion tok/s
        means the model/quant/GPU is the bottleneck, not the prompt.
        """
        um = getattr(crew, "usage_metrics", None)
        if um is None:
            return
        gen_secs = sum(secs for _, secs in timings) or 1e-9
        pt = getattr(um, "prompt_tokens", 0) or 0
        cached = getattr(um, "cached_prompt_tokens", 0) or 0
        ct = getattr(um, "completion_tokens", 0) or 0
        reqs = getattr(um, "successful_requests", 0) or 0
        self._diag(
            f"[tokens] prompt={pt} (cached {cached})  completion={ct}  "
            f"requests={reqs} vs tasks={task_count}  {ct / gen_secs:.1f} completion tok/s"
        )

    def _party_for(self, speaker_id: str, human_player_name: str) -> str:
        """The party roster from this companion's seat: the human + the OTHER
        companions. Excluding themselves stops third-person self-references
        (ported from the old version, which did the same)."""
        others = [
            str(v.get("character_name", sid))
            for sid, v in self.companion_prompt_vars.items()
            if sid != speaker_id
        ]
        return ", ".join([human_player_name] + others)

    def _beat_directives(
        self,
        speaker_id: str,
        intent: BeatIntent | None,
        selected_speakers: list[str],
        said_lines: list[str] | None = None,
        player_input: str = "",
        stance_hint: str = "",
    ) -> str:
        """A code-built THIS BEAT block telling one companion what the moment is.

        Small models infer poorly from raw transcript, so the engine states it
        outright: what the player's input means (intent), who else speaks, what
        was already said, and not to echo it. WHICH lines fire is decided here;
        the wording comes from config/directives.yaml (live-editable).
        """
        name = self.companion_prompt_vars.get(speaker_id, {}).get("character_name", speaker_id)
        lines = [directive("header")]
        quoted = player_input.strip()

        if intent is not None:
            if intent.addressed_id == speaker_id:
                key = "direct_question" if intent.kind == "direct_question" else "direct_address"
                lines.append(directive(key, name=name, player_input=quoted))
            elif intent.kind in ("question", "direct_question"):
                lines.append(directive("question"))
            elif intent.kind == "action":
                lines.append(directive("action"))
            if intent.situations:
                lines.append(directive("situation", situations=", ".join(intent.situations)))

        if stance_hint:
            lines.append(stance_hint)

        others = [
            str(self.companion_prompt_vars.get(sid, {}).get("character_name", sid))
            for sid in selected_speakers
            if sid != speaker_id
        ]
        if others:
            lines.append(directive("also_speaking", others=", ".join(others)))
        if said_lines:
            lines.append(directive("already_said"))
            lines.extend(f"    {said}" for said in said_lines)
        if others or said_lines:
            lines.append(directive("anti_echo", name=name))
        return "\n".join(lines) if len(lines) > 1 else ""

    def _companion_prompt(
        self,
        speaker_id: str,
        common_vars: dict[str, Any],
        game_state_name: str,
        said_so_far: list[str],
    ) -> tuple[str, str, str]:
        """Assemble (system, user, name) for a direct companion call.

        Reuses the same persona (agent role/goal/backstory) and scene/rules
        (companion task) as the CrewAI path, so an A/B isolates the ReAct wrapper
        rather than the prompt content.
        """
        agent = self.companion_agents[speaker_id]
        merged = {**common_vars, **self.companion_prompt_vars[speaker_id]}
        name = str(merged.get("character_name", speaker_id))

        cfg = load_task_for_game_state(
            "companion",
            merged,
            game_state_name,
            tier=self.tier,
        )

        system = "\n\n".join(
            part
            for part in (
                (agent.role or "").strip(),
                ("Your goal: " + agent.goal.strip()) if (agent.goal or "").strip() else "",
                (agent.backstory or "").strip(),
            )
            if part
        )

        user_parts: list[str] = []
        if said_so_far:
            user_parts.append(directive("said_so_far") + "\n" + "\n".join(said_so_far))
        user_parts.append(cfg["description"].strip())
        if (cfg.get("expected_output") or "").strip():
            user_parts.append("Output: " + cfg["expected_output"].strip())
        user_parts.append(directive("respond_now", name=name))
        return system, "\n\n".join(user_parts), name

    def _direct_companion_call(
        self, system: str, user: str, prefill: str = ""
    ) -> tuple[str, float, Any, Any]:
        """One companion turn as a plain chat completion — no CrewAI/ReAct wrapper.

        `prefill` pre-shapes the reply by seeding the assistant turn (small
        models hold format much better mid-answer). Not every OpenAI-compatible
        server supports a trailing assistant message, so an empty or failed
        prefilled call quietly retries without it.

        Returns (text, seconds, prompt_tokens, completion_tokens). Token counts
        are None when the call path can't report usage.
        """
        base_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        start = time.perf_counter()
        text, prompt_tokens, completion_tokens = "", None, None

        def _call(messages):
            import litellm

            params: dict[str, Any] = {}
            for attr in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "timeout"):
                value = getattr(self.llm, attr, None)
                if value is not None:
                    params[attr] = value
            return litellm.completion(
                model=getattr(self.llm, "model", None),
                api_base=getattr(self.llm, "base_url", None),
                api_key=getattr(self.llm, "api_key", None),
                messages=messages,
                **params,
            )

        try:
            resp = None
            if prefill:
                try:
                    resp = _call(base_messages + [{"role": "assistant", "content": prefill}])
                    if not (resp.choices[0].message.content or "").strip():
                        resp = None  # server ignored the prefill; retry plain
                except Exception:
                    resp = None
            if resp is None:
                prefill = ""
                resp = _call(base_messages)
            text = (prefill + (resp.choices[0].message.content or "")).strip()
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
        except Exception as exc:  # fall back to the crewai wrapper (text only)
            if self.debug:
                print(f"[direct] litellm.completion failed ({exc}); using llm.call")
            text = str(self.llm.call(base_messages)).strip()
        return text, time.perf_counter() - start, prompt_tokens, completion_tokens

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
            verbose=self.debug,
        )
        task_config = load_task_config(
            "editor_task.yaml",
            editor_vars,
            tier=self.tier,
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

    # The GM's closing hand-back-to-player line (in optional quotes/markdown),
    # at the very end of its output. The leading wrapper only matches right
    # after whitespace so a quote that closes the preceding dialogue isn't
    # swallowed into the lifted prompt.
    _CLOSING_PROMPT_RE = re.compile(
        r'(?:(?<=\s)["*_]+|^["*_]+|(?<=\s)|^)'
        r"(?:what (?:do|will) you do|what do you say|how do you respond"
        r"|what'?s your (?:move|next move))"
        r'[?!.]*["*_]*\s*$',
        re.I,
    )

    def _assemble_beat(self, outputs: list[tuple[str, str]]) -> str:
        """Join the (possibly corrected) per-agent outputs into the beat text.

        Reading order: GM frame, then companions, then the GM's hand-back
        prompt LAST. The GM generates "What do you do?" at the end of its own
        output, but companions speak after it, so the prompt is lifted off the
        GM text and re-appended after the final companion line.
        """
        lines: list[str] = []
        closing_prompt = ""
        for label, text in outputs:
            if label == "GM":
                # The story-so-far is injected as "GM: ..." transcript lines and
                # models sometimes imitate the format; strip a leaked label.
                text = re.sub(r"^\s*GM:\s*", "", text)
                m = self._CLOSING_PROMPT_RE.search(text)
                if m:
                    closing_prompt = m.group(0).strip(' "*_\n')
                    text = text[: m.start()].rstrip()
                if text:
                    lines.append(text)
            else:
                name = self.companion_prompt_vars.get(label, {}).get(
                    "character_name", label
                )
                lines.append(f"{name}: {text}")
        if closing_prompt:
            lines.append(closing_prompt)
        return "\n\n".join(lines)