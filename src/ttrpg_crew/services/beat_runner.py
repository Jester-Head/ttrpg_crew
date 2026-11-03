from __future__ import annotations

import time
from typing import Any, Callable

from crewai import Agent, Crew, Process, Task

from ttrpg_crew import session_log
from ttrpg_crew.game.beat import Beat, BeatSegment
from ttrpg_crew.game.intent import BeatIntent
from ttrpg_crew.game.validator import (
    audit,
    duplicate_companion_line,
    extract_closing_prompt,
    normalize_rewrite,
    strip_companion_prompt,
    strip_speaker_label,
)
from ttrpg_crew.prompts.directives import directive
from ttrpg_crew.prompts.prompt_loader import (
    load_agent_config,
    load_task_config,
    load_task_for_game_state,
)
from ttrpg_crew.services import direct_companions
from ttrpg_crew.services.llm_call import chat_completion


class PromptBackedBeatRunner:
    def __init__(
        self,
        gm_agent: Agent,
        companion_agents: dict[str, Agent],
        companion_prompt_vars: dict[str, dict[str, Any]],
        *,
        tier: str | None = None,
        llm: Any = None,
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
        scene_memory: str = "",
        campaign_notes: str,
        session_recap: str,
        combat_state: str,
        party_members: str,
        protagonist_name: str = "Player",
        selected_speakers: list[str],
        gm_intent: str,
        gm_should_frame_first: bool = True,
        beat_intent: BeatIntent | None = None,
        stance_hints: dict[str, str] | None = None,
        on_segment: Callable[[BeatSegment], None] | None = None,
    ) -> Beat:
        stance_hints = stance_hints or {}

        # Per-task timing + output capture. Tasks run sequentially, so each
        # callback fires in order; we record the gap since the previous task
        # and stash the raw text for the audit.
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
            "scene_memory": scene_memory,
            "recent_story": recent_story,
            "player_input": player_input,
            "campaign_notes": campaign_notes,
            "session_recap": session_recap,
            "combat_state": combat_state,
            "party_members": party_members,
            "protagonist_name": protagonist_name,
            "selected_speakers": selected_speakers,
            "companions_speaking": companions_speaking,
            "gm_intent": gm_intent,
        }

        beat = Beat()
        prior_companion_lines: list[str] = []
        companion_names = tuple(
            str(v.get("character_name")).strip()
            for v in self.companion_prompt_vars.values()
            if v.get("character_name")
        )

        def finalize(label: str, text: str) -> str:
            return self._finalize(
                label,
                text,
                beat=beat,
                protagonist_name=protagonist_name,
                companion_names=companion_names,
                prior_companion_lines=prior_companion_lines,
                on_segment=on_segment,
            )

        # The GM runs in its own crew and is finalized before any companion
        # starts, so its narration reaches the player without waiting on them.
        # It always goes through CrewAI for now (keeps its search-tool option
        # at the large tier).
        gm_task: Task | None = None
        gm_text = ""
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
            gm_crew = Crew(
                agents=[self.gm_agent], tasks=[gm_task], process=Process.sequential, verbose=False
            )
            clock["last"] = time.perf_counter()
            gm_crew.kickoff()
            self._log_crew_tokens(gm_crew, timings, 1)
            gm_raw = next((text for label, text in outputs if label == "GM"), "")
            gm_text = finalize("GM", gm_raw)

        if self.direct_companions:
            # Feed each companion what has been said this beat (GM frame, then
            # prior companions) so they react in context, mirroring the CrewAI
            # path's task `context`. Prompt assembly + the LLM call live in
            # services/direct_companions.py.
            said: list[str] = [f"GM: {gm_text}"] if gm_text else []
            companion_said: list[str] = []
            for speaker_id in selected_speakers:
                merged_vars = {
                    **common_vars,
                    **self.companion_prompt_vars[speaker_id],
                    "party_members": self._party_for(speaker_id, protagonist_name),
                    "beat_directives": self._beat_directives(
                        speaker_id, beat_intent, selected_speakers, companion_said,
                        player_input=player_input,
                        stance_hint=stance_hints.get(speaker_id, ""),
                    ),
                }
                system, user, name = direct_companions.companion_messages(
                    agent=self.companion_agents[speaker_id],
                    merged_vars=merged_vars,
                    game_state_name=game_state_name,
                    tier=self.tier,
                    said_so_far=said,
                )
                text, secs, prompt_tokens, completion_tokens = (
                    direct_companions.companion_call(
                        self.llm, system, user, debug=self.debug
                    )
                )
                timings.append((speaker_id, secs))
                tps = (completion_tokens / secs) if (completion_tokens and secs) else 0.0
                self._diag(
                    f"[direct] {speaker_id}  prompt={prompt_tokens}  "
                    f"completion={completion_tokens}  {secs:.1f}s  {tps:.1f} tok/s"
                )
                text = finalize(speaker_id, text)
                said.append(f"{name}: {text}")
                companion_said.append(f"{name}: {text}")
        elif selected_speakers:
            tasks: list[Task] = []
            for speaker_id in selected_speakers:
                companion_vars = {
                    **common_vars,
                    **self.companion_prompt_vars[speaker_id],
                    "party_members": self._party_for(speaker_id, protagonist_name),
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

                # The GM task already ran in its own crew; CrewAI reads a
                # context task's stored output, so it still reaches companions.
                companion_task = Task(
                    description=companion_task_config["description"],
                    expected_output=companion_task_config["expected_output"],
                    agent=self.companion_agents[speaker_id],
                    context=([gm_task] if gm_task else []) + tasks,
                    callback=_timed_cb(speaker_id),
                )
                tasks.append(companion_task)

            crew = Crew(
                agents=[self.companion_agents[sid] for sid in selected_speakers],
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
            )
            clock["last"] = time.perf_counter()
            crew.kickoff()
            self._log_crew_tokens(crew, timings, len(tasks))
            for label, text in outputs:
                if label != "GM":
                    finalize(label, text)

        if timings:
            parts = "  ".join(f"{label}={secs:.1f}s" for label, secs in timings)
            self._diag(f"[timing] beat breakdown: {parts}")

        session_log.beat(player_input.strip() or "Opening", str(beat))
        return beat

    def _finalize(
        self,
        label: str,
        text: str,
        *,
        beat: Beat,
        protagonist_name: str,
        companion_names: tuple[str, ...],
        prior_companion_lines: list[str],
        on_segment: Callable[[BeatSegment], None] | None,
    ) -> str:
        """Audit one output, edit it if flagged, and deliver it as a segment.

        The edit is kept only when it has strictly fewer violations, so it can
        never make the beat worse. Returns the final text.
        """
        self._diag(f"[output] {label}: {text}")
        character_name = ""
        if label != "GM":
            # A trailing "What do you do?" is never the companion's to say
            # (transcript-format echo); strip it before auditing.
            text = strip_companion_prompt(text)
            character_name = str(
                self.companion_prompt_vars.get(label, {}).get("character_name", label)
            )
        violations = audit(
            label,
            text,
            protagonist_name=protagonist_name,
            companions=companion_names,
            character_name=character_name,
        )
        if label != "GM":
            dup = duplicate_companion_line(text, prior_companion_lines)
            if dup:
                violations.append(dup)
        for v in violations:
            self._diag(f"[audit] {label}: {v.rule} — {v.detail}")

        if violations and self.auto_retry:
            edited = self._edit_output(label, text, violations, protagonist_name)
            remaining = audit(
                label,
                edited,
                protagonist_name=protagonist_name,
                companions=companion_names,
                character_name=character_name,
            )
            if label != "GM":
                dup = duplicate_companion_line(edited, prior_companion_lines)
                if dup:
                    remaining.append(dup)
            # An empty edit passes every check, so it would always "win".
            if edited.strip() and len(remaining) < len(violations):
                text = edited
                fixed = len(violations) - len(remaining)
                self._diag(f"[audit] {label}: editor fixed {fixed}/{len(violations)} issue(s)")
            else:
                self._diag(f"[audit] {label}: editor no better, kept original")

        if label != "GM":
            prior_companion_lines.append(text)

        # Contract note: typing indicators will eventually need a "speaker
        # started" signal too; expect this single-event hook to grow a sibling
        # rather than treating its shape as settled.
        segment = self._build_segment(label, text, beat)
        if segment and on_segment is not None:
            try:
                on_segment(segment)
            except Exception as exc:  # a UI bug must never kill the beat
                self._diag(f"[segment] on_segment callback failed: {exc}")
        return text

    def _build_segment(self, label: str, text: str, beat: Beat) -> BeatSegment | None:
        """Finalize one output into a BeatSegment appended to the beat.

        GM text gets its leaked transcript label stripped and its closing
        hand-back prompt lifted into beat.closing_prompt (rendered last, after
        every companion has spoken).
        """
        if label == "GM":
            text = strip_speaker_label(text, "GM")
            text, prompt = extract_closing_prompt(text)
            if prompt:
                beat.closing_prompt = prompt
            if not text:
                return None
            segment = BeatSegment("GM", "GM", text)
        else:
            name = str(
                self.companion_prompt_vars.get(label, {}).get("character_name", label)
            )
            segment = BeatSegment(label, name, text)
        beat.segments.append(segment)
        return segment

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

    def _party_for(self, speaker_id: str, protagonist_name: str) -> str:
        """The party roster from this companion's seat: the human + the OTHER
        companions. Excluding themselves stops third-person self-references
        (ported from the old version, which did the same)."""
        others = [
            str(v.get("character_name", sid))
            for sid, v in self.companion_prompt_vars.items()
            if sid != speaker_id
        ]
        return ", ".join([protagonist_name] + others)

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

    def _edit_output(
        self, label: str, flagged_text: str, violations: list, protagonist_name: str
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

        # NPCs (campaign cast) are GM-played and distinct from companions: the
        # editor's examples use a real NPC name so the boundary reads concrete.
        from ttrpg_crew.services.knowledge import load_campaign_cast

        cast = load_campaign_cast()
        npc_name = str(cast[0].get("name")) if cast else "the NPC"
        companion_example = next(
            (str(v.get("character_name")) for v in self.companion_prompt_vars.values()
             if v.get("character_name")),
            "the companion",
        )
        editor_vars = {
            "role_kind": "GM" if is_gm else "companion",
            "character_name": character_name or "the companion",
            "npc_name": npc_name,
            "companion_example": companion_example,
            "protagonist_name": protagonist_name,
            "violations": "\n".join(fix_lines),
            "flagged_text": flagged_text,
            "voice_examples": voice_examples,
        }

        # A plain chat call, not a CrewAI agent: a one-shot rewrite gains
        # nothing from the agent loop, and only a direct call can be capped
        # with max_tokens (see models.yaml `llm:`).
        agent_config = load_agent_config("editor_agent.yaml", editor_vars, tier=self.tier)
        task_config = load_task_config(
            "editor_task.yaml",
            editor_vars,
            tier=self.tier,
            task_key="base",
        )
        system = "\n\n".join(
            part
            for part in (
                agent_config["role"].strip(),
                "Your goal: " + agent_config["goal"].strip(),
                agent_config["backstory"].strip(),
            )
            if part
        )
        user = task_config["description"].strip()
        if (task_config.get("expected_output") or "").strip():
            user += "\n\nOutput: " + task_config["expected_output"].strip()
        try:
            resp = chat_completion(
                self.editor_llm,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            self._diag(f"[audit] {label}: editor call failed ({exc})")
            return flagged_text
        return normalize_rewrite(raw, character_name)

