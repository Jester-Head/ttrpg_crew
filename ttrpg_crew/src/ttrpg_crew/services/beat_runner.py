from __future__ import annotations

import re
import time
from typing import Any, Callable

from crewai import Agent, Crew, Process, Task
from crewai.tasks.task_output import TaskOutput

from ttrpg_crew import session_log
from ttrpg_crew.game.beat import Beat, BeatSegment
from ttrpg_crew.game.intent import BeatIntent
from ttrpg_crew.game.speaker_policy import selection_reason
from ttrpg_crew.game.validator import (
    audit,
    duplicate_companion_line,
    extract_closing_prompt,
    normalize_rewrite,
    repeats_player_action,
    strip_companion_prompt,
    strip_speaker_label,
    trim_runaway,
)
from ttrpg_crew.prompts.directives import directive
from ttrpg_crew.prompts.prompt_loader import (
    load_task_config,
    load_task_for_game_state,
    resolve_task_key,
)
from ttrpg_crew.services import direct_companions
from ttrpg_crew.services.agent_factory import build_agent

COMPANION_MODES = ("crewai", "direct", "direct_think")


def _quoted(player_input: str) -> str:
    """The player's words, ending in punctuation, for quoting inside a prompt line."""
    text = player_input.strip()
    return text + "." if text and text[-1] not in ".!?\"'" else text


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
        companion_mode: str = "crewai",
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
        # "crewai": companions share the GM's crew. "direct": plain chat
        # completions, no ReAct wrapper. "direct_think": direct, but the model
        # writes a private NOTE: before its SAY: line. The GM runs through
        # CrewAI in every mode.
        if companion_mode not in COMPANION_MODES:
            raise ValueError(f"companion_mode must be one of {COMPANION_MODES}")
        self.companion_mode = companion_mode
        # Raw per-speaker outputs and think-first notes from the last beat,
        # before audit/edit (read by the A/B harness).
        self.last_outputs: list[tuple[str, str]] = []
        self.last_notes: dict[str, str] = {}
        self.last_missing_marker: set[str] = set()

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
        protagonist_name: str = "Player",
        selected_speakers: list[str],
        gm_intent: str,
        gm_should_frame_first: bool = True,
        beat_intent: BeatIntent | None = None,
        stance_hints: dict[str, str] | None = None,
        on_segment: Callable[[BeatSegment], None] | None = None,
        gm_text: str | None = None,
        gm_only: bool = False,
    ) -> Beat:
        """Run one beat. `gm_text` replays a pre-generated GM frame instead of
        calling the GM, so A/B arms can share identical narration. `gm_only`
        runs just the GM (still told who will speak)."""
        stance_hints = stance_hints or {}
        speakers_to_run = [] if gm_only else selected_speakers
        self.last_notes = {}
        self.last_missing_marker = set()
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
                text = trim_runaway(getattr(output, "raw", str(output)), is_gm=label == "GM")
                # Later tasks read their context from this same object, so the
                # trim also keeps a runaway out of the next companion's prompt.
                if hasattr(output, "raw"):
                    output.raw = text
                outputs.append((label, text))

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
            "protagonist_name": protagonist_name,
            "selected_speakers": selected_speakers,
            "companions_speaking": companions_speaking,
            "gm_intent": gm_intent,
        }

        # GM always runs through CrewAI for now (keeps its search-tool option at
        # the large tier). Companions either share that crew (default) or run
        # as plain chat completions below.
        gm_replayed = gm_should_frame_first and gm_text is not None
        # Tasks whose output each CrewAI companion sees as context.
        context_pool: list[Task] = []
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
                callback=None if gm_replayed else _timed_cb("GM"),
            )
            context_pool.append(gm_task)

            if gm_replayed:
                # Context is read from task.output, so a replayed frame reaches
                # CrewAI companions exactly as a live GM task's output would.
                gm_task.output = TaskOutput(
                    description=gm_task.description, raw=gm_text, agent=self.gm_agent.role
                )
                outputs.append(("GM", gm_text))
            else:
                tasks.append(gm_task)
                agents.append(self.gm_agent)

        if self.companion_mode != "crewai":
            # GM (if live) runs in its own crew; companions are direct calls.
            if tasks:
                gm_crew = Crew(
                    agents=agents, tasks=tasks, process=Process.sequential, verbose=False
                )
                clock["last"] = time.perf_counter()
                gm_crew.kickoff()
                self._log_crew_tokens(gm_crew, timings, len(tasks))

            # Feed each companion what has been said this beat (GM frame, then
            # prior companions) so they react in context, mirroring the CrewAI
            # path's task `context`. Prompt assembly + the LLM call live in
            # services/direct_companions.py.
            gm_frame = next((text for label, text in outputs if label == "GM"), "")
            said: list[str] = [f"GM: {gm_frame}"] if gm_frame else []
            companion_said: list[str] = []
            for speaker_id in speakers_to_run:
                merged_vars = {
                    **common_vars,
                    **self.companion_prompt_vars[speaker_id],
                    "party_members": self._party_for(speaker_id, protagonist_name),
                    "beat_directives": self._beat_directives(
                        speaker_id, beat_intent, selected_speakers, companion_said,
                        player_input=player_input,
                        protagonist_name=protagonist_name,
                        stance_hint=stance_hints.get(speaker_id, ""),
                        recent_story=recent_story,
                    ),
                }
                think_first = self.companion_mode == "direct_think"
                system, user, name = direct_companions.companion_messages(
                    agent=self.companion_agents[speaker_id],
                    merged_vars=merged_vars,
                    game_state_name=game_state_name,
                    tier=self.tier,
                    said_so_far=said,
                    think_first=think_first,
                    closing=self._answer_first(
                        speaker_id, beat_intent, player_input, protagonist_name
                    ),
                )
                result = direct_companions.companion_call(
                    self.llm, system, user, debug=self.debug
                )
                text = result.text
                if think_first:
                    note, text, found = direct_companions.split_thought(text)
                    self.last_notes[speaker_id] = note
                    if not found:
                        self.last_missing_marker.add(speaker_id)
                    self._diag(f"[think] {speaker_id} note: {note or '(none)'}")
                text = trim_runaway(text, is_gm=False)
                timings.append((speaker_id, result.seconds))
                outputs.append((speaker_id, text))
                said.append(f"{name}: {text}")
                companion_said.append(f"{name}: {text}")
                ct, secs = result.completion_tokens, result.seconds
                tps = (ct / secs) if (ct and secs) else 0.0
                self._diag(
                    f"[direct] {speaker_id}  prompt={result.prompt_tokens}  "
                    f"completion={ct}  {secs:.1f}s  {tps:.1f} tok/s"
                )
        else:
            for speaker_id in speakers_to_run:
                companion_agent = self.companion_agents[speaker_id]
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
                        protagonist_name=protagonist_name,
                        stance_hint=stance_hints.get(speaker_id, ""),
                        recent_story=recent_story,
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
                    expected_output="\n\n".join(
                        part for part in (
                            companion_task_config["expected_output"],
                            self._answer_first(
                                speaker_id, beat_intent, player_input, protagonist_name
                            ),
                        ) if part
                    ),
                    agent=companion_agent,
                    context=context_pool.copy(),
                    callback=_timed_cb(speaker_id),
                )

                context_pool.append(companion_task)
                tasks.append(companion_task)
                agents.append(companion_agent)

            if tasks:
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=False,
                )

                clock["last"] = time.perf_counter()
                crew.kickoff()
                self._log_crew_tokens(crew, timings, len(tasks))

        self.last_outputs = list(outputs)

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

        presence = self._presence_line(game_state_name)
        beat = Beat()
        prior_companion_lines: list[str] = []
        for label, text in outputs:
            self._diag(f"[output] {label}: {text}")
            if label != "GM":
                # A trailing "What do you do?" is never the companion's to say
                # (transcript-format echo); strip it before auditing.
                text = strip_companion_prompt(text)
            check = dict(
                protagonist_name=protagonist_name,
                companion_names=companion_names,
                prior_lines=prior_companion_lines,
                player_input=player_input,
            )
            violations = self._violations(label, text, **check)
            for v in violations:
                self._diag(f"[audit] {label}: {v.rule} — {v.detail}")

            if violations and self.auto_retry:
                edited = self._edit_output(label, text, violations, protagonist_name)
                remaining = self._violations(label, edited, **check)
                if len(remaining) < len(violations):
                    text = edited
                    fixed = len(violations) - len(remaining)
                    self._diag(f"[audit] {label}: editor fixed {fixed}/{len(violations)} issue(s)")
                else:
                    self._diag(f"[audit] {label}: editor no better, kept original")

            if label != "GM":
                prior_companion_lines.append(text)

            # The segment is final (audited + edited) the moment it leaves this
            # loop iteration — deliver it to the streaming consumer immediately.
            # Contract note: typing indicators will eventually need a
            # "speaker started" signal too; expect this single-event hook to
            # grow a sibling rather than treating its shape as settled.
            segment = self._build_segment(label, text, beat, presence=presence)
            if segment and on_segment is not None:
                try:
                    on_segment(segment)
                except Exception as exc:  # a UI bug must never kill the beat
                    self._diag(f"[segment] on_segment callback failed: {exc}")

        session_log.beat(player_input.strip() or "Opening", str(beat))
        return beat

    def _violations(
        self,
        label: str,
        text: str,
        *,
        protagonist_name: str,
        companion_names: tuple[str, ...],
        prior_lines: list[str],
        player_input: str,
    ) -> list:
        """Every auditor check for one output (rerun on the editor's rewrite)."""
        if label == "GM":
            return audit(label, text, protagonist_name=protagonist_name, companions=companion_names)
        found = audit(
            label,
            text,
            protagonist_name=protagonist_name,
            companions=companion_names,
            character_name=str(
                self.companion_prompt_vars.get(label, {}).get("character_name", label)
            ),
        )
        for extra in (
            duplicate_companion_line(text, prior_lines),
            repeats_player_action(text, player_input),
        ):
            if extra:
                found.append(extra)
        return found

    def _presence_line(self, game_state_name: str) -> str:
        """The engine-written "who is here" line for opening and resume beats.

        Models skip party placement even when told to, so the engine states it.
        """
        if resolve_task_key("gm", game_state_name) not in ("opening_scene", "session_resume"):
            return ""
        names = [
            str(v.get("character_name")).strip()
            for v in self.companion_prompt_vars.values()
            if v.get("character_name")
        ]
        if not names:
            return ""
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return directive("party_present", names=joined)

    def _build_segment(
        self, label: str, text: str, beat: Beat, *, presence: str = ""
    ) -> BeatSegment | None:
        """Finalize one output into a BeatSegment appended to the beat.

        GM text gets its leaked transcript label stripped, its closing
        hand-back prompt lifted into beat.closing_prompt (rendered last, after
        every companion has spoken), and the presence line added unless the GM
        already named every companion.
        """
        if label == "GM":
            text = strip_speaker_label(text, "GM")
            text, prompt = extract_closing_prompt(text)
            if prompt:
                beat.closing_prompt = prompt
            names = [
                str(v.get("character_name")).split()[0]
                for v in self.companion_prompt_vars.values()
                if v.get("character_name")
            ]
            if presence and not all(re.search(rf"\b{re.escape(n)}\b", text) for n in names):
                text = f"{text}\n\n{presence}".strip()
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

    def _answer_first(
        self,
        speaker_id: str,
        intent: BeatIntent | None,
        player_input: str,
        protagonist_name: str,
    ) -> str:
        """The player's direct question again, placed last in the prompt.

        Stated only inside the THIS BEAT block, it gets lost under the scene
        and the companion talks about something else."""
        if intent is None or intent.addressed_id != speaker_id or intent.kind != "direct_question":
            return ""
        return directive(
            "answer_first", protagonist=protagonist_name, player_input=_quoted(player_input)
        )

    def _beat_directives(
        self,
        speaker_id: str,
        intent: BeatIntent | None,
        selected_speakers: list[str],
        said_lines: list[str] | None = None,
        player_input: str = "",
        protagonist_name: str = "Player",
        stance_hint: str = "",
        recent_story: str = "",
    ) -> str:
        """A code-built THIS BEAT block telling one companion what the moment is.

        Small models infer poorly from raw transcript, so the engine states it
        outright: what the player's input means (intent), why this companion
        is speaking, who else speaks, what was already said, and not to echo
        it. WHICH lines fire is decided here; the wording comes from
        config/directives.yaml (live-editable).
        """
        name = self.companion_prompt_vars.get(speaker_id, {}).get("character_name", speaker_id)
        lines = [directive("header")]
        # The player's words go in every intent line: companion task prompts
        # carry no player_input, so without this they guess from the GM's text.
        fields = {
            "name": name,
            "player_input": _quoted(player_input),
            "protagonist": protagonist_name,
        }

        if intent is not None:
            if intent.addressed_id == speaker_id:
                key = "direct_question" if intent.kind == "direct_question" else "direct_address"
                lines.append(directive(key, **fields))
            elif intent.addressed_id:
                lines.append(
                    directive("addressed_other", other=intent.addressed_name or "", **fields)
                )
            elif intent.kind == "question":
                lines.append(directive("question", **fields))
            elif intent.kind == "action":
                lines.append(directive("action", **fields))

            # The selector already decided this companion speaks; say why, so
            # the model gets an angle instead of a speak-or-not judgment call.
            # Same text window the selector scores against.
            reason = ""
            if intent.addressed_id != speaker_id:
                reason = selection_reason(
                    self.companion_prompt_vars.get(speaker_id, {}),
                    f"{player_input}\n{recent_story[-600:]}",
                )
            if reason:
                lines.append(directive("why_you", reason=reason, **fields))
            elif intent.situations:
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
        return normalize_rewrite(str(crew.kickoff()), character_name)

