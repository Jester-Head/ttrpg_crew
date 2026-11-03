from __future__ import annotations

import random
import re
import time
from typing import Any, Callable

from crewai import Agent, Crew, Process, Task
from crewai.events import crewai_event_bus
from crewai.events.types.llm_events import LLMCallStartedEvent
from crewai.tasks.task_output import TaskOutput

from ttrpg_crew import session_log
from ttrpg_crew.game.beat import Beat, BeatSegment
from ttrpg_crew.game.intent import BeatIntent
from ttrpg_crew.game.moves import companion_move
from ttrpg_crew.game.rolls import parse_roll_request, strip_roll_tag
from ttrpg_crew.game.speaker_policy import selection_reason
from ttrpg_crew.game.stance import Stance, stance_directive
from ttrpg_crew.game.validator import (
    audit,
    duplicate_companion_line,
    extract_closing_prompt,
    normalize_rewrite,
    repeats_player_action,
    repeats_silent_action,
    silent_action,
    strip_companion_prompt,
    strip_observes,
    strip_speaker_label,
    trim_runaway,
)
from ttrpg_crew.prompts.directives import directive
from ttrpg_crew.prompts.prompt_loader import (
    load_agent_config,
    load_task_config,
    resolve_task_key,
)
from ttrpg_crew.services import direct_companions, llm_client
from ttrpg_crew.services.character_registry import examples_for

COMPANION_MODES = ("crewai", "direct", "direct_think")
# How many of a companion's recent silences a new one is checked against.
_SILENCE_MEMORY = 4


# Handlers run on a worker thread, but Crew.kickoff flushes them before it
# returns, so the prompt log's tags still match the call.
@crewai_event_bus.on(LLMCallStartedEvent)
def _log_crewai_prompt(source: Any, event: LLMCallStartedEvent) -> None:
    session_log.prompt(
        getattr(event, "agent_role", None) or "crewai",
        event.messages,
        source="crewai",
        model=event.model or "",
    )


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
        llm: Any = None,
        editor_llm: Any = None,
        auto_retry: bool = True,
        debug: bool = False,
        companion_mode: str = "crewai",
        recent_silences: dict[str, list[str]] | None = None,
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
        # Each companion's recent silent actions; owned by the flow so it
        # lasts across beats.
        self.recent_silences = recent_silences if recent_silences is not None else {}
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
        stances: dict[str, Stance | None] | None = None,
        gm_task_key: str = "narrative",
        roll_result: str = "",
        allow_roll: bool = True,
        on_segment: Callable[[BeatSegment], None] | None = None,
        gm_text: str | None = None,
        gm_only: bool = False,
        speakers_forced: bool = False,
    ) -> Beat:
        """Run one beat. `gm_text` replays a pre-generated GM frame instead of
        calling the GM, so A/B arms can share identical narration. `gm_only`
        runs just the GM (still told who will speak). `speakers_forced` means
        the caller fixed the speakers, so none of them may answer [observes]."""
        stances = stances or {}
        situations = beat_intent.situations if beat_intent else []
        speakers_to_run = [] if gm_only else selected_speakers
        self.last_notes = {}
        self.last_missing_marker = set()

        # Which prompt each companion gets this beat, and voice lines picked
        # for that moment rather than the same two every beat.
        moves = {
            sid: companion_move(sid, beat_intent, stances.get(sid), game_state_name)
            for sid in selected_speakers
        }
        for sid, move in moves.items():
            pool = self.companion_prompt_vars[sid].get("example_pool")
            self.companion_prompt_vars[sid]["examples"] = examples_for(pool, move, situations)

        def stance_vars(sid: str) -> dict[str, Any]:
            stance = stances.get(sid)
            return {
                "stance_approach": stance.approach if stance else "",
                "stance_strength": stance.strength if stance else 0,
            }

        def stance_line(sid: str) -> str:
            # An agree/disagree prompt already states the stance in full.
            if moves[sid] in ("agree", "disagree"):
                return ""
            return stance_directive(stances.get(sid))

        prompt_keys = [f"GM={gm_task_key}"] if gm_should_frame_first else []
        prompt_keys += [f"{sid}={move}" for sid, move in moves.items()]
        self._diag("[prompt] " + "  ".join(prompt_keys))

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
            "roll_result": roll_result,
        }

        companion_names = tuple(
            str(v.get("character_name")).strip()
            for v in self.companion_prompt_vars.values()
            if v.get("character_name")
        )
        presence = self._presence_line(game_state_name)
        beat = Beat()
        prior_companion_lines: list[str] = []

        def finalize(label: str, text: str, *, require_prompt: bool = True) -> str:
            return self._finalize(
                label,
                text,
                beat=beat,
                protagonist_name=protagonist_name,
                companion_names=companion_names,
                prior_companion_lines=prior_companion_lines,
                player_input=player_input,
                presence=presence,
                on_segment=on_segment,
                require_prompt=require_prompt,
            )

        # The GM runs in its own crew and is finalized before any companion
        # starts, so its narration reaches the player without waiting on them.
        # It always goes through CrewAI for now (keeps its search-tool option
        # at the large tier).
        gm_replayed = gm_should_frame_first and gm_text is not None
        # Tasks whose output each CrewAI companion sees as context.
        context_pool: list[Task] = []
        gm_frame = ""
        if gm_should_frame_first:
            gm_task_config = load_task_config(
                "gm_tasks.yaml",
                common_vars,
                tier=self.tier,
                task_key=gm_task_key,
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
                gm_crew = Crew(
                    agents=[self.gm_agent], tasks=[gm_task], process=Process.sequential,
                    verbose=False,
                )
                clock["last"] = time.perf_counter()
                gm_crew.kickoff()
                self._log_crew_tokens(gm_crew, timings, 1)

            gm_raw = next((text for label, text in outputs if label == "GM"), "")
            # A roll call ends the beat: the roll line is the hand-back, and
            # companions wait for the result instead of talking over it. Only
            # the player's own action can call for a roll, so a tag in the
            # opening or in a roll's result beat is dropped.
            roll = parse_roll_request(gm_raw)
            gm_raw = strip_roll_tag(gm_raw)
            beat.pending_roll = roll if allow_roll else None
            if beat.pending_roll is not None:
                self._diag(
                    f"[roll] requested {beat.pending_roll.skill} "
                    f"DC {beat.pending_roll.dc}; companions wait"
                )
            gm_frame = finalize("GM", gm_raw, require_prompt=beat.pending_roll is None)
            if beat.pending_roll is not None:
                beat.closing_prompt = ""

        if beat.pending_roll is not None:
            speakers_to_run = []

        if self.companion_mode != "crewai":
            # Feed each companion what has been said this beat (GM frame, then
            # prior companions) so they react in context, mirroring the CrewAI
            # path's task `context`. Prompt assembly + the LLM call live in
            # services/direct_companions.py.
            said: list[str] = [f"GM: {gm_frame}"] if gm_frame else []
            companion_said: list[str] = []
            for speaker_id in speakers_to_run:
                merged_vars = {
                    **common_vars,
                    **self.companion_prompt_vars[speaker_id],
                    **stance_vars(speaker_id),
                    "may_observe": self._may_observe(speaker_id, beat_intent, speakers_forced),
                    "party_members": self._party_for(speaker_id, protagonist_name),
                    "beat_directives": self._beat_directives(
                        speaker_id, beat_intent, selected_speakers, companion_said,
                        player_input=player_input,
                        protagonist_name=protagonist_name,
                        stance_hint=stance_line(speaker_id),
                        recent_story=recent_story,
                    ),
                }
                think_first = self.companion_mode == "direct_think"
                system, user, name = direct_companions.companion_messages(
                    agent=self.companion_agents[speaker_id],
                    merged_vars=merged_vars,
                    task_key=moves[speaker_id],
                    tier=self.tier,
                    said_so_far=said,
                    think_first=think_first,
                    closing=self._answer_first(
                        speaker_id, beat_intent, player_input, protagonist_name
                    ),
                )
                result = direct_companions.companion_call(
                    self.llm, system, user, label=speaker_id, debug=self.debug
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
                ct, secs = result.completion_tokens, result.seconds
                tps = (ct / secs) if (ct and secs) else 0.0
                self._diag(
                    f"[direct] {speaker_id}  prompt={result.prompt_tokens}  "
                    f"completion={ct}  {secs:.1f}s  {tps:.1f} tok/s"
                )
                final = finalize(speaker_id, text)
                if final:
                    said.append(f"{name}: {final}")
                    companion_said.append(f"{name}: {final}")
        elif speakers_to_run:
            tasks: list[Task] = []
            for speaker_id in speakers_to_run:
                companion_vars = {
                    **common_vars,
                    **self.companion_prompt_vars[speaker_id],
                    **stance_vars(speaker_id),
                    "may_observe": self._may_observe(speaker_id, beat_intent, speakers_forced),
                    "party_members": self._party_for(speaker_id, protagonist_name),
                    # Tasks are built before any output exists, so this carries
                    # intent + who else speaks + anti-echo (no said-lines yet;
                    # those arrive at runtime via the task context).
                    "beat_directives": self._beat_directives(
                        speaker_id, beat_intent, selected_speakers,
                        player_input=player_input,
                        protagonist_name=protagonist_name,
                        stance_hint=stance_line(speaker_id),
                        recent_story=recent_story,
                    ),
                }

                companion_task_config = load_task_config(
                    "companion_tasks.yaml",
                    companion_vars,
                    tier=self.tier,
                    task_key=moves[speaker_id],
                )

                # The GM task already ran in its own crew; CrewAI reads a
                # context task's stored output, so it still reaches companions.
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
                    agent=self.companion_agents[speaker_id],
                    context=context_pool.copy(),
                    callback=_timed_cb(speaker_id),
                )
                context_pool.append(companion_task)
                tasks.append(companion_task)

            crew = Crew(
                agents=[self.companion_agents[sid] for sid in speakers_to_run],
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

        self.last_outputs = list(outputs)

        if timings:
            parts = "  ".join(f"{label}={secs:.1f}s" for label, secs in timings)
            self._diag(f"[timing] beat breakdown: {parts}")

        logged = str(beat)
        if beat.pending_roll is not None:
            logged += f"\n\nRoll {beat.pending_roll.skill} (DC {beat.pending_roll.dc})"
        session_log.beat(player_input.strip() or "Opening", logged)
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
        player_input: str,
        presence: str,
        on_segment: Callable[[BeatSegment], None] | None,
        require_prompt: bool = True,
    ) -> str:
        """Audit one output, edit it if flagged, and deliver it as a segment.

        The edit is kept only when it has strictly fewer violations, so it can
        never make the beat worse. Returns the final text ("" when a companion
        ends up with no line).
        """
        self._diag(f"[output] {label}: {text}")
        silent = False
        if label != "GM":
            # A trailing "What do you do?" is never the companion's to say
            # (transcript-format echo); strip it before auditing.
            text = strip_companion_prompt(text)
            text, silent = self._settle_silence(label, text)
            if not text:
                self._diag(f"[silence] {label}: no line this beat")
                return ""
        check = dict(
            protagonist_name=protagonist_name,
            companion_names=companion_names,
            prior_lines=prior_companion_lines,
            player_input=player_input,
            require_prompt=require_prompt,
        )
        violations = self._violations(label, text, **check)
        for v in violations:
            self._diag(f"[audit] {label}: {v.rule} — {v.detail}")

        if violations and self.auto_retry:
            edited = self._edit_output(label, text, violations, protagonist_name)
            remaining = self._violations(label, edited, **check)
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
        segment = self._build_segment(label, text, beat, presence=presence, silent=silent)
        if segment and on_segment is not None:
            try:
                on_segment(segment)
            except Exception as exc:  # a UI bug must never kill the beat
                self._diag(f"[segment] on_segment callback failed: {exc}")
        return text

    def _violations(
        self,
        label: str,
        text: str,
        *,
        protagonist_name: str,
        companion_names: tuple[str, ...],
        prior_lines: list[str],
        player_input: str,
        require_prompt: bool = True,
    ) -> list:
        """Every auditor check for one output (rerun on the editor's rewrite)."""
        if label == "GM":
            return audit(
                label,
                text,
                protagonist_name=protagonist_name,
                companions=companion_names,
                require_prompt=require_prompt,
            )
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
        self, label: str, text: str, beat: Beat, *, presence: str = "", silent: bool = False
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
            segment = BeatSegment(label, name, text, silent=silent)
        beat.segments.append(segment)
        return segment

    def _settle_silence(self, speaker_id: str, text: str) -> tuple[str, bool]:
        """Return (text, silent) for a companion's reply.

        A reply that is one short bracketed action counts as staying quiet and
        is kept unless it repeats this companion's recent silences. A bare
        [observes] or a repeat is swapped for an unused line from the
        character's silent_beats; with none left the companion has no line
        ("" returned). Stray [observes] tags in a spoken reply are dropped."""
        text = strip_observes(text)
        action = silent_action(text) if text else None
        if text and action is None:
            return text, False

        recent = self.recent_silences.setdefault(speaker_id, [])
        if action is None or repeats_silent_action(action, recent):
            beats = self.companion_prompt_vars.get(speaker_id, {}).get("silent_beats") or []
            fresh = [
                b for b in (str(b).strip().strip("[]").strip() for b in beats)
                if b and not repeats_silent_action(b, recent)
            ]
            if not fresh:
                return "", True
            action = random.choice(fresh)
        recent.append(action)
        del recent[:-_SILENCE_MEMORY]
        return f"[{action}]", True

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

    @staticmethod
    def _may_observe(
        speaker_id: str, intent: BeatIntent | None, speakers_forced: bool
    ) -> bool:
        """Whether this companion is offered [observes] instead of a reply.

        Offering it to someone who must answer makes models speak and then
        tack the tag on, or go silent on a direct question."""
        addressed = intent is not None and intent.addressed_id == speaker_id
        return not (speakers_forced or addressed)

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
            result = llm_client.chat(
                self.editor_llm,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=getattr(self.editor_llm, "max_tokens", None),
                label="editor",
            )
        except Exception as exc:
            self._diag(f"[audit] {label}: editor call failed ({exc})")
            return flagged_text
        return normalize_rewrite(result.text, character_name)
