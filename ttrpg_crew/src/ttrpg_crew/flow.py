"""Top-level orchestration for a single TTRPG beat.

Each kickoff() call runs one beat:

    1. load character profiles + campaign grounding (character_registry, knowledge)
    2. game state (in-memory; save/load persistence is still future work)
    3. build prompt variables                        (character_registry)
    4. pick speakers: deterministic selector by default, LLM moderator opt-in
    5. classify player intent + assign companion stances (intent, stance)
    6. run the GM and selected companions            (beat_runner)
    7. audit/edit output, update transcript and scene state
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from crewai import Crew, Process, Task
from pydantic import ValidationError

from ttrpg_crew.config_loader import config_section
from ttrpg_crew.game.intent import classify_intent
from ttrpg_crew.game.schemas import ModeratorSelection
from ttrpg_crew.game.stance import assign_stance, stance_directive
from ttrpg_crew.game.speaker_policy import (
    calculate_soft_speaker_weights,
    select_speakers,
)
from ttrpg_crew.game.state import CharacterState, GameState
from ttrpg_crew.game.transcript import Transcript
from ttrpg_crew.prompts.prompt_loader import load_task_for_game_state
from ttrpg_crew.services.agent_factory import build_agent
from ttrpg_crew.services.beat_runner import PromptBackedBeatRunner
from ttrpg_crew.services.character_registry import (
    build_all_character_prompt_vars,
    load_character_profiles,
)
from ttrpg_crew.services.knowledge import (
    ensure_indexed,
    format_cast,
    format_protagonist,
    load_campaign_cast,
    load_campaign_frame,
    load_latest_session_recap,
    load_player_character,
    load_premise,
    require_data_root,
    sample_name_pool,
    sample_style_cues,
    search_knowledge,
)
from ttrpg_crew.tools.dice_tool import roll_dice
from ttrpg_crew.tools.knowledge_tool import search_world_knowledge


class TTRPGFlow:
    """Runs one beat of the session. Construct, then call ``kickoff()``."""

    def __init__(
        self,
        *,
        game_state: GameState | None = None,
        tier: str | None = None,
        llm: Any = None,
        manager_llm: Any = None,
        editor_llm: Any = None,
        human_player_name: str = "Player",
        debug: bool = False,
        use_llm_moderator: bool = False,
        use_direct_companions: bool = False,
    ) -> None:
        # Speaker selection: deterministic by default (fast, consistent, no LLM
        # call). Set use_llm_moderator=True to fall back to the LLM manager agent
        # (richer subtext-reading, but slow and inconsistent on small models).
        self.use_llm_moderator = use_llm_moderator
        # When True, companions skip the CrewAI Agent/ReAct wrapper and run as
        # direct chat completions (faster, no discarded reasoning). A/B toggle.
        self.use_direct_companions = use_direct_companions
        # debug=True shows engine internals (agent panels, [timing]/[audit]
        # lines); play mode keeps it False so only the story is printed.
        self.debug = debug
        self.tier = tier
        self.llm = llm
        # The moderator only emits speaker-selection JSON — it doesn't need the
        # big narration model. Route it to a faster model when one is provided,
        # otherwise reuse the main llm.
        self.manager_llm = manager_llm or llm
        # The editor fixes flagged output; model-agnostic, defaults to the main
        # model. Point it at a faster model later if desired.
        self.editor_llm = editor_llm or llm
        self.transcript = Transcript()

        # Fail fast with guidance if the campaign/lore/SRD corpus is missing
        # (e.g. installed wrong, or launched from a layout where the data root
        # can't be found). Better a clear error here than a silently empty game.
        require_data_root()

        # The protagonist anchors every scene. If a player character is defined,
        # the GM refers to the player by that name (so the boundary rules and
        # framing reference the real character, not a generic "Player").
        pc = load_player_character()
        self.protagonist = format_protagonist(pc)
        self.human_player_name = (pc.get("name") or "").strip() or human_player_name

        # The recurring campaign cast — injected each beat so these NPCs stay
        # consistent. Incidental NPCs are still generated from the name banks.
        self.cast_brief = format_cast(load_campaign_cast())

        # 1. Load character profiles (TODO: requires config/characters.yaml).
        self.character_profiles = load_character_profiles()

        # 2. Load current game state (TODO: read from a save via storage.save_game).
        self.game_state = game_state or self._new_game_state()

        # Ensure every enabled character has a CharacterState entry.
        for character_id, profile in self.character_profiles.items():
            self.game_state.characters.setdefault(
                character_id,
                CharacterState(id=character_id, name=profile.get("name", character_id)),
            )

        # Index lore + rules once (local embeddings). If the embedding server is
        # unreachable, disable retrieval and keep playing — direct injection of
        # premise/recap still works without it.
        self.knowledge_ready = False
        try:
            ensure_indexed(verbose=self.debug)
            self.knowledge_ready = True
        except Exception as exc:
            if self.debug:
                print(f"[knowledge] retrieval disabled ({exc})")

    def _new_game_state(self) -> GameState:
        """Opening state — the starting location comes from the campaign frame."""
        opening = (load_campaign_frame().get("opening") or {})
        location = str(opening.get("location") or "").strip() or "Gloamvere City"
        return GameState(
            campaign_premise="",
            location=location,
            scene_type="opening",
        )

    @staticmethod
    def _recent_story_chars() -> int:
        """Cap on the story-so-far injected each beat (models.yaml context:),
        read live so it can be tuned mid-session."""
        try:
            return int(config_section("context").get("recent_story_chars", 4000))
        except (TypeError, ValueError):
            return 4000

    def _common_vars(self, *, player_input: str = "") -> dict[str, Any]:
        gs = self.game_state
        return {
            # Direct-injected knowledge: premise + latest session recap are small
            # and high-value, so they go straight into the prompt every beat.
            "campaign_premise": gs.campaign_premise or load_premise(),
            "protagonist": self.protagonist,
            "cast": self.cast_brief,
            "name_pool": sample_name_pool(),
            "style_cues": sample_style_cues(),
            "scene_context": gs.location,
            # The story so far — the continuity the GM and companions write
            # against. Tail-capped (models.yaml context:) so long sessions
            # don't blow the model's window; newest beats always survive.
            "recent_story": self.transcript.tail(self._recent_story_chars()),
            "player_input": player_input,
            # campaign_notes carries auto-retrieved lore for tool-less tiers.
            "campaign_notes": self._grounding_lore(player_input=player_input),
            "session_recap": load_latest_session_recap(),
            "combat_state": "",
            "party_members": ", ".join(
                c.name for c in self.game_state.characters.values()
            ),
            "human_player_name": self.human_player_name,
        }

    def _grounding_lore(self, *, player_input: str = "") -> str:
        """Auto-retrieve relevant lore for small/medium tiers.

        Capable (large) tiers drive the search tool themselves, so they get
        nothing here. Weaker tiers can't use the ReAct tool reliably, so we
        fetch the lore for them and inject it — keeping faction roles and NPCs
        accurate without the fragile tool protocol.
        """
        if self.tier not in ("medium", "small") or not self.knowledge_ready:
            return ""
        seed = player_input.strip() or self.game_state.location
        query = (
            f"{seed}. Factions, the memory economy, and key figures of "
            f"Gloamvere City relevant to this scene."
        )
        try:
            return search_knowledge(query, k=3)
        except Exception:
            return ""

    def _select_speakers(
        self,
        common_vars: dict[str, Any],
    ) -> ModeratorSelection:
        """Decide who speaks this beat.

        Default: deterministic selection (game.speaker_policy.select_speakers) —
        no LLM call. With use_llm_moderator=True, run the manager agent instead
        (richer subtext, but a slow per-beat call that small models do poorly).
        """
        available_speakers = list(self.character_profiles.keys())

        if not self.use_llm_moderator:
            return select_speakers(
                character_profiles=self.character_profiles,
                game_state=self.game_state,
                player_input=common_vars["player_input"],
                recent_story=common_vars["recent_story"],
                available_speakers=available_speakers,
                scene_type=self.game_state.scene_type,
                human_player_name=self.human_player_name,
            )

        weights = calculate_soft_speaker_weights(
            self.game_state,
            self.transcript.visible_text(),
            available_speakers,
        )

        manager_vars = {
            **common_vars,
            "available_speakers": available_speakers,
            "speaker_weights": weights,
        }

        try:
            manager_agent = build_agent(
                "manager",
                manager_vars,
                tier=self.tier,
                llm=self.manager_llm,
                verbose=self.debug,
            )
            task_config = load_task_for_game_state(
                "manager",
                manager_vars,
                self.game_state.scene_type,
                tier=self.tier,
            )
            task = Task(
                description=task_config["description"],
                expected_output=task_config["expected_output"],
                agent=manager_agent,
            )
            crew = Crew(
                agents=[manager_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            raw = str(crew.kickoff())
            selection = self._parse_selection(raw, available_speakers)
        except (ValidationError, ValueError, KeyError):
            selection = None

        if selection is None:
            return ModeratorSelection(
                selected_speakers=[],
                gm_should_frame_first=True,
            )
        return selection

    def _parse_selection(
        self,
        raw: str,
        available_speakers: list[str],
    ) -> ModeratorSelection | None:
        """Parse the moderator's raw output into a validated ModeratorSelection.

        Tolerates code fences and surrounding prose, and drops any selected
        ids that are not actually available speakers.
        """
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None

        data = json.loads(match.group(0))
        selection = ModeratorSelection.model_validate(data)

        allowed = set(available_speakers)
        selection.selected_speakers = [
            speaker_id
            for speaker_id in selection.selected_speakers
            if speaker_id in allowed
        ]
        return selection

    def kickoff(self, *, player_input: str = "", on_segment: Any = None) -> Any:
        beat_start = time.perf_counter()
        common_vars = self._common_vars(player_input=player_input)

        # 3. Build prompt variables for each companion.
        companion_prompt_vars = build_all_character_prompt_vars(
            character_profiles=self.character_profiles,
            common_vars=common_vars,
            game_state=self.game_state,
        )

        # 4. Run the moderator to select speakers.
        moderator_start = time.perf_counter()
        selection = self._select_speakers(common_vars)
        moderator_secs = time.perf_counter() - moderator_start

        # Build the agents for this beat.
        # Agentic tools use CrewAI's ReAct text protocol, which small/medium
        # models routinely fumble (e.g. emitting `Action: none` and stalling).
        # Gate tools to capable (large) tiers; weaker tiers rely on the
        # direct-injected premise/recap and plain narration instead.
        gm_tools: list[Any] = []
        if self.tier in (None, "large"):
            gm_tools.append(roll_dice)
            if self.knowledge_ready:
                gm_tools.append(search_world_knowledge)

        gm_agent = build_agent(
            "gm",
            common_vars,
            tier=self.tier,
            llm=self.llm,
            tools=gm_tools,
            verbose=self.debug,
        )
        companion_agents = {
            speaker_id: build_agent(
                "companion",
                companion_prompt_vars[speaker_id],
                tier=self.tier,
                llm=self.llm,
                verbose=self.debug,
            )
            for speaker_id in selection.selected_speakers
        }

        # 5. Run the selected agents.
        runner = PromptBackedBeatRunner(
            gm_agent=gm_agent,
            companion_agents=companion_agents,
            companion_prompt_vars=companion_prompt_vars,
            tier=self.tier,
            llm=self.llm,
            editor_llm=self.editor_llm,
            debug=self.debug,
            direct_companions=self.use_direct_companions,
        )

        # What the player's input MEANS, computed in code (no LLM call) and told
        # to the agents outright, so small models don't have to infer it.
        beat_intent = classify_intent(
            player_input=player_input,
            character_profiles=self.character_profiles,
            available_speakers=list(self.character_profiles.keys()),
        )

        # Each selected companion's stance toward the player's approach (the old
        # AutoGen BanterGenerator, deterministic): the engine picks the move,
        # the model performs it. Empty string = no stance this beat.
        stance_hints = {
            sid: stance_directive(
                assign_stance(self.character_profiles.get(sid, {}), player_input)
            )
            for sid in selection.selected_speakers
        }

        beat_call_start = time.perf_counter()
        result = runner.run_beat(
            game_state_name=self.game_state.scene_type,
            campaign_premise=common_vars["campaign_premise"],
            protagonist=common_vars["protagonist"],
            cast=common_vars["cast"],
            name_pool=common_vars["name_pool"],
            style_cues=common_vars["style_cues"],
            scene_context=common_vars["scene_context"],
            recent_story=common_vars["recent_story"],
            player_input=player_input,
            campaign_notes=common_vars["campaign_notes"],
            session_recap=common_vars["session_recap"],
            combat_state=common_vars["combat_state"],
            party_members=common_vars["party_members"],
            human_player_name=self.human_player_name,
            selected_speakers=selection.selected_speakers,
            gm_intent=selection.gm_intent,
            gm_should_frame_first=selection.gm_should_frame_first,
            beat_intent=beat_intent,
            stance_hints=stance_hints,
            on_segment=on_segment,
        )
        beat_secs = time.perf_counter() - beat_call_start

        # 6. Update state and record continuity so the next beat remembers this
        # turn (the player's action, then the beat) and the scene advances.
        self._update_turn_counts(selection.selected_speakers)
        if player_input.strip():
            self.transcript.append(self.human_player_name, player_input.strip())
        self.transcript.append("GM", str(result))
        self._advance_scene(str(result))

        if self.debug:
            n_speakers = len(selection.selected_speakers)
            gm_calls = 1 if selection.gm_should_frame_first else 0
            print(
                f"[timing] moderator={moderator_secs:.1f}s  "
                f"beat={beat_secs:.1f}s ({gm_calls} GM + {n_speakers} companions)  "
                f"total={time.perf_counter() - beat_start:.1f}s"
            )

        return result

    def _advance_scene(self, beat_text: str) -> None:
        """Light scene progression from GM tags in the narration."""
        loc = re.search(r"\[LOCATION:\s*([^\]]+)\]", beat_text)
        if loc:
            self.game_state.location = loc.group(1).strip()
        if "[COMBAT" in beat_text:
            self.game_state.scene_type = "combat"
        elif self.game_state.scene_type == "opening":
            self.game_state.scene_type = "narrative"

    def _update_turn_counts(self, spoke_this_beat: list[str]) -> None:
        spoke = set(spoke_this_beat)
        for character_id, state in self.game_state.characters.items():
            if character_id in spoke:
                state.recent_turns += 1
                state.total_turns += 1
            else:
                # Decay so the spotlight rotates back over time.
                state.recent_turns = max(0, state.recent_turns - 1)
