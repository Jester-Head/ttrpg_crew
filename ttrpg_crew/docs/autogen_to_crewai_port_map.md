# AutoGen -> CrewAI Adaptation Map

Survey of the old version (`expert-waffle/project`, AutoGen) and what it takes to
bring each foundational system into the new version (`ttrpg_crew`, CrewAI).

## Headline finding

Almost nothing is actually AutoGen-coupled. The save system, the **full
Daggerheart resolver**, the fine-tune logger, the Hope/Fear economy, and the UI
are all framework-agnostic Python. The *only* truly AutoGen-specific piece is the
turn-driving / agent layer (`GameEngine` in `app/cli.py` using
`agent.on_messages` + `TextMessage` + `SpotlightManager`), and the new repo has
**already re-implemented that** as `flow.py` + `services/beat_runner.py` +
`game/speaker_policy.py` + `game/stance.py`.

So "port to CrewAI" mostly means: lift the plain-Python systems into `ttrpg_crew`,
reshape them onto the new `GameState`/`CharacterState`, and wire them to the beat
loop. It is not a rewrite.

## Effort key

- **1:1** — lift and reshape onto new state objects; logic unchanged.
- **Light** — port, then rewire to `flow.py` / `beat_runner`.
- **Rebuild** — orchestration that was AutoGen-coupled; new repo already has the
  replacement, old code is the behavior reference.
- **New** — not built in either version; future/team work.

## System-by-system

| System | Old location | Coupling | New home | Effort |
|---|---|---|---|---|
| Save / load | `game/persistence.py` (`GameSession`, `SaveManager`) | None | new module over `GameState` | 1:1 |
| **Roll resolver** | `game/combat.py` (`DiceRoller.roll_action`, `RollOutcome`) | None | `game/` resolver | 1:1 |
| Combat data | `game/combat.py` (`Combatant`, `CombatState`, `CombatManager`) | None | `game/` | 1:1 |
| Adversary loader | `game/combat.py` (`AdversaryLoader`) | None | already have `adversaries.yaml`; port loader | 1:1 |
| Consequence cues | `data/combat_reactions.yaml` (`CombatReactionsLoader`) | None | sampled cue bank | 1:1 |
| Fine-tune logger | `telemetry/conversation_logger.py` | None | `telemetry/` | 1:1 + enrich |
| Hope / Fear economy | session + character (`spend/gain_hope`, `add/spend_fear`) | None | `GameState`/`CharacterState` | 1:1 |
| Rolling continuity | session `story_summary` + `session_summaries` (LLM on save) | Light (LLM call) | `flow.py` save path | Light |
| Scene memory | `memory/game_memory.py` | None | mostly already in `GameState` | 1:1 (subsumed) |
| Correction store | `rag/correction_store.py` (human edits -> few-shot) | None | optional; new repo uses auditor/editor | Light |
| Party mgmt | `game/party.py` (minus recruitment) | None | over the roster | Light |
| UI / state views | `ui/*` + cli `/status /sheet /notes /roll /hope` | None (terminal) | presentation over `flow.py` | Light |
| Turn driver | `app/cli.py` `GameEngine.run_turn` etc. | **Heavy (AutoGen)** | `flow.py` + `beat_runner` + `speaker_policy` | Rebuild (done) |
| GM/companion agents | `agents/factory.py`, `game_agents.py`, `companions.py` | **Heavy (AutoGen)** | CrewAI agents in `beat_runner` | Rebuild (done) |

## The pieces worth knowing in detail

### Save / load (`game/persistence.py`) — 1:1
`GameSession` is one big serializable dataclass (`to_dict`/`from_dict`);
`SaveManager` does manual save + rotating auto-save (every 5 turns) + list/load/
delete, plain JSON. Already includes `story_summary`, `session_summaries`,
`story_notes`, `important_npcs`, `plot_threads`. **The save file is the
anti-cheat boundary** we designed around — this is exactly that layer. Port the
manager as-is; rewrite `to_dict` to serialize the new `GameState`/
`CharacterState` instead of the old session shape.

### The resolver already exists (`game/combat.py`) — 1:1
`DiceRoller.roll_action(trait_modifier, difficulty)` does the Daggerheart duality
dice and returns one of five `RollOutcome` tiers (crit / success-hope /
success-fear / failure-hope / failure-fear), with advantage/disadvantage,
crit-doubles, and the Hope/Fear bookkeeping per SRD. `process_attack` applies
damage through armor, generates Fear, adds the crit bonus. **This IS the
deterministic resolver from `roll_resolution_design.md`** — we re-derived
something already built. Porting it is lift-and-drop (pure `random` + math). The
*new* part is only the GM-narration handoff (the nudge / `resolve` task) wired
into the CrewAI beat, not the dice.

### Consequence cues already exist (`data/combat_reactions.yaml`)
`CombatReactionsLoader.get_roll_reaction(personality, outcome)` returns a random
line keyed by personality and the five outcomes — i.e. the rotating
per-tier cue bank from the anti-mimicry design, already in data form. Port the
data + the random-sample accessor; it slots straight into the `resolve` nudge.

### Fine-tune logger (`telemetry/conversation_logger.py`) — 1:1 + enrich
`ConversationLogger.log_turn` already writes OpenAI-format JSONL per turn with
`system_prompt` + full messages + `rating`/`edited_response`/`score`/`issue_tags`,
plus filtered `export_training_data` and `merge_training_files`. This is the
fine-tune capture layer from `social_exploration_design.md`, already built. Port
it and call `log_turn` from `beat_runner`. **Enrichment (new):** also stamp the
validator verdict, the roll record, and state deltas so SFT (filter unflagged)
and DPO (flagged original vs `edited_response`) datasets fall out — the
`edited_response`/`rating` fields are already shaped for the preference pairs.

### Hope / Fear economy — 1:1, and it underpins the headline mechanic
`fear_tokens` on the session, `hope` on the character, with `spend_hope`,
`gain_hope`, `add_fear`, `spend_fear` already implemented, and combat generating
Fear per SRD. So the *plumbing* for "Fear as the GM's escalation budget" exists;
what's a team decision is the GM *behavior* (when/how it spends Fear narratively),
not the counters.

### Continuity (rolling summaries) — Light
On save, the old `GameEngine.generate_save_summary` asks the GM for a summary,
stores it as `story_summary`, and appends to `session_summaries` (keeps last 10).
The new repo has `load_latest_session_recap` but not the generate-on-save half.
Port that one method into the `flow.py` save path.

## Not built in either version (future / team)

These showed up as gaps in the design review and are **not** ports — there's
nothing to lift:

- **Director / pacing / arc enforcement.** Old version had only phases + speaker
  spotlight + static `plot_threads`. New repo has `frame.yaml` clocks as *data*
  but no engine that advances the story toward the arc.
- **Progression / leveling.** Characters carry `level`/`tier` fields in both
  versions; neither has level-up logic.
- **Memory-as-currency as a player mechanic.** Lore only, both versions.
- **Fear-as-escalation-budget as GM behavior.** Counters exist; the spending
  doctrine is undesigned (team decision).

## Suggested port order

1. **Hope/Fear onto `GameState`/`CharacterState`** — tiny, unblocks the resolver
   and combat.
2. **Resolver** (`DiceRoller`/`RollOutcome`) as a standalone `game/` module —
   pure, testable, no framework.
3. **Save/load** over the new state — makes everything persistent and locks in the
   anti-cheat boundary.
4. **Fine-tune logger** wired into `beat_runner`, enriched with verdict + roll +
   deltas — start accruing data immediately (cheap now, expensive to retrofit).
5. **The `resolve` nudge** (from `roll_resolution_design.md`) tying the resolver
   to the GM beat, using `combat_reactions.yaml` as the cue bank.
6. **Combat loop** orchestration over `flow.py` (turn order, enemy activation,
   companions) — the one piece with real AutoGen-seam rework, using old
   `CombatManager` as the behavior reference.
7. **UI / state views** (`/status`, `/sheet`, Hope/Fear/clock displays) as
   read-only presentation — the "player sees but can't edit" surface.

Items 1-5 are mostly mechanical ports. Item 6 is where the framework difference
actually bites. Item 7 is presentation.

## The one real AutoGen seam

Old `GameEngine.run_turn` drove agents with
`await agent.on_messages([TextMessage(content, source=...)], cancellation_token)`
and picked speakers via `SpotlightManager.select_next_speaker`. CrewAI doesn't
use that turn model — `flow.py` + `beat_runner` already orchestrate beats and
`game/speaker_policy.py` already does deterministic speaker selection. So when
porting combat/party/etc., the rule is: take the **logic and data** from the old
classes, drop the **`on_messages`/`TextMessage`/`SpotlightManager` calls**, and
invoke the new beat loop instead.
```
