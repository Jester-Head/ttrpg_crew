# Roll Resolution + GM Narration — Design Notes

Status: design only, not implemented. Captures the architecture we worked out
for deterministic mechanics, GM narration of fixed outcomes, anti-cheat,
transparency logging, and graceful degradation of inevitable agent failures.

## Core principle

**State is truth. The engine owns the roll and every number it touches. The GM
narrates; narration is cosmetic and can never move state.**

Everything below follows from that one rule.

## Ownership split

| Concern | Owner |
|---|---|
| Dice, success/fail, Hope/Fear, Stress/HP arithmetic, damage | Engine (deterministic) |
| Writing those into `GameState` | Engine only |
| *Calling* for a roll, setting Difficulty | GM (its traditional job) |
| Narrating the resolved outcome | GM (constrained) |
| Choosing the *fiction* of a consequence on a Fear/failure | GM |
| Applying that consequence's *mechanical* effect | Engine (GM never stamps numbers) |
| Player HP / resources / whether an action succeeded | Engine state, never prose |

## Beat + roll lifecycle (pseudocode)

```
on player_input:
    beat = GM.narrate(context, player_input)          # 1 LLM call
    if beat contains [ROLL: trait vs difficulty]:
        result = resolver.resolve(trait, difficulty, actor, seed)   # deterministic, NO LLM
        state.apply(result.deltas)                     # ONLY the engine writes state
        log.write_resolution(result, context_window)   # transparency receipt
        nudge = build_nudge(result, tier, sample_consequence_cues(result.tier))
        outcome_beat = GM.narrate_outcome(nudge)        # 2nd call OR fold into next beat
        transcript.append(outcome_beat)
    transcript.append(beat)
    # NOTE: leaked GM puppeteering is harmless here — narration never wrote
    # state, so the player's next declared action just resolves against state.
```

The GM's authority is "declare the stakes" (`[ROLL: ...]`), never "produce the
result."

## The resolver (Daggerheart duality dice)

```
resolve(trait, difficulty, actor, seed):
    rng  = seeded(seed)
    hope = rng.d12()
    fear = rng.d12()
    total = hope + fear + actor.traits[trait]
    crit       = (hope == fear)
    success    = crit or (total >= difficulty)
    with_hope  = hope > fear
    with_fear  = fear > hope

    deltas = {}
    if crit:           deltas.player_hope += 1; deltas.actor_stress -= 1
    elif with_hope:    deltas.player_hope += 1      # success or fail, player gains Hope
    elif with_fear:    deltas.gm_fear     += 1      # GM gains a Fear to spend

    tier = name(success, with_hope, with_fear, crit)
    return Result(hope, fear, total, difficulty, success,
                  with_hope, with_fear, crit, deltas, tier)
```

Five outcome tiers: `critical`, `success_hope`, `success_fear`, `failure_hope`,
`failure_fear`.

## The GM nudge ("resolve" task)

A dedicated `task_key: resolve` in `gm_tasks.yaml`, rendered with the result as
variables. It **fixes the outcome (closed) and leaves the fiction (open).**

Split the prompt by what you want obeyed vs. what you don't want fixated:

- **Static / rigid (copy it, please):** the outcome fact + discipline.
  e.g. "FAILURE with Fear. They do NOT succeed. Do not convert it into an
  advantage. Don't reroll. Don't recite dice/numbers. Hand the turn back."
- **Sampled / rotating (the flavor):** 2 consequence keywords drawn fresh each
  beat from a per-tier bank (same mechanism as `sample_style_cues`). Generic
  categories the GM dramatizes with the current scene's nouns.

### Tier-specific (model size = prompt weight)

- **small:** rigid outcome fact + "don't soften" + 2 rotating sampled cues.
  NO vivid-scene example (that's the fixation magnet).
- **medium:** adds one bad->good contrast.
- **large:** rule + full bad->good range (generalizes from the rule; examples
  just add range).

### Example bad->good (for medium/large), per the two most-botched tiers

```
Failure with Fear
  BAD : "Your blade nearly finds the gap; the guard stumbles back, rattled."
        (converted a failure into momentum)
  GOOD: "Your blade glances off the pauldron. The guard's already shouting,
         boots answer from the stairwell."

Success with Fear
  BAD : "The lock gives and the door swings wide. Easy." (ignored the Fear)
  GOOD: "The lock gives, but the tumblers clack loud in the quiet, and a
         chair scrapes back somewhere above."
```

## Anti-mimicry via rotating cues

Small models copy few-shot examples verbatim (our prior finding: in-context
examples cause mimicry / fixation). Don't fight the copy — make the copied thing
**rotate**, so a verbatim copy is generic AND different each beat.

```
consequence_cues:           # new bank, shape like style_palette / name_banks
  failure_fear: [alarm raised, footing lost, weapon fouled, reinforcements,
                 an opening for the enemy, a cost to a companion, ...]
  success_fear: [noise made, attention drawn, time lost, a tell left behind, ...]
  failure_hope: [...]   success_hope: [...]   critical: [...]

sample_consequence_cues(tier, n=2) -> random.sample(bank[tier], n)   # fresh per beat
```

Copy of a rotating input = variety, not fixation. Degrades to "bland-but-correct"
on the weakest models, which beats "vivid-but-fixated." Phrase entries as
evocative fragments and tell the model "seed, not a line."

## Anti-cheat model

- `GameState` is mutated ONLY by `resolver.apply()` (and explicit save load).
  Narration cannot call state mutations.
- The chat interface exposes NO state-editing command. The only cheat path is
  editing the save file — out of band, high friction, self-selected. (Accepted:
  "if they want it that badly they can edit the save, but not while playing.")
- **Player authority is scoped to intent, never outcome.** The player owns what
  their character attempts / says / feels (volition + inner life). They never
  own whether it worked, NPC/world reactions, established facts, or resolved
  rolls. This is structural, not a player-facing "override" command — outcomes
  come from the resolver, so the player simply can't author them.
- This boundary is the SAME line the `player_puppeteering` auditor already
  draws (volitional verbs + inner-life nouns flagged; world-on-body consequence
  not). The player may retcon exactly the category the GM was never allowed to
  author.

## Graceful degradation (assume failures leak; make them land softly)

| Failure mode | How it lands softly |
|---|---|
| GM puppeteers the player's action | Harmless: narration didn't move state; player's next declared action resolves against state; self-heals, no rewrite |
| Editor rewrite is poor | Avoided for mechanics — engine owns them; editor only touches prose flavor |
| Small model copies examples | Defused — flavor cues rotate per beat |
| Model softens a failure into a win | Fenced — rigid outcome fact + bad->good + tier anchor; validator flags success-language on a failure tier |
| Context truncation / forgotten detail | Absorbed in-fiction (memory economy / Zizi / the Fraying) — SMALL details only, used sparingly |

Throughline: stop depending on any LLM doing a good repair. Deterministic +
free layers (state ownership, rotating cues, player-intent-only) carry the load;
the editor LLM is the last resort, not the thing between a glitch and the player.

## Validator upgrades

- **New — mechanical fabrication:** flag a GM beat that asserts dice values,
  success/fail, or resource deltas the resolver did NOT produce this turn.
  (Now deterministically checkable, because numbers live in state.)
- **New — outcome contradiction:** on a `resolve` beat, flag narration whose
  tier contradicts `result.tier` (at least guard explicit success/fail words on
  a failure tier).
- **Keep — player_puppeteering** (agency verbs + inner-life nouns), now
  reinforced because outcomes are no longer prose-owned.

## Transparency log (per resolution, JSON, beside session logs)

The receipt that proves the engine rolled it, not the GM. Also fine-tune/debug
data.

```
{
  "beat_id": ..., "timestamp": ...,
  "actor": ..., "trait": ..., "difficulty": ..., "reason": ...,   # inputs
  "seed": ..., "hope_die": ..., "fear_die": ..., "modifier": ..., "total": ...,  # rng
  "success": bool, "with_hope": bool, "with_fear": bool,
  "critical": bool, "tier": ...,                                  # outcome
  "state_before": {hope, fear, stress, hp},
  "state_after":  {hope, fear, stress, hp},                       # deltas
  "transcript_window": [ ...N beats before, ...N after ]          # context
}
```

## Open decisions to pin before coding

1. **Difficulty source** — GM-set (flexible, slightly inconsistent) vs.
   table-driven from action type (consistent, less flavorful). Start GM-set.
2. **Seedable RNG** — yes; log the seed. Free now, makes eval + bug repros
   deterministic later.
3. **Consequence application** — GM picks the move's fiction; engine applies the
   mechanical effect. GM never stamps numbers.
4. **Cadence** — 2-call (narrate setup -> resolve -> narrate outcome) vs 1-call
   (fold the resolved outcome into the next beat's context like player input).
   Nudge content is identical either way; choose per overhead budget.
5. **Log window size** — how many beats before/after each resolution to bottle.
6. **Roll trigger** — exact `[ROLL: ...]` tag grammar and where DC selection
   lives.

## Reuses existing machinery

- `[ROLL:]` follows the existing `[LOCATION:]` / `[COMBAT:]` tag pattern.
- consequence-cue sampling mirrors `sample_name_pool()` / `sample_style_cues()`
  in `services/knowledge.py`, knobs in `models.yaml` `sampling:`.
- `resolve` is just another `task_key` + tier set in `gm_tasks.yaml`.
- result deltas write through `GameState` / `CharacterState`.
```
