# Roleplay / Social / Exploration — Design Notes

Status: design only, not implemented. The non-combat pillar. Shares the resolver,
nudge, anti-cheat, and graceful-degradation architecture from
`roll_resolution_design.md`; this doc covers what's different — the soft state,
when to roll, and how the data feeds eventual fine-tuning.

Scope cut: **recruitment is out for now.** Companions are a pre-set party (the
roster in `config/characters.yaml`); no acquisition flow. `knowledge/campaign/
recruitment.yaml` stays as dormant data — parked, not wired.

## Core principle (same as combat)

**State is truth. The GM narrates around state; narration can never move state.**

Combat's "truth" is HP/Stress/dice. This pillar's truth is softer but the rule is
identical: the things worth being consistent about live in state, not prose.

## What "state" means here

| Tracked in state (truth) | Mutated by |
|---|---|
| Location / scene | `[LOCATION:]` tag -> `game_state.location` |
| Who is present (NPCs, companions) | scene transitions |
| NPC disposition toward the party (the social "HP") | resolved social rolls + scripted beats only |
| Clocks (The Fraying, The Clade Closes In, ...) | outcomes (see below); already in `frame.yaml` |
| Known facts / discoveries | revealed on success; prevents re-reveal/contradiction |
| Hope / Fear / Stress | carried from the shared resolver |
| Companion stance / approval | `game/stance.py`, selection |

Everything else — NPC voice, room texture, the fiction — is GM narration, and is
allowed to glitch harmlessly because it isn't authoritative.

## The loop (narration-default, roll-as-exception)

Most of this pillar is free narration: player declares intent, GM narrates the
world/NPC response. A roll is the **exception**, not the rhythm.

```
on player_input:
    if needs_roll(intent):
        GM emits [ROLL: trait vs difficulty]      # -> shared resolver + nudge
    else:
        GM.narrate(context, intent)               # just say yes / say no, move on

needs_roll(intent):
    return uncertain(intent) AND consequential(intent)
    # uncertain  : outcome genuinely in doubt (NPC might refuse; search might fail)
    # consequential: failure changes the fiction or ticks a clock
    # not both -> narrate it; don't tax flow with a pointless roll
```

Over-rolling kills conversation; under-rolling removes stakes. "Uncertain AND
consequential" is the gate.

## Rolls in this pillar (shared resolver, different consequences)

Same duality-dice resolver and five tiers as combat. What changes is the
**consequence space** — instead of damage/stress, social/exploration rolls move
the soft state:

- **Social** (Presence to persuade, Instinct to read, Finesse to deceive):
  consequence = disposition shift, a price named, info gained/withheld, attention.
- **Exploration** (Instinct/Knowledge to search/recall, Agility to traverse):
  consequence = discovery, a clock tick, a hazard, time burned, getting noticed.

```
on resolved_social_roll(npc, result):
    shift = tier_to_disposition_shift(result.tier)   # engine table, NOT GM whim
    npc.disposition = clamp(npc.disposition + shift)
    # GM then narrates the NPC consistent with the new disposition;
    # it cannot exceed that disposition in prose.

on any_outcome(result):
    if result.with_fear:        tick(active_danger_clock)
    if investigation_success:   tick(progress_clock)
    # clocks live in frame/game_state; GM narrates the PRESSURE, never invents a tick
```

## Consequence cue banks (rotating, per tier — same anti-mimicry trick)

Parallel to combat's `consequence_cues`, with social/exploration variants,
sampled fresh each beat:

```
social_cues:
  failure_fear: [they take offense, a price is named, a lie catches up,
                 an ally is embarrassed, the room turns cold, ...]
  success_fear: [they agree but remember it, a favor is owed, you're watched, ...]
explore_cues:
  failure_fear: [you're turned around, time burned, a hazard springs,
                 a clue spoiled, you're noticed, ...]
  success_fear: [you find it but you're not alone, a noise gives you away, ...]
```

Small tier: rigid outcome fact + 2 sampled cues, no vivid example. Medium/large:
add bad->good. Same tier weighting as combat.

## Anti-cheat applied here

The soft state is the anchor:

- **NPC disposition lives in state, not prose.** The player can't sweet-talk an
  NPC into anything for free — belief/agreement is a *resolved outcome*, not
  something the player or a leaked GM line can assert. Player owns what they
  *say*; the engine owns whether it *lands*. (Same intent-not-outcome line as
  combat.)
- **Known facts live in state.** The GM can't retcon what's been discovered, and
  the player can't claim knowledge they haven't earned.
- **Clocks live in state.** Progress/tension can't be hand-waved in either
  direction.

So the cheat surface is the same as everywhere: edit the save, out of band. The
chat interface can't move disposition, facts, or clocks.

## Graceful degradation (this pillar)

| Failure mode | How it lands softly |
|---|---|
| GM flip-flops an NPC's attitude | Anchored by `npc.disposition`; narration that exceeds it is the bug, and is checkable |
| GM re-reveals / contradicts a known fact | Known-facts state re-grounds it |
| GM invents progress ("you've nearly cracked it") | Clocks are state; unearned progress has no tick behind it |
| Leaked puppeteering in a social beat | Harmless — the player's actual next line resolves against state |
| Small-model mimicry of social examples | Rotating cues defuse it |
| Forgotten earlier detail (truncation) | Absorbed in-fiction (memory economy / Zizi / the Fraying), small details only |

## Companions in scenes

Already handled: deterministic speaker selection (`game/speaker_policy.py`) +
stance (`game/stance.py`) decide who chimes in and how warmly, from the
`selection` blocks on each roster entry. Social/exploration is where that system
mostly lives. Companion approval can be tracked state too (a future hook), moved
by the player's choices — same disposition pattern as NPCs.

## Exploration specifics

- **Clocks are the progress engine.** `frame.yaml` already defines them. Success
  on investigation ticks progress; Fear ticks danger. The session's tension is a
  function of clock state, not GM mood.
- **Location/discovery** ride the existing `[LOCATION:]` tag and a known-facts
  set. Moving somewhere new = scene transition; finding something = a fact flips.

## Fine-tuning: design the capture NOW

(Per the side note — you'll fine-tune eventually. The cost of logging in a
trainable shape from the start is ~zero; bolting it on later is expensive.)

Every beat should already emit a structured record. Stamp it so it doubles as a
training example:

```
{
  "scene_type": narrative|social|exploration|combat,
  "tier": small|medium|large,                 # which prompt variant ran
  "state_before": {...}, "state_after": {...},
  "player_input": ...,
  "output_segments": [ {role: gm|companion, text, tags} ],
  "roll": {...} | null,                        # the resolver record if any
  "audit": { flagged: bool, violations: [...] },   # validator verdict = a LABEL
  "edit": { rewritten: bool, before, after } | null
}
```

Two datasets fall out for free:

- **SFT** — every *unflagged* beat is "given this context+state, produce
  rule-compliant GM/companion output." Filter on `audit.flagged == false`.
- **Preference / DPO** — every flagged original vs. its editor rewrite (or vs. a
  clean beat) is a preference pair. The auditor verdict is the label; you don't
  have to hand-annotate.

Why it matters beyond data: a model fine-tuned on this internalizes the rules
(don't puppeteer, honor the tier, voice discipline, don't soften failures), which
is the *long-term* version of "shape the failure distribution" — it shrinks how
often the auditor/editor has to fire at all. Fine-tuning and the
graceful-degradation stack are the same goal at different time horizons.

Keep role-separation, state stamps, and tags clean in the logs from day one so
the dataset is a filter away, not a parsing project.

## Open decisions to pin

1. **Disposition model** — numeric (-3..+3) vs named states (hostile/wary/
   neutral/warm/ally). Named is easier for the GM prompt to consume.
2. **Disposition shift table** — how each roll tier moves it; caps per scene so
   one roll can't swing an NPC end to end.
3. **Known-facts granularity** — freeform flags vs a structured discovery list
   per scene/mystery.
4. **Roll gate wording** — how to teach "uncertain AND consequential" to the GM
   so it stops rolling for trivia (small models over-roll or never roll).
5. **Companion approval** — track now or later; same disposition machinery if so.
6. **Clock triggers** — exact mapping of outcomes -> which clock ticks.

## Reuses existing machinery

- This pillar *is* the `narrative` task_key in `gm_tasks.yaml` (+ social/explore
  variants if you split them).
- Clocks already in `knowledge/campaign/frame.yaml`.
- NPC `state` field already on `knowledge/campaign/cast.yaml` entries.
- Speaker/stance already in `game/speaker_policy.py` / `game/stance.py`.
- Cue sampling mirrors `sample_style_cues()`; knobs in `models.yaml`.
- Shared resolver + nudge from `roll_resolution_design.md`.

## Parked

- `knowledge/campaign/recruitment.yaml` — dormant. Companion acquisition is out
  of scope; revisit after the core loop is solid.
```
