"""Fixed-scenario A/B harness for comparing models and settings.

Runs the SAME short list of player inputs through the engine with diagnostics
on, so you can swap the model in LM Studio (or flip a setting) and compare
apples-to-apples. Whatever chat model is resident in LM Studio is auto-detected,
exactly as in normal play.

What to read in the output, per beat:
  * the beat text                  — judge fit-to-context (tone / personality /
                                      situation), then approve or reject.
  * [output] <agent>: ...          — each agent's raw line before any edit.
  * [tokens] / [direct] ...        — cost: completion tokens, requests, tok/s.
  * [audit] ...                    — rule/register/AI-ism flags.

Usage (from the ttrpg_crew project dir):
    python scripts/ab_beats.py
    TTRPG_DIRECT_COMPANIONS=1 python scripts/ab_beats.py   # direct-companion path

Edit SCENARIO to match your campaign. The Python-side sampling (name pool, style
cues) is seeded so two runs draw the same samples; only the model/settings vary.
"""

from __future__ import annotations

import os
import random

# Force diagnostics on regardless of the caller's environment.
os.environ["TTRPG_DEBUG"] = "1"

from dotenv import load_dotenv

# A fixed beat list chosen to exercise: a GM-only opening, a player action that
# should pull in companions, a direct address to a named companion, and a
# higher-tension action. Inputs are deliberately fiction-agnostic (no assumed
# NPCs or staging) because each run's opening differs — a canned input that
# contradicts the live fiction confuses the GM. The direct-address beat fills in
# a real companion from the live roster, so it never names someone who isn't in
# the party. Edit freely; keep it identical across the runs you're comparing.
def build_scenario(companion_name: str) -> list[str]:
    return [
        "",  # opening: GM frames the scene, player has not acted yet
        "I take the vial and ask what the Conclave will want in return.",
        f"I ask {companion_name} whether the canals are safe to move through tonight.",
        "I refuse, hand the vial back, and turn to leave.",
    ]


def main() -> None:
    load_dotenv()
    random.seed(0)  # reproducible name/style sampling across runs

    from ttrpg_crew.main import _build_flow

    flow = _build_flow()
    print(
        f"\n[ab] model={getattr(flow.llm, 'model', '?')}  tier={flow.tier}  "
        f"direct_companions={flow.use_direct_companions}\n"
    )

    companion_name = next(
        (c.name for c in flow.game_state.characters.values()), "my companion"
    )
    scenario = build_scenario(companion_name)

    for i, player_input in enumerate(scenario):
        header = "OPENING" if (i == 0 and not player_input) else f"INPUT: {player_input}"
        print("\n" + "=" * 72)
        print(f"BEAT {i + 1}  {header}")
        print("=" * 72)
        beat = flow.kickoff(player_input=player_input)
        print("\n--- beat text ---")
        print(str(beat).strip())


if __name__ == "__main__":
    main()
