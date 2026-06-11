from __future__ import annotations

import random
import re

from crewai.tools import tool


@tool("Dice Roller")
def roll_dice(expression: str) -> str:
    """Roll dice using notation like 1d20+5, 2d6, or 4d6-1."""

    pattern = r"^(\d+)d(\d+)([+-]\d+)?$"
    match = re.match(pattern, expression.strip())

    if not match:
        return "Invalid dice expression. Use notation like 1d20+5 or 2d6."

    count = int(match.group(1))
    sides = int(match.group(2))
    modifier = int(match.group(3) or 0)

    if count <= 0 or sides <= 0 or count > 100:
        return "Invalid dice expression. Dice count and sides must be positive."

    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls) + modifier

    return f"Rolled {expression}: rolls={rolls}, modifier={modifier}, total={total}"
