"""The structured result of one beat.

Per-speaker segments plus the GM's closing hand-back prompt, kept structured
all the way out of the engine so a UI can render each speaker its own way
(chat bubbles, avatars, styling) without parsing text. str(beat) flattens to
the plain-text form the CLI prints and the transcript/session log store.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BeatSegment:
    speaker_id: str  # "GM" or the companion's id (e.g. "mireille")
    speaker_name: str  # display name: "GM" or the character's full name
    text: str

    @property
    def is_gm(self) -> bool:
        return self.speaker_id == "GM"

    def render(self) -> str:
        """The flat-text form of this segment (GM bare, companions labeled).

        The single source for segment formatting — used by str(beat), the CLI,
        and any other plain-text consumer.
        """
        return self.text if self.is_gm else f"{self.speaker_name}: {self.text}"


@dataclass
class Beat:
    segments: list[BeatSegment] = field(default_factory=list)
    closing_prompt: str = ""  # the hand-back line, rendered last (may be "")
    # Next moves the player can pick instead of typing. Filled after the beat
    # by the scene tracker; deliberately left out of str(beat), so they never
    # enter the transcript or the story the models read.
    suggested_actions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [seg.render() for seg in self.segments if seg.text]
        if self.closing_prompt:
            lines.append(self.closing_prompt)
        return "\n\n".join(lines)
