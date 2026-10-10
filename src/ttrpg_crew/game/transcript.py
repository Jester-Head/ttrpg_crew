from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TranscriptEntry:
    speaker: str
    message: str
    kind: str = "visible"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Transcript:
    entries: list[TranscriptEntry] = field(default_factory=list)

    def append(self, speaker: str, message: str, kind: str = "visible") -> None:
        self.entries.append(
            TranscriptEntry(
                speaker=speaker,
                message=message,
                kind=kind,
            )
        )

    def visible_text(self) -> str:
        lines = []
        for entry in self.entries:
            if entry.kind == "visible":
                lines.append(f"{entry.speaker}: {entry.message}")
        return "\n".join(lines)

    def tail(self, max_chars: int) -> str:
        """The most recent visible transcript, capped at max_chars.

        Keeps whole entries (newest backwards) so the cut never lands
        mid-sentence; the newest entries always survive.
        """
        lines: list[str] = []
        used = 0
        for entry in reversed(self.entries):
            if entry.kind != "visible":
                continue
            line = f"{entry.speaker}: {entry.message}"
            if used + len(line) > max_chars and lines:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(reversed(lines))

    def full_text(self) -> str:
        return "\n".join(
            f"[{entry.kind}] {entry.speaker}: {entry.message}"
            for entry in self.entries
        )
