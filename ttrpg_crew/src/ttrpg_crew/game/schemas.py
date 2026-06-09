from __future__ import annotations

from pydantic import BaseModel, Field


class ModeratorSelection(BaseModel):
    selected_speakers: list[str] = Field(default_factory=list)
    gm_should_frame_first: bool = True
    gm_intent: str = "Frame the next beat naturally."
    reasoning: str = ""
    spotlight_notes: dict = Field(default_factory=dict)
