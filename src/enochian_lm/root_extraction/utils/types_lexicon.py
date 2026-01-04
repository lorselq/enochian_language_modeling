from __future__ import annotations

from typing import NotRequired, TypedDict

class AltRecord(TypedDict):
    value: str
    confidence: float

class SenseRecord(TypedDict):
    definition: str
    confidence: float

class EntryRecord(TypedDict):
    canonical: str
    alternates: list[AltRecord]
    senses: NotRequired[list[SenseRecord]]
    context_tags: NotRequired[list[str]]
    pos: NotRequired[str]
    normalized: NotRequired[str]
    enhanced_definition: NotRequired[str]
    key_citations: NotRequired[list[dict]]
    commentary: NotRequired[str]
    canon_word: NotRequired[bool]
