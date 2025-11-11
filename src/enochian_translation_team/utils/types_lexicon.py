from typing import TypedDict, NotRequired, List

class AltRecord(TypedDict):
    value: str
    confidence: float

class SenseRecord(TypedDict):
    definition: str
    confidence: float

class EntryRecord(TypedDict):
    canonical: str
    alternates: List[AltRecord]
    senses: NotRequired[List[SenseRecord]]  # allow None/omitted
    context_tags: NotRequired[List[str]]
    pos: NotRequired[str]
    normalized: NotRequired[str]
    enhanced_definition: NotRequired[str]
    key_citations: NotRequired[List[dict]]
    commentary: NotRequired[str]
    canon_word: NotRequired[bool]