import json
import os
import unicodedata
import logging
from typing import List, Optional, Literal, Any, Dict, Tuple, runtime_checkable, Sequence, Protocol
from pydantic import BaseModel, field_validator, Field
from dataclasses import dataclass
from pathlib import Path



@runtime_checkable
class AltLike(Protocol):
    value: str
    confidence: float

@runtime_checkable
class SenseLike(Protocol):
    definition: str
    confidence: float

@runtime_checkable
class EntryLike(Protocol):
    canonical: str
    alternates: Sequence[AltLike]
    senses: Sequence[SenseLike] | None


@dataclass
class _Alt:
    value: str
    confidence: float = 1.0

@dataclass
class _Sense:
    definition: str
    confidence: float = 1.0

@dataclass
class _EntryLike:
    canonical: str
    alternates: list[_Alt]
    senses: list[_Sense]


# Module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)

class Alternate(BaseModel):
    value: str
    confidence: float
    direction: Literal['to', 'from', 'both']

    @field_validator('value', mode='before')
    def normalize_value(cls, v: str) -> str:
        return unicodedata.normalize('NFKC', v.strip().lower())

    @field_validator('confidence', mode='after')
    def check_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            logger.warning(f"Alternate confidence {v} out of [0,1]")
        return v

class Sense(BaseModel):
    definition: str
    confidence: float

    @field_validator('confidence', mode='after')
    def check_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            logger.warning(f"Sense confidence {v} out of [0,1]")
        return v

class Entry(BaseModel):
    canonical: str
    alternates: List['Alternate'] = Field(default_factory=list)
    senses: Optional[List['Sense']] = None
    context_tags: Optional[List[str]] = None
    pos: Optional[str] = None
    normalized: str = ""
    enhanced_definition: str = ""
    # Additional metadata
    key_citations: Optional[List[Dict[str, Any]]] = None
    commentary: Optional[str] = None
    canon_word: Optional[bool] = None

    @field_validator('canonical', mode='before')
    def normalize_canonical(cls, v: str) -> str:
        return unicodedata.normalize('NFKC', v.strip().lower())

    @field_validator('context_tags', mode='before')
    def normalize_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        return [tag.strip().lower() for tag in v]



def load_dictionary(
    json_path: str,
    cache_dir: str = ".cache",
) -> List[Entry]:
    """
    Load dictionary JSON (list or dict) into validated Entry objects.
    - Supports 'alternates', 'senses', or single 'definition'.
    - Captures 'key_citations', 'commentary', and 'canon_word' if present.
    - Normalizes, deduplicates alternates, and caches by JSON hash (cache currently disabled).
    """
    os.makedirs(cache_dir, exist_ok=True)

    with open(json_path, 'rb') as f:
        data_bytes = f.read()

    raw = json.loads(data_bytes)

    # Support both list-of-dicts and dict-of-dicts
    items: List[Tuple[str, dict]] = []
    if isinstance(raw, list):
        logger.info(f"Loading {len(raw)} entries from list JSON")
        for item in raw:
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict entry: {item}")
                continue
            key = item.get('canonical') or item.get('word') or ''
            items.append((key, item))
    elif isinstance(raw, dict):
        logger.info(f"Loading {len(raw)} entries from dict JSON")
        for key, val in raw.items():
            items.append((key, val))
    else:
        logger.error("Unsupported JSON root type: %s", type(raw))
        return []

    # --- Group/merge phase ---
    grouped: Dict[str, Dict[str, Any]] = {}
    for key, val in items:
        canonical = unicodedata.normalize('NFKC', str(key).strip().lower())
        norm = canonical
        bin = grouped.setdefault(norm, {
            'canonical': canonical,
            'alternates': [],
            'senses': [],
            'context_tags': [],
            'key_citations': [],
            'commentaries': [],
            'canon_word': False,
            'pos': None,
        })

        if val.get('canon_word'):
            bin['canon_word'] = True

        if not bin['pos'] and val.get('pos'):
            bin['pos'] = val.get('pos')

        for tag in val.get('context_tags', []):
            t = (tag or "").strip().lower()
            if t and t not in bin['context_tags']:
                bin['context_tags'].append(t)

        for cit in val.get('key_citations', []):
            if cit not in bin['key_citations']:
                bin['key_citations'].append(cit)

        comm = val.get('commentary')
        if comm and comm not in bin['commentaries']:
            bin['commentaries'].append(comm)

        # Senses
        if 'senses' in val and isinstance(val['senses'], list):
            for s in val['senses']:
                try:
                    bin['senses'].append(Sense(**s))
                except Exception as e:
                    logger.warning(f"Invalid sense for '{canonical}': {s} ({e})")
        elif 'definition' in val:
            try:
                bin['senses'].append(Sense(definition=val['definition'], confidence=1.0))
            except Exception as e:
                logger.warning(f"Invalid definition for '{canonical}': {val.get('definition')} ({e})")

        # Alternates
        for alt in val.get('alternates', []):
            try:
                a = Alternate(**alt)
                if all(a.value != existing.value for existing in bin['alternates']):
                    bin['alternates'].append(a)
            except Exception as e:
                logger.warning(f"Invalid alternate for '{canonical}': {alt} ({e})")

    # --- Build phase (AFTER grouping) ---
    entries: List[Entry] = []
    for norm, data in grouped.items():
        commentary = "; ".join(data['commentaries']) if data['commentaries'] else None
        entry = Entry(
            canonical=data['canonical'],
            alternates=data['alternates'],
            senses=data['senses'] or None,
            context_tags=data['context_tags'] or None,
            pos=data['pos'],
            normalized=norm,
            key_citations=data['key_citations'] or None,
            commentary=commentary,
            canon_word=data['canon_word'],
        )
        entries.append(entry)

    logger.info(f"Loaded {len(entries)} merged dictionary entries")
    return entries


def load_dictionary_v2(json_path: str) -> list[_EntryLike]:
    """
    Adapt the NEW dictionary.json schema to the minimal shape used by training.
    Keeps: normalized/word -> canonical, top-level definition, senses[*].definition.
    Assigns confidence=1.0 if not present.
    Ignores: citations/commentary.
    """
    raw = json.loads(Path(json_path).read_text())
    if not isinstance(raw, list):
        logger.error("v2 expects a list root; got %s", type(raw))
        return []

    entries: list[_EntryLike] = []
    for it in raw:
        if not isinstance(it, dict):
            continue

        canonical = (it.get("normalized") or it.get("word") or "").lower().strip()
        if not canonical:
            continue

        # Optional original orthography as an alternate token
        alts: list[_Alt] = []
        w = (it.get("word") or "").lower().strip()
        if w and w != canonical:
            alts.append(_Alt(value=w, confidence=1.0))

        # Collect definitions with default confidence
        senses: list[_Sense] = []
        top_def = (it.get("definition") or "").strip()
        if top_def:
            senses.append(_Sense(definition=top_def, confidence=1.0))

        for s in (it.get("senses") or []):
            try:
                d = (s.get("definition") or "").strip()
                if d:
                    conf = float(s.get("confidence", 1.0))
                    # Avoid duplicate of top_def with same text
                    if not (top_def and d == top_def and conf == 1.0):
                        senses.append(_Sense(definition=d, confidence=conf))
            except Exception:
                continue

        entries.append(_EntryLike(canonical=canonical, alternates=alts, senses=senses))
    return entries
