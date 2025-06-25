import json
import os
import hashlib
import unicodedata
import logging
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, field_validator
import joblib

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
    alternates: List[Alternate] = []
    senses: Optional[List[Sense]] = None
    context_tags: Optional[List[str]] = None
    pos: Optional[str] = None
    normalized: str = ""
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

    def build_enhanced_definition(self) -> str:
        parts = []
        if self.senses:
            for s in self.senses:
                parts.append(f"{s.definition} [{s.confidence:.2f}]")
        else:
            for a in self.alternates:
                parts.append(f"{a.value} [{a.confidence:.2f}]")
        return f"{self.canonical}: " + "; ".join(parts)


def load_dictionary(
    json_path: str,
    cache_dir: str = ".cache",
) -> List[Entry]:
    """
    Load dictionary JSON (list or dict) into validated Entry objects.
    - Supports 'alternates', 'senses', or single 'definition'.
    - Captures 'key_citations', 'commentary', and 'canon_word' if present.
    - Normalizes, deduplicates alternates, and caches by JSON hash.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Hash the raw JSON for cache invalidation
    with open(json_path, 'rb') as f:
        data_bytes = f.read()
    # digest = hashlib.md5(data_bytes).hexdigest()
    # cache_file = os.path.join(cache_dir, f'dict_{digest}.joblib')

    # if os.path.exists(cache_file):
    #     entries = joblib.load(cache_file)
    #     logger.info(f"Loaded dictionary from cache: {cache_file}")
    #     return entries

    raw = json.loads(data_bytes)
    items = []
    # Support both list-of-dicts and dict-of-dicts
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

    entries: List[Entry] = []
    for key, val in items:
        canonical = unicodedata.normalize('NFKC', str(key).strip().lower())
        # Alternates
        alts: List[Alternate] = []
        for alt in val.get('alternates', []):
            try:
                alts.append(Alternate(**alt))
            except Exception as e:
                logger.warning(f"Invalid alternate for '{canonical}': {alt} ({e})")
        # Senses: explicit or from 'definition'
        senses: List[Sense] = []
        if 'senses' in val:
            for s in val['senses']:
                try:
                    senses.append(Sense(**s))
                except Exception as e:
                    logger.warning(f"Invalid sense for '{canonical}': {s} ({e})")
        elif 'definition' in val:
            senses.append(Sense(definition=val['definition'], confidence=1.0))
        # Build Entry
        entry = Entry(
            canonical=canonical,
            alternates=alts,
            senses=senses or None,
            context_tags=val.get('context_tags'),
            pos=val.get('pos'),
            key_citations=val.get('key_citations'),
            commentary=val.get('commentary'),
            canon_word=val.get('canon_word'),
        )
        # Deduplicate alternates by value
        unique: Dict[str, Alternate] = {}
        for a in entry.alternates:
            if a.value in unique:
                existing = unique[a.value]
                existing.confidence = (existing.confidence + a.confidence) / 2
            else:
                unique[a.value] = a
        entry.alternates = list(unique.values())
        # Warn only if truly no sense or alternate
        if not entry.senses and not entry.alternates:
            logger.warning(f"Entry '{canonical}' has neither alternates nor definition/senses")
        entries.append(entry)

    # Cache
    # joblib.dump(entries, cache_file)
    # logger.info(f"Cached {len(entries)} entries to {cache_file}")
    return entries
