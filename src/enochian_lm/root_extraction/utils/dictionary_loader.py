import json
import logging
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from enochian_lm.root_extraction.utils.types_lexicon import (
    AltRecord,
    EntryRecord,
    SenseRecord,
)

# Backwards-compatibility alias. Downstream code should prefer EntryRecord.
EntryLike = EntryRecord


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
    confidence: float = 1.0
    direction: Literal["to", "from", "both"] = "both"

    @field_validator("value", mode="before")
    def normalize_value(cls, v: str) -> str:
        return unicodedata.normalize("NFKC", v.strip().lower())

    @field_validator("confidence", mode="after")
    def check_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            logger.warning(f"Alternate confidence {v} out of [0,1]")
        return v


class Sense(BaseModel):
    definition: str
    confidence: float = 1.0

    @field_validator("confidence", mode="after")
    def check_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            logger.warning(f"Sense confidence {v} out of [0,1]")
        return v


class Entry(BaseModel):
    canonical: str
    alternates: List["Alternate"] = Field(default_factory=list)
    senses: Optional[List["Sense"]] = None
    context_tags: Optional[List[str]] = None
    pos: Optional[str] = None
    normalized: str = ""
    enhanced_definition: str = ""
    # Additional metadata
    key_citations: Optional[List[Dict[str, Any]]] = None
    commentary: Optional[str] = None
    canon_word: Optional[bool] = None

    @field_validator("canonical", mode="before")
    def normalize_canonical(cls, v: str) -> str:
        return unicodedata.normalize("NFKC", v.strip().lower())

    @field_validator("context_tags", mode="before")
    def normalize_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        return [tag.strip().lower() for tag in v]


def load_dictionary(
    json_path: str,
    cache_dir: str = ".cache",
) -> List[EntryRecord]:
    """
    Load dictionary JSON (list or dict) into validated Entry objects.
    - Supports 'alternates', 'senses', or single 'definition'.
    - Captures 'key_citations', 'commentary', and 'canon_word' if present.
    - Normalizes, deduplicates alternates, and caches by JSON hash (cache currently disabled).
    """
    os.makedirs(cache_dir, exist_ok=True)

    with open(json_path, "rb") as f:
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
            key = item.get("canonical") or item.get("word") or ""
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
        canonical = unicodedata.normalize("NFKC", str(key).strip().lower())
        norm = canonical
        bin = grouped.setdefault(
            norm,
            {
                "canonical": canonical,
                "alternates": [],
                "senses": [],
                "context_tags": [],
                "key_citations": [],
                "commentaries": [],
                "canon_word": False,
                "pos": None,
            },
        )

        if val.get("canon_word"):
            bin["canon_word"] = True

        if not bin["pos"] and val.get("pos"):
            bin["pos"] = val.get("pos")

        for tag in val.get("context_tags", []):
            t = (tag or "").strip().lower()
            if t and t not in bin["context_tags"]:
                bin["context_tags"].append(t)

        for cit in val.get("key_citations", []):
            if cit not in bin["key_citations"]:
                bin["key_citations"].append(cit)

        comm = val.get("commentary")
        if comm and comm not in bin["commentaries"]:
            bin["commentaries"].append(comm)

        # Senses (normalize to plain dict records with default confidence)
        if "senses" in val and isinstance(val["senses"], list):
            for s in val["senses"]:
                try:
                    definition = (s.get("definition") or "").strip()
                    if not definition:
                        continue
                    conf = float(s.get("confidence", 1.0))
                    bin["senses"].append({"definition": definition, "confidence": conf})
                except Exception as e:
                    logger.warning(f"Invalid sense for '{canonical}': {s} ({e})")
        elif "definition" in val:
            try:
                definition = (val.get("definition") or "").strip()
                if definition:
                    bin["senses"].append({"definition": definition, "confidence": 1.0})
            except Exception as e:
                logger.warning(
                    f"Invalid definition for '{canonical}': {val.get('definition')} ({e})"
                )

        # Alternates (normalize to plain dict records with default confidence)
        for alt in val.get("alternates", []):
            try:
                value = (alt.get("value") or "").strip().lower()
                if not value:
                    continue
                conf = float(alt.get("confidence", 1.0))
                if all(
                    value != (existing.get("value") or "")
                    for existing in bin["alternates"]
                ):
                    bin["alternates"].append({"value": value, "confidence": conf})
            except Exception as e:
                logger.warning(f"Invalid alternate for '{canonical}': {alt} ({e})")

    # --- Build phase (AFTER grouping) ---
    entries: List[EntryRecord] = []
    for norm, data in grouped.items():
        entry: EntryRecord = {
            "canonical": data["canonical"],
            "alternates": list(data["alternates"]),
            "normalized": norm,
        }
        if data["senses"]:
            entry["senses"] = list(data["senses"])
        if data["context_tags"]:
            entry["context_tags"] = list(data["context_tags"])
        if data["pos"]:
            entry["pos"] = data["pos"]
        if data["key_citations"]:
            entry["key_citations"] = list(data["key_citations"])
        if data.get("commentaries"):
            entry["commentary"] = "; ".join(data["commentaries"])
        if data.get("canon_word"):
            entry["canon_word"] = True
        entries.append(entry)

    return entries


def load_dictionary_v2(json_path: str) -> list[EntryRecord]:
    raw = json.loads(Path(json_path).read_text())
    if not isinstance(raw, list):
        return []
    out: list[EntryRecord] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        canonical = (it.get("normalized") or it.get("word") or "").lower().strip()
        if not canonical:
            continue
        alts: list[AltRecord] = []
        w = (it.get("word") or "").lower().strip()
        if w and w != canonical:
            alts.append({"value": w, "confidence": 1.0})
        senses: list[SenseRecord] = []
        top = (it.get("definition") or "").strip()
        if top:
            senses.append({"definition": top, "confidence": 1.0})
        for s in it.get("senses") or []:
            d = (s.get("definition") or "").strip()
            if d:
                c = float(s.get("confidence", 1.0))
                senses.append({"definition": d, "confidence": c})
        rec: EntryRecord = {
            "canonical": canonical,
            "alternates": alts,
        }
        if senses:
            rec["senses"] = senses
        out.append(rec)
    return out
