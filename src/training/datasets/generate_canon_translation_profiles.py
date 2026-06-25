"""Generate human-editable translation profiles for canon dictionary entries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DICTIONARY = (
    PROJECT_ROOT / "src" / "enochian_lm" / "root_extraction" / "data" / "dictionary.json"
)
DEFAULT_ENRICHED = (
    PROJECT_ROOT
    / "src"
    / "enochian_lm"
    / "root_extraction"
    / "data"
    / "dictionary_enriched.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "translation" / "canon_dictionary_profiles.yml"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "with",
    "you",
    "your",
}


def generate_profiles(
    *,
    dictionary_path: Path = DEFAULT_DICTIONARY,
    enriched_path: Path = DEFAULT_ENRICHED,
) -> dict[str, object]:
    """Build profile payloads from raw canon dictionary plus enriched metadata."""

    raw_entries = _load_json_list(dictionary_path)
    enriched_entries = _load_json_list(enriched_path) if enriched_path.exists() else []
    enriched_by_key = {
        _entry_key(entry): entry
        for entry in enriched_entries
        if _entry_key(entry)
    }
    profiles: dict[str, dict[str, object]] = {}
    missing_enriched = 0
    for entry in raw_entries:
        if entry.get("canon_word") is not True:
            continue
        key = _entry_key(entry)
        if not key:
            continue
        enriched = enriched_by_key.get(key)
        if enriched is None:
            missing_enriched += 1
        profile = _profile_for_entry(entry, enriched)
        profiles[key.upper()] = profile
    return {
        "metadata": {
            "source": "canon_dictionary_seed",
            "dictionary": _display_path(dictionary_path),
            "enriched_dictionary": _display_path(enriched_path),
            "canon_profile_count": len(profiles),
            "missing_enriched_count": missing_enriched,
            "notes": (
                "Generated seed for manual curation. Translation blind mode may "
                "use this checked-in YAML without querying live dictionary data."
            ),
        },
        "canon_dictionary_profiles": dict(sorted(profiles.items())),
    }


def write_profiles(payload: Mapping[str, object], output_path: Path = DEFAULT_OUTPUT) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=False,
        width=100,
    )
    output_path.write_text(text, encoding="utf-8")


def _profile_for_entry(
    entry: Mapping[str, object],
    enriched: Mapping[str, object] | None,
) -> dict[str, object]:
    word = str(entry.get("word") or entry.get("normalized") or "").upper()
    definition = _clean_text(entry.get("definition"))
    senses = _sense_definitions(entry)
    enriched_senses = _sense_rows(enriched) if enriched else []
    parts_of_speech = _unique(
        str(pos)
        for sense in enriched_senses
        for pos in _as_list(sense.get("parts_of_speech"))
    )
    semantic_domains = _unique(
        str(domain)
        for sense in enriched_senses
        for domain in _as_list(sense.get("semantic_domains"))
    )
    citation_text = " ".join(_citation_contexts(entry))
    motif_terms = _extract_terms(" ".join([definition, *senses, citation_text]))
    head_terms = _extract_terms(" ".join([definition, *senses]))
    return {
        "word": word,
        "definition": definition,
        "senses": senses,
        "parts_of_speech": parts_of_speech,
        "semantic_domains": semantic_domains,
        "motif_terms": motif_terms,
        "head_terms": head_terms,
        "source": "canon_dictionary_seed",
        "canon_word": True,
        "enriched": enriched is not None,
    }


def _load_json_list(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [entry for entry in data if isinstance(entry, dict)]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _entry_key(entry: Mapping[str, object]) -> str:
    return str(entry.get("normalized") or entry.get("word") or "").strip().upper()


def _sense_rows(entry: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    if not entry:
        return []
    senses = entry.get("senses")
    if not isinstance(senses, Sequence) or isinstance(senses, (str, bytes)):
        return []
    return [sense for sense in senses if isinstance(sense, Mapping)]


def _sense_definitions(entry: Mapping[str, object]) -> list[str]:
    values = [_clean_text(sense.get("definition")) for sense in _sense_rows(entry)]
    values = [value for value in values if value]
    definition = _clean_text(entry.get("definition"))
    if definition and definition not in values:
        values.insert(0, definition)
    return _unique(values)


def _citation_contexts(entry: Mapping[str, object]) -> list[str]:
    contexts: list[str] = []
    for citation in _as_list(entry.get("key_citations")):
        if isinstance(citation, Mapping):
            context = _clean_text(citation.get("context"))
            if context:
                contexts.append(context)
    for sense in _sense_rows(entry):
        for citation in _as_list(sense.get("key_citations")):
            if isinstance(citation, Mapping):
                context = _clean_text(citation.get("context"))
                if context:
                    contexts.append(context)
    return contexts


def _extract_terms(text: str, *, limit: int = 12) -> list[str]:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    return [term for term, _count in ranked[:limit]]


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def _as_list(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _unique(values: Iterable[object] | object) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    iterable = values if isinstance(values, Iterable) and not isinstance(values, (str, bytes)) else []
    for value in iterable:
        text = str(value).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = generate_profiles(
        dictionary_path=args.dictionary,
        enriched_path=args.enriched,
    )
    write_profiles(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
