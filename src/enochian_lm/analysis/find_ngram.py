"""Dictionary-backed ngram lookup utilities for analysis commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class DictionaryEntry(TypedDict, total=False):
    """Raw dictionary entry shape used by the lookup command."""

    word: str
    canonical: str
    normalized: str
    canon_word: bool
    definition: str
    senses: list[dict[str, Any]]
    key_citations: list[dict[str, Any]]
    commentary: str
    alternates: list[Any]
    pos: str
    context_tags: list[str]


def normalize_ngram(value: str) -> str:
    """Normalize user and dictionary lookup text into comparable ngram text.

    Why:
    the CLI is meant for quick linguistic lookup, and callers will often paste
    forms with punctuation, spaces, or mixed case.

    How:
    retain only alphabetic characters and lowercase them. This mirrors the
    requested "letters only" search behavior while remaining agnostic about
    whether the source text came from `word`, `normalized`, or an alternate.

    Responsibility:
    provide the single normalization rule used for query validation, canonical
    matching, alternate matching, and match-position reporting.
    """

    return "".join(ch.lower() for ch in value if ch.isalpha())


def load_dictionary_entries(path: Path) -> list[DictionaryEntry]:
    """Load raw dictionary records without applying canon filtering.

    Why:
    the existing dictionary loader intentionally skips `canon_word: false`
    records, but this lookup command must be able to include non-canon words
    unless the caller asks for `--canon-only`.

    How:
    read the JSON artifact directly and accept only object entries from the
    top-level list. A dict-root fallback is included for compatibility with
    older dictionary shapes used elsewhere in the project.

    Responsibility:
    return the closest raw records to the on-disk dictionary so downstream
    formatting can preserve senses, citations, commentary, and metadata.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, dict)]
    if isinstance(raw, dict):
        entries: list[DictionaryEntry] = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            entry = dict(value)
            entry.setdefault("word", key)
            entries.append(entry)
        return entries
    raise ValueError(f"Dictionary '{path}' must contain a list or object root")


def find_ngram_matches(
    entries: list[DictionaryEntry],
    ngram: str,
    *,
    canon_only: bool = False,
    include_alternates: bool = False,
) -> list[dict[str, Any]]:
    """Find dictionary words whose normalized surfaces contain an ngram.

    Why:
    analysis workflows need a fast, transparent way to see every dictionary
    sense attached to words that contain a proposed morph/ngram.

    How:
    normalize the query, scan each canonical surface, optionally scan
    alternates, and retain enough match metadata for both readable text and
    JSON output.

    Responsibility:
    produce stable, serialization-friendly match records while leaving display
    choices to the formatter functions.
    """

    needle = normalize_ngram(ngram)
    if not needle:
        raise ValueError("NGRAM must contain at least one alphabetic character")

    matches: list[dict[str, Any]] = []
    for entry in entries:
        if canon_only and entry.get("canon_word") is not True:
            continue
        surfaces = _entry_surfaces(entry, include_alternates=include_alternates)
        matched_surfaces = [
            surface for surface in surfaces if needle in surface["normalized"]
        ]
        if not matched_surfaces:
            continue
        matches.append(_build_match_record(entry, needle, matched_surfaces))

    return sorted(
        matches,
        key=lambda item: (
            str(item.get("normalized") or "").lower(),
            str(item.get("word") or "").lower(),
        ),
    )


def format_ngram_matches_text(
    matches: list[dict[str, Any]],
    *,
    query: str,
    citations: bool = False,
    verbose: bool = False,
) -> str:
    """Render ngram matches as readable CLI text.

    Why:
    the default CLI output should be pleasant for direct inspection while still
    exposing every sense and optional evidence when requested.

    How:
    group output by matched dictionary word, always list all senses, and add
    citations or verbose metadata only behind their corresponding flags.

    Responsibility:
    keep human-facing presentation deterministic and compact enough for normal
    terminal use.
    """

    normalized_query = normalize_ngram(query)
    lines = [
        f'Matches for "{query}" (normalized: "{normalized_query}"): {len(matches)}'
    ]
    for match in matches:
        lines.append("")
        canon_label = "canon" if match["canon_word"] else "non-canon"
        lines.append(f"{match['word']} [{canon_label}]")
        if verbose:
            _append_verbose_lines(lines, match)

        senses = match.get("senses") or []
        if senses:
            for index, sense in enumerate(senses, start=1):
                definition = str(sense.get("definition") or "").strip()
                if definition:
                    lines.append(f"  {index}. {definition}")
                elif verbose:
                    lines.append(f"  {index}. [no definition]")
        else:
            definition = str(match.get("definition") or "").strip()
            if definition:
                lines.append(f"  1. {definition}")
            elif verbose:
                lines.append("  1. [no definition]")

        if citations:
            _append_citation_lines(lines, collect_citations(match))

    return "\n".join(lines) + "\n"


def format_ngram_matches_json(matches: list[dict[str, Any]], *, query: str) -> str:
    """Render ngram matches as deterministic JSON.

    Why:
    analysis commands are often piped into later scripts, so the command needs a
    machine-readable format with the same match records used for text output.

    How:
    wrap the list in a small envelope containing the raw and normalized query,
    match count, and records.

    Responsibility:
    provide stable JSON output without leaking CLI-only text formatting into
    automation paths.
    """

    payload = {
        "query": query,
        "normalized_query": normalize_ngram(query),
        "match_count": len(matches),
        "matches": matches,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def collect_citations(match: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect deduplicated entry-level and sense-level citations.

    Why:
    dictionary data can carry the same citation at the entry and sense levels,
    and repeated evidence makes CLI output harder to scan.

    How:
    serialize citation dictionaries with sorted keys to create a stable
    de-duplication key while preserving first-seen order.

    Responsibility:
    return the exact citation objects the formatter should display or serialize.
    """

    seen: set[str] = set()
    citations: list[dict[str, Any]] = []
    for citation in match.get("key_citations") or []:
        _append_unique_citation(citations, seen, citation)
    for sense in match.get("senses") or []:
        for citation in sense.get("key_citations") or []:
            _append_unique_citation(citations, seen, citation)
    return citations


def _entry_surfaces(
    entry: DictionaryEntry, *, include_alternates: bool
) -> list[dict[str, Any]]:
    """Build searchable surfaces for one dictionary entry.

    Why:
    lookup behavior needs to distinguish canonical matches from alternate
    matches without losing which surface caused a result.

    How:
    always include the canonical surface derived from `normalized` or `word`;
    append alternate values only when requested.

    Responsibility:
    isolate dictionary-shape quirks from the main search loop.
    """

    surfaces: list[dict[str, Any]] = []
    canonical = _canonical_surface(entry)
    if canonical:
        surfaces.append(
            {
                "kind": "canonical",
                "value": canonical,
                "normalized": normalize_ngram(canonical),
            }
        )
    if include_alternates:
        for alternate in entry.get("alternates") or []:
            value = _alternate_value(alternate)
            if value:
                surfaces.append(
                    {
                        "kind": "alternate",
                        "value": value,
                        "normalized": normalize_ngram(value),
                    }
                )
    return [surface for surface in surfaces if surface["normalized"]]


def _canonical_surface(entry: DictionaryEntry) -> str:
    """Choose the canonical text used for default matching.

    Why:
    the dictionary has carried both `normalized` and `word` fields over time,
    and the command should prefer the field meant for normalized lookup.

    How:
    return `normalized` first, then `word`, then `canonical`.

    Responsibility:
    keep canonical surface precedence consistent across search and output.
    """

    return str(
        entry.get("normalized") or entry.get("word") or entry.get("canonical") or ""
    ).strip()


def _display_word(entry: DictionaryEntry) -> str:
    """Choose the word label shown to humans and JSON consumers.

    Why:
    users expect the displayed word to preserve the dictionary's actual entry
    spelling where available rather than always seeing the normalized form.

    How:
    prefer `word`, then `canonical`, then `normalized`.

    Responsibility:
    keep output labels stable even when source dictionary shapes differ.
    """

    return str(
        entry.get("word") or entry.get("canonical") or entry.get("normalized") or ""
    ).strip()


def _alternate_value(alternate: Any) -> str:
    """Extract an alternate spelling from supported raw alternate shapes.

    Why:
    alternate records may be plain strings or dictionaries depending on which
    dictionary artifact produced them.

    How:
    return the alternate itself for strings or its `value` field for dicts.

    Responsibility:
    give alternate matching one clean value extraction path.
    """

    if isinstance(alternate, str):
        return alternate.strip()
    if isinstance(alternate, dict):
        return str(alternate.get("value") or "").strip()
    return ""


def _build_match_record(
    entry: DictionaryEntry, needle: str, surfaces: list[dict[str, Any]]
) -> dict[str, Any]:
    """Create a serializable match record from a raw dictionary entry.

    Why:
    both text and JSON renderers need the same normalized, deduplicated view of
    dictionary data and match metadata.

    How:
    copy relevant source fields, normalize missing booleans conservatively, and
    enrich matching surfaces with ngram start positions.

    Responsibility:
    define the public JSON shape emitted by the CLI command.
    """

    matched_surfaces = [
        {
            "kind": surface["kind"],
            "value": surface["value"],
            "normalized": surface["normalized"],
            "positions": _find_positions(surface["normalized"], needle),
        }
        for surface in surfaces
    ]
    record: dict[str, Any] = {
        "word": _display_word(entry),
        "normalized": _canonical_surface(entry),
        "canon_word": entry.get("canon_word") is True,
        "definition": entry.get("definition"),
        "senses": list(entry.get("senses") or []),
        "key_citations": list(entry.get("key_citations") or []),
        "matched_surfaces": matched_surfaces,
    }
    for key in (
        "commentary",
        "alternates",
        "pos",
        "context_tags",
        "enhanced_definition",
    ):
        if key in entry:
            record[key] = entry[key]
    return record


def _find_positions(haystack: str, needle: str) -> list[int]:
    """Return all overlapping start positions for a normalized ngram match.

    Why:
    verbose output should explain where the ngram appears, including repeated
    or overlapping occurrences within a single word.

    How:
    repeatedly call `str.find` and advance by one character after every match.

    Responsibility:
    provide deterministic zero-based match positions for text and JSON output.
    """

    positions: list[int] = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            return positions
        positions.append(index)
        start = index + 1


def _append_verbose_lines(lines: list[str], match: dict[str, Any]) -> None:
    """Append verbose metadata lines for a text match.

    Why:
    `--verbose` should expose metadata useful to analysis without cluttering
    the default word-and-senses view.

    How:
    add compact labeled lines only for metadata present in the match record.

    Responsibility:
    keep verbose text formatting centralized and predictable.
    """

    if match.get("normalized"):
        lines.append(f"  normalized: {match['normalized']}")
    for surface in match.get("matched_surfaces") or []:
        positions = ", ".join(str(pos) for pos in surface.get("positions") or [])
        lines.append(
            f"  match: {surface.get('kind')} {surface.get('value')} at [{positions}]"
        )
    if match.get("alternates"):
        alternates = ", ".join(
            value for value in (_alternate_value(alt) for alt in match["alternates"]) if value
        )
        if alternates:
            lines.append(f"  alternates: {alternates}")
    if match.get("pos"):
        lines.append(f"  pos: {match['pos']}")
    if match.get("context_tags"):
        lines.append(f"  context_tags: {', '.join(match['context_tags'])}")
    if match.get("commentary"):
        lines.append(f"  commentary: {match['commentary']}")


def _append_citation_lines(
    lines: list[str], citations: list[dict[str, Any]]
) -> None:
    """Append citation lines for a text match.

    Why:
    citation formatting should be consistent whether evidence came from the
    entry or individual senses.

    How:
    show location plus context when available, falling back to JSON for unusual
    citation shapes.

    Responsibility:
    keep evidence display readable without dropping unsupported citation fields.
    """

    if not citations:
        lines.append("  citations: none")
        return
    lines.append("  citations:")
    for citation in citations:
        location = str(citation.get("location") or "").strip()
        context = str(citation.get("context") or "").strip()
        if location and context:
            lines.append(f"    - {location}: {context}")
        elif location:
            lines.append(f"    - {location}")
        elif context:
            lines.append(f"    - {context}")
        else:
            lines.append(f"    - {json.dumps(citation, ensure_ascii=False)}")


def _append_unique_citation(
    citations: list[dict[str, Any]], seen: set[str], citation: Any
) -> None:
    """Append one citation object if it has not already appeared.

    Why:
    citation records can be repeated across levels of the same entry.

    How:
    accept dictionary citations only and use JSON serialization as the
    de-duplication key.

    Responsibility:
    preserve first-seen citation order while removing duplicate evidence rows.
    """

    if not isinstance(citation, dict):
        return
    key = json.dumps(citation, sort_keys=True, ensure_ascii=False)
    if key in seen:
        return
    seen.add(key)
    citations.append(citation)
