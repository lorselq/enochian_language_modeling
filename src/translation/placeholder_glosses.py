from __future__ import annotations

"""Normalize placeholder glosses before they leak into human-facing output.

Residual-only whole-word anchors are still useful as evidence, but raw
headlines like ``Top residuals: nacro:1.00`` are not readable translations.
This module centralizes detection and cleanup so ranking, phrase rendering, and
footnote formatting can all make the same decision about when a gloss is real
enough to show to a user.
"""

from collections.abc import Mapping, Sequence
import re

_PLACEHOLDER_PREFIXES = ("top residuals:",)
_BRACKET_PLACEHOLDER_RE = re.compile(r"^\[[A-Z][A-Z' ?-]*\]$")


def is_placeholder_gloss(text: object) -> bool:
    """Return whether ``text`` is a raw evidence placeholder rather than a gloss.

    Phrase rendering and candidate ranking both need to distinguish between
    usable semantics and diagnostic residue. Keeping that distinction here lets
    the translation stack demote placeholder anchors without throwing away the
    underlying evidence.
    """

    if not isinstance(text, str):
        return False
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return False
    if normalized.startswith(_PLACEHOLDER_PREFIXES):
        return True
    return bool(_BRACKET_PLACEHOLDER_RE.fullmatch(text.strip()))


def sanitize_human_gloss(text: object, *, token: str | None = None) -> str | None:
    """Return a user-safe gloss or ``None`` when the value is placeholder text.

    Human-facing phrase output should remain explicit when evidence is weak, but
    it should not echo raw residual diagnostics as if they were translations.
    Callers can use the ``None`` return value to render a visible unresolved
    placeholder such as ``[NACRO]`` while preserving raw diagnostics elsewhere.
    """

    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned or is_placeholder_gloss(cleaned):
        return None
    return cleaned


def unresolved_token_gloss(token: str) -> str:
    """Render an unresolved token in a stable bracketed form.

    The phrase pipeline needs a deterministic fallback when no grounded gloss
    survives cleanup. A shared bracketed format keeps reports, footnotes, and
    JSON payloads aligned instead of inventing slightly different placeholders.
    """

    normalized = (token or "").strip().upper() or "?"
    return f"[{normalized}]"


def candidate_has_placeholder_gloss(candidate: Mapping[str, object]) -> bool:
    """Return whether any definition attached to ``candidate`` is placeholder text.

    Ranking policy needs a candidate-level signal so residual-only anchors and
    partially grounded decompositions can be demoted consistently before phrase
    search consumes them.
    """

    meanings = candidate.get("meanings")
    if not isinstance(meanings, Sequence):
        return False
    for meaning in meanings:
        if not isinstance(meaning, Mapping):
            continue
        definition = meaning.get("definition")
        if is_placeholder_gloss(definition):
            return True
        definitions = meaning.get("definitions")
        if not isinstance(definitions, Sequence) or isinstance(definitions, (str, bytes)):
            continue
        for alternate in definitions:
            if is_placeholder_gloss(alternate):
                return True
    return False


def candidate_is_residual_placeholder_anchor(candidate: Mapping[str, object]) -> bool:
    """Return whether ``candidate`` is a residual-only whole-word placeholder.

    Whole-word anchors receive strong default scores because they often reflect
    authoritative evidence. This helper carves out the narrow case where that
    shortcut becomes harmful: an exact whole-word candidate whose only visible
    gloss is a residual placeholder.
    """

    if str(candidate.get("analysis_type") or "") != "whole_word_anchor":
        return False
    meanings = candidate.get("meanings")
    if not isinstance(meanings, Sequence) or not meanings:
        return False
    has_placeholder = False
    for meaning in meanings:
        if not isinstance(meaning, Mapping):
            return False
        if str(meaning.get("provenance") or "") != "residual":
            return False
        if is_placeholder_gloss(meaning.get("definition")):
            has_placeholder = True
    return has_placeholder


def candidate_is_placeholder_anchor(candidate: Mapping[str, object]) -> bool:
    """Return whether a whole-word anchor exposes only opaque placeholder text.

    Residual headlines are not the only anchors that can outrank better
    decompositions. Some exact whole-word candidates surface unresolved
    placeholders such as ``[ASCLAD]``; this helper lets ranking treat those
    anchors as weak evidence instead of polished semantics.
    """

    if str(candidate.get("analysis_type") or "") != "whole_word_anchor":
        return False
    meanings = candidate.get("meanings")
    if not isinstance(meanings, Sequence) or not meanings:
        return False
    saw_definition = False
    for meaning in meanings:
        if not isinstance(meaning, Mapping):
            return False
        definition = meaning.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            continue
        saw_definition = True
        if not is_placeholder_gloss(definition):
            return False
    return saw_definition
