from __future__ import annotations

"""Normalize placeholder glosses before they leak into human-facing output.

Residual-only whole-word anchors are still useful as evidence, but raw
headlines like ``Top residuals: nacro:1.00`` are not readable translations.
This module centralizes detection and cleanup so ranking, phrase rendering, and
footnote formatting can all make the same decision about when a gloss is real
enough to show to a user.
"""

from collections.abc import Mapping, Sequence
import json
import re

_PLACEHOLDER_PREFIXES = ("top residuals:",)
_BRACKET_PLACEHOLDER_RE = re.compile(r"^\[[A-Z][A-Z' ?-]*\]$")
_META_LINGUISTIC_MARKERS = (
    "morpheme",
    "marker",
    "prefix",
    "suffix",
    "relativizer",
    "deictic",
    "derivational",
    "grammatical",
    "particle",
    "pronoun",
    "specifier",
    "reference",
    "subordinate clause",
)
_DESCRIPTOR_PREFIXES = {
    "physical",
    "terrestrial",
    "tangible",
    "solid",
    "material",
    "core",
    "fundamental",
    "singular",
    "plural",
}
_CLAUSE_SPLIT_RE = re.compile(
    r"(?i)\b(?:used to|functioning as|functions as|serving as|serves as|"
    r"applicable to|appears in|extended to|extends to|as evidenced by|"
    r"when combined with|in compounds|through affixation|marking|encoding|"
    r"denoting|signifying|underlying)\b"
)
_QUOTED_PHRASE_RE = re.compile(r"[\"'`](.{1,48}?)[\"'`]")
_FIRST_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_GLOSS_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "through",
    "upon",
    "which",
    "with",
}


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


def normalize_semantic_terms(value: object) -> list[str]:
    """Normalize semantic-core style payloads into short lexical terms.

    Glossator payloads carry `semantic_core` and `negative_contrast` in a few
    different shapes. Translation needs one stable representation so candidate
    ranking, traces, and phrase rendering can all consume the same cleaned
    lexical hints.
    """

    if value is None:
        return []
    if isinstance(value, str):
        raw_text = value.strip()
        if not raw_text:
            return []
        if raw_text.startswith("[") and raw_text.endswith("]"):
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                parsed = raw_text
            else:
                return normalize_semantic_terms(parsed)
        pieces = re.split(r"[,/;]|(?:\bor\b)", raw_text, flags=re.IGNORECASE)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pieces = [str(item) for item in value]
    else:
        pieces = [str(value)]

    normalized: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        cleaned = _normalize_term(piece)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def semantic_core_gloss(value: object) -> str | None:
    """Return the primary human-facing gloss from semantic-core data."""

    terms = normalize_semantic_terms(value)
    if not terms:
        return None
    return terms[0]


def is_meta_linguistic_gloss(text: object) -> bool:
    """Return whether a gloss is technical scaffolding rather than meaning."""

    if not isinstance(text, str):
        return False
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return False
    if any(marker in normalized for marker in _META_LINGUISTIC_MARKERS):
        if _QUOTED_PHRASE_RE.search(normalized):
            return False
        return True
    if re.match(r"^[A-Z][A-Z0-9'-]*\s+(?:denotes|encodes|signifies)\b", text.strip()):
        return True
    return normalized.startswith(("a root ", "the root ", "root denoting "))


def clean_lexical_gloss(text: object, *, token: str | None = None) -> str | None:
    """Extract a lexical gloss from verbose definition prose.

    When semantic-core data is missing, the fallback should still recover the
    token's meaning in speech rather than surface an analysis label. This
    helper strips examples and meta-linguistic scaffolding, then keeps the
    smallest supported lexical phrase it can recover.
    """

    sanitized = sanitize_human_gloss(text, token=token)
    if sanitized is None:
        return None

    normalized = re.sub(r"(?is)\busage:\s*`[^`]*`", "", sanitized).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    quoted = _quoted_gloss_candidate(normalized)
    if quoted is not None:
        return quoted

    trimmed = _strip_meta_scaffolding(normalized)
    trimmed = re.split(r"[.;:]", trimmed, maxsplit=1)[0].strip()
    trimmed = _CLAUSE_SPLIT_RE.split(trimmed, maxsplit=1)[0].strip()
    if not trimmed:
        return None

    lowered = trimmed.lower()
    if " as " in lowered:
        trimmed = trimmed[: lowered.index(" as ")].strip()
        lowered = trimmed.lower()
    if " of " in lowered and len(trimmed.split()) > 3:
        trimmed = trimmed.rsplit(" of ", 1)[-1].strip()
    if " or " in lowered and len(trimmed.split()) > 4:
        trimmed = trimmed.split(" or ", 1)[0].strip()

    trimmed = _drop_article_prefix(trimmed)
    trimmed = _drop_descriptor_prefixes(trimmed)
    trimmed = re.sub(r"\s+", " ", trimmed).strip(" ,;-")
    lowered = trimmed.lower()
    if lowered.startswith("that which is not"):
        return "not"
    if lowered.startswith("that is not"):
        return "not"
    if lowered.startswith("(be) "):
        trimmed = trimmed[5:].strip()
    elif lowered.startswith("to ") and len(trimmed.split()) <= 4:
        trimmed = trimmed[3:].strip()

    if not trimmed:
        return None
    if is_meta_linguistic_gloss(trimmed):
        return None
    return trimmed


def surface_gloss_from_sources(
    *,
    semantic_core: object,
    definition: object,
    negative_contrast: object | None = None,
    token: str | None = None,
) -> tuple[str | None, str]:
    """Return the best human-facing gloss and the strategy used."""

    semantic_terms = normalize_semantic_terms(semantic_core)
    specific_gloss = specific_gloss_from_definition_and_semantic_core(
        semantic_core=semantic_terms,
        definition=definition,
        negative_contrast=negative_contrast,
        token=token,
    )
    if specific_gloss is not None:
        return specific_gloss, "semantic_core_guided_definition"
    semantic_gloss = semantic_core_gloss(semantic_terms)
    if semantic_gloss is not None and not is_meta_linguistic_gloss(semantic_gloss):
        return semantic_gloss, "semantic_core"
    cleaned = clean_lexical_gloss(definition, token=token)
    if cleaned is not None:
        return cleaned, "cleaned_definition"
    return None, "unresolved"


def specific_gloss_from_definition_and_semantic_core(
    *,
    semantic_core: object,
    definition: object,
    negative_contrast: object | None = None,
    token: str | None = None,
) -> str | None:
    """Prefer a more specific definition span when it clearly fits semantic_core.

    `semantic_core` is the safest human-facing anchor, but some definitions
    contain a short lexical span that is both more specific and still tightly
    aligned with that core idea. This helper mines those spans locally so the
    translator can say `divine source` instead of merely `source` without ever
    falling back to meta-linguistic scaffolding.
    """

    semantic_terms = normalize_semantic_terms(semantic_core)
    if not semantic_terms:
        return None

    base_gloss = semantic_core_gloss(semantic_terms)
    if base_gloss is None:
        return None

    base_score = _definition_span_similarity_score(
        base_gloss,
        semantic_terms=semantic_terms,
        negative_contrast=negative_contrast,
    )
    best_gloss: str | None = None
    best_score = base_score
    for candidate in _definition_span_candidates(definition, token=token):
        score = _definition_span_similarity_score(
            candidate,
            semantic_terms=semantic_terms,
            negative_contrast=negative_contrast,
        )
        if score <= best_score + 0.12:
            continue
        best_gloss = candidate
        best_score = score
    return best_gloss


def gloss_overlaps_negative_contrast(
    gloss: object,
    negative_contrast: object,
) -> list[str]:
    """Return negative-contrast terms that conflict lexically with a gloss."""

    if not isinstance(gloss, str) or not gloss.strip():
        return []
    gloss_tokens = {
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]*", gloss)
    }
    if not gloss_tokens:
        return []
    overlaps: list[str] = []
    for candidate in normalize_semantic_terms(negative_contrast):
        candidate_tokens = {
            token.lower()
            for token in re.findall(r"[a-zA-Z][a-zA-Z'-]*", candidate)
        }
        if candidate_tokens and gloss_tokens & candidate_tokens:
            overlaps.append(candidate)
    return overlaps


def unresolved_token_gloss(token: str) -> str:
    """Render an unresolved token in a stable bracketed form.

    The phrase pipeline needs a deterministic fallback when no grounded gloss
    survives cleanup. A shared bracketed format keeps reports, footnotes, and
    JSON payloads aligned instead of inventing slightly different placeholders.
    """

    normalized = (token or "").strip().upper() or "?"
    return f"[{normalized}]"


def opaque_token_fallback_gloss(token: str) -> str:
    """Render a last-resort opaque token form for human-facing phrase output.

    Some phrase reads survive scoring with only placeholder-strength evidence.
    The phrase layer still needs a visible best-effort chunk in those cases,
    but the older bracketed ``[TOKEN]`` form reads like an internal diagnostic
    instead of an honest fallback gloss. Returning the normalized token in
    lowercase preserves the surviving opaque evidence without pretending it is
    a grounded English translation.
    """

    normalized = (token or "").strip().upper() or "?"
    return normalized.lower()


def _quoted_gloss_candidate(text: str) -> str | None:
    for match in _QUOTED_PHRASE_RE.finditer(text):
        candidate = _normalize_term(match.group(1))
        if candidate and not is_meta_linguistic_gloss(candidate):
            return candidate
    return None


def _definition_span_candidates(text: object, *, token: str | None = None) -> list[str]:
    """Extract short lexical spans from a verbose definition for scoring.

    The definition field often contains both a usable lexical gloss and a lot
    of explanatory prose. We split it into plausible short spans so
    semantic-core-guided selection can compare meaning-bearing pieces instead of
    judging the whole paragraph as one blob.
    """

    sanitized = sanitize_human_gloss(text, token=token)
    if sanitized is None:
        return []
    normalized = re.sub(r"(?is)\busage:\s*`[^`]*`", "", sanitized).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return []

    first_sentence = _definition_first_sentence(normalized)
    fragments: list[str] = []
    if first_sentence is not None:
        fragments.append(first_sentence)
    fragments.extend([normalized, _strip_meta_scaffolding(normalized)])
    quoted = _quoted_gloss_candidate(normalized)
    if quoted is not None:
        fragments.append(quoted)

    for fragment in list(fragments):
        for piece in re.split(r"[.;:]", fragment):
            piece = piece.strip()
            if piece:
                fragments.append(piece)
            for comma_piece in piece.split(","):
                comma_piece = comma_piece.strip()
                if comma_piece:
                    fragments.append(comma_piece)
            lowered = piece.lower()
            if " as " in lowered:
                fragments.append(piece[: lowered.index(" as ")].strip())
            if " underlying " in lowered:
                fragments.append(piece[: lowered.index(" underlying ")].strip())
            if " or " in lowered:
                fragments.extend(
                    subpiece.strip()
                    for subpiece in piece.split(" or ")
                    if subpiece.strip()
                )
            if " and " in lowered and len(piece.split()) <= 8:
                fragments.extend(
                    subpiece.strip()
                    for subpiece in piece.split(" and ")
                    if subpiece.strip()
                )

    normalized_candidates: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        candidate = _normalize_definition_span_candidate(fragment, token=token)
        if candidate is None:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized_candidates.append(candidate)
    return normalized_candidates


def _definition_first_sentence(text: str) -> str | None:
    """Return the leading sentence so lexical selection sees the primary gloss first.

    Multi-sentence glossary definitions usually place the shortest lexical
    meaning in sentence one and reserve later sentences for commentary. Keeping
    that opening sentence at the front of the candidate list helps
    semantic-core span scoring break ties toward user-facing meanings.
    """

    sentence = _FIRST_SENTENCE_SPLIT_RE.split(text, maxsplit=1)[0].strip()
    if not sentence:
        return None
    if sentence.endswith((".", "!", "?")):
        sentence = sentence[:-1].strip()
    return sentence or None


def _normalize_definition_span_candidate(
    text: object,
    *,
    token: str | None = None,
) -> str | None:
    """Clean one candidate definition span without forcing global gloss policy.

    Semantic-core-guided span scoring needs candidate phrases that still retain
    useful specificity. This helper therefore does lighter cleanup than
    `clean_lexical_gloss`, keeping concise modifiers when they help meaning.
    """

    if isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
        text = semantic_core_gloss(text)
    sanitized = sanitize_human_gloss(text, token=token)
    if sanitized is None:
        return None

    quoted = _quoted_gloss_candidate(sanitized)
    if quoted is not None:
        return quoted

    cleaned = re.sub(r"(?is)\busage:\s*`[^`]*`", "", sanitized).strip()
    cleaned = _strip_meta_scaffolding(cleaned)
    cleaned = _CLAUSE_SPLIT_RE.split(cleaned, maxsplit=1)[0].strip()
    cleaned = re.split(r"[.;:]", cleaned, maxsplit=1)[0].strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;-")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if " as " in lowered:
        cleaned = cleaned[: lowered.index(" as ")].strip()
        lowered = cleaned.lower()
    if lowered.startswith("(be) "):
        cleaned = cleaned[5:].strip()
    elif lowered.startswith("to ") and len(cleaned.split()) <= 4:
        cleaned = cleaned[3:].strip()
    elif lowered.startswith("that which is not"):
        cleaned = "not"
    elif lowered.startswith("that is not"):
        cleaned = "not"

    cleaned = _drop_article_prefix(cleaned)
    if " or " in cleaned.lower() and len(cleaned.split()) > 3:
        return None
    if " and " in cleaned.lower() and len(cleaned.split()) > 3:
        return None
    if len(cleaned.split()) > 5:
        return None

    if not cleaned:
        return None
    if is_meta_linguistic_gloss(cleaned):
        return None
    return cleaned


def _definition_span_similarity_score(
    candidate: object,
    *,
    semantic_terms: Sequence[str],
    negative_contrast: object | None = None,
) -> float:
    """Score a definition span against semantic_core and negative contrast.

    This is intentionally lexical and deterministic. We reward spans that cover
    the semantic core and add a small amount of specific, non-generic detail,
    while penalizing spans that echo negative-contrast terms or drift back into
    technical scaffolding.
    """

    if not isinstance(candidate, str) or not candidate.strip():
        return -1.0
    if is_meta_linguistic_gloss(candidate):
        return -1.0

    candidate_tokens = _meaningful_gloss_tokens(candidate)
    if not candidate_tokens:
        return -1.0

    best_score = -1.0
    for term in semantic_terms:
        term_tokens = _meaningful_gloss_tokens(term)
        if not term_tokens:
            continue
        overlap = len(candidate_tokens & term_tokens) / float(len(term_tokens))
        score = overlap
        exact_match = candidate.strip().lower() == term.strip().lower()
        if exact_match:
            score += 0.05

        extra_specific = {
            token
            for token in candidate_tokens - term_tokens
            if token not in _DESCRIPTOR_PREFIXES and token not in _GLOSS_STOPWORDS
        }
        if overlap >= 1.0 and extra_specific:
            score += 0.22
            score += min(0.18, 0.09 * len(extra_specific))

        if len(candidate_tokens) > 5:
            score -= 0.04 * float(len(candidate_tokens) - 5)

        score -= 0.25 * float(
            len(gloss_overlaps_negative_contrast(candidate, negative_contrast))
        )
        best_score = max(best_score, score)

    return best_score


def _strip_meta_scaffolding(text: str) -> str:
    stripped = re.sub(
        r"^[A-Z][A-Z0-9'-]*\s+(?:denotes|encodes|signifies|means)\s+",
        "",
        text,
    )
    stripped = re.sub(
        r"(?i)^(?:a|an|the)\s+(?:root(?:\s+morpheme)?|morpheme|marker|particle|"
        r"prefix(?:al)?|suffix(?:al)?|pronoun|specifier|demonstrative|relativizer|"
        r"deictic\s+morpheme|bound\s+morpheme|grammatical\s+morpheme|"
        r"derivational\s+morpheme)\s+",
        "",
        stripped,
    )
    stripped = re.sub(
        r"(?i)^(?:denoting|meaning|means|marking|encoding|signifying|functioning as|"
        r"functions as|serving as|serves as|relating to)\s+",
        "",
        stripped,
    )
    stripped = re.sub(
        r"(?i)^(?:the\s+root\s+'?[^']+'?\s+)?(?:denotes|encodes|signifies|means)\s+",
        "",
        stripped,
    )
    return stripped.strip()


def _drop_article_prefix(text: str) -> str:
    words = text.split()
    if words and words[0].lower() in {"a", "an", "the"}:
        return " ".join(words[1:]).strip()
    return text


def _drop_descriptor_prefixes(text: str) -> str:
    words = text.split()
    while len(words) > 1 and words[0].lower() in _DESCRIPTOR_PREFIXES:
        words = words[1:]
    return " ".join(words).strip()


def _normalize_term(value: object) -> str | None:
    if not isinstance(value, str):
        value = str(value)
    cleaned = re.sub(r"\s+", " ", value.strip().strip("\"'`")).strip(" ,;-")
    cleaned = re.sub(r"\s*\[[^\]]+\]", "", cleaned).strip(" ,;-")
    if not cleaned:
        return None
    if is_meta_linguistic_gloss(cleaned):
        return None
    return cleaned


def _meaningful_gloss_tokens(text: str) -> set[str]:
    """Tokenize a gloss while dropping high-frequency function words."""

    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]*", text)
        if token.lower() not in _GLOSS_STOPWORDS
    }


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
