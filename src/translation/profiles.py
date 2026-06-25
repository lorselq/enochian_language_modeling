"""Translation profile loading and head-modifier motif scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping, Sequence
import re

import yaml


PROFILE_DIR = Path(__file__).resolve().parent
MANUAL_PROFILE_PATH = PROFILE_DIR / "translation_profiles.yml"
CANON_PROFILE_PATH = PROFILE_DIR / "canon_dictionary_profiles.yml"


@dataclass(frozen=True)
class TranslationProfile:
    """Semantic profile used as a soft translation prior."""

    name: str = "default"
    motif_terms: tuple[str, ...] = ()
    head_terms: tuple[str, ...] = ()
    transformation_hint: str | None = None
    source: str = "built_in"
    metadata: Mapping[str, object] = field(default_factory=dict)


BUILT_IN_PROFILES: dict[str, TranslationProfile] = {
    "default": TranslationProfile(
        name="default",
        motif_terms=(
            "separation",
            "separate",
            "division",
            "divide",
            "distinction",
            "distinct",
            "force",
            "pressure",
            "disturbance",
            "affliction",
        ),
        head_terms=(
            "pillar",
            "pillars",
            "column",
            "support",
            "structure",
            "geometry",
            "form",
        ),
        transformation_hint=(
            "Leading structural heads may be transformed by repeated modifier "
            "motifs into an artifact/function reading."
        ),
    ),
    "separation_artifact": TranslationProfile(
        name="separation_artifact",
        motif_terms=(
            "separation",
            "separate",
            "separated",
            "separating",
            "division",
            "divide",
            "dividing",
            "divided",
            "cut",
            "cutting",
            "sever",
            "severing",
            "distinction",
            "distinct",
            "apart",
            "force",
            "pressure",
            "disturbance",
            "vexing",
            "affliction",
        ),
        head_terms=(
            "pillar",
            "pillars",
            "column",
            "support",
            "rectangular",
            "linear",
            "geometry",
            "geometric",
            "form",
            "structure",
            "structural",
        ),
        transformation_hint=(
            "Rigid or linear head + separative modifiers can yield a "
            "sword-like blade/artifact reading."
        ),
    ),
}


def load_translation_profile(name: str | None = None) -> TranslationProfile:
    """Load a named translation profile from YAML, falling back to built-ins."""

    key = _normalize_profile_name(name)
    profiles = dict(BUILT_IN_PROFILES)
    profiles.update(_load_manual_profiles(MANUAL_PROFILE_PATH))
    profiles.update(_load_canon_profiles(CANON_PROFILE_PATH))
    return profiles.get(key) or profiles["default"]


def analyze_head_modifier(
    morphs: Sequence[str],
    meanings: Sequence[Mapping[str, object]],
    *,
    profile: TranslationProfile,
) -> dict[str, object]:
    """Analyze a decomposition as leading head roots plus modifier roots."""

    normalized_morphs = [str(morph).upper() for morph in morphs if str(morph).strip()]
    if not normalized_morphs:
        return _empty_analysis(profile)

    head_roots = _leading_head_roots(normalized_morphs)
    modifier_roots = normalized_morphs[len(head_roots) :] if head_roots else normalized_morphs[1:]
    if not head_roots and normalized_morphs:
        head_roots = [normalized_morphs[0]]

    meaning_text = {
        str(meaning.get("morph") or meaning.get("canonical") or "").upper(): _meaning_text(meaning)
        for meaning in meanings
        if isinstance(meaning, Mapping)
    }
    head_matches = {
        morph: _matched_terms(meaning_text.get(morph, ""), profile.head_terms)
        for morph in head_roots
    }
    modifier_matches = {
        morph: _matched_terms(meaning_text.get(morph, ""), profile.motif_terms)
        for morph in modifier_roots
    }
    distributed_roots = [
        morph for morph, terms in modifier_matches.items() if terms
    ]
    head_match_count = sum(1 for terms in head_matches.values() if terms)
    modifier_match_count = len(distributed_roots)
    modifier_denominator = max(1, len(modifier_roots))
    score = (
        0.30 * min(1.0, head_match_count / float(max(1, len(head_roots))))
        + 0.50 * min(1.0, modifier_match_count / float(modifier_denominator))
        + (0.20 if modifier_match_count >= 2 else 0.0)
    )
    score = max(0.0, min(1.0, score))
    return {
        "profile": profile.name,
        "head_roots": head_roots,
        "modifier_roots": modifier_roots,
        "head_matches": head_matches,
        "modifier_matches": modifier_matches,
        "distributed_motif_roots": distributed_roots,
        "distributed_motif_count": modifier_match_count,
        "score": score,
        "transformation_hint": profile.transformation_hint,
        "transformational_gloss": _transformational_gloss(
            head_roots=head_roots,
            modifier_roots=distributed_roots,
            profile=profile,
        ),
    }


def score_decomposition_profile(
    morphs: Sequence[str],
    definition_candidates: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    profile: TranslationProfile,
) -> dict[str, object]:
    """Score a decomposition against a profile before final sense selection."""

    meanings: list[dict[str, object]] = []
    normalized_morphs = [str(morph).upper() for morph in morphs if str(morph).strip()]
    head_roots = _leading_head_roots(normalized_morphs)
    if not head_roots and normalized_morphs:
        head_roots = [normalized_morphs[0]]
    for morph in normalized_morphs:
        terms = profile.head_terms if morph in head_roots else profile.motif_terms
        candidate = preferred_definition_candidate(
            definition_candidates.get(morph, []),
            profile=profile,
            preferred_terms=terms,
        )
        if candidate is None:
            candidate = _first_candidate(definition_candidates.get(morph, []))
        meanings.append(
            {
                "morph": morph,
                "definition": _display_definition(candidate) if candidate else "",
                "raw_definition": _clean_text(
                    candidate.get("raw_definition") if candidate else ""
                ),
                "semantic_core_terms": _as_string_list(
                    candidate.get("semantic_core_terms") if candidate else []
                ),
                "negative_contrast": _as_string_list(
                    candidate.get("negative_contrast") if candidate else []
                ),
            }
        )
    return analyze_head_modifier(normalized_morphs, meanings, profile=profile)


def apply_profile_preferred_meanings(
    candidate: dict[str, object],
    definition_candidates: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    profile: TranslationProfile,
) -> None:
    """Replace candidate meanings with senses that best fit the active profile."""

    morphs = [str(morph).upper() for morph in candidate.get("morphs", []) if isinstance(morph, str)]
    head_roots = _leading_head_roots(morphs)
    if not head_roots and morphs:
        head_roots = [morphs[0]]
    meanings_raw = candidate.get("meanings")
    meanings = meanings_raw if isinstance(meanings_raw, list) else []
    by_morph = {
        str(meaning.get("morph") or meaning.get("canonical") or "").upper(): meaning
        for meaning in meanings
        if isinstance(meaning, dict)
    }
    for morph in morphs:
        terms = profile.head_terms if morph in head_roots else profile.motif_terms
        preferred = preferred_definition_candidate(
            definition_candidates.get(morph, []),
            profile=profile,
            preferred_terms=terms,
        )
        if preferred is None:
            continue
        meaning = by_morph.get(morph)
        if meaning is None:
            meaning = {"morph": morph, "canonical": morph}
            meanings.append(meaning)
            by_morph[morph] = meaning
        _update_meaning_from_candidate(meaning, morph, preferred)
    candidate["meanings"] = meanings
    candidate["head_modifier_analysis"] = analyze_head_modifier(
        morphs,
        [meaning for meaning in meanings if isinstance(meaning, Mapping)],
        profile=profile,
    )


def preferred_definition_candidate(
    candidates: Sequence[Mapping[str, object]],
    *,
    profile: TranslationProfile,
    preferred_terms: Sequence[str] | None = None,
) -> Mapping[str, object] | None:
    """Return the candidate that best matches profile terms."""

    if not candidates:
        return None
    terms = tuple(preferred_terms or (*profile.motif_terms, *profile.head_terms))
    scored: list[tuple[float, Mapping[str, object]]] = []
    for item in candidates:
        text = _candidate_text(item)
        matches = _matched_terms(text, terms)
        quality = _safe_float(item.get("quality"), default=0.0)
        source_bonus = 0.25 if str(item.get("source") or "") in {"cluster", "residual"} else 0.0
        scored.append((len(matches) * 10.0 + source_bonus + quality, item))
    scored.sort(key=lambda row: row[0], reverse=True)
    if scored[0][0] <= 0:
        return None
    return scored[0][1]


def _load_manual_profiles(path: Path) -> dict[str, TranslationProfile]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("profiles") if isinstance(payload, Mapping) else None
    if not isinstance(rows, Mapping):
        return {}
    return {
        _normalize_profile_name(name): _profile_from_mapping(
            _normalize_profile_name(name),
            row,
            source="manual_yaml",
        )
        for name, row in rows.items()
        if isinstance(row, Mapping)
    }


def _load_canon_profiles(path: Path) -> dict[str, TranslationProfile]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("canon_dictionary_profiles") if isinstance(payload, Mapping) else None
    if not isinstance(rows, Mapping):
        return {}
    profiles: dict[str, TranslationProfile] = {}
    for name, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        key = _normalize_profile_name(name)
        profiles[key] = TranslationProfile(
            name=key,
            motif_terms=tuple(_as_string_list(row.get("motif_terms"))),
            head_terms=tuple(_as_string_list(row.get("head_terms"))),
            transformation_hint=str(row.get("definition") or "").strip() or None,
            source="canon_dictionary_seed",
            metadata=dict(row),
        )
    return profiles


def _profile_from_mapping(
    name: str,
    row: Mapping[str, object],
    *,
    source: str,
) -> TranslationProfile:
    return TranslationProfile(
        name=name,
        motif_terms=tuple(_as_string_list(row.get("motif_terms"))),
        head_terms=tuple(_as_string_list(row.get("head_terms"))),
        transformation_hint=str(row.get("transformation_hint") or "").strip() or None,
        source=source,
        metadata=dict(row),
    )


def _leading_head_roots(morphs: Sequence[str]) -> list[str]:
    heads: list[str] = []
    for morph in morphs:
        if len(morph) < 3:
            break
        heads.append(morph)
        if len(heads) >= 2:
            break
    return heads


def _update_meaning_from_candidate(
    meaning: dict[str, object],
    morph: str,
    candidate: Mapping[str, object],
) -> None:
    definition = _display_definition(candidate)
    raw_definition = _clean_text(candidate.get("raw_definition") or definition)
    semantic_core = _as_string_list(candidate.get("semantic_core_terms"))
    negative = _as_string_list(candidate.get("negative_contrast"))
    source = str(candidate.get("source") or "derived")
    meaning.update(
        {
            "morph": morph,
            "canonical": morph,
            "definition": definition,
            "raw_definition": raw_definition,
            "surface_gloss": definition,
            "surface_gloss_strategy": candidate.get("surface_gloss_strategy")
            or "translation_profile",
            "semantic_core": semantic_core,
            "semantic_core_terms": semantic_core,
            "negative_contrast": negative,
            "provenance": source,
            "cluster_id": candidate.get("cluster_id"),
            "source_cluster_id": candidate.get("source_cluster_id")
            or candidate.get("cluster_id"),
            "source_variant": candidate.get("source_variant"),
            "definition_trace": {
                "selected_definition": definition,
                "raw_selected_definition": raw_definition,
                "selected_source": source,
                "selected_quality": candidate.get("quality"),
                "selected_semantic_core": semantic_core,
                "selected_negative_contrast": negative,
                "surface_gloss": definition,
                "surface_gloss_strategy": "translation_profile",
                "runner_ups": [],
                "suppressed": [],
                "blind_dictionary_fallback": False,
                "negative_contrast_penalties": [],
                "meta_linguistic_rejections": [],
                "selected_source_detail": {
                    key: candidate.get(key)
                    for key in (
                        "cluster_id",
                        "source_cluster_id",
                        "source_variant",
                        "source_run_id",
                        "source_cluster_index",
                    )
                    if candidate.get(key) is not None
                },
            },
        }
    )


def _transformational_gloss(
    *,
    head_roots: Sequence[str],
    modifier_roots: Sequence[str],
    profile: TranslationProfile,
) -> str | None:
    if not head_roots or not modifier_roots:
        return None
    root_text = " + ".join([*head_roots, *modifier_roots])
    if profile.name == "separation_artifact":
        return f"{root_text}: rigid/linear head + separative modifiers => sword-like artifact"
    if profile.transformation_hint:
        return f"{root_text}: {profile.transformation_hint}"
    return None


def _empty_analysis(profile: TranslationProfile) -> dict[str, object]:
    return {
        "profile": profile.name,
        "head_roots": [],
        "modifier_roots": [],
        "head_matches": {},
        "modifier_matches": {},
        "distributed_motif_roots": [],
        "distributed_motif_count": 0,
        "score": 0.0,
        "transformation_hint": profile.transformation_hint,
        "transformational_gloss": None,
    }


def _candidate_text(candidate: Mapping[str, object]) -> str:
    pieces: list[str] = []
    for key in ("definition", "raw_definition", "surface_gloss"):
        value = candidate.get(key)
        if isinstance(value, str):
            pieces.append(value)
    for key in ("semantic_core", "semantic_core_terms", "negative_contrast"):
        value = candidate.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            pieces.extend(str(item) for item in value)
    return " ".join(piece for piece in pieces if piece).lower()


def _meaning_text(meaning: Mapping[str, object]) -> str:
    return _candidate_text(meaning)


def _matched_terms(text: str, terms: Sequence[str]) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    matched: list[str] = []
    for term in terms:
        term_text = str(term or "").strip().lower()
        if not term_text:
            continue
        if re.search(rf"\b{re.escape(term_text)}\b", lowered):
            matched.append(term_text)
    return matched


def _display_definition(candidate: Mapping[str, object]) -> str:
    raw_definition = _clean_text(candidate.get("raw_definition"))
    definition = _clean_text(candidate.get("definition"))
    if raw_definition and (
        not definition
        or len(raw_definition.split()) > len(definition.split())
        or len(raw_definition) > len(definition) + 12
    ):
        return raw_definition
    return definition or raw_definition


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .;:")


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _first_candidate(
    candidates: Sequence[Mapping[str, object]] | None,
) -> Mapping[str, object] | None:
    if not candidates:
        return None
    return candidates[0]


def _normalize_profile_name(name: object) -> str:
    value = str(name or "default").strip().lower()
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_") or "default"


def _safe_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
