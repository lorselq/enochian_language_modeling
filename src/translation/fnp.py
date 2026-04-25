"""Deterministic Functional-Nature Profile helpers for translation.

FNP is a soft structural prior derived from accepted root-gloss metadata. This
module keeps the math local, typed, and auditable so the word and phrase
pipelines can consume functional evidence without turning it into a hard POS
label or an LLM guess layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
import re


FNP_AXES = ("nounness", "modifier", "verbness")

OPERATOR_HINTS = {
    "association",
    "clause",
    "clause-introducer",
    "comparison",
    "conjunction",
    "deictic",
    "demonstrative",
    "directional",
    "locative",
    "negation",
    "ordinal",
    "polarity",
    "possessive",
    "pronoun",
    "quantifier",
    "relative",
    "relativizer",
    "relation",
    "relational",
    "sequence",
}


@dataclass(slots=True)
class RootFNPObservation:
    """Carry one accepted root-gloss row into the FNP aggregation layer.

    Root glosses may disagree because roots are polysemous, short, or
    attachment-sensitive. Keeping each observation intact lets aggregation
    preserve uncertainty instead of collapsing the root into a single hard
    grammatical category.
    """

    root: str
    variant: str
    source_cluster_id: int | None = None
    nounness: float | None = None
    modifier: float | None = None
    verbness: float | None = None
    confidence: float | None = None
    definition: str | None = None
    semantic_core_terms: list[str] = field(default_factory=list)
    decoding_guide: str | None = None
    evidence_examples: list[str] = field(default_factory=list)
    attachment_prefix_likelihood: float | None = None
    attachment_suffix_likelihood: float | None = None
    attachment_free_likelihood: float | None = None
    attachment_productivity: float | None = None
    estimated_attachment_profile: str | None = None
    observed_prefix_count: int = 0
    observed_suffix_count: int = 0
    observed_infix_count: int = 0
    observed_free_count: int = 0

    def as_dict(self) -> dict[str, object]:
        """Serialize one observation for repository and debug payloads."""

        return asdict(self)


@dataclass(slots=True)
class RootFNPProfile:
    """Aggregate soft functional evidence for one root.

    The profile is intentionally probabilistic-looking but not a probability
    distribution: nounness, modifier, and verbness are independent evidence
    axes that may all be high or all be weak.
    """

    root: str
    profile: dict[str, float]
    variance: dict[str, float]
    instability: float
    effective_support: float
    observation_count: int
    ambiguity_score: float
    authority_weight: float
    attachment_profile: dict[str, object] = field(default_factory=dict)
    representative_definitions: list[str] = field(default_factory=list)
    semantic_core_terms: list[str] = field(default_factory=list)
    decoding_guides: list[str] = field(default_factory=list)
    representative_evidence_examples: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Serialize the aggregate profile for public JSON outputs."""

        return asdict(self)


@dataclass(slots=True)
class FNPFunctionHypothesis:
    """Represent one soft whole-candidate function hypothesis.

    Candidate and phrase ranking need ranked alternatives rather than one
    forced label. This shape keeps the chosen label explainable while leaving
    room for close alternatives and mixed readings.
    """

    label: str
    score: float

    def as_dict(self) -> dict[str, object]:
        """Serialize the hypothesis for candidate payloads."""

        return asdict(self)


@dataclass(slots=True)
class CandidateFNPAnalysis:
    """Describe FNP reasoning for one word decomposition candidate.

    Word ranking consumes `fnp_raw` as a small scoring prior, while public
    outputs consume the richer fields for auditability and phrase parsing.
    """

    root_fnp_profiles: dict[str, dict[str, object]]
    candidate_fnp_profile: dict[str, float]
    candidate_function_hypotheses: list[FNPFunctionHypothesis]
    inferred_word_function: str
    fnp_confidence: float
    head_analysis: dict[str, object]
    fnp_evidence: list[str]
    fnp_warnings: list[str]
    attachment_fit_score: float
    local_fnp_score: float
    uncertainty_penalty: float
    fnp_raw: float

    def as_dict(self) -> dict[str, object]:
        """Serialize the candidate analysis using stable public keys."""

        return {
            "root_fnp_profiles": self.root_fnp_profiles,
            "candidate_fnp_profile": self.candidate_fnp_profile,
            "candidate_function_hypotheses": [
                hypothesis.as_dict()
                for hypothesis in self.candidate_function_hypotheses
            ],
            "inferred_word_function": self.inferred_word_function,
            "fnp_confidence": self.fnp_confidence,
            "head_analysis": self.head_analysis,
            "fnp_evidence": self.fnp_evidence,
            "fnp_warnings": self.fnp_warnings,
            "attachment_fit_score": self.attachment_fit_score,
            "local_fnp_score": self.local_fnp_score,
            "uncertainty_penalty": self.uncertainty_penalty,
            "fnp_raw": self.fnp_raw,
        }


@dataclass(slots=True)
class PhraseFNPAnalysis:
    """Carry the FNP-facing summary for one phrase token candidate.

    Phrase parsing should not reach back into word-service internals. This
    compact shape provides the token-level function and confidence values used
    by the grammar prior and renderer payloads.
    """

    token: str
    label: str
    confidence: float
    profile: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Serialize one phrase-token FNP summary."""

        return asdict(self)


@dataclass(slots=True)
class PhraseGrammarAnalysis:
    """Summarize soft FNP grammar plausibility for a phrase parse.

    The score is a prior, not a grammar gate. It is designed to nudge close
    parse alternatives toward plausible function sequences while preserving
    ambiguous or semantically strong alternatives.
    """

    phrase_function_sequence: list[dict[str, object]]
    grammar_score: float
    grammar_evidence: list[str] = field(default_factory=list)
    grammar_warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Serialize phrase-level grammar diagnostics."""

        return asdict(self)


def aggregate_root_fnp_profile(
    root: str,
    observations: Sequence[RootFNPObservation],
) -> RootFNPProfile | None:
    """Aggregate accepted gloss observations into one soft root profile.

    The translation pipeline queries many roots repeatedly. This function
    centralizes the weighted means, variance, ambiguity, and warning logic so
    repository caching can reuse the same deterministic profile everywhere.
    """

    cleaned_root = str(root or "").strip().upper()
    usable = [obs for obs in observations if str(obs.root or "").strip()]
    if not cleaned_root or not usable:
        return None

    weights = [_observation_weight(obs) for obs in usable]
    weight_total = sum(weights)
    if weight_total <= 0:
        weights = [1.0 for _obs in usable]
        weight_total = float(len(weights))

    profile = {
        axis: _weighted_mean(
            [_axis_value(obs, axis) for obs in usable],
            weights,
        )
        for axis in FNP_AXES
    }
    variance = {
        axis: _weighted_variance(
            [_axis_value(obs, axis) for obs in usable],
            weights,
            profile[axis],
        )
        for axis in FNP_AXES
    }
    instability = _clamp(
        sum(math.sqrt(max(0.0, value)) for value in variance.values())
        / float(len(FNP_AXES))
    )
    ambiguity = _ambiguity_score(profile, instability)
    average_confidence = _weighted_mean(
        [_safe_float(obs.confidence, default=0.5) for obs in usable],
        weights,
    )
    support_factor = _clamp(math.log1p(len(usable)) / math.log1p(8.0))
    authority_weight = _clamp(
        (0.55 * average_confidence + 0.45 * support_factor)
        * (1.0 - 0.45 * instability)
        * (1.0 - 0.20 * ambiguity)
    )
    attachment_profile = _aggregate_attachment_profile(usable, weights)
    semantic_core_terms = _unique_limited(
        term
        for obs in sorted(usable, key=_observation_weight, reverse=True)
        for term in obs.semantic_core_terms
    )
    definitions = _unique_limited(
        obs.definition
        for obs in sorted(usable, key=_observation_weight, reverse=True)
        if obs.definition
    )
    guides = _unique_limited(
        obs.decoding_guide
        for obs in sorted(usable, key=_observation_weight, reverse=True)
        if obs.decoding_guide
    )
    evidence_examples = _unique_limited(
        example
        for obs in sorted(usable, key=_observation_weight, reverse=True)
        for example in obs.evidence_examples
        if example
    )
    warnings = _root_profile_warnings(
        cleaned_root,
        profile=profile,
        instability=instability,
        ambiguity_score=ambiguity,
        observation_count=len(usable),
    )

    return RootFNPProfile(
        root=cleaned_root,
        profile={key: round(value, 4) for key, value in profile.items()},
        variance={key: round(value, 4) for key, value in variance.items()},
        instability=round(instability, 4),
        effective_support=round(weight_total, 4),
        observation_count=len(usable),
        ambiguity_score=round(ambiguity, 4),
        authority_weight=round(authority_weight, 4),
        attachment_profile=attachment_profile,
        representative_definitions=definitions,
        semantic_core_terms=semantic_core_terms,
        decoding_guides=guides,
        representative_evidence_examples=evidence_examples,
        warnings=warnings,
    )


def analyze_candidate_fnp(
    *,
    word: str,
    morphs: Sequence[str],
    root_profiles: Mapping[str, RootFNPProfile],
    morph_support: Mapping[str, str] | None = None,
    meaning_quality: Mapping[str, float] | None = None,
) -> CandidateFNPAnalysis:
    """Infer a soft whole-word FNP analysis for one decomposition.

    Candidate analysis blends root profiles with support, attachment fit,
    stability, and morph position. Segment length participates only as a small
    feature so large chunks do not automatically become heads.
    """

    normalized_morphs = [str(morph or "").upper() for morph in morphs if str(morph or "")]
    support = {str(key).upper(): str(value) for key, value in (morph_support or {}).items()}
    qualities = {
        str(key).upper(): _safe_float(value, default=0.0)
        for key, value in (meaning_quality or {}).items()
    }
    root_fnp_profiles: dict[str, dict[str, object]] = {}
    segment_rows: list[dict[str, object]] = []
    no_profile_count = 0

    for index, morph in enumerate(normalized_morphs):
        profile = root_profiles.get(morph)
        if profile is None:
            no_profile_count += 1
            segment_rows.append(
                _neutral_segment_row(
                    morph=morph,
                    index=index,
                    count=len(normalized_morphs),
                )
            )
            continue
        root_fnp_profiles[morph] = profile.as_dict()
        row = _segment_row(
            morph=morph,
            index=index,
            count=len(normalized_morphs),
            profile=profile,
            support_label=support.get(morph),
            meaning_quality=qualities.get(morph),
        )
        segment_rows.append(row)

    aggregate_profile = _candidate_profile(segment_rows)
    head_rows = sorted(segment_rows, key=lambda row: float(row["head_score"]), reverse=True)
    best_head = head_rows[0] if head_rows else None
    head_threshold = max(0.0, float(best_head["head_score"]) - 0.08) if best_head else 0.0
    head_roots = [
        str(row["morph"])
        for row in head_rows
        if float(row["head_score"]) >= head_threshold and float(row["head_score"]) >= 0.45
    ][:2]
    operator_roots = [
        str(row["morph"])
        for row in segment_rows
        if float(row["operator_score"]) >= 0.55
        and str(row["morph"]) not in set(head_roots)
    ]
    modifier_roots = [
        str(row["morph"])
        for row in segment_rows
        if float(row["modifier_score"]) >= 0.50
        and str(row["morph"]) not in set(head_roots)
        and str(row["morph"]) not in set(operator_roots)
    ]
    uncertain_roots = [
        str(row["morph"])
        for row in segment_rows
        if row.get("has_profile") is False
        or float(row["uncertainty"]) >= 0.55
        or float(row["attachment_fit"]) <= 0.25
    ]

    attachment_fit = _safe_average(row["attachment_fit"] for row in segment_rows)
    local_score = _safe_average(row["head_score"] for row in head_rows[:2])
    uncertainty_penalty = _clamp(
        _safe_average(row["uncertainty"] for row in segment_rows)
        + 0.15 * (no_profile_count / float(max(1, len(normalized_morphs))))
    )
    authority = _safe_average(row["authority"] for row in segment_rows)
    top_axis, second_axis, top_gap = _top_axis_gap(aggregate_profile)
    confidence = _clamp(
        0.35 * max(0.0, local_score)
        + 0.25 * authority
        + 0.25 * attachment_fit
        + 0.15 * _clamp(top_gap + 0.25)
        - 0.30 * uncertainty_penalty
    )
    hypotheses = _candidate_function_hypotheses(
        aggregate_profile,
        best_head=best_head,
        head_roots=head_roots,
        operator_roots=operator_roots,
        modifier_roots=modifier_roots,
        confidence=confidence,
    )
    inferred = hypotheses[0].label if hypotheses else "mixed_or_uncertain"
    fnp_raw = _clamp(
        (local_score - 0.50) * 1.10
        + (attachment_fit - 0.50) * 0.60
        + (confidence - 0.50) * 0.60
        - uncertainty_penalty * 0.35,
        low=-1.0,
        high=1.0,
    )
    evidence = _candidate_evidence_lines(
        word=word,
        segment_rows=segment_rows,
        inferred=inferred,
        top_axis=top_axis,
    )
    warnings = _candidate_warnings(segment_rows, no_profile_count=no_profile_count)

    return CandidateFNPAnalysis(
        root_fnp_profiles=root_fnp_profiles,
        candidate_fnp_profile={
            key: round(value, 4) for key, value in aggregate_profile.items()
        },
        candidate_function_hypotheses=hypotheses,
        inferred_word_function=inferred,
        fnp_confidence=round(confidence, 4),
        head_analysis={
            "head_roots": head_roots,
            "operator_roots": operator_roots,
            "modifier_roots": modifier_roots,
            "uncertain_roots": sorted(set(uncertain_roots)),
            "segment_roles": [
                {
                    "morph": row["morph"],
                    "position": row["position"],
                    "role": row["role"],
                    "head_score": round(float(row["head_score"]), 4),
                    "operator_score": round(float(row["operator_score"]), 4),
                    "modifier_score": round(float(row["modifier_score"]), 4),
                    "attachment_fit": round(float(row["attachment_fit"]), 4),
                }
                for row in segment_rows
            ],
        },
        fnp_evidence=evidence,
        fnp_warnings=warnings,
        attachment_fit_score=round(attachment_fit, 4),
        local_fnp_score=round(local_score, 4),
        uncertainty_penalty=round(uncertainty_penalty, 4),
        fnp_raw=round(fnp_raw, 4),
    )


def candidate_analysis_payload(analysis: CandidateFNPAnalysis) -> dict[str, object]:
    """Return only the public fields that belong on a word candidate."""

    payload = analysis.as_dict()
    payload.pop("attachment_fit_score", None)
    payload.pop("local_fnp_score", None)
    payload.pop("uncertainty_penalty", None)
    payload.pop("fnp_raw", None)
    return payload


def function_family(label: str | None) -> str:
    """Map detailed FNP labels onto phrase-level function families."""

    normalized = str(label or "").strip().lower()
    if not normalized:
        return "unknown"
    if "relational" in normalized:
        return "relational_operator_like"
    if "modifier" in normalized:
        return "modifier_like"
    if "operator" in normalized:
        return "relational_operator_like"
    if "predicate" in normalized or "verb" in normalized:
        return "predicate_like"
    if "nominal" in normalized or "noun" in normalized:
        return "noun_like"
    if "mixed" in normalized or "uncertain" in normalized:
        return "mixed"
    return "unknown"


def role_hint_from_fnp(label: str | None, confidence: float | None = None) -> str | None:
    """Translate FNP function families into the legacy phrase role hints.

    Phrase parsing already understands `noun`, `verb`, `modifier`, and
    `relational`. This adapter lets FNP steer that existing surface without
    requiring a large parse-layer rewrite.
    """

    if confidence is not None and confidence < 0.20:
        return None
    family = function_family(label)
    if family == "noun_like":
        return "noun"
    if family == "predicate_like":
        return "verb"
    if family == "modifier_like":
        return "modifier"
    if family == "relational_operator_like":
        return "relational"
    return None


def phrase_fnp_analysis(
    *,
    token: str,
    label: str | None,
    confidence: float | None,
    profile: Mapping[str, object] | None = None,
    warnings: Sequence[str] | None = None,
) -> PhraseFNPAnalysis:
    """Normalize word-level FNP payloads for phrase parsing."""

    return PhraseFNPAnalysis(
        token=str(token or "").upper(),
        label=function_family(label),
        confidence=_clamp(_safe_float(confidence, default=0.0)),
        profile={
            key: _safe_float(value, default=0.0)
            for key, value in (profile or {}).items()
            if key in FNP_AXES
        },
        warnings=[str(warning) for warning in (warnings or []) if str(warning).strip()],
    )


def analyze_phrase_fnp_sequence(
    token_profiles: Sequence[PhraseFNPAnalysis],
) -> PhraseGrammarAnalysis:
    """Score a phrase function sequence with a soft FNP grammar prior."""

    sequence = [profile.as_dict() for profile in token_profiles]
    score = 0.0
    evidence: list[str] = []
    warnings: list[str] = []
    for left, right in zip(token_profiles, token_profiles[1:], strict=False):
        transition_score, transition_evidence, transition_warning = score_fnp_transition(
            left.label,
            right.label,
            left_confidence=left.confidence,
            right_confidence=right.confidence,
            left_token=left.token,
            right_token=right.token,
        )
        score += transition_score
        if transition_evidence:
            evidence.append(transition_evidence)
        if transition_warning:
            warnings.append(transition_warning)
    for profile in token_profiles:
        warnings.extend(profile.warnings[:2])
    return PhraseGrammarAnalysis(
        phrase_function_sequence=sequence,
        grammar_score=round(score, 4),
        grammar_evidence=evidence,
        grammar_warnings=_unique_limited(warnings, limit=8),
    )


def score_fnp_transition(
    left_label: str | None,
    right_label: str | None,
    *,
    left_confidence: float,
    right_confidence: float,
    left_token: str = "",
    right_token: str = "",
) -> tuple[float, str, str | None]:
    """Score one adjacent phrase transition without hard-rejecting it."""

    left = function_family(left_label)
    right = function_family(right_label)
    confidence = _clamp((left_confidence + right_confidence) / 2.0)
    if confidence <= 0.0 or "unknown" in {left, right}:
        return 0.0, "", None

    base = 0.0
    evidence = ""
    warning: str | None = None
    if left == "modifier_like" and right == "noun_like":
        base = 1.10
        evidence = "modifier-like -> noun-like transition is plausible"
    elif left == "relational_operator_like" and right == "predicate_like":
        base = 1.25
        evidence = "relational/operator-like -> predicate-like transition is plausible"
    elif left == "predicate_like" and right in {"noun_like", "mixed"}:
        base = 0.85
        evidence = "predicate-like -> argument-like transition is plausible"
    elif left == "noun_like" and right == "relational_operator_like":
        base = 0.70
        evidence = "noun-like -> relational/operator-like transition is plausible"
    elif left == "relational_operator_like" and right == "noun_like":
        base = 0.75
        evidence = "relational/operator-like -> noun-like transition is plausible"
    elif left == "noun_like" and right == "predicate_like":
        base = 0.55
        evidence = "noun-like -> predicate-like transition is plausible"
    elif left == "noun_like" and right == "noun_like":
        base = 0.20
        evidence = "noun-like -> noun-like apposition is possible but weak"
    elif "mixed" in {left, right}:
        base = 0.05
        evidence = "mixed FNP transition preserved with low structural confidence"
        warning = "At least one token remains functionally mixed."
    else:
        base = -0.65
        evidence = "FNP sequence is structurally weak"
        warning = f"Weak FNP transition: {left} -> {right}."

    scaled = _clamp(base * (0.35 + 0.65 * confidence), low=-0.8, high=1.4)
    token_prefix = ""
    if left_token or right_token:
        token_prefix = f"{str(left_token).upper()} -> {str(right_token).upper()}: "
    return round(scaled, 4), token_prefix + evidence + ".", warning


def _segment_row(
    *,
    morph: str,
    index: int,
    count: int,
    profile: RootFNPProfile,
    support_label: str | None,
    meaning_quality: float | None,
) -> dict[str, object]:
    """Build the per-morph scoring row used by candidate FNP inference."""

    fnp = profile.profile
    position = _position_label(index, count)
    attachment_fit = _attachment_fit(profile, position)
    support_strength = _support_strength(support_label)
    quality = _clamp(meaning_quality if meaning_quality is not None else support_strength)
    authority = _safe_float(profile.authority_weight, default=0.0)
    stability = _clamp(1.0 - _safe_float(profile.instability, default=0.0))
    lexical_axis = max(
        _safe_float(fnp.get("nounness"), default=0.0),
        _safe_float(fnp.get("verbness"), default=0.0),
    )
    modifier_axis = _safe_float(fnp.get("modifier"), default=0.0)
    operator_score = _operator_score(profile)
    length_score = _clamp(math.log1p(len(morph)) / math.log1p(7.0))
    head_score = _clamp(
        0.22 * quality
        + 0.20 * lexical_axis
        + 0.17 * stability
        + 0.16 * attachment_fit
        + 0.15 * authority
        + 0.10 * length_score
        - 0.18 * operator_score
    )
    modifier_score = _clamp(
        0.45 * modifier_axis
        + 0.20 * operator_score
        + 0.18 * attachment_fit
        + 0.10 * quality
        + 0.07 * stability
    )
    uncertainty = _clamp(
        0.45 * _safe_float(profile.ambiguity_score, default=0.0)
        + 0.35 * _safe_float(profile.instability, default=0.0)
        + 0.20 * (1.0 - authority)
    )
    return {
        "morph": morph,
        "position": position,
        "has_profile": True,
        "profile": fnp,
        "authority": authority,
        "attachment_fit": attachment_fit,
        "head_score": head_score,
        "modifier_score": modifier_score,
        "operator_score": operator_score,
        "uncertainty": uncertainty,
        "role": _segment_role(
            head_score=head_score,
            modifier_score=modifier_score,
            operator_score=operator_score,
            profile=fnp,
            uncertainty=uncertainty,
        ),
        "warnings": list(profile.warnings),
    }


def _neutral_segment_row(*, morph: str, index: int, count: int) -> dict[str, object]:
    """Build a neutral row when no accepted FNP data exists for a morph."""

    return {
        "morph": morph,
        "position": _position_label(index, count),
        "has_profile": False,
        "profile": {"nounness": 0.0, "modifier": 0.0, "verbness": 0.0},
        "authority": 0.0,
        "attachment_fit": 0.5,
        "head_score": 0.0,
        "modifier_score": 0.0,
        "operator_score": 0.0,
        "uncertainty": 0.75,
        "role": "unknown",
        "warnings": [f"{morph} has no accepted FNP profile."],
    }


def _candidate_profile(segment_rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Combine segment profiles into a whole-candidate soft profile."""

    weighted: dict[str, float] = {axis: 0.0 for axis in FNP_AXES}
    total = 0.0
    for row in segment_rows:
        profile = row.get("profile")
        if not isinstance(profile, Mapping):
            continue
        role = str(row.get("role") or "")
        role_weight = 1.0
        if role in {"operator", "modifier"}:
            role_weight = 0.65
        if role == "unknown":
            role_weight = 0.30
        weight = (
            0.35
            + 0.45 * _safe_float(row.get("head_score"), default=0.0)
            + 0.20 * _safe_float(row.get("authority"), default=0.0)
        ) * role_weight
        total += weight
        for axis in FNP_AXES:
            weighted[axis] += weight * _safe_float(profile.get(axis), default=0.0)
    if total <= 0:
        return {axis: 0.0 for axis in FNP_AXES}
    return {axis: _clamp(value / total) for axis, value in weighted.items()}


def _candidate_function_hypotheses(
    profile: Mapping[str, float],
    *,
    best_head: Mapping[str, object] | None,
    head_roots: Sequence[str],
    operator_roots: Sequence[str],
    modifier_roots: Sequence[str],
    confidence: float,
) -> list[FNPFunctionHypothesis]:
    """Rank whole-word FNP hypotheses from aggregate and head evidence."""

    nounness = _safe_float(profile.get("nounness"), default=0.0)
    modifier = _safe_float(profile.get("modifier"), default=0.0)
    verbness = _safe_float(profile.get("verbness"), default=0.0)
    head_role = str(best_head.get("role") or "") if best_head else ""
    head_axis = "verbness" if verbness >= nounness else "nounness"
    hypotheses: list[FNPFunctionHypothesis] = []
    if head_roots and head_axis == "nounness":
        hypotheses.append(
            FNPFunctionHypothesis(
                "nominal_head_with_modification",
                _clamp(0.45 * nounness + 0.30 * confidence + 0.15 * bool(modifier_roots) + 0.10 * bool(operator_roots)),
            )
        )
    if head_roots and head_axis == "verbness":
        hypotheses.append(
            FNPFunctionHypothesis(
                "predicate_head_with_modification",
                _clamp(0.45 * verbness + 0.30 * confidence + 0.15 * bool(modifier_roots) + 0.10 * bool(operator_roots)),
            )
        )
    if operator_roots or head_role == "operator":
        operator_bonus = 0.25 if head_role == "operator" or not head_roots else 0.05
        hypotheses.append(
            FNPFunctionHypothesis(
                "relational_operator",
                _clamp(0.30 * modifier + 0.25 * confidence + operator_bonus),
            )
        )
    if modifier_roots:
        hypotheses.append(
            FNPFunctionHypothesis(
                "modifier_operator",
                _clamp(0.55 * modifier + 0.25 * confidence + 0.10),
            )
        )
    top_axis, second_axis, gap = _top_axis_gap(profile)
    if gap < 0.16 or not hypotheses:
        hypotheses.append(
            FNPFunctionHypothesis(
                "mixed_or_uncertain",
                _clamp(0.65 - gap + 0.20 * (1.0 - confidence)),
            )
        )
    hypotheses.sort(key=lambda item: item.score, reverse=True)
    return hypotheses[:4]


def _candidate_evidence_lines(
    *,
    word: str,
    segment_rows: Sequence[Mapping[str, object]],
    inferred: str,
    top_axis: str,
) -> list[str]:
    """Render compact evidence notes for candidate debug payloads."""

    lines = [
        f"{str(word).upper()} FNP inference favored {inferred} from {top_axis} evidence."
    ]
    for row in segment_rows[:6]:
        morph = str(row.get("morph") or "")
        role = str(row.get("role") or "unknown")
        position = str(row.get("position") or "unknown")
        attachment = _safe_float(row.get("attachment_fit"), default=0.0)
        head = _safe_float(row.get("head_score"), default=0.0)
        if row.get("has_profile") is False:
            lines.append(f"{morph} has no accepted FNP profile; treated as uncertain.")
            continue
        lines.append(
            f"{morph} acts as {role} in {position} position "
            f"(head={head:.2f}, attachment_fit={attachment:.2f})."
        )
    return lines


def _candidate_warnings(
    segment_rows: Sequence[Mapping[str, object]],
    *,
    no_profile_count: int,
) -> list[str]:
    """Collect FNP warnings without duplicating repeated root messages."""

    warnings: list[str] = []
    if no_profile_count:
        warnings.append(f"{no_profile_count} morph(s) lack accepted FNP profiles.")
    for row in segment_rows:
        morph = str(row.get("morph") or "")
        if _safe_float(row.get("attachment_fit"), default=0.5) <= 0.25:
            warnings.append(f"{morph} has weak attachment fit in this candidate.")
        for warning in row.get("warnings") or []:
            if isinstance(warning, str) and warning.strip():
                warnings.append(f"{morph}: {warning.strip()}")
    return _unique_limited(warnings, limit=8)


def _aggregate_attachment_profile(
    observations: Sequence[RootFNPObservation],
    weights: Sequence[float],
) -> dict[str, object]:
    """Merge attachment likelihoods and evidence counts across observations."""

    likelihoods = {
        "prefix": _weighted_mean(
            [_safe_float(obs.attachment_prefix_likelihood, default=0.0) for obs in observations],
            weights,
        ),
        "suffix": _weighted_mean(
            [_safe_float(obs.attachment_suffix_likelihood, default=0.0) for obs in observations],
            weights,
        ),
        "free": _weighted_mean(
            [_safe_float(obs.attachment_free_likelihood, default=0.0) for obs in observations],
            weights,
        ),
        "infix": 0.0,
    }
    evidence_counts = {
        "prefix": max((obs.observed_prefix_count for obs in observations), default=0),
        "suffix": max((obs.observed_suffix_count for obs in observations), default=0),
        "infix": max((obs.observed_infix_count for obs in observations), default=0),
        "free": max((obs.observed_free_count for obs in observations), default=0),
    }
    count_total = sum(evidence_counts.values())
    if count_total > 0:
        likelihoods["infix"] = evidence_counts["infix"] / float(count_total)
    productivity = _weighted_mean(
        [_safe_float(obs.attachment_productivity, default=0.0) for obs in observations],
        weights,
    )
    estimated_profiles = _unique_limited(
        obs.estimated_attachment_profile
        for obs in observations
        if obs.estimated_attachment_profile
    )
    return {
        "likelihoods": {key: round(_clamp(value), 4) for key, value in likelihoods.items()},
        "evidence_counts": evidence_counts,
        "prefix_likelihood": round(_clamp(likelihoods["prefix"]), 4),
        "suffix_likelihood": round(_clamp(likelihoods["suffix"]), 4),
        "infix_likelihood": round(_clamp(likelihoods["infix"]), 4),
        "free_likelihood": round(_clamp(likelihoods["free"]), 4),
        "observed_prefix_count": evidence_counts["prefix"],
        "observed_suffix_count": evidence_counts["suffix"],
        "observed_infix_count": evidence_counts["infix"],
        "observed_free_count": evidence_counts["free"],
        "productivity": round(_clamp(productivity), 4),
        "estimated_profiles": estimated_profiles,
    }


def _attachment_fit(profile: RootFNPProfile, position: str) -> float:
    """Estimate whether a root's observed attachment behavior fits this slot."""

    attachment = profile.attachment_profile or {}
    likelihoods = attachment.get("likelihoods")
    counts = attachment.get("evidence_counts")
    likelihood = 0.5
    if isinstance(likelihoods, Mapping):
        likelihood = _safe_float(likelihoods.get(position), default=0.5)
    count_score = 0.0
    if isinstance(counts, Mapping):
        total = sum(_safe_float(value, default=0.0) for value in counts.values())
        if total > 0:
            count_score = _safe_float(counts.get(position), default=0.0) / total
    if count_score <= 0:
        return _clamp(likelihood)
    return _clamp(0.55 * likelihood + 0.45 * count_score)


def _operator_score(profile: RootFNPProfile) -> float:
    """Estimate relational/operator behavior from modifier and lexical hints."""

    modifier = _safe_float(profile.profile.get("modifier"), default=0.0)
    terms = " ".join(
        [
            *profile.semantic_core_terms,
            *profile.representative_definitions[:2],
            *profile.decoding_guides[:1],
        ]
    ).lower()
    hint_hits = sum(1 for hint in OPERATOR_HINTS if re.search(rf"\b{re.escape(hint)}\b", terms))
    hint_score = _clamp(0.20 * hint_hits)
    profiles = {
        str(item).lower()
        for item in profile.attachment_profile.get("estimated_profiles", [])
        if isinstance(item, str)
    }
    attachment_operator = 0.15 if any("prefix" in item or "suffix" in item for item in profiles) else 0.0
    return _clamp(0.65 * modifier + hint_score + attachment_operator)


def _segment_role(
    *,
    head_score: float,
    modifier_score: float,
    operator_score: float,
    profile: Mapping[str, float],
    uncertainty: float,
) -> str:
    """Classify a segment role softly for candidate head analysis."""

    if uncertainty >= 0.72:
        return "uncertain"
    if operator_score >= 0.58 and operator_score >= head_score + 0.04:
        return "operator"
    if modifier_score >= 0.55 and modifier_score >= head_score:
        return "modifier"
    nounness = _safe_float(profile.get("nounness"), default=0.0)
    verbness = _safe_float(profile.get("verbness"), default=0.0)
    if head_score >= 0.42:
        return "predicate_head" if verbness > nounness else "nominal_head"
    return "uncertain"


def _root_profile_warnings(
    root: str,
    *,
    profile: Mapping[str, float],
    instability: float,
    ambiguity_score: float,
    observation_count: int,
) -> list[str]:
    """Generate root-level warnings for noisy or overloaded FNP evidence."""

    warnings: list[str] = []
    high_axes = [
        axis for axis, value in profile.items()
        if _safe_float(value, default=0.0) >= 0.45
    ]
    if len(high_axes) >= 2:
        warnings.append(
            "FNP axes are mixed: " + ", ".join(axis for axis in high_axes) + "."
        )
    if len(root) <= 2 and (instability >= 0.24 or ambiguity_score >= 0.50):
        warnings.append(
            "Short root has high FNP variability; treat as overloaded/noisy."
        )
    if observation_count <= 1:
        warnings.append("FNP support is based on a single accepted observation.")
    return warnings


def _ambiguity_score(profile: Mapping[str, float], instability: float) -> float:
    """Compute how mixed the independent FNP axes are."""

    values = sorted((_safe_float(profile.get(axis), default=0.0) for axis in FNP_AXES), reverse=True)
    if not values:
        return 0.0
    top = values[0]
    second = values[1] if len(values) > 1 else 0.0
    high_axis_count = sum(1 for value in values if value >= 0.40)
    close_axis_score = _clamp(1.0 - max(0.0, top - second) / 0.55)
    mixed_axis_score = _clamp((high_axis_count - 1) / 2.0)
    return _clamp(0.45 * close_axis_score + 0.35 * mixed_axis_score + 0.20 * instability)


def _position_label(index: int, count: int) -> str:
    """Map a segment index to the attachment role used by FNP profiles."""

    if count <= 1:
        return "free"
    if index <= 0:
        return "prefix"
    if index >= count - 1:
        return "suffix"
    return "infix"


def _support_strength(label: str | None) -> float:
    """Convert existing morph support labels into bounded FNP evidence weight."""

    normalized = str(label or "").strip().lower()
    if normalized == "cluster":
        return 1.0
    if normalized in {"dictionary", "attested"}:
        return 0.85
    if normalized == "residual":
        return 0.75
    if normalized == "hypothesis":
        return 0.60
    return 0.20


def _top_axis_gap(profile: Mapping[str, float]) -> tuple[str, str, float]:
    """Return the strongest FNP axis, runner-up axis, and their gap."""

    ranked = sorted(
        ((axis, _safe_float(profile.get(axis), default=0.0)) for axis in FNP_AXES),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return "nounness", "modifier", 0.0
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else ("", 0.0)
    return top[0], second[0], max(0.0, top[1] - second[1])


def _axis_value(obs: RootFNPObservation, axis: str) -> float:
    """Read an FNP axis from one observation with neutral missing behavior."""

    if axis == "nounness":
        return _safe_float(obs.nounness, default=0.0)
    if axis == "modifier":
        return _safe_float(obs.modifier, default=0.0)
    if axis == "verbness":
        return _safe_float(obs.verbness, default=0.0)
    return 0.0


def _observation_weight(obs: RootFNPObservation) -> float:
    """Return the confidence weight used by root-profile aggregation."""

    return max(0.05, _safe_float(obs.confidence, default=0.5))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Compute a safe weighted mean for FNP aggregation."""

    total = sum(weights)
    if total <= 0:
        return 0.0
    return _clamp(sum(value * weight for value, weight in zip(values, weights, strict=False)) / total)


def _weighted_variance(
    values: Sequence[float],
    weights: Sequence[float],
    mean: float,
) -> float:
    """Compute a bounded weighted variance for one FNP axis."""

    total = sum(weights)
    if total <= 0:
        return 0.0
    variance = sum(
        weight * ((value - mean) ** 2)
        for value, weight in zip(values, weights, strict=False)
    ) / total
    return _clamp(variance)


def _safe_average(values: Iterable[object]) -> float:
    """Average arbitrary numeric-ish values while ignoring invalid entries."""

    cleaned = [_safe_float(value, default=0.0) for value in values]
    if not cleaned:
        return 0.0
    return sum(cleaned) / float(len(cleaned))


def _unique_limited(values: Iterable[object], *, limit: int = 6) -> list[str]:
    """Return unique non-empty strings in source order with a small limit."""

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _safe_float(value: object, *, default: float) -> float:
    """Best-effort float coercion used at repository and payload boundaries."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return default
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return default
        return parsed if math.isfinite(parsed) else default
    return default


def _clamp(value: object, *, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric-ish value into the requested interval."""

    numeric = _safe_float(value, default=low)
    return max(low, min(high, numeric))
