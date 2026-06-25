"""Agentic translation dossiers built from read-only evidence tools.

This module adds a researcher-style layer above the existing deterministic
translation services. The layer does not mutate memory or insights databases;
it gathers source-aware evidence packets, explores plausible alternate reads,
and returns a structured dossier that can be rendered as Markdown or JSON.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from types import TracebackType
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from enochian_lm.common.config import get_config_paths
from .llm_synthesis import DEFAULT_LLM_CONTEXT
from .tokenization import tokenize_words


class WordTranslator(Protocol):
    """Describe the word-service surface the dossier harness depends on."""

    repository: Any

    def translate_word(self, word: str, **kwargs: object) -> dict[str, object]:
        """Return deterministic word translation evidence for one source."""

    def close(self) -> None:
        """Release word-service resources owned by the harness."""


class PhraseTranslator(Protocol):
    """Describe the phrase-service surface the dossier harness depends on."""

    def translate_phrase(self, phrase: str, **kwargs: object) -> dict[str, object]:
        """Return deterministic phrase translation evidence for one source."""

    def close(self) -> None:
        """Release phrase-service resources owned by the harness."""


class AgenticTranslationService:
    """Build read-only translation dossiers from curated evidence tools.

    The service sits above the existing translation pipeline. It calls the
    deterministic baseline first, then performs bounded source-aware follow-up
    lookups so an LLM or human reader can see why a less likely reading might
    still deserve attention.
    """

    VALID_SOURCES = ("solo", "debate")

    def __init__(
        self,
        *,
        word_service: WordTranslator,
        phrase_service: PhraseTranslator | None = None,
        dictionary_entries: Sequence[Mapping[str, object]] | None = None,
        raw_dictionary_entries: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """Keep the harness dependency-injected and testable.

        Word and phrase services remain the canonical baseline engines. The
        dictionary snapshots are stored separately because the harness must
        preserve canon/non-canon labels even when the runtime dictionary loader
        filters non-canon entries out of normal translation.
        """

        self.word_service = word_service
        self.phrase_service = phrase_service
        self.dictionary_entries = list(dictionary_entries or [])
        self.raw_dictionary_entries = list(raw_dictionary_entries or [])
        self._dictionary_by_norm = self._build_dictionary_index(
            self.dictionary_entries,
            self.raw_dictionary_entries,
        )

    @classmethod
    def from_config(
        cls,
        *,
        sources: Iterable[str] | None = None,
        llm_enabled: bool = False,
        llm_use_remote: bool = False,
    ) -> "AgenticTranslationService":
        """Construct a dossier harness from canonical project paths.

        The harness reuses the single-word service inside phrase translation so
        DB handles and caches stay variant-aware without opening a second copy
        of the large insights databases.
        """

        selected = _resolve_sources(sources)
        paths = get_config_paths()
        dictionary_path = Path(paths["dictionary"])
        from enochian_lm.root_extraction.utils.dictionary_loader import load_dictionary
        from .phrase_service import PhraseTranslationService
        from .service import SingleWordTranslationService

        dictionary_entries = load_dictionary(str(dictionary_path))
        raw_dictionary_entries = _load_raw_dictionary_entries(dictionary_path)
        word_service = SingleWordTranslationService.from_config(
            variants=selected,
            llm_enabled=llm_enabled,
            llm_use_remote=llm_use_remote,
        )
        phrase_service = PhraseTranslationService(word_service=word_service)
        return cls(
            word_service=word_service,
            phrase_service=phrase_service,
            dictionary_entries=dictionary_entries,
            raw_dictionary_entries=raw_dictionary_entries,
        )

    def close(self) -> None:
        """Close owned translation resources once dossier generation is done."""

        if self.phrase_service is not None:
            self.phrase_service.close()
            return
        self.word_service.close()

    def __enter__(self) -> "AgenticTranslationService":
        """Support `with` blocks so CLI callers cannot leak DB handles."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release resources when the surrounding CLI command exits."""

        self.close()

    def build_dossier(
        self,
        text: str,
        *,
        kind: str = "auto",
        analysis_source: str = "both",
        top_k: int = 3,
        exploration_budget: int = 4,
        llm: bool = False,
        llm_context: str | None = None,
        allow_dictionary: bool = True,
        allow_whole_word: bool = True,
        speculative: bool = True,
        speculative_profile: str = "default",
        translation_profile: str = "default",
    ) -> dict[str, object]:
        """Run fixed research phases with bounded alternate exploration.

        The method keeps the process deterministic enough for regression tests:
        baseline calls happen first, evidence packets are gathered from named
        tools, then the exploratory branch is capped by `exploration_budget`.
        """

        normalized_text = (text or "").strip()
        if not normalized_text:
            raise ValueError("Text must be a non-empty string.")

        resolved_kind = self._resolve_kind(normalized_text, kind)
        sources = _resolve_analysis_source(analysis_source)
        top_k = max(1, int(top_k))
        exploration_budget = max(0, int(exploration_budget))
        trace: list[dict[str, object]] = []

        baselines = self._run_baselines(
            normalized_text,
            kind=resolved_kind,
            sources=sources,
            top_k=top_k,
            llm=llm,
            llm_context=llm_context or DEFAULT_LLM_CONTEXT,
            allow_dictionary=allow_dictionary,
            allow_whole_word=allow_whole_word,
            speculative=speculative,
            speculative_profile=speculative_profile,
            translation_profile=translation_profile,
            trace=trace,
        )
        evidence_packets = self._gather_evidence_packets(
            normalized_text,
            kind=resolved_kind,
            sources=sources,
            baselines=baselines,
            top_k=top_k,
            exploration_budget=exploration_budget,
            allow_dictionary=allow_dictionary,
            trace=trace,
        )
        source_comparison = self._compare_sources(
            sources,
            baselines=baselines,
            evidence_packets=evidence_packets,
        )
        hypotheses = self._build_hypotheses(sources, baselines=baselines)
        explorations = self._build_otherwise_explorations(
            sources,
            baselines=baselines,
            evidence_packets=evidence_packets,
            source_comparison=source_comparison,
            budget=exploration_budget,
        )
        recommendation = self._build_final_recommendation(
            hypotheses=hypotheses,
            source_comparison=source_comparison,
            explorations=explorations,
        )
        uncertainty = self._build_uncertainty(
            hypotheses=hypotheses,
            explorations=explorations,
            source_comparison=source_comparison,
        )
        translation_summary = self._build_translation_summary(
            recommendation=recommendation,
            hypotheses=hypotheses,
            evidence_packets=evidence_packets,
            baselines=baselines,
            explorations=explorations,
        )

        return {
            "request": {
                "text": normalized_text,
                "kind": resolved_kind,
                "analysis_source": analysis_source,
                "sources": sources,
                "top_k": top_k,
                "exploration_budget": exploration_budget,
                "llm_enabled": bool(llm),
                "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                "dictionary_enabled": bool(allow_dictionary),
                "whole_word_enabled": bool(allow_whole_word),
                "blind_mode_enabled": not (bool(allow_dictionary) and bool(allow_whole_word)),
                "speculative_enabled": bool(speculative),
                "speculative_profile": speculative_profile,
                "translation_profile": translation_profile,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "translation_summary": translation_summary,
            "final_recommendation": recommendation,
            "hypotheses": hypotheses,
            "otherwise_explorations": explorations,
            "source_comparison": source_comparison,
            "uncertainty": uncertainty,
            "research_trace": trace,
            "evidence_packets": evidence_packets,
            "baseline_by_source": baselines,
        }

    def render_markdown(self, dossier: Mapping[str, object]) -> str:
        """Render the four-question dossier summary for interactive use.

        The full JSON payload can be large enough to bury the useful work. This
        renderer leads with the final definition, morphology, root-choice
        reasoning, and extra clues so the output reads like a research note
        instead of a database dump.
        """

        request = _as_mapping(dossier.get("request"))
        summary = _as_mapping(dossier.get("translation_summary"))
        recommendation = _as_mapping(dossier.get("final_recommendation"))
        uncertainty = _as_mapping(dossier.get("uncertainty"))
        comparison = _as_mapping(dossier.get("source_comparison"))
        breakdown = _as_mapping(summary.get("etymological_breakdown"))
        justification = _as_mapping(summary.get("root_choice_justification"))
        considerations = _as_list(summary.get("root_considerations"))
        exploratory = _as_list(summary.get("exploratory_hypotheses"))
        clues = _as_list(summary.get("additional_clues"))

        lines = [
            f"# Agentic Translation Dossier: {request.get('text', '')}",
            "",
            "## Final Definition",
            f"{summary.get('final_definition') or recommendation.get('translation') or 'unresolved'}",
            f"- Analysis source: `{recommendation.get('analysis_source', 'unknown')}`",
            f"- Confidence: `{uncertainty.get('confidence_label', 'unknown')}`",
        ]
        rationale = recommendation.get("rationale")
        if rationale:
            lines.append(f"- Recommendation note: {rationale}")

        lines.extend(["", "## Etymological Breakdown"])
        surface = breakdown.get("surface")
        if surface:
            lines.append(f"- Breakdown: `{surface}`")
        for morph in _as_list(breakdown.get("morphs")):
            row = _as_mapping(morph)
            label = row.get("morph", "?")
            definition = _compact_text(row.get("definition", "unresolved"))
            evidence_class = row.get("evidence_class", "unknown")
            lines.append(f"- `{label}`: {definition} ({evidence_class})")

        lines.extend(["", "## Root Choice Justification"])
        candidate_rationale = justification.get("candidate_rationale")
        if candidate_rationale:
            lines.append(f"- Candidate rationale: {candidate_rationale}")
        for root_choice in _as_list(justification.get("root_choices")):
            row = _as_mapping(root_choice)
            lines.append(
                f"- `{row.get('morph', '?')}` selected as "
                f"{_compact_text(row.get('selected_definition', 'unresolved'))}: "
                f"{_compact_text(row.get('why_selected', 'no local rationale recorded'))}"
            )
            alternatives = _as_list(row.get("alternatives"))
            if alternatives:
                alt_text = "; ".join(
                    _compact_text(_as_mapping(alt).get("definition") or alt)
                    for alt in alternatives[:3]
                )
                lines.append(f"  - Alternatives considered: {alt_text}")

        lines.extend(["", "## Roots Considered"])
        if considerations:
            for consideration in considerations:
                row = _as_mapping(consideration)
                lines.append(
                    f"- `{row.get('morph', '?')}`: "
                    f"{_compact_text(row.get('selected_definition', 'unresolved'))} "
                    f"[{row.get('analysis_source', 'unknown')} rank "
                    f"{row.get('candidate_rank', '?')}; "
                    f"{row.get('candidate_role', 'candidate')}]"
                )
                why = row.get("why_considered")
                if why:
                    lines.append(f"  - Why considered: {_compact_text(why)}")
                alternatives = _as_list(row.get("alternate_definitions"))
                if alternatives:
                    alt_text = "; ".join(
                        _compact_text(_as_mapping(alt).get("definition") or alt)
                        for alt in alternatives[:3]
                    )
                    lines.append(f"  - Other possibilities: {alt_text}")
                counts = _as_mapping(row.get("definition_counts"))
                if counts:
                    count_text = ", ".join(
                        f"{key}={value}" for key, value in counts.items()
                    )
                    lines.append(f"  - Definition-count pressure: {count_text}")
        else:
            lines.append("- No root-level candidates were surfaced.")

        lines.extend(["", "## Exploratory Hypotheses"])
        if exploratory:
            for hypothesis in exploratory:
                row = _as_mapping(hypothesis)
                lines.append(
                    f"- `{row.get('surface', 'unresolved')}`: "
                    f"{_compact_text(row.get('translation', 'unresolved'))} "
                    f"[{row.get('analysis_source', 'unknown')}]"
                )
                rationale = row.get("rationale")
                if rationale:
                    lines.append(f"  - Why considered: {_compact_text(rationale)}")
                head_roots = _as_list(row.get("head_roots"))
                modifier_roots = _as_list(row.get("modifier_roots"))
                if head_roots or modifier_roots:
                    lines.append(
                        "  - Head/modifiers: "
                        f"head={'+'.join(str(root) for root in head_roots) or 'unresolved'}; "
                        f"modifiers={'+'.join(str(root) for root in modifier_roots) or 'none'}"
                    )
                motif_count = row.get("distributed_motif_count")
                if motif_count is not None:
                    lines.append(f"  - Distributed motif roots: {motif_count}")
                transformational = row.get("transformational_gloss")
                if transformational:
                    lines.append(f"  - Transformational gloss: {_compact_text(transformational)}")
                evidence = _as_list(row.get("evidence"))
                if evidence:
                    evidence_text = "; ".join(
                        _compact_text(_as_mapping(item).get("summary") or item)
                        for item in evidence[:4]
                    )
                    lines.append(f"  - Evidence: {evidence_text}")
        else:
            lines.append("- No speculative full-cover hypotheses were surfaced.")

        lines.extend(["", "## Additional Clues"])
        if clues:
            for clue in clues:
                row = _as_mapping(clue)
                lines.append(f"- {_compact_text(row.get('clue', ''))}")
                detail = row.get("detail")
                if detail:
                    lines.append(f"  - {_compact_text(detail)}")
        else:
            lines.append("- No additional bounded-digging clues were surfaced.")

        agreements = _as_list(comparison.get("agreements"))
        disagreements = _as_list(comparison.get("disagreements"))
        gaps = _as_list(comparison.get("source_specific_gaps"))
        if agreements or disagreements or gaps:
            lines.extend(["", "## Source Notes"])
            for item in agreements:
                lines.append(f"- Agreement: {item}")
            for item in disagreements:
                lines.append(f"- Disagreement: {item}")
            for item in gaps:
                lines.append(f"- Gap: {item}")

        return "\n".join(lines).rstrip() + "\n"

    def compact_dossier(self, dossier: Mapping[str, object]) -> dict[str, object]:
        """Return the compact JSON shape users need for normal inspection.

        Full baseline payloads remain available through `--detail full`, but the
        default machine-readable output should answer the same four questions as
        the Markdown renderer without burying them under raw translator data.
        """

        return {
            "request": dossier.get("request", {}),
            "translation_summary": dossier.get("translation_summary", {}),
            "final_recommendation": dossier.get("final_recommendation", {}),
            "source_comparison": dossier.get("source_comparison", {}),
            "uncertainty": dossier.get("uncertainty", {}),
            "research_trace_summary": _summarize_research_trace(
                _as_list(dossier.get("research_trace"))
            ),
        }

    def _run_baselines(
        self,
        text: str,
        *,
        kind: str,
        sources: Sequence[str],
        top_k: int,
        llm: bool,
        llm_context: str,
        allow_dictionary: bool,
        allow_whole_word: bool,
        speculative: bool,
        speculative_profile: str,
        translation_profile: str,
        trace: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        """Collect deterministic baselines separately for every source."""

        baselines: dict[str, dict[str, object]] = {}
        with self._dictionary_visibility(allow_dictionary):
            for source in sources:
                trace.append(
                    {
                        "phase": "baseline",
                        "tool": f"translate_{kind}",
                        "analysis_source": source,
                    }
                )
                if kind == "word":
                    baselines[source] = self.word_service.translate_word(
                        text,
                        variants=[source],
                        top_k=top_k,
                        llm=llm,
                        llm_context=llm_context,
                        allow_dictionary=allow_dictionary,
                        allow_whole_word=allow_whole_word,
                        speculative=speculative,
                        speculative_compete=bool(speculative),
                        speculative_profile=speculative_profile,
                        translation_profile=translation_profile,
                        use_beam_search=True,
                        with_root_groups=True,
                    )
                    continue
                if self.phrase_service is None:
                    raise ValueError("Phrase translation is unavailable for this harness.")
                baselines[source] = self.phrase_service.translate_phrase(
                    text,
                    variants=[source],
                    top_k=top_k,
                    llm=llm,
                    llm_context=llm_context,
                    allow_dictionary=allow_dictionary,
                    allow_whole_word=allow_whole_word,
                    speculative=speculative,
                    speculative_profile=speculative_profile,
                    translation_profile=translation_profile,
                    memory_update=False,
                    use_memory=False,
                    with_root_groups=True,
                )
        return baselines

    @contextmanager
    def _dictionary_visibility(self, allow_dictionary: bool):
        """Temporarily hide dictionary entries during blind baseline calls.

        The existing word translator accepts `allow_dictionary=False`, but it
        still receives a dictionary snapshot for weak substring support. Agentic
        blind mode is stricter: the baseline should behave as though no
        dictionary entries were available, while restoring the service
        immediately afterward for normal calls.
        """

        if allow_dictionary:
            yield
            return

        candidate_finder = getattr(self.word_service, "candidate_finder", None)
        if candidate_finder is None or not hasattr(candidate_finder, "dictionary"):
            yield
            return

        original_dictionary = candidate_finder.dictionary
        candidate_finder.dictionary = {}
        try:
            yield
        finally:
            candidate_finder.dictionary = original_dictionary


    def _gather_evidence_packets(
        self,
        text: str,
        *,
        kind: str,
        sources: Sequence[str],
        baselines: Mapping[str, Mapping[str, object]],
        top_k: int,
        exploration_budget: int,
        allow_dictionary: bool,
        trace: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Gather dictionary, candidate, cluster, residual, and FNP packets."""

        packets: list[dict[str, object]] = []
        tokens = [text.upper()] if kind == "word" else [token.upper() for token in tokenize_words(text)]
        for source in sources:
            baseline = baselines.get(source, {})
            packets.append(self._baseline_summary_packet(source, baseline))
            if allow_dictionary:
                for token in tokens:
                    dictionary_packet = self._dictionary_packet(source, token)
                    if dictionary_packet is not None:
                        packets.append(dictionary_packet)
            candidate_rows = self._candidate_rows(baseline)[:top_k]
            for candidate in candidate_rows:
                packets.extend(
                    self._candidate_packets(
                        source,
                        candidate,
                        allow_dictionary=allow_dictionary,
                    )
                )
            morphs = self._morphs_for_followup(tokens, candidate_rows, exploration_budget)
            packets.extend(
                self._repository_packets_for_morphs(
                    source,
                    morphs,
                    allow_dictionary=allow_dictionary,
                    trace=trace,
                )
            )
        return packets

    def _baseline_summary_packet(
        self,
        source: str,
        baseline: Mapping[str, object],
    ) -> dict[str, object]:
        """Summarize baseline evidence counts without merging sources."""

        evidence = _as_mapping(baseline.get("evidence"))
        counts = {
            key: evidence.get(key, 0)
            for key in (
                "direct_clusters",
                "residual_semantics",
                "morph_hypotheses",
                "attested_definitions",
                "dictionary_morphs",
            )
        }
        return {
            "analysis_source": source,
            "packet_type": "baseline_summary",
            "label": "baseline evidence counts",
            "evidence_class": "derived",
            "canon_status": "mixed_or_unknown",
            "counts": counts,
            "summary": ", ".join(f"{key}={value}" for key, value in counts.items()),
        }

    def _dictionary_packet(self, source: str, token: str) -> dict[str, object] | None:
        """Return a dictionary packet that preserves canon/non-canon metadata."""

        entry = self._dictionary_by_norm.get(token.lower())
        if not entry:
            return None
        definitions = _dictionary_definitions(entry)
        citations = entry.get("key_citations")
        return {
            "analysis_source": source,
            "packet_type": "dictionary_entry",
            "label": token,
            "token": token,
            "evidence_class": "dictionary-backed",
            "canon_status": _canon_status(entry),
            "definitions": definitions,
            "citations": citations if isinstance(citations, list) else [],
            "summary": "; ".join(definitions[:3]) if definitions else "dictionary entry",
        }

    def _candidate_packets(
        self,
        source: str,
        candidate: Mapping[str, object],
        *,
        allow_dictionary: bool,
    ) -> list[dict[str, object]]:
        """Convert baseline candidate meanings into source-aware packets."""

        packets: list[dict[str, object]] = []
        meanings = _as_list(candidate.get("meanings"))
        morphs = _as_list(candidate.get("morphs"))
        if not meanings and morphs:
            meanings = [{"morph": morph, "definition": candidate.get("translation_gloss")} for morph in morphs]
        for meaning in meanings:
            row = _as_mapping(meaning)
            morph = str(row.get("morph") or row.get("canonical") or "").upper()
            if not morph:
                continue
            provenance = str(row.get("provenance") or row.get("selected_source") or "derived")
            definition = _first_string(
                row.get("surface_gloss"),
                row.get("definition"),
                row.get("raw_definition"),
            )
            packets.append(
                {
                    "analysis_source": source,
                    "packet_type": "candidate_meaning",
                    "label": morph,
                    "morph": morph,
                    "evidence_class": _evidence_class(provenance),
                    "canon_status": self._canon_status_for_token(
                        morph,
                        allow_dictionary=allow_dictionary,
                    ),
                    "definition": definition,
                    "semantic_core": _as_list(row.get("semantic_core") or row.get("semantic_core_terms")),
                    "negative_contrast": _as_list(row.get("negative_contrast")),
                    "source_detail": row.get("source_cluster_id") or row.get("cluster_id"),
                    "summary": definition or "candidate morph meaning",
                }
            )
        return packets

    def _repository_packets_for_morphs(
        self,
        source: str,
        morphs: Sequence[str],
        *,
        allow_dictionary: bool,
        trace: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Perform bounded read-only repository lookups for serious root digging."""

        repository = getattr(self.word_service, "repository", None)
        if repository is None:
            return []

        packets: list[dict[str, object]] = []
        for morph in morphs:
            trace.append(
                {
                    "phase": "evidence",
                    "tool": "fetch_word_evidence",
                    "analysis_source": source,
                    "morph": morph,
                }
            )
            evidence = self._safe_fetch_word_evidence(repository, morph, source)
            if evidence is not None:
                packets.extend(
                    self._residual_packets(
                        source,
                        evidence,
                        allow_dictionary=allow_dictionary,
                    )
                )
            clusters = self._safe_fetch_clusters(repository, morph, source)
            for cluster in clusters[:2]:
                packets.append(
                    self._cluster_packet(
                        source,
                        morph,
                        cluster,
                        allow_dictionary=allow_dictionary,
                    )
                )
            fnp_profiles = self._safe_fetch_fnp_profiles(repository, [morph], source)
            profile = fnp_profiles.get(morph)
            if profile is not None:
                packets.append(
                    {
                        "analysis_source": source,
                        "packet_type": "fnp_profile",
                        "label": morph,
                        "morph": morph,
                        "evidence_class": "derived",
                        "canon_status": self._canon_status_for_token(
                            morph,
                            allow_dictionary=allow_dictionary,
                        ),
                        "profile": _to_plain(profile),
                        "summary": f"Functional/attachment profile available for {morph}.",
                    }
                )
        return packets

    def _safe_fetch_word_evidence(
        self,
        repository: Any,
        morph: str,
        source: str,
    ) -> object | None:
        """Read residual and support evidence without letting optional tables fail."""

        try:
            return repository.fetch_word_evidence(morph, variants=[source])
        except Exception:
            return None

    def _safe_fetch_clusters(
        self,
        repository: Any,
        morph: str,
        source: str,
    ) -> list[Any]:
        """Read accepted clusters for one morph through the repository API."""

        try:
            return list(repository.fetch_clusters(morph, variants=[source]))
        except Exception:
            return []

    def _safe_fetch_fnp_profiles(
        self,
        repository: Any,
        morphs: Sequence[str],
        source: str,
    ) -> dict[str, object]:
        """Read root functional profiles when the underlying view is present."""

        try:
            return dict(repository.fetch_root_fnp_profiles(morphs, variants=[source]))
        except Exception:
            return {}

    def _residual_packets(
        self,
        source: str,
        evidence: object,
        *,
        allow_dictionary: bool,
    ) -> list[dict[str, object]]:
        """Serialize solo/debate residual evidence as weaker source packets."""

        packets: list[dict[str, object]] = []
        for residual in evidence.residual_semantics[:2]:
            packets.append(
                {
                    "analysis_source": source,
                    "packet_type": "residual_semantics",
                    "label": residual.residual,
                    "morph": residual.residual,
                    "parent_word": residual.parent_word,
                    "evidence_class": "residual",
                    "canon_status": self._canon_status_for_token(
                        residual.residual,
                        allow_dictionary=allow_dictionary,
                    ),
                    "definition": residual.glossator_def,
                    "semantic_core": residual.semantic_core,
                    "negative_contrast": residual.negative_contrast,
                    "summary": residual.residual_headline
                    or residual.glossator_def
                    or "residual semantic evidence",
                }
            )
        return packets

    def _cluster_packet(
        self,
        source: str,
        morph: str,
        cluster: Any,
        *,
        allow_dictionary: bool,
    ) -> dict[str, object]:
        """Serialize an accepted cluster with compact raw-definition context."""

        raw_defs = [
            {
                "source_word": raw_def.source_word,
                "definition": raw_def.definition,
                "tier": raw_def.tier,
            }
            for raw_def in cluster.raw_definitions[:4]
        ]
        return {
            "analysis_source": source,
            "packet_type": "accepted_cluster",
            "label": morph,
            "morph": morph,
            "cluster_id": cluster.cluster_id,
            "evidence_class": "derived",
            "canon_status": self._canon_status_for_token(
                morph,
                allow_dictionary=allow_dictionary,
            ),
            "definition": cluster.glossator_def,
            "semantic_core": cluster.semantic_core,
            "negative_contrast": cluster.negative_contrast,
            "raw_definitions": raw_defs,
            "summary": _cluster_summary(cluster),
        }

    def _compare_sources(
        self,
        sources: Sequence[str],
        *,
        baselines: Mapping[str, Mapping[str, object]],
        evidence_packets: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Compare selected sources without collapsing their confidence."""

        top_by_source = {
            source: self._top_translation(baselines.get(source, {}))
            for source in sources
        }
        agreements: list[str] = []
        disagreements: list[str] = []
        source_specific_gaps: list[str] = []
        if len(sources) > 1:
            normalized = {
                source: str(row.get("translation") or "").strip().lower()
                for source, row in top_by_source.items()
            }
            unique = {value for value in normalized.values() if value}
            if len(unique) == 1 and unique:
                agreements.append(
                    "Top readings agree on " + next(iter(unique))
                )
            elif unique:
                disagreements.append(
                    "; ".join(
                        f"{source}={top_by_source[source].get('translation', 'unresolved')}"
                        for source in sources
                    )
                )

        for source in sources:
            evidence = _as_mapping(baselines.get(source, {}).get("evidence"))
            if not any(
                evidence.get(key, 0)
                for key in ("direct_clusters", "residual_semantics", "attested_definitions")
            ):
                source_specific_gaps.append(f"{source} has little direct DB evidence.")
            if _source_has_noncanon_only_packets(source, evidence_packets):
                source_specific_gaps.append(
                    f"{source} has support that is non-canon or absent from the canonical dictionary."
                )

        return {
            "top_by_source": top_by_source,
            "agreements": agreements,
            "disagreements": disagreements,
            "source_specific_gaps": source_specific_gaps,
        }

    def _build_hypotheses(
        self,
        sources: Sequence[str],
        *,
        baselines: Mapping[str, Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Turn baseline candidates into candidate translation hypotheses."""

        hypotheses: list[dict[str, object]] = []
        for source in sources:
            rows = self._candidate_rows(baselines.get(source, {}))
            if not rows:
                top = self._top_translation(baselines.get(source, {}))
                if top.get("translation"):
                    rows = [top]
            for index, candidate in enumerate(rows[:3], start=1):
                translation = self._candidate_translation(candidate)
                hypotheses.append(
                    {
                        "analysis_source": source,
                        "rank": candidate.get("rank", index),
                        "translation": translation,
                        "analysis_type": candidate.get("analysis_type"),
                        "confidence": candidate.get("confidence"),
                        "score": candidate.get("score"),
                        "morphs": _as_list(candidate.get("morphs")),
                        "rationale": self._candidate_rationale(candidate),
                    }
                )
        return hypotheses

    def _build_translation_summary(
        self,
        *,
        recommendation: Mapping[str, object],
        hypotheses: Sequence[Mapping[str, object]],
        evidence_packets: Sequence[Mapping[str, object]],
        baselines: Mapping[str, Mapping[str, object]],
        explorations: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Build the user-facing four-part summary from the full dossier.

        The agentic harness still preserves raw evidence, but this summary is
        the practical workbench: final definition, chosen decomposition,
        root-choice reasoning, and clues discovered while digging.
        """

        chosen = self._chosen_hypothesis(recommendation, hypotheses)
        return {
            "final_definition": recommendation.get("translation") or "unresolved",
            "etymological_breakdown": self._build_etymological_breakdown(
                chosen,
                evidence_packets,
            ),
            "root_choice_justification": self._build_root_choice_justification(
                chosen,
                baselines,
                evidence_packets,
            ),
            "root_considerations": self._build_root_considerations(
                recommendation,
                baselines,
                evidence_packets,
            ),
            "exploratory_hypotheses": self._build_exploratory_hypotheses(
                baselines,
                evidence_packets,
            ),
            "additional_clues": self._build_additional_clues(
                explorations,
                evidence_packets,
            ),
        }

    def _chosen_hypothesis(
        self,
        recommendation: Mapping[str, object],
        hypotheses: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        """Find the hypothesis that backs the final recommendation."""

        source = recommendation.get("analysis_source")
        rank = recommendation.get("rank")
        for hypothesis in hypotheses:
            if hypothesis.get("analysis_source") == source and hypothesis.get("rank") == rank:
                return hypothesis
        return hypotheses[0] if hypotheses else {}

    def _build_etymological_breakdown(
        self,
        chosen: Mapping[str, object],
        evidence_packets: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Summarize the decomposition used by the selected hypothesis."""

        morphs = [str(morph).upper() for morph in _as_list(chosen.get("morphs"))]
        source = str(chosen.get("analysis_source") or "")
        packet_lookup = {
            str(packet.get("morph") or packet.get("label") or "").upper(): packet
            for packet in evidence_packets
            if packet.get("packet_type") == "candidate_meaning"
            and packet.get("analysis_source") == source
        }
        rows: list[dict[str, object]] = []
        for morph in morphs:
            packet = _as_mapping(packet_lookup.get(morph))
            rows.append(
                {
                    "morph": morph,
                    "definition": packet.get("definition") or "unresolved",
                    "evidence_class": packet.get("evidence_class") or "unknown",
                    "canon_status": packet.get("canon_status") or "unknown",
                    "source_detail": packet.get("source_detail"),
                }
            )
        return {
            "surface": " + ".join(morphs) if morphs else "unresolved",
            "analysis_source": chosen.get("analysis_source"),
            "morphs": rows,
        }

    def _build_root_choice_justification(
        self,
        chosen: Mapping[str, object],
        baselines: Mapping[str, Mapping[str, object]],
        evidence_packets: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Explain why selected roots beat nearby alternate readings."""

        source = str(chosen.get("analysis_source") or "")
        selected_morphs = [str(morph).upper() for morph in _as_list(chosen.get("morphs"))]
        packet_lookup = {
            str(packet.get("morph") or packet.get("label") or "").upper(): packet
            for packet in evidence_packets
            if packet.get("packet_type") == "candidate_meaning"
            and packet.get("analysis_source") == source
        }
        candidate = self._matching_candidate(baselines.get(source, {}), chosen)
        meaning_lookup = {
            str(_as_mapping(meaning).get("morph") or _as_mapping(meaning).get("canonical") or "").upper(): _as_mapping(meaning)
            for meaning in _as_list(candidate.get("meanings"))
        }
        root_choices: list[dict[str, object]] = []
        for morph in selected_morphs:
            packet = _as_mapping(packet_lookup.get(morph))
            meaning = _as_mapping(meaning_lookup.get(morph))
            trace = _as_mapping(meaning.get("definition_trace"))
            alternatives = [
                {
                    "definition": _as_mapping(alt).get("definition"),
                    "source": _as_mapping(alt).get("source"),
                    "quality": _as_mapping(alt).get("quality"),
                    "cluster_id": _as_mapping(alt).get("cluster_id"),
                }
                for alt in _as_list(trace.get("runner_ups"))[:4]
            ]
            root_choices.append(
                {
                    "morph": morph,
                    "selected_definition": packet.get("definition") or meaning.get("definition") or "unresolved",
                    "why_selected": trace.get("bundle_selection_reason")
                    or trace.get("selected_source")
                    or packet.get("summary")
                    or "Selected by baseline ranking.",
                    "selected_quality": trace.get("selected_quality"),
                    "source_detail": packet.get("source_detail"),
                    "alternatives": alternatives,
                }
            )
        return {
            "candidate_rationale": chosen.get("rationale"),
            "score": chosen.get("score"),
            "confidence": chosen.get("confidence"),
            "root_choices": root_choices,
        }

    def _build_root_considerations(
        self,
        recommendation: Mapping[str, object],
        baselines: Mapping[str, Mapping[str, object]],
        evidence_packets: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Summarize every bounded root the baseline seriously examined."""

        selected_source = recommendation.get("analysis_source")
        selected_rank = recommendation.get("rank")
        packet_lookup = _candidate_packet_lookup(evidence_packets)
        rows: list[dict[str, object]] = []
        rows_by_key: dict[tuple[str, str], dict[str, object]] = {}
        for source, baseline in baselines.items():
            for index, candidate in enumerate(self._candidate_rows(baseline)[:3], start=1):
                rank = int(candidate.get("rank") or index)
                candidate_role = (
                    "selected_final"
                    if source == selected_source and rank == selected_rank
                    else "alternate_candidate"
                )
                candidate_reason = self._candidate_rationale(candidate)
                score_breakdown = _as_mapping(candidate.get("score_breakdown"))
                definition_counts = _as_mapping(score_breakdown.get("definition_counts"))
                meanings_by_morph = {
                    str(
                        _as_mapping(meaning).get("morph")
                        or _as_mapping(meaning).get("canonical")
                        or ""
                    ).upper(): _as_mapping(meaning)
                    for meaning in _as_list(candidate.get("meanings"))
                }
                for morph in [str(item).upper() for item in _as_list(candidate.get("morphs"))]:
                    if not morph:
                        continue
                    key = (source, morph)
                    occurrence = {
                        "rank": rank,
                        "role": candidate_role,
                        "analysis_type": candidate.get("analysis_type"),
                        "translation": self._candidate_translation(candidate),
                    }
                    existing = rows_by_key.get(key)
                    if existing is not None:
                        _as_list(existing.get("candidate_occurrences")).append(occurrence)
                        if candidate_role == "selected_final":
                            existing["candidate_role"] = candidate_role
                            existing["candidate_rank"] = rank
                            existing["candidate_analysis_type"] = candidate.get("analysis_type")
                            existing["candidate_translation"] = self._candidate_translation(candidate)
                        continue
                    meaning = _as_mapping(meanings_by_morph.get(morph))
                    packet = _as_mapping(packet_lookup.get((source, morph)))
                    trace = _as_mapping(meaning.get("definition_trace"))
                    row = {
                        "morph": morph,
                        "analysis_source": source,
                        "candidate_rank": rank,
                        "candidate_role": candidate_role,
                        "candidate_analysis_type": candidate.get("analysis_type"),
                        "candidate_translation": self._candidate_translation(candidate),
                        "candidate_occurrences": [occurrence],
                        "selected_definition": _first_string(
                            packet.get("definition"),
                            meaning.get("definition"),
                            meaning.get("surface_gloss"),
                        )
                        or "unresolved",
                        "why_considered": candidate_reason,
                        "why_definition_selected": trace.get("bundle_selection_reason")
                        or trace.get("selected_source")
                        or packet.get("summary")
                        or "Definition inherited from this candidate's selected meaning.",
                        "evidence_class": packet.get("evidence_class")
                        or _evidence_class(
                            str(
                                meaning.get("provenance")
                                or meaning.get("selected_source")
                                or "derived"
                            )
                        ),
                        "canon_status": packet.get("canon_status") or "unknown",
                        "definition_counts": (
                            {morph: definition_counts.get(morph)}
                            if morph in definition_counts
                            else {}
                        ),
                        "alternate_definitions": _definition_alternatives(
                            morph,
                            source,
                            trace,
                            evidence_packets,
                        ),
                    }
                    rows.append(row)
                    rows_by_key[key] = row
        return rows[:12]

    def _build_exploratory_hypotheses(
        self,
        baselines: Mapping[str, Mapping[str, object]],
        evidence_packets: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Project speculative word candidates into the public dossier summary."""

        packet_lookup = _candidate_packet_lookup(evidence_packets)
        hypotheses: list[dict[str, object]] = []
        for source, baseline in baselines.items():
            speculative_rows = _as_list(baseline.get("speculative_hypotheses"))
            for candidate in speculative_rows[:5]:
                row = _as_mapping(candidate)
                morphs = [str(morph).upper() for morph in _as_list(row.get("morphs"))]
                if not morphs:
                    continue
                head_modifier = _as_mapping(
                    row.get("head_modifier_analysis")
                    or _as_mapping(row.get("decision_trace")).get("head_modifier_analysis")
                )
                evidence_rows: list[dict[str, object]] = []
                for morph in morphs:
                    packet = _as_mapping(packet_lookup.get((source, morph)))
                    evidence_rows.append(
                        {
                            "morph": morph,
                            "definition": packet.get("definition")
                            or self._definition_for_candidate_morph(row, morph),
                            "evidence_class": packet.get("evidence_class") or "derived",
                            "cluster_id": packet.get("source_detail"),
                            "summary": (
                                f"{morph}: "
                                f"{packet.get('definition') or self._definition_for_candidate_morph(row, morph) or 'unresolved'}"
                            ),
                        }
                    )
                hypotheses.append(
                    {
                        "analysis_source": source,
                        "surface": " + ".join(morphs),
                        "translation": self._candidate_translation(row),
                        "score": row.get("score"),
                        "rationale": self._candidate_rationale(row),
                        "head_roots": _as_list(head_modifier.get("head_roots")),
                        "modifier_roots": _as_list(head_modifier.get("modifier_roots")),
                        "distributed_motif_count": head_modifier.get(
                            "distributed_motif_count"
                        ),
                        "transformational_gloss": head_modifier.get(
                            "transformational_gloss"
                        ),
                        "evidence": evidence_rows,
                    }
                )
        return hypotheses[:8]

    @staticmethod
    def _definition_for_candidate_morph(
        candidate: Mapping[str, object],
        morph: str,
    ) -> str | None:
        for meaning in _as_list(candidate.get("meanings")):
            row = _as_mapping(meaning)
            if str(row.get("morph") or row.get("canonical") or "").upper() == morph:
                return _first_string(
                    row.get("surface_gloss"),
                    row.get("definition"),
                    row.get("raw_definition"),
                )
        return None

    def _matching_candidate(
        self,
        baseline: Mapping[str, object],
        chosen: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Locate the raw candidate row behind a chosen summary hypothesis."""

        chosen_rank = chosen.get("rank")
        chosen_morphs = [str(morph).upper() for morph in _as_list(chosen.get("morphs"))]
        for candidate in self._candidate_rows(baseline):
            candidate_morphs = [str(morph).upper() for morph in _as_list(candidate.get("morphs"))]
            if candidate.get("rank") == chosen_rank or candidate_morphs == chosen_morphs:
                return candidate
        return {}

    def _build_additional_clues(
        self,
        explorations: Sequence[Mapping[str, object]],
        evidence_packets: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Collect the extra clues surfaced during bounded digging."""

        clues: list[dict[str, object]] = []
        for exploration in explorations:
            clues.append(
                {
                    "kind": exploration.get("trigger") or "exploration",
                    "clue": exploration.get("finding") or exploration.get("question") or "Alternate-reading clue.",
                    "detail": exploration.get("question"),
                }
            )
        for packet in evidence_packets:
            packet_type = packet.get("packet_type")
            if packet_type not in {"accepted_cluster", "residual_semantics", "fnp_profile"}:
                continue
            label = packet.get("label") or packet.get("morph") or "evidence"
            clues.append(
                {
                    "kind": packet_type,
                    "clue": f"{label}: {packet.get('summary', '')}",
                    "detail": packet.get("definition") if packet_type != "fnp_profile" else None,
                }
            )
            if len(clues) >= 8:
                break
        return clues[:8]

    def _build_otherwise_explorations(
        self,
        sources: Sequence[str],
        *,
        baselines: Mapping[str, Mapping[str, object]],
        evidence_packets: Sequence[Mapping[str, object]],
        source_comparison: Mapping[str, object],
        budget: int,
    ) -> list[dict[str, object]]:
        """Surface bounded reasons to consider a less likely reading."""

        explorations: list[dict[str, object]] = []
        if budget <= 0:
            return explorations

        if _as_list(source_comparison.get("disagreements")):
            explorations.append(
                {
                    "trigger": "solo/debate disagreement",
                    "question": "Do the sources point toward different readings?",
                    "finding": "The selected analysis sources disagree at the top reading, so the dossier preserves alternatives rather than merging them.",
                }
            )

        for source in sources:
            rows = self._candidate_rows(baselines.get(source, {}))
            if len(rows) >= 2 and self._dictionary_vs_compositional(rows):
                explorations.append(
                    {
                        "trigger": "dictionary-vs-compositional conflict",
                        "analysis_source": source,
                        "question": "Could a lower-ranked compositional read matter?",
                        "finding": "A dictionary/exact reading outranks a compositional reading; inspect the compositional packet before treating the exact gloss as exhaustive.",
                    }
                )
            if len(rows) >= 2 and self._score_margin(rows[0], rows[1]) < 0.25:
                explorations.append(
                    {
                        "trigger": "strong runner-up",
                        "analysis_source": source,
                        "question": "Is the runner-up close enough to preserve?",
                        "finding": "The top two candidate scores are close enough that the lower-ranked reading remains plausible.",
                    }
                )
            rejected = _as_list(_as_mapping(baselines.get(source, {}).get("evidence")).get("rejected_morphs"))
            if rejected:
                explorations.append(
                    {
                        "trigger": "rejected morph pressure",
                        "analysis_source": source,
                        "question": "Did any tempting splits already receive negative evidence?",
                        "finding": "Rejected morphs are present: " + ", ".join(str(item) for item in rejected[:5]),
                    }
                )
            short_ambiguous = self._short_ambiguous_morphs(rows)
            if short_ambiguous:
                explorations.append(
                    {
                        "trigger": "ambiguous short roots",
                        "analysis_source": source,
                        "question": "Could a short root be over-explaining the word?",
                        "finding": "Short/high-count roots need caution: " + ", ".join(short_ambiguous[:5]),
                    }
                )

        if any(
            packet.get("canon_status") in {"non_canon", "non_canon_or_unmatched"}
            for packet in evidence_packets
        ):
            explorations.append(
                {
                    "trigger": "non-canon-only support",
                    "question": "Is any support outside the canonical dictionary?",
                    "finding": "At least one evidence packet is non-canon or unmatched, which is especially important for solo analysis.",
                }
            )

        return explorations[:budget]

    def _build_final_recommendation(
        self,
        *,
        hypotheses: Sequence[Mapping[str, object]],
        source_comparison: Mapping[str, object],
        explorations: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Choose a recommendation while preserving source disagreement."""

        if not hypotheses:
            return {
                "translation": "unresolved",
                "analysis_source": None,
                "rationale": "No candidate hypothesis survived baseline translation.",
                "alternatives": [],
            }

        chosen = max(
            hypotheses,
            key=lambda row: (
                _safe_float(row.get("confidence")),
                _safe_float(row.get("score")),
                -int(row.get("rank") or 99),
            ),
        )
        alternatives = [
            {
                "analysis_source": row.get("analysis_source"),
                "translation": row.get("translation"),
                "rank": row.get("rank"),
            }
            for row in hypotheses
            if row is not chosen
        ][:4]
        rationale = "Selected the strongest source-specific hypothesis by confidence and score."
        if _as_list(source_comparison.get("disagreements")):
            rationale += " Source disagreement remains explicit in the dossier."
        if explorations:
            rationale += " Alternate-reading triggers should be reviewed before treating this as final."
        return {
            "translation": chosen.get("translation"),
            "analysis_source": chosen.get("analysis_source"),
            "rank": chosen.get("rank"),
            "rationale": rationale,
            "alternatives": alternatives,
        }

    def _build_uncertainty(
        self,
        *,
        hypotheses: Sequence[Mapping[str, object]],
        explorations: Sequence[Mapping[str, object]],
        source_comparison: Mapping[str, object],
    ) -> dict[str, object]:
        """Summarize uncertainty without pretending sources were merged."""

        reasons: list[str] = []
        if _as_list(source_comparison.get("disagreements")):
            reasons.append("Solo/debate top readings disagree.")
        if explorations:
            reasons.extend(str(item.get("trigger")) for item in explorations[:4])
        if not hypotheses:
            reasons.append("No baseline hypothesis was available.")

        top_confidence = max((_safe_float(row.get("confidence")) for row in hypotheses), default=0.0)
        if reasons and top_confidence < 0.9:
            label = "low"
        elif reasons:
            label = "medium"
        elif top_confidence >= 0.9:
            label = "high"
        else:
            label = "medium"
        return {
            "confidence_label": label,
            "top_confidence": top_confidence,
            "reasons": reasons,
        }

    def _candidate_rows(self, baseline: Mapping[str, object]) -> list[Mapping[str, object]]:
        """Extract word or phrase candidates from known baseline shapes."""

        candidates = _as_list(baseline.get("candidates"))
        if not candidates:
            candidates = _as_list(baseline.get("senses"))
        if candidates:
            return [_as_mapping(candidate) for candidate in candidates]

        chosen_parse = _as_mapping(baseline.get("chosen_parse"))
        token_choices = _as_list(chosen_parse.get("token_choices"))
        if token_choices:
            return [_as_mapping(choice) for choice in token_choices]
        if chosen_parse:
            return [chosen_parse]
        return []

    def _candidate_translation(self, candidate: Mapping[str, object]) -> str:
        """Pick the most human-readable translation field from a candidate."""

        return _first_string(
            candidate.get("synthesized_definition"),
            candidate.get("translation_gloss"),
            candidate.get("bundle_surface_gloss"),
            candidate.get("translation_skeleton"),
            candidate.get("definition"),
            candidate.get("raw_definition"),
            candidate.get("concatenated_meanings"),
        ) or "unresolved"

    def _candidate_rationale(self, candidate: Mapping[str, object]) -> str:
        """Produce a compact explanation for one baseline hypothesis."""

        trace = _as_mapping(candidate.get("decision_trace"))
        if trace.get("selection_reason"):
            return str(trace["selection_reason"])
        morphs = _as_list(candidate.get("morphs"))
        if morphs:
            return "Built from morphs: " + ", ".join(str(morph) for morph in morphs)
        return "Baseline candidate from deterministic translation."

    def _top_translation(self, baseline: Mapping[str, object]) -> dict[str, object]:
        """Return a compact top-reading summary for one source."""

        rows = self._candidate_rows(baseline)
        if rows:
            row = rows[0]
            return {
                "translation": self._candidate_translation(row),
                "rank": row.get("rank", 1),
                "confidence": row.get("confidence"),
                "score": row.get("score"),
                "analysis_type": row.get("analysis_type"),
            }
        return {
            "translation": _first_string(
                baseline.get("rendered_translation"),
                baseline.get("translation_skeleton"),
            ),
            "rank": 1,
            "confidence": baseline.get("render_confidence") or baseline.get("confidence"),
            "score": baseline.get("score"),
            "analysis_type": baseline.get("analysis_type"),
        }

    def _resolve_kind(self, text: str, kind: str) -> str:
        """Resolve auto kind from token count while validating explicit kinds."""

        if kind not in {"auto", "word", "phrase"}:
            raise ValueError(f"Unsupported agentic translation kind: {kind}")
        if kind != "auto":
            return kind
        tokens = tokenize_words(text)
        return "phrase" if len(tokens) > 1 else "word"

    def _morphs_for_followup(
        self,
        tokens: Sequence[str],
        candidate_rows: Sequence[Mapping[str, object]],
        exploration_budget: int,
    ) -> list[str]:
        """Choose a bounded morph list for repository follow-up tools."""

        morphs: list[str] = []
        for token in tokens:
            if token not in morphs:
                morphs.append(token)
        for candidate in candidate_rows:
            for morph in _as_list(candidate.get("morphs")):
                raw = str(morph).strip().upper()
                if raw and raw not in morphs:
                    morphs.append(raw)
            for meaning in _as_list(candidate.get("meanings")):
                row = _as_mapping(meaning)
                raw = str(row.get("morph") or row.get("canonical") or "").upper()
                if raw and raw not in morphs:
                    morphs.append(raw)
        limit = max(1, exploration_budget + len(tokens))
        return morphs[:limit]

    def _dictionary_vs_compositional(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> bool:
        """Detect exact-vs-compositional tension in ranked candidates."""

        first_type = str(rows[0].get("analysis_type") or "")
        later_types = {str(row.get("analysis_type") or "") for row in rows[1:]}
        return "dictionary" in first_type and any("compositional" in item for item in later_types)

    def _score_margin(
        self,
        first: Mapping[str, object],
        second: Mapping[str, object],
    ) -> float:
        """Compute a conservative score margin for runner-up exploration."""

        first_score = _safe_float(first.get("score"))
        second_score = _safe_float(second.get("score"))
        if first_score == 0 and second_score == 0:
            first_score = _safe_float(first.get("confidence"))
            second_score = _safe_float(second.get("confidence"))
        return abs(first_score - second_score)

    def _short_ambiguous_morphs(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> list[str]:
        """Find short roots whose definition counts suggest overreach risk."""

        ambiguous: list[str] = []
        for row in rows[:3]:
            score_breakdown = _as_mapping(row.get("score_breakdown"))
            counts = _as_mapping(score_breakdown.get("definition_counts"))
            for morph, count in counts.items():
                if len(str(morph)) <= 2 and _safe_float(count) >= 10:
                    ambiguous.append(
                        f"{str(morph).upper()} "
                        f"({int(_safe_float(count))} definitions)"
                    )
        return ambiguous

    def _canon_status_for_token(
        self,
        token: str,
        *,
        allow_dictionary: bool = True,
    ) -> str:
        """Classify a token while respecting dictionary suppression.

        Blind dossier runs should not consult dictionary metadata to label
        otherwise unknown DB-derived morphs. Returning an explicit suppressed
        label keeps the output honest without losing source provenance.
        """

        if not allow_dictionary:
            return "unclassified_without_dictionary"
        entry = self._dictionary_by_norm.get((token or "").strip().lower())
        if not entry:
            return "non_canon_or_unmatched"
        return _canon_status(entry)

    def _build_dictionary_index(
        self,
        entries: Sequence[Mapping[str, object]],
        raw_entries: Sequence[Mapping[str, object]],
    ) -> dict[str, Mapping[str, object]]:
        """Index canonical and raw dictionary rows without losing non-canon rows."""

        index: dict[str, Mapping[str, object]] = {}
        for entry in entries:
            key = str(entry.get("normalized") or entry.get("canonical") or "").lower()
            if key:
                index[key] = entry
            canonical = str(entry.get("canonical") or "").lower()
            if canonical:
                index.setdefault(canonical, entry)
        for entry in raw_entries:
            key = str(
                entry.get("normalized")
                or entry.get("word")
                or entry.get("canonical")
                or ""
            ).lower()
            if key:
                index[key] = entry
        return index


def render_agentic_dossier_markdown(dossier: Mapping[str, object]) -> str:
    """Render dossiers without requiring callers to keep a service instance."""

    service = AgenticTranslationService(
        word_service=_NullWordTranslator(),
        raw_dictionary_entries=[],
        dictionary_entries=[],
    )
    return service.render_markdown(dossier)


class _NullWordTranslator:
    """Provide a harmless renderer-only stand-in for Markdown formatting."""

    repository: Any

    def translate_word(self, word: str, **kwargs: object) -> dict[str, object]:
        """Reject accidental translation attempts from renderer-only helpers."""

        raise RuntimeError("Renderer-only translator cannot translate words.")

    def close(self) -> None:
        """No-op close for renderer-only helpers."""


def _resolve_analysis_source(raw: str) -> list[str]:
    """Expand the public analysis-source selector into concrete sources."""

    if raw == "both":
        return ["solo", "debate"]
    if raw in AgenticTranslationService.VALID_SOURCES:
        return [raw]
    raise ValueError(f"Unsupported analysis source: {raw}")


def _resolve_sources(sources: Iterable[str] | None) -> list[str]:
    """Normalize optional source iterables for service construction."""

    selected = list(sources or AgenticTranslationService.VALID_SOURCES)
    for source in selected:
        if source not in AgenticTranslationService.VALID_SOURCES:
            raise ValueError(f"Unsupported analysis source: {source}")
    return selected


def _load_raw_dictionary_entries(path: Path) -> list[Mapping[str, object]]:
    """Load raw dictionary rows so non-canon metadata remains visible."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, Mapping)]
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, Mapping)]
    return []


def _dictionary_definitions(entry: Mapping[str, object]) -> list[str]:
    """Extract dictionary definitions from both raw and validated row shapes."""

    definitions: list[str] = []
    direct = entry.get("definition") or entry.get("enhanced_definition")
    if isinstance(direct, str) and direct.strip():
        definitions.append(direct.strip())
    for sense in _as_list(entry.get("senses")):
        row = _as_mapping(sense)
        definition = row.get("definition")
        if isinstance(definition, str) and definition.strip():
            definitions.append(definition.strip())
    return _dedupe_strings(definitions)


def _canon_status(entry: Mapping[str, object]) -> str:
    """Map dictionary metadata into a stable public canon label."""

    if entry.get("canon_word") is False:
        return "non_canon"
    if entry.get("canon_word") is True:
        return "canon"
    return "canon_unknown"


def _evidence_class(provenance: str) -> str:
    """Normalize internal provenance labels into dossier evidence classes."""

    lowered = provenance.lower()
    if "dictionary" in lowered:
        return "dictionary-backed"
    if "residual" in lowered:
        return "residual"
    if "cluster" in lowered:
        return "derived"
    return lowered or "derived"


def _cluster_summary(cluster: Any) -> str:
    """Choose a compact cluster summary from glossator and raw-definition text."""

    if cluster.semantic_core:
        return "semantic core: " + ", ".join(cluster.semantic_core[:4])
    if cluster.glossator_def:
        return str(cluster.glossator_def).strip().replace("\n", " ")[:180]
    if cluster.raw_definitions:
        return "; ".join(
            str(raw_def.definition)
            for raw_def in cluster.raw_definitions[:3]
            if raw_def.definition
        )
    return f"accepted cluster {cluster.cluster_id}"


def _source_has_noncanon_only_packets(
    source: str,
    packets: Sequence[Mapping[str, object]],
) -> bool:
    """Detect source support that is outside canonical dictionary coverage."""

    return any(
        packet.get("analysis_source") == source
        and packet.get("packet_type") != "baseline_summary"
        and packet.get("canon_status") in {"non_canon", "non_canon_or_unmatched"}
        for packet in packets
    )


def _candidate_packet_lookup(
    packets: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    """Index candidate meaning packets by source and morph."""

    lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    for packet in packets:
        if packet.get("packet_type") != "candidate_meaning":
            continue
        source = str(packet.get("analysis_source") or "")
        morph = str(packet.get("morph") or packet.get("label") or "").upper()
        if source and morph:
            lookup.setdefault((source, morph), packet)
    return lookup


def _definition_alternatives(
    morph: str,
    source: str,
    trace: Mapping[str, object],
    packets: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Collect bounded alternate definitions for one considered morph."""

    alternatives: list[dict[str, object]] = []
    for alt in _as_list(trace.get("runner_ups"))[:4]:
        row = _as_mapping(alt)
        definition = row.get("definition")
        if definition:
            alternatives.append(
                {
                    "definition": definition,
                    "source": row.get("source"),
                    "quality": row.get("quality"),
                    "cluster_id": row.get("cluster_id"),
                }
            )

    for packet in packets:
        if packet.get("analysis_source") != source:
            continue
        if str(packet.get("morph") or packet.get("label") or "").upper() != morph:
            continue
        if packet.get("packet_type") not in {"accepted_cluster", "residual_semantics"}:
            continue
        definition = packet.get("definition") or packet.get("summary")
        if definition:
            alternatives.append(
                {
                    "definition": definition,
                    "source": packet.get("packet_type"),
                    "quality": packet.get("evidence_class"),
                    "cluster_id": packet.get("cluster_id"),
                }
            )

    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for alt in alternatives:
        definition = str(alt.get("definition") or "").strip()
        if not definition or definition.lower() in seen:
            continue
        seen.add(definition.lower())
        deduped.append(alt)
    return deduped[:4]


def _summarize_research_trace(trace: Sequence[object]) -> dict[str, object]:
    """Summarize read-only tool usage without emitting the full trace."""

    phase_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    for item in trace:
        row = _as_mapping(item)
        phase = str(row.get("phase") or "unknown")
        tool = str(row.get("tool") or "unknown")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    return {
        "event_count": len(trace),
        "phase_counts": phase_counts,
        "tool_counts": tool_counts,
    }


def _as_mapping(value: object) -> Mapping[str, object]:
    """Coerce unknown JSON-ish values into mapping form."""

    if isinstance(value, Mapping):
        return value
    return {}


def _as_list(value: object) -> list[Any]:
    """Coerce unknown JSON-ish values into list form."""

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _first_string(*values: object) -> str | None:
    """Return the first non-empty string from several candidate fields."""

    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _compact_text(value: object, *, limit: int = 180) -> str:
    """Return one-line, terminal-safe text for dossier Markdown."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("{") and '"DEFINITION"' in text:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, Mapping):
            definition = parsed.get("DEFINITION") or parsed.get("definition")
            if isinstance(definition, str) and definition.strip():
                text = definition.strip()
    text = " ".join(text.replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _safe_float(value: object) -> float:
    """Convert loose numeric payloads into floats for ranking comparisons."""

    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    """Preserve order while removing duplicate strings."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        output.append(normalized)
    return output


def _to_plain(value: object) -> object:
    """Convert dataclasses and nested containers into JSON-safe values."""

    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
