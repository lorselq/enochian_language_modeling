"""Read-only root sense grouping diagnostics for accepted root glosses.

Why this module exists:
the translation pipeline already uses accepted root glosses as evidence, but
short roots can carry several positional or semantic readings that are easy to
flatten into one vague `semantic_core`. This module builds an inspectable
diagnostic read model that groups surviving root-gloss rows without changing
translation ranking.

Architecture responsibility:
`RootSenseGroupService` reads `root_glosses` and `root_attachment_profile`,
constructs evidence packets, groups them deterministically, and returns stable
JSON/text report payloads for the `enlm report root-groups` command.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import math
import re
from pathlib import Path

import numpy as np

from enochian_lm.common.sqlite_bootstrap import sqlite3
from enochian_lm.root_extraction.utils.embeddings import (
    get_sentence_transformer_if_available,
)

GROUPING_VERSION = "root-sense-groups-v1"

GENERIC_TERMS = {
    "being",
    "existence",
    "marker",
    "morpheme",
    "relation",
    "state",
}

DEFAULT_PARAMETERS: dict[str, object] = {
    "merge_similarity_threshold": 0.78,
    "strong_merge_similarity_threshold": 0.86,
    "attachment_axis_divergence_threshold": 0.35,
    "observed_attachment_ratio_divergence_threshold": 0.45,
    "negative_contrast_overlap_threshold": 1,
    "exception_density_needs_review_threshold": 0.25,
    "low_confidence_threshold": 0.65,
    "stable_min_packets": 2,
    "stable_min_avg_confidence": 0.78,
    "stable_max_attachment_divergence": 0.25,
    "max_groups_default": 12,
    "semantic_text_weights": {
        "semantic_core": 3.0,
        "nested_effect": 3.0,
        "nested_sense": 2.0,
        "definition": 2.0,
        "decoding_guide": 1.0,
        "examples": 1.0,
    },
}

DEFAULT_ALIGNMENT_PARAMETERS: dict[str, float] = {
    "source_cluster_id_match_weight": 0.45,
    "nested_evidence_word_match_weight": 0.25,
    "attachment_role_match_weight": 0.15,
    "semantic_overlap_weight": 0.10,
    "group_rank_weight": 0.05,
    "needs_review_penalty": 0.08,
    "split_recommended_penalty": 0.12,
    "provisional_penalty": 0.02,
    "max_alignment_score": 1.0,
}


@dataclass(slots=True)
class RootGroupOptions:
    """Carry root-group report options from CLI or tests into the service.

    The service is intentionally read-only and deterministic. Keeping options in
    one object prevents command defaults, report parameters, and tests from
    drifting apart as the diagnostic layer grows.
    """

    variant_paths: Mapping[str, Path | str | None]
    variants: Sequence[str] = ("both",)
    detail: str = "full"
    max_groups: int = 12
    pretty: bool = False
    parameters: Mapping[str, object] = field(default_factory=dict)


class RootGroupError(RuntimeError):
    """Raised when a root-group report cannot be built from available inputs."""


class MissingRootGlossesViewError(RootGroupError):
    """Raised when the selected insights DB lacks the required root_glosses view."""


class RootSenseGroupService:
    """Build root sense group reports from accepted SQLite root-gloss rows.

    Why:
    root-gloss summaries lose important information carried by examples,
    exceptions, attachment behavior, and nested evidence effects.

    How:
    the service converts each accepted row into a packet, compares packets with
    a deterministic vector backend, applies conservative conflict rules, then
    emits JSON-compatible dictionaries.

    Responsibility:
    expose diagnostic grouping only. The service must not mutate databases or
    decide translation output.
    """

    def __init__(self, options: RootGroupOptions) -> None:
        self.options = options
        self.parameters = {**DEFAULT_PARAMETERS, **dict(options.parameters)}
        self.embedding_backend = "uninitialized"

    def build_report(self, root: str) -> dict[str, object]:
        root_lookup = _normalize_lookup_root(root)
        root_public = _normalize_public_root(root)
        if not root_lookup:
            raise ValueError("Root must be a non-empty string.")

        variants = _resolve_variants(self.options.variants)
        packets: list[dict[str, object]] = []
        diagnostics_warnings: list[str] = []
        for variant in variants:
            path = self.options.variant_paths.get(variant)
            if path is None:
                raise FileNotFoundError(f"Missing insights database path for {variant}.")
            db_path = Path(path)
            if not db_path.exists():
                raise FileNotFoundError(f"Missing insights database file: {db_path}")
            packets.extend(
                self._packets_for_variant(
                    variant=variant,
                    db_path=db_path,
                    root_lookup=root_lookup,
                    root_public=root_public,
                )
            )

        if not packets:
            return {
                "root": root_public,
                "variants_queried": list(variants),
                "generated_at": _utc_now(),
                "grouping_version": GROUPING_VERSION,
                "parameters": self._serialized_parameters(),
                "groups": [],
                "ungrouped_packets": [],
                "diagnostics": {
                    "packet_count": 0,
                    "group_count": 0,
                    "ungrouped_count": 0,
                    "pre_cap_group_count": 0,
                    "rejected_merges": [],
                    "embedding_backend": "none",
                    "warnings": diagnostics_warnings,
                    "empty_reason": "no_accepted_root_glosses",
                },
            }

        similarity, backend = self._similarity_matrix(packets)
        self.embedding_backend = backend
        groups, rejected_merges = self._build_groups(
            root_public=root_public,
            packets=packets,
            similarity=similarity,
        )
        groups.sort(
            key=lambda group: (
                -float(group["ranking"]["rank_score"]),
                str(group["label"]),
                min(group["source_cluster_ids"] or [10**9]),
            )
        )
        for index, group in enumerate(groups, start=1):
            label_slug = _slug(str(group["label"] or "group"))
            group["group_id"] = f"{root_lookup}:{label_slug}:{index}"
            for member in group["members"]:
                member["assignment_reasons"] = list(member["assignment_reasons"])

        pre_cap_group_count = len(groups)
        max_groups = max(1, int(self.options.max_groups or 12))
        capped_groups = groups[:max_groups]
        detail = str(self.options.detail or "full")
        if detail == "compact":
            capped_groups = [_compact_group_for_report(group) for group in capped_groups]
        elif detail != "full":
            raise ValueError("--detail must be 'full' or 'compact'.")

        return {
            "root": root_public,
            "variants_queried": list(variants),
            "generated_at": _utc_now(),
            "grouping_version": GROUPING_VERSION,
            "parameters": self._serialized_parameters(),
            "groups": capped_groups,
            "ungrouped_packets": [],
            "diagnostics": {
                "packet_count": len(packets),
                "group_count": len(capped_groups),
                "ungrouped_count": 0,
                "pre_cap_group_count": pre_cap_group_count,
                "rejected_merges": rejected_merges,
                "embedding_backend": backend,
                "warnings": diagnostics_warnings,
                "empty_reason": None,
            },
        }

    def _packets_for_variant(
        self,
        *,
        variant: str,
        db_path: Path,
        root_lookup: str,
        root_public: str,
    ) -> list[dict[str, object]]:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            if not _table_or_view_exists(conn, "root_glosses"):
                raise MissingRootGlossesViewError(
                    f"{db_path} does not expose root_glosses."
                )
            attachment_rows = _fetch_attachment_rows(conn, root_lookup)
            rows = conn.execute(
                """
                SELECT *
                FROM root_glosses
                WHERE LOWER(TRIM(root)) = ?
                  AND LOWER(COALESCE(TRIM(evaluation), 'accepted')) = 'accepted'
                ORDER BY source_cluster_id ASC
                """,
                (root_lookup,),
            ).fetchall()
            packets: list[dict[str, object]] = []
            for ordinal, row in enumerate(rows, start=1):
                packets.append(
                    self._packet_from_row(
                        row=row,
                        variant=variant,
                        root_public=root_public,
                        ordinal=ordinal,
                        attachment_rows=attachment_rows,
                    )
                )
            return packets
        finally:
            conn.close()

    def _packet_from_row(
        self,
        *,
        row: sqlite3.Row,
        variant: str,
        root_public: str,
        ordinal: int,
        attachment_rows: Mapping[int | None, Mapping[str, object]],
    ) -> dict[str, object]:
        warnings: list[str] = []
        source_cluster_id = _safe_int(_row_get(row, "source_cluster_id"))
        if source_cluster_id is None:
            packet_id = f"{variant}:{root_public}:row-{ordinal}"
            warnings.append("missing_source_cluster_id")
        else:
            packet_id = f"{variant}:{root_public}:{source_cluster_id}"

        semantic_core, semantic_core_raw = _parse_json_list(
            _row_get(row, "semantic_core"),
            warnings=warnings,
            warning_name="malformed_semantic_core",
        )
        negative_contrast, negative_contrast_raw = _parse_json_list(
            _row_get(row, "negative_contrast"),
            warnings=warnings,
            warning_name="malformed_negative_contrast",
        )
        examples, examples_raw = _parse_json_list(
            _row_get(row, "examples_json"),
            warnings=warnings,
            warning_name="malformed_examples_json",
        )
        confidence_drivers, confidence_drivers_raw = _parse_json_list(
            _row_get(row, "confidence_drivers"),
            warnings=warnings,
            warning_name="malformed_confidence_drivers",
        )
        confidence_risks, confidence_risks_raw = _parse_json_list(
            _row_get(row, "confidence_risks"),
            warnings=warnings,
            warning_name="malformed_confidence_risks",
        )
        attachment_exceptions, attachment_exceptions_raw = _parse_json_list(
            _row_get(row, "attachment_exceptions"),
            warnings=warnings,
            warning_name="malformed_attachment_exceptions",
        )
        contribution, contribution_raw = _parse_json_object(
            _row_get(row, "contribution_json"),
            warnings=warnings,
            warning_name="malformed_contribution_json",
        )
        raw_glossator_json = _row_get(row, "raw_glossator_json")
        raw_payload, raw_payload_warning = _parse_raw_payload(raw_glossator_json)
        if raw_payload_warning:
            warnings.append(raw_payload_warning)
        nested_evidence = _nested_evidence_from_payload(raw_payload, warnings)

        attachment = _attachment_summary(
            row=row,
            attachment_row=attachment_rows.get(source_cluster_id)
            or attachment_rows.get(None),
        )
        definition = _safe_str(_row_get(row, "definition"))
        decoding_guide = _safe_str(_row_get(row, "decoding_guide"))
        reason = _safe_str(_row_get(row, "reason"))
        surface_examples = _surface_examples(examples, nested_evidence)
        semantic_text_components = _semantic_text_components(
            semantic_core=semantic_core,
            nested_evidence=nested_evidence,
            definition=definition,
            decoding_guide=decoding_guide,
            examples=examples,
        )

        packet = {
            "packet_id": packet_id,
            "root": root_public,
            "variant": variant,
            "evaluation": _safe_str(_row_get(row, "evaluation")) or "accepted",
            "source_cluster_id": source_cluster_id,
            "definition": definition,
            "semantic_core": semantic_core,
            "decoding_guide": decoding_guide,
            "negative_contrast": negative_contrast,
            "examples": examples,
            "nested_evidence": nested_evidence,
            "confidence_score": _safe_float(_row_get(row, "confidence_score")),
            "examples_in_cluster": _safe_int(_row_get(row, "examples_in_cluster")),
            "attachment": attachment,
            "raw_glossator_json": raw_payload,
            "reason": reason,
            "contribution": contribution,
            "confidence_drivers": confidence_drivers,
            "confidence_risks": confidence_risks,
            "attachment_exceptions": attachment_exceptions,
            "pos_bias": {
                "nounness": _safe_float(_row_get(row, "pos_bias_nounness")),
                "modifier": _safe_float(_row_get(row, "pos_bias_modifier")),
                "verbness": _safe_float(_row_get(row, "pos_bias_verbness")),
            },
            "source_run_id": _safe_str(_row_get(row, "run_id")),
            "semantic_text": _semantic_text_for_tfidf(semantic_text_components),
            "semantic_text_components": semantic_text_components,
            "surface_examples": surface_examples,
            "exception_terms": _unique_terms(
                [*attachment_exceptions, *confidence_risks]
            ),
            "packet_warnings": sorted(set(warnings)),
        }
        if semantic_core_raw is not None:
            packet["semantic_core_raw"] = semantic_core_raw
        if negative_contrast_raw is not None:
            packet["negative_contrast_raw"] = negative_contrast_raw
        if examples_raw is not None:
            packet["examples_raw"] = examples_raw
        if confidence_drivers_raw is not None:
            packet["confidence_drivers_raw"] = confidence_drivers_raw
        if confidence_risks_raw is not None:
            packet["confidence_risks_raw"] = confidence_risks_raw
        if attachment_exceptions_raw is not None:
            packet["attachment_exceptions_raw"] = attachment_exceptions_raw
        if contribution_raw is not None:
            packet["contribution_raw"] = contribution_raw
        if raw_payload_warning == "malformed_raw_glossator_json":
            packet["raw_glossator_json_raw"] = _safe_str(raw_glossator_json)
        return packet

    def _similarity_matrix(
        self,
        packets: Sequence[dict[str, object]],
    ) -> tuple[np.ndarray, str]:
        if len(packets) == 1:
            return np.ones((1, 1), dtype=float), "single-packet"

        embedder = get_sentence_transformer_if_available(
            "paraphrase-MiniLM-L6-v2",
            local_files_only=True,
        )
        if embedder is not None:
            vectors = []
            weights = self.parameters["semantic_text_weights"]
            for packet in packets:
                components = packet.get("semantic_text_components")
                vectors.append(_weighted_sentence_vector(embedder, components, weights))
            matrix = np.vstack(vectors)
            return _cosine_similarity_matrix(matrix), "sentence-transformer"

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception:
            return np.eye(len(packets), dtype=float), "single-packet"

        texts = [str(packet.get("semantic_text") or "") for packet in packets]
        if not any(text.strip() for text in texts):
            return np.eye(len(packets), dtype=float), "single-packet"
        matrix = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
        ).fit_transform(texts)
        return np.asarray(cosine_similarity(matrix), dtype=float), "tfidf-word-1-2"

    def _build_groups(
        self,
        *,
        root_public: str,
        packets: list[dict[str, object]],
        similarity: np.ndarray,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        threshold = float(self.parameters["merge_similarity_threshold"])
        strong_threshold = float(self.parameters["strong_merge_similarity_threshold"])
        active_edges: set[tuple[int, int]] = set()
        rejected_merges: list[dict[str, object]] = []
        for left in range(len(packets)):
            for right in range(left + 1, len(packets)):
                score = float(similarity[left, right])
                if score < threshold:
                    continue
                conflict_flags = _pair_conflict_flags(
                    packets[left],
                    packets[right],
                    similarity=score,
                    parameters=self.parameters,
                )
                strong_conflicts = [
                    flag
                    for flag in conflict_flags
                    if flag
                    in {
                        "attachment_divergence",
                        "negative_contrast_conflict",
                        "effect_domain_divergence",
                    }
                ]
                if strong_conflicts and score < strong_threshold:
                    rejected_merges.append(
                        {
                            "left_packet_id": packets[left]["packet_id"],
                            "right_packet_id": packets[right]["packet_id"],
                            "similarity": round(score, 4),
                            "flags": strong_conflicts,
                            "reason": "strong_conflict_below_strong_merge_threshold",
                        }
                    )
                    continue
                active_edges.add((left, right))

        components = _connected_components(len(packets), active_edges)
        groups = [
            self._group_from_component(
                root_public=root_public,
                indexes=component,
                packets=packets,
                similarity=similarity,
            )
            for component in components
        ]
        return groups, rejected_merges

    def _group_from_component(
        self,
        *,
        root_public: str,
        indexes: list[int],
        packets: list[dict[str, object]],
        similarity: np.ndarray,
    ) -> dict[str, object]:
        component_packets = [packets[index] for index in indexes]
        pair_flags: set[str] = set()
        for left_pos, left in enumerate(indexes):
            for right in indexes[left_pos + 1 :]:
                pair_flags.update(
                    _pair_conflict_flags(
                        packets[left],
                        packets[right],
                        similarity=float(similarity[left, right]),
                        parameters=self.parameters,
                    )
                )
        ranking = _ranking_for_packets(
            component_packets,
            similarity=similarity,
            indexes=indexes,
            pair_flags=pair_flags,
            parameters=self.parameters,
        )
        semantic_terms = _semantic_terms(component_packets)
        surface_examples = _surface_examples_for_group(component_packets)
        label = _label_for_group(semantic_terms, surface_examples)
        status = _status_for_group(component_packets, ranking, pair_flags, self.parameters)
        warnings = _warnings_for_group(component_packets, pair_flags)
        source_ids = [
            packet.get("source_cluster_id")
            for packet in component_packets
            if packet.get("source_cluster_id") is not None
        ]
        variants = sorted({str(packet["variant"]) for packet in component_packets})
        return {
            "group_id": "",
            "root": root_public,
            "label": label,
            "status": status,
            "semantic_terms": semantic_terms,
            "surface_examples": surface_examples,
            "source_cluster_ids": source_ids,
            "variants": variants,
            "members": [
                {
                    "packet_id": packet["packet_id"],
                    "source_cluster_id": packet.get("source_cluster_id"),
                    "variant": packet["variant"],
                    "assignment_score": 1.0,
                    "assignment_reasons": [
                        "single-packet group"
                        if len(component_packets) == 1
                        else "connected-component assignment"
                    ],
                    "split_flags": sorted(pair_flags | set(packet["packet_warnings"])),
                }
                for packet in component_packets
            ],
            "attachment_profile": _aggregate_attachment(component_packets),
            "ranking": ranking,
            "warnings": warnings,
            "evidence_packets": component_packets,
        }

    def _serialized_parameters(self) -> dict[str, object]:
        payload = dict(self.parameters)
        payload["grouping_version"] = GROUPING_VERSION
        payload["max_groups"] = self.options.max_groups
        payload["detail"] = self.options.detail
        return payload


def render_report_json(report: Mapping[str, object], *, pretty: bool = False) -> str:
    """Serialize a root-group report as stable JSON text."""

    return json.dumps(
        report,
        indent=2 if pretty else None,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"


def render_report_text(report: Mapping[str, object]) -> str:
    """Render a compact human-facing root-group report.

    Text output is intentionally derived from the JSON report shape so Phase 1
    and Phase 2 cannot drift apart.
    """

    root = str(report.get("root") or "")
    variants = ", ".join(str(v) for v in report.get("variants_queried", []))
    groups = report.get("groups")
    group_list = groups if isinstance(groups, list) else []
    lines = [f"Root: {root}", f"Variants: {variants or 'none'}"]
    if not group_list:
        diagnostics = report.get("diagnostics")
        empty_reason = (
            diagnostics.get("empty_reason")
            if isinstance(diagnostics, Mapping)
            else None
        )
        lines.append(f"No accepted groups found ({empty_reason or 'no groups'}).")
        return "\n".join(lines) + "\n"

    for index, group in enumerate(group_list, start=1):
        if not isinstance(group, Mapping):
            continue
        ranking = group.get("ranking")
        rank_score = (
            ranking.get("rank_score")
            if isinstance(ranking, Mapping)
            else None
        )
        lines.append("")
        lines.append(
            f"{index}. {group.get('label') or 'unknown'} "
            f"[{group.get('status') or 'unknown'}] "
            f"score={_format_number(rank_score)}"
        )
        source_ids = ", ".join(str(v) for v in group.get("source_cluster_ids", []))
        lines.append(f"   source_cluster_ids: {source_ids or 'none'}")
        terms = ", ".join(str(v) for v in group.get("semantic_terms", []))
        lines.append(f"   semantic_terms: {terms or 'none'}")
        examples = ", ".join(str(v) for v in group.get("surface_examples", [])[:5])
        lines.append(f"   surface_examples: {examples or 'none'}")
        attachment = group.get("attachment_profile")
        if isinstance(attachment, Mapping):
            lines.append(
                "   attachment: "
                f"prefix={_format_number(attachment.get('prefix_likelihood'))}, "
                f"suffix={_format_number(attachment.get('suffix_likelihood'))}, "
                f"free={_format_number(attachment.get('free_likelihood'))}, "
                f"profile={attachment.get('estimated_profile') or 'unknown'}"
            )
        warnings = [str(w) for w in group.get("warnings", []) if str(w).strip()]
        if warnings:
            lines.append(f"   warnings: {'; '.join(warnings)}")
        effects = _top_effects_from_group(group)
        if effects:
            lines.append(f"   top_effects: {'; '.join(effects[:5])}")
    diagnostics = report.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        rejected = diagnostics.get("rejected_merges")
        if isinstance(rejected, list) and rejected:
            lines.append("")
            lines.append(f"Rejected merges: {len(rejected)}")
    return "\n".join(lines) + "\n"


def compact_group_summary(group: Mapping[str, object]) -> dict[str, object]:
    """Return the Phase 3-safe compact group summary shape."""

    ranking = group.get("ranking")
    rank_score = ranking.get("rank_score") if isinstance(ranking, Mapping) else 0.0
    return {
        "root": group.get("root"),
        "group_id": group.get("group_id"),
        "label": group.get("label"),
        "status": group.get("status"),
        "rank_score": rank_score,
        "semantic_terms": list(group.get("semantic_terms") or []),
        "surface_examples": list(group.get("surface_examples") or [])[:5],
        "source_cluster_ids": list(group.get("source_cluster_ids") or []),
        "warnings": list(group.get("warnings") or []),
    }


def compact_report_groups(
    report: Mapping[str, object],
    *,
    max_groups: int = 3,
) -> list[dict[str, object]]:
    """Return bounded compact summaries from a full root group report.

    Why:
    translation diagnostics and LLM render payloads need enough sense inventory
    context to explain a reading, but they must not carry full evidence packets
    by default.

    How:
    this helper projects the report's ranked groups through
    ``compact_group_summary`` and enforces the Phase 3 payload limit.

    Responsibility:
    keep report output, translation diagnostics, and LLM context on the same
    compact public shape.
    """

    groups = report.get("groups")
    group_list = groups if isinstance(groups, list) else []
    return [
        compact_group_summary(group)
        for group in group_list[: max(0, int(max_groups))]
        if isinstance(group, Mapping)
    ]


def align_root_groups_for_morph(
    *,
    morph: str,
    span_role: str,
    report: Mapping[str, object] | None,
    source_cluster_ids: Sequence[int] | None = None,
    evidence_word: str | None = None,
    semantic_text: str | None = None,
    max_alternates: int = 2,
    parameters: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Align one morph occurrence to the most relevant root sense group.

    Why:
    a short root such as D or I can survive as many plausible groups. Translation
    needs a local alignment decision for a particular decomposition occurrence,
    not a single canonical root meaning.

    How:
    the scorer combines row provenance, nested evidence word matches,
    attachment role compatibility, lexical overlap, and group rank. Status
    penalties keep exception-heavy or split-recommended groups visible without
    letting them dominate by accident.

    Responsibility:
    produce advisory diagnostics and bounded ranking features only. It must not
    mutate reports or erase alternate senses.
    """

    if report is None:
        return _empty_alignment(morph=morph, span_role=span_role)

    params = {**DEFAULT_ALIGNMENT_PARAMETERS, **dict(parameters or {})}
    groups_raw = report.get("groups")
    groups = [group for group in groups_raw if isinstance(group, Mapping)] if isinstance(groups_raw, list) else []
    if not groups:
        return _empty_alignment(morph=morph, span_role=span_role)

    source_ids = {
        parsed
        for value in (source_cluster_ids or [])
        for parsed in [_safe_int(value)]
        if parsed is not None
    }
    scored = [
        _score_group_alignment(
            morph=morph,
            span_role=span_role,
            group=group,
            source_cluster_ids=source_ids,
            evidence_word=evidence_word,
            semantic_text=semantic_text,
            parameters=params,
        )
        for group in groups
    ]
    scored.sort(
        key=lambda item: (
            -float(item["alignment_score"]),
            str(item["group_summary"].get("label") or ""),
            str(item["group_id"] or ""),
        )
    )
    primary = scored[0]
    alternates = scored[1 : 1 + max(0, int(max_alternates))]
    return {
        "morph": str(morph or "").upper(),
        "span_role": _normalize_role(span_role),
        "primary_group_id": primary["group_id"],
        "alignment_score": primary["alignment_score"],
        "reasons": primary["reasons"],
        "warnings": primary["warnings"],
        "source_cluster_ids": sorted(source_ids),
        "primary_group": primary["group_summary"],
        "alternates": [
            {
                "group_id": alternate["group_id"],
                "alignment_score": alternate["alignment_score"],
                "reasons": alternate["reasons"],
                "warnings": alternate["warnings"],
                "group": alternate["group_summary"],
            }
            for alternate in alternates
        ],
    }


def _table_or_view_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE name = ?
          AND type IN ('table', 'view')
        LIMIT 1
        """,
        (name,),
    ).fetchone()
    return row is not None


def _fetch_attachment_rows(
    conn: sqlite3.Connection,
    root_lookup: str,
) -> dict[int | None, Mapping[str, object]]:
    if not _table_or_view_exists(conn, "root_attachment_profile"):
        return {}
    rows = conn.execute(
        """
        SELECT *
        FROM root_attachment_profile
        WHERE LOWER(TRIM(root)) = ?
        """,
        (root_lookup,),
    ).fetchall()
    result: dict[int | None, Mapping[str, object]] = {}
    aggregate_counts = {
        "observed_prefix_count": 0,
        "observed_suffix_count": 0,
        "observed_infix_count": 0,
        "observed_free_count": 0,
    }
    profiles: Counter[str] = Counter()
    for row in rows:
        source_id = _safe_int(_row_get(row, "source_cluster_id"))
        data = {key: row[key] for key in row.keys()}
        result[source_id] = data
        for key in aggregate_counts:
            aggregate_counts[key] += _safe_int(_row_get(row, key)) or 0
        profile = _safe_str(_row_get(row, "estimated_profile"))
        if profile:
            profiles[profile] += 1
    if rows:
        result[None] = {
            **aggregate_counts,
            "estimated_profile": profiles.most_common(1)[0][0]
            if profiles
            else "unknown",
            "source_cluster_id": None,
        }
    return result


def _attachment_summary(
    *,
    row: sqlite3.Row,
    attachment_row: Mapping[str, object] | None,
) -> dict[str, object]:
    attachment_row = attachment_row or {}
    return {
        "prefix_likelihood": _safe_float(_row_get(row, "attachment_prefix_likelihood")),
        "suffix_likelihood": _safe_float(_row_get(row, "attachment_suffix_likelihood")),
        "infix_likelihood": None,
        "free_likelihood": _safe_float(_row_get(row, "attachment_free_likelihood")),
        "productivity": _safe_float(_row_get(row, "attachment_productivity")),
        "estimated_profile": _safe_str(attachment_row.get("estimated_profile"))
        or "unknown",
        "observed_prefix_count": _safe_int(
            attachment_row.get("observed_prefix_count")
        )
        or 0,
        "observed_suffix_count": _safe_int(
            attachment_row.get("observed_suffix_count")
        )
        or 0,
        "observed_infix_count": _safe_int(
            attachment_row.get("observed_infix_count")
        )
        or 0,
        "observed_free_count": _safe_int(attachment_row.get("observed_free_count"))
        or 0,
    }


def _parse_json_list(
    value: object,
    *,
    warnings: list[str],
    warning_name: str,
) -> tuple[list[str], str | None]:
    if value is None:
        return [], None
    if isinstance(value, list):
        return [_stringify(item) for item in value if _stringify(item)], None
    if not isinstance(value, str):
        text = _stringify(value)
        return ([text] if text else []), None
    stripped = value.strip()
    if not stripped:
        return [], None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        warnings.append(warning_name)
        return [], stripped
    if isinstance(parsed, list):
        return [_stringify(item) for item in parsed if _stringify(item)], None
    if isinstance(parsed, str):
        return [parsed] if parsed.strip() else [], None
    warnings.append(warning_name)
    return [], stripped


def _parse_json_object(
    value: object,
    *,
    warnings: list[str],
    warning_name: str,
) -> tuple[dict[str, object], str | None]:
    if value is None:
        return {}, None
    if isinstance(value, dict):
        return dict(value), None
    if not isinstance(value, str):
        return {}, None
    stripped = value.strip()
    if not stripped:
        return {}, None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        warnings.append(warning_name)
        return {}, stripped
    if isinstance(parsed, dict):
        return dict(parsed), None
    warnings.append(warning_name)
    return {}, stripped


def _parse_raw_payload(value: object) -> tuple[dict[str, object], str | None]:
    if isinstance(value, dict):
        return dict(value), None
    if not isinstance(value, str) or not value.strip():
        return {}, "missing_raw_glossator_json"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}, "malformed_raw_glossator_json"
    return (dict(parsed), None) if isinstance(parsed, dict) else ({}, "malformed_raw_glossator_json")


def _nested_evidence_from_payload(
    payload: Mapping[str, object],
    warnings: list[str],
) -> list[dict[str, object]]:
    raw = payload.get("EVIDENCE", payload.get("evidence"))
    if raw is None:
        warnings.append("missing_nested_evidence")
        return []
    if not isinstance(raw, list):
        warnings.append("missing_nested_evidence")
        return []
    evidence: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        note = item.get("note")
        note_map = note if isinstance(note, Mapping) else {}
        role = _safe_str(note_map.get("role")) or "unknown"
        evidence.append(
            {
                "word": (_safe_str(item.get("word")) or "").upper() or None,
                "sense": _safe_str(item.get("sense")),
                "role": role.lower(),
                "effect": _safe_str(note_map.get("effect")),
                "confidence": _safe_float(note_map.get("confidence")),
                "sense_alignment": _safe_float(note_map.get("sense_alignment")),
                "loc": _safe_str(item.get("loc")),
                "note": dict(note_map),
            }
        )
    if not evidence:
        warnings.append("missing_nested_evidence")
    return evidence


def _semantic_text_components(
    *,
    semantic_core: Sequence[str],
    nested_evidence: Sequence[Mapping[str, object]],
    definition: str | None,
    decoding_guide: str | None,
    examples: Sequence[str],
) -> dict[str, list[str]]:
    return {
        "semantic_core": [str(item) for item in semantic_core if str(item).strip()],
        "nested_effect": [
            str(item.get("effect"))
            for item in nested_evidence
            if str(item.get("effect") or "").strip()
        ],
        "nested_sense": [
            str(item.get("sense"))
            for item in nested_evidence
            if str(item.get("sense") or "").strip()
        ],
        "definition": [definition] if definition else [],
        "decoding_guide": [decoding_guide] if decoding_guide else [],
        "examples": [str(item) for item in examples if str(item).strip()],
    }


def _semantic_text_for_tfidf(components: Mapping[str, Sequence[str]]) -> str:
    weights = DEFAULT_PARAMETERS["semantic_text_weights"]
    pieces: list[str] = []
    for key, values in components.items():
        repeat = int(math.ceil(float(weights.get(key, 1.0)))) if isinstance(weights, Mapping) else 1
        for value in values:
            normalized = _normalize_text(value)
            if normalized:
                pieces.extend([normalized] * max(1, repeat))
    return " ".join(pieces)


def _weighted_sentence_vector(
    embedder: object,
    components: object,
    weights: object,
) -> np.ndarray:
    if not isinstance(components, Mapping) or not isinstance(weights, Mapping):
        text = _semantic_text_for_tfidf({})
        return np.asarray(embedder.encode([text or " "])[0], dtype=float)
    vectors: list[np.ndarray] = []
    vector_weights: list[float] = []
    for key, values in components.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        texts = [str(value).strip() for value in values if str(value).strip()]
        if not texts:
            continue
        encoded = embedder.encode(texts)
        weight = float(weights.get(key, 1.0))
        for vector in encoded:
            vectors.append(np.asarray(vector, dtype=float))
            vector_weights.append(weight)
    if not vectors:
        return np.zeros(384, dtype=float)
    matrix = np.vstack(vectors)
    return np.average(matrix, axis=0, weights=np.asarray(vector_weights, dtype=float))


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def _connected_components(
    count: int,
    edges: Iterable[tuple[int, int]],
) -> list[list[int]]:
    graph: dict[int, set[int]] = {index: set() for index in range(count)}
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    seen: set[int] = set()
    components: list[list[int]] = []
    for index in range(count):
        if index in seen:
            continue
        stack = [index]
        component: list[int] = []
        seen.add(index)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(graph[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _pair_conflict_flags(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    similarity: float,
    parameters: Mapping[str, object],
) -> set[str]:
    del similarity
    flags: set[str] = set()
    attachment_divergence = _attachment_divergence(
        left.get("attachment"),
        right.get("attachment"),
    )
    if attachment_divergence >= float(parameters["attachment_axis_divergence_threshold"]):
        flags.add("attachment_divergence")
    if _negative_contrast_conflict(left, right):
        flags.add("negative_contrast_conflict")
    if _effect_domain_divergence(left, right):
        flags.add("effect_domain_divergence")
    if _roles_mixed(left, right):
        flags.add("mixed_role_evidence")
    return flags


def _attachment_divergence(left: object, right: object) -> float:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return 0.0
    left_axis, left_value = _dominant_attachment_axis(left)
    right_axis, right_value = _dominant_attachment_axis(right)
    if left_axis == "unknown" or right_axis == "unknown" or left_axis == right_axis:
        return 0.0
    return abs(left_value - right_value)


def _dominant_attachment_axis(attachment: Mapping[str, object]) -> tuple[str, float]:
    values = {
        "prefix": _safe_float(attachment.get("prefix_likelihood")),
        "suffix": _safe_float(attachment.get("suffix_likelihood")),
        "free": _safe_float(attachment.get("free_likelihood")),
    }
    usable = {key: value for key, value in values.items() if value is not None}
    if not usable:
        counts = {
            "prefix": float(_safe_int(attachment.get("observed_prefix_count")) or 0),
            "suffix": float(_safe_int(attachment.get("observed_suffix_count")) or 0),
            "free": float(_safe_int(attachment.get("observed_free_count")) or 0),
        }
        total = sum(counts.values())
        if total <= 0:
            return "unknown", 0.0
        usable = {key: value / total for key, value in counts.items()}
    axis, value = max(usable.items(), key=lambda item: item[1])
    return axis, float(value)


def _negative_contrast_conflict(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    return bool(
        _positive_terms(left) & _contrast_terms(right)
        or _positive_terms(right) & _contrast_terms(left)
    )


def _positive_terms(packet: Mapping[str, object]) -> set[str]:
    terms: set[str] = set()
    for item in packet.get("semantic_core") or []:
        terms.update(_term_tokens(str(item)))
    contribution = packet.get("contribution")
    if isinstance(contribution, Mapping):
        for key in contribution:
            terms.update(_term_tokens(str(key)))
    for item in packet.get("nested_evidence") or []:
        if isinstance(item, Mapping):
            terms.update(_term_tokens(str(item.get("effect") or "")))
    return terms - GENERIC_TERMS


def _contrast_terms(packet: Mapping[str, object]) -> set[str]:
    terms: set[str] = set()
    for item in packet.get("negative_contrast") or []:
        terms.update(_term_tokens(str(item)))
    return terms - GENERIC_TERMS


def _effect_domain_divergence(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    left_effects = [
        str(item.get("effect"))
        for item in left.get("nested_evidence") or []
        if isinstance(item, Mapping) and str(item.get("effect") or "").strip()
    ]
    right_effects = [
        str(item.get("effect"))
        for item in right.get("nested_evidence") or []
        if isinstance(item, Mapping) and str(item.get("effect") or "").strip()
    ]
    if not left_effects or not right_effects:
        return False
    if _positive_terms(left) & _positive_terms(right):
        return False
    similarities = [
        _jaccard(_term_tokens(left_effect), _term_tokens(right_effect))
        for left_effect in left_effects
        for right_effect in right_effects
    ]
    average = sum(similarities) / float(len(similarities)) if similarities else 0.0
    return average < 0.45


def _roles_mixed(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_roles = _roles(left)
    right_roles = _roles(right)
    if not left_roles or not right_roles:
        return False
    return bool({"prefix", "suffix"} <= (left_roles | right_roles))


def _roles(packet: Mapping[str, object]) -> set[str]:
    return {
        str(item.get("role") or "").lower()
        for item in packet.get("nested_evidence") or []
        if isinstance(item, Mapping) and str(item.get("role") or "").strip()
    }


def _ranking_for_packets(
    packets: Sequence[Mapping[str, object]],
    *,
    similarity: np.ndarray,
    indexes: Sequence[int],
    pair_flags: set[str],
    parameters: Mapping[str, object],
) -> dict[str, object]:
    confidences = [
        float(value)
        for packet in packets
        for value in [_safe_float(packet.get("confidence_score"))]
        if value is not None
    ]
    nested_conf = [
        float(value)
        for packet in packets
        for item in packet.get("nested_evidence") or []
        if isinstance(item, Mapping)
        for value in [_safe_float(item.get("confidence"))]
        if value is not None
    ]
    nested_align = [
        float(value)
        for packet in packets
        for item in packet.get("nested_evidence") or []
        if isinstance(item, Mapping)
        for value in [_safe_float(item.get("sense_alignment"))]
        if value is not None
    ]
    examples_total = sum(
        _safe_int(packet.get("examples_in_cluster")) or len(packet.get("examples") or [])
        for packet in packets
    )
    semantic_coherence = _semantic_coherence(similarity, indexes)
    attachment_divergence = _max_attachment_divergence(packets)
    attachment_consistency = max(0.0, min(1.0, 1.0 - attachment_divergence))
    exception_density = _density(packets, "attachment_exceptions")
    risk_density = _density(packets, "confidence_risks")
    review_penalty = max(
        exception_density,
        risk_density,
        attachment_divergence,
        1.0 if "effect_domain_divergence" in pair_flags else 0.0,
    )
    packet_count = len(packets)
    support_factor = min(1.0, math.log1p(packet_count) / math.log1p(8))
    examples_factor = min(1.0, math.log1p(examples_total) / math.log1p(24))
    avg_conf = _average(confidences)
    avg_nested_conf = _average(nested_conf)
    avg_nested_align = _average(nested_align)
    rank_score = (
        0.25 * avg_conf
        + 0.20 * semantic_coherence
        + 0.15 * attachment_consistency
        + 0.15 * avg_nested_conf
        + 0.10 * avg_nested_align
        + 0.10 * support_factor
        + 0.05 * examples_factor
        - 0.20 * review_penalty
    )
    return {
        "packet_count": packet_count,
        "avg_confidence": round(avg_conf, 4),
        "max_confidence": round(max(confidences) if confidences else 0.0, 4),
        "examples_total": examples_total,
        "nested_evidence_count": sum(len(packet.get("nested_evidence") or []) for packet in packets),
        "avg_nested_evidence_confidence": round(avg_nested_conf, 4),
        "avg_nested_sense_alignment": round(avg_nested_align, 4),
        "semantic_coherence": round(semantic_coherence, 4),
        "attachment_consistency": round(attachment_consistency, 4),
        "exception_density": round(exception_density, 4),
        "risk_density": round(risk_density, 4),
        "review_penalty": round(review_penalty, 4),
        "rank_score": round(max(0.0, rank_score), 4),
        "support_factor": round(support_factor, 4),
        "examples_factor": round(examples_factor, 4),
        "flags": sorted(pair_flags),
        "parameters": {
            "stable_min_packets": parameters["stable_min_packets"],
            "stable_min_avg_confidence": parameters["stable_min_avg_confidence"],
        },
    }


def _semantic_coherence(similarity: np.ndarray, indexes: Sequence[int]) -> float:
    if len(indexes) <= 1:
        return 1.0
    values = [
        float(similarity[left, right])
        for left_pos, left in enumerate(indexes)
        for right in indexes[left_pos + 1 :]
    ]
    return _average(values)


def _max_attachment_divergence(packets: Sequence[Mapping[str, object]]) -> float:
    values = [
        _attachment_divergence(left.get("attachment"), right.get("attachment"))
        for left_pos, left in enumerate(packets)
        for right in packets[left_pos + 1 :]
    ]
    return max(values) if values else 0.0


def _density(packets: Sequence[Mapping[str, object]], field: str) -> float:
    if not packets:
        return 0.0
    count = sum(1 for packet in packets if packet.get(field))
    return count / float(len(packets))


def _status_for_group(
    packets: Sequence[Mapping[str, object]],
    ranking: Mapping[str, object],
    flags: set[str],
    parameters: Mapping[str, object],
) -> str:
    if "effect_domain_divergence" in flags or "negative_contrast_conflict" in flags:
        return "split_recommended"
    if (
        float(ranking.get("exception_density") or 0.0)
        >= float(parameters["exception_density_needs_review_threshold"])
        or float(ranking.get("risk_density") or 0.0)
        >= float(parameters["exception_density_needs_review_threshold"])
        or "mixed_role_evidence" in flags
        or float(ranking.get("avg_confidence") or 0.0)
        < float(parameters["low_confidence_threshold"])
    ):
        return "needs_review"
    if len(packets) == 1:
        packet = packets[0]
        if (
            (_safe_float(packet.get("confidence_score")) or 0.0) >= 0.85
            and packet.get("nested_evidence")
            and not packet.get("attachment_exceptions")
            and not packet.get("confidence_risks")
        ):
            return "stable"
        return "provisional"
    if (
        len(packets) >= int(parameters["stable_min_packets"])
        and float(ranking.get("avg_confidence") or 0.0)
        >= float(parameters["stable_min_avg_confidence"])
        and float(ranking.get("attachment_consistency") or 0.0)
        >= 1.0 - float(parameters["stable_max_attachment_divergence"])
    ):
        return "stable"
    return "provisional"


def _warnings_for_group(
    packets: Sequence[Mapping[str, object]],
    flags: set[str],
) -> list[str]:
    warnings = set(flags)
    for packet in packets:
        warnings.update(str(w) for w in packet.get("packet_warnings") or [])
        warnings.update(str(w) for w in packet.get("attachment_exceptions") or [])
        warnings.update(str(w) for w in packet.get("confidence_risks") or [])
    return sorted(w for w in warnings if w)


def _semantic_terms(packets: Sequence[Mapping[str, object]]) -> list[str]:
    counter: Counter[str] = Counter()
    for packet in packets:
        for term in packet.get("semantic_core") or []:
            normalized = _normalize_text(term)
            if normalized:
                counter[normalized] += 3
        contribution = packet.get("contribution")
        if isinstance(contribution, Mapping):
            for key in contribution:
                normalized = _normalize_text(str(key))
                if normalized:
                    counter[normalized] += 2
    return [term for term, _count in counter.most_common(8)]


def _surface_examples_for_group(packets: Sequence[Mapping[str, object]]) -> list[str]:
    seen: set[str] = set()
    examples: list[str] = []
    for packet in packets:
        for item in packet.get("surface_examples") or []:
            text = str(item).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                examples.append(text)
            if len(examples) >= 8:
                return examples
    return examples


def _label_for_group(
    semantic_terms: Sequence[str],
    surface_examples: Sequence[str],
) -> str:
    content_terms = [term for term in semantic_terms if term not in GENERIC_TERMS]
    if len(content_terms) >= 2:
        return "/".join(content_terms[:2])
    if content_terms:
        return content_terms[0]
    if surface_examples:
        words = _term_tokens(surface_examples[0])
        return " ".join(list(words)[:3]) or "unlabeled"
    return "unlabeled"


def _aggregate_attachment(packets: Sequence[Mapping[str, object]]) -> dict[str, object]:
    likelihood_keys = [
        "prefix_likelihood",
        "suffix_likelihood",
        "infix_likelihood",
        "free_likelihood",
        "productivity",
    ]
    output: dict[str, object] = {}
    for key in likelihood_keys:
        values = [
            value
            for packet in packets
            for attachment in [packet.get("attachment")]
            if isinstance(attachment, Mapping)
            for value in [_safe_float(attachment.get(key))]
            if value is not None
        ]
        output[key] = round(_average(values), 4) if values else None
    counts = defaultdict(int)
    profiles = Counter()
    for packet in packets:
        attachment = packet.get("attachment")
        if not isinstance(attachment, Mapping):
            continue
        for key in (
            "observed_prefix_count",
            "observed_suffix_count",
            "observed_infix_count",
            "observed_free_count",
        ):
            counts[key] += _safe_int(attachment.get(key)) or 0
        profile = _safe_str(attachment.get("estimated_profile"))
        if profile and profile != "unknown":
            profiles[profile] += 1
    output.update(counts)
    output["estimated_profile"] = (
        profiles.most_common(1)[0][0] if profiles else "unknown"
    )
    return output


def _compact_group_for_report(group: Mapping[str, object]) -> dict[str, object]:
    payload = dict(group)
    payload["evidence_packets"] = []
    return payload


def _top_effects_from_group(group: Mapping[str, object]) -> list[str]:
    effects: list[str] = []
    for packet in group.get("evidence_packets") or []:
        if not isinstance(packet, Mapping):
            continue
        for item in packet.get("nested_evidence") or []:
            if not isinstance(item, Mapping):
                continue
            effect = _safe_str(item.get("effect"))
            if effect and effect not in effects:
                effects.append(effect)
    return effects


def _empty_alignment(*, morph: str, span_role: str) -> dict[str, object]:
    return {
        "morph": str(morph or "").upper(),
        "span_role": _normalize_role(span_role),
        "primary_group_id": None,
        "alignment_score": 0.0,
        "reasons": ["no_root_groups_available"],
        "warnings": [],
        "source_cluster_ids": [],
        "primary_group": None,
        "alternates": [],
    }


def _score_group_alignment(
    *,
    morph: str,
    span_role: str,
    group: Mapping[str, object],
    source_cluster_ids: set[int],
    evidence_word: str | None,
    semantic_text: str | None,
    parameters: Mapping[str, float],
) -> dict[str, object]:
    score = 0.0
    reasons: list[str] = []
    warnings = [str(w) for w in group.get("warnings") or [] if str(w).strip()]

    group_source_ids = {
        parsed
        for value in group.get("source_cluster_ids") or []
        for parsed in [_safe_int(value)]
        if parsed is not None
    }
    if source_cluster_ids and group_source_ids & source_cluster_ids:
        score += float(parameters["source_cluster_id_match_weight"])
        reasons.append("source_cluster_id_match")

    if evidence_word and _group_contains_evidence_word(group, evidence_word):
        score += float(parameters["nested_evidence_word_match_weight"])
        reasons.append("nested_evidence_word_match")

    role_score, role_reason = _attachment_role_alignment(group, span_role)
    if role_score > 0:
        score += float(parameters["attachment_role_match_weight"]) * role_score
        reasons.append(role_reason)

    semantic_overlap = _group_semantic_overlap(group, semantic_text)
    if semantic_overlap > 0:
        score += float(parameters["semantic_overlap_weight"]) * semantic_overlap
        reasons.append("semantic_overlap")

    rank_score = 0.0
    ranking = group.get("ranking")
    if isinstance(ranking, Mapping):
        rank_score = max(0.0, min(1.0, _safe_float(ranking.get("rank_score")) or 0.0))
    score += float(parameters["group_rank_weight"]) * rank_score
    if rank_score > 0:
        reasons.append("group_rank_support")

    status = str(group.get("status") or "unknown")
    if status == "needs_review":
        score -= float(parameters["needs_review_penalty"])
        warnings.append("alignment_penalized_needs_review")
    elif status == "split_recommended":
        score -= float(parameters["split_recommended_penalty"])
        warnings.append("alignment_penalized_split_recommended")
    elif status == "provisional":
        score -= float(parameters["provisional_penalty"])

    max_score = float(parameters["max_alignment_score"])
    bounded = max(0.0, min(max_score, score))
    if not reasons:
        reasons.append("ranked_as_visible_alternate")

    return {
        "group_id": group.get("group_id"),
        "alignment_score": round(bounded, 4),
        "reasons": reasons,
        "warnings": sorted(set(warnings)),
        "group_summary": compact_group_summary(group),
    }


def _group_contains_evidence_word(
    group: Mapping[str, object],
    evidence_word: str,
) -> bool:
    target = str(evidence_word or "").strip().upper()
    if not target:
        return False
    for packet in group.get("evidence_packets") or []:
        if not isinstance(packet, Mapping):
            continue
        for item in packet.get("nested_evidence") or []:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("word") or "").strip().upper() == target:
                return True
        for example in packet.get("examples") or []:
            if target in str(example or "").upper():
                return True
    return False


def _attachment_role_alignment(
    group: Mapping[str, object],
    span_role: str,
) -> tuple[float, str]:
    role = _normalize_role(span_role)
    if role == "unknown":
        return 0.0, ""
    attachment = group.get("attachment_profile")
    if not isinstance(attachment, Mapping):
        return 0.0, ""
    key = f"{role}_likelihood"
    likelihood = _safe_float(attachment.get(key))
    if likelihood is not None:
        return max(0.0, min(1.0, likelihood)), f"{role}_attachment_match"
    count_key = f"observed_{role}_count"
    count = _safe_int(attachment.get(count_key)) or 0
    total = sum(
        _safe_int(attachment.get(name)) or 0
        for name in (
            "observed_prefix_count",
            "observed_suffix_count",
            "observed_infix_count",
            "observed_free_count",
        )
    )
    if total <= 0:
        return 0.0, ""
    return count / float(total), f"{role}_observed_attachment_match"


def _group_semantic_overlap(
    group: Mapping[str, object],
    semantic_text: str | None,
) -> float:
    source_terms = _term_tokens(str(semantic_text or ""))
    if not source_terms:
        return 0.0
    group_terms: set[str] = set()
    for field in ("semantic_terms", "surface_examples"):
        for item in group.get(field) or []:
            group_terms.update(_term_tokens(str(item)))
    for packet in group.get("evidence_packets") or []:
        if not isinstance(packet, Mapping):
            continue
        for item in packet.get("nested_evidence") or []:
            if isinstance(item, Mapping):
                group_terms.update(_term_tokens(str(item.get("effect") or "")))
                group_terms.update(_term_tokens(str(item.get("sense") or "")))
    return _jaccard(source_terms - GENERIC_TERMS, group_terms - GENERIC_TERMS)


def _normalize_role(role: str | None) -> str:
    value = str(role or "").strip().lower()
    if value in {"prefix", "suffix", "infix", "free"}:
        return value
    return "unknown"


def _surface_examples(
    examples: Sequence[str],
    nested_evidence: Sequence[Mapping[str, object]],
) -> list[str]:
    values: list[str] = []
    for item in examples:
        text = str(item).strip()
        if text:
            values.append(text)
    for item in nested_evidence:
        sense = _safe_str(item.get("sense"))
        if sense:
            cleaned = re.sub(r"\s*usage:\s*`.*?`", "", sense, flags=re.I).strip()
            values.append(cleaned or sense)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped[:12]


def _unique_terms(values: Sequence[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _term_tokens(value):
            if token not in seen:
                seen.add(token)
                terms.append(token)
    return terms


def _term_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z']*", _normalize_text(text))
        if token and token not in GENERIC_TERMS
    }


def _normalize_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"(?!\b\w+'\w+\b)[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _average(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _row_get(row: sqlite3.Row, key: str) -> object:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stringify(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _normalize_lookup_root(root: str) -> str:
    return str(root or "").strip().lower()


def _normalize_public_root(root: str) -> str:
    return str(root or "").strip().upper()


def _resolve_variants(raw_variants: Sequence[str]) -> list[str]:
    variants = list(raw_variants or ("both",))
    if variants == ["both"] or "both" in variants:
        return ["solo", "debate"]
    allowed = {"solo", "debate"}
    resolved = []
    for variant in variants:
        if variant not in allowed:
            raise ValueError(f"Unsupported variant: {variant}")
        if variant not in resolved:
            resolved.append(variant)
    return resolved


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "group"


def _format_number(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}"
