from __future__ import annotations

"""Regression tests for root-level diagnostic sense grouping.

These tests protect the evidence-preservation contract described in
``docs/translation_interpretation_schema.md``. The grouping report is meant to
surface nested evidence and ambiguity without changing translation ranking.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from translation.root_groups import (
    MissingRootGlossesViewError,
    RootGroupOptions,
    RootSenseGroupService,
    render_report_text,
)


def _create_root_group_db(path: Path) -> None:
    """Create the minimal read-only schema consumed by root group diagnostics.

    Why:
    the service reads production views, but tests need a small deterministic DB
    that carries the same public columns.

    How:
    this helper creates tables named like the production views because the
    service only requires read-compatible row shapes.

    Responsibility:
    keep tests focused on report behavior rather than root-extraction setup.
    """

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE root_glosses (
              root TEXT,
              evaluation TEXT,
              definition TEXT,
              reason TEXT,
              decoding_guide TEXT,
              semantic_core TEXT,
              negative_contrast TEXT,
              examples_json TEXT,
              contribution_json TEXT,
              pos_bias_nounness REAL,
              pos_bias_modifier REAL,
              pos_bias_verbness REAL,
              attachment_prefix_likelihood REAL,
              attachment_suffix_likelihood REAL,
              attachment_free_likelihood REAL,
              attachment_productivity REAL,
              attachment_exceptions TEXT,
              confidence_score REAL,
              confidence_drivers TEXT,
              confidence_risks TEXT,
              examples_in_cluster INTEGER,
              source_cluster_id INTEGER,
              raw_glossator_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE root_attachment_profile (
              root TEXT,
              observed_prefix_count INTEGER,
              observed_suffix_count INTEGER,
              observed_infix_count INTEGER,
              observed_free_count INTEGER,
              estimated_profile TEXT,
              source_cluster_id INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_gloss(
    path: Path,
    *,
    root: str,
    cluster_id: int,
    definition: str,
    semantic_core: list[str],
    evidence: list[dict[str, object]],
    confidence: float = 0.82,
    attachment_prefix: float = 0.2,
    attachment_suffix: float = 0.1,
    attachment_free: float = 0.7,
    examples: list[str] | None = None,
    negative_contrast: list[str] | None = None,
    attachment_exceptions: list[str] | None = None,
    confidence_risks: list[str] | None = None,
) -> None:
    """Insert one accepted root gloss row with nested glossator evidence.

    Why:
    root group behavior depends on several fields interacting, especially raw
    evidence effects that are easy to lose if only definitions are compared.

    How:
    callers provide the semantic pieces relevant to each test and this helper
    fills the remaining root-gloss columns with stable defaults.

    Responsibility:
    make each test's semantic intent readable at the call site.
    """

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            INSERT INTO root_glosses (
              root, evaluation, definition, reason, decoding_guide,
              semantic_core, negative_contrast, examples_json, contribution_json,
              pos_bias_nounness, pos_bias_modifier, pos_bias_verbness,
              attachment_prefix_likelihood, attachment_suffix_likelihood,
              attachment_free_likelihood, attachment_productivity,
              attachment_exceptions, confidence_score, confidence_drivers,
              confidence_risks, examples_in_cluster, source_cluster_id,
              raw_glossator_json
            )
            VALUES (?, 'accepted', ?, 'accepted by test', 'test guide',
                    ?, ?, ?, ?,
                    0.4, 0.2, 0.4,
                    ?, ?, ?, 0.5,
                    ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                root,
                definition,
                json.dumps(semantic_core),
                json.dumps(negative_contrast or []),
                json.dumps(examples or []),
                json.dumps({"test": 1.0}),
                attachment_prefix,
                attachment_suffix,
                attachment_free,
                json.dumps(attachment_exceptions or []),
                confidence,
                json.dumps(["test support"]),
                json.dumps(confidence_risks or []),
                len(examples or evidence),
                cluster_id,
                json.dumps({"EVIDENCE": evidence}),
            ),
        )
        conn.execute(
            """
            INSERT INTO root_attachment_profile (
              root, observed_prefix_count, observed_suffix_count,
              observed_infix_count, observed_free_count, estimated_profile,
              source_cluster_id
            )
            VALUES (?, ?, ?, 0, ?, ?, ?)
            """,
            (
                root,
                4 if attachment_prefix >= attachment_free else 0,
                4 if attachment_suffix >= attachment_free else 0,
                4 if attachment_free > max(attachment_prefix, attachment_suffix) else 0,
                "free" if attachment_free > max(attachment_prefix, attachment_suffix) else "bound",
                cluster_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _force_lexical_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable optional sentence-transformer loading for deterministic tests.

    Why:
    the implementation prefers a local embedding model when present, but unit
    tests should not depend on machine-specific model caches.

    How:
    patch the service module's imported helper to force the documented TF-IDF
    fallback.

    Responsibility:
    keep grouping assertions stable across developer and CI environments.
    """

    monkeypatch.setattr(
        "translation.root_groups.get_sentence_transformer_if_available",
        lambda *_args, **_kwargs: None,
    )


def test_io_report_preserves_negative_nested_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "debate.sqlite3"
    _create_root_group_db(db_path)
    _insert_gloss(
        db_path,
        root="io",
        cluster_id=3069,
        definition="An existential base indicating a state of being.",
        semantic_core=["existential state", "manifestation"],
        examples=["OHIO: woe"],
        evidence=[
            {
                "word": "OHIO",
                "sense": "woe; sorrowful exclamation. usage: `OHIO`",
                "loc": "test:1",
                "note": {
                    "role": "suffix",
                    "effect": "negative manifestation of a state: woe or sorrow",
                    "confidence": 0.91,
                    "sense_alignment": 0.88,
                },
            }
        ],
        confidence=0.9,
    )

    report = RootSenseGroupService(
        RootGroupOptions(
            variant_paths={"debate": db_path},
            variants=("debate",),
            detail="full",
        )
    ).build_report("IO")

    assert report["root"] == "IO"
    assert report["diagnostics"]["embedding_backend"] == "single-packet"
    group = report["groups"][0]
    packet = group["evidence_packets"][0]
    nested = packet["nested_evidence"][0]

    assert nested["word"] == "OHIO"
    assert nested["effect"] == "negative manifestation of a state: woe or sorrow"
    assert packet["raw_glossator_json"]["EVIDENCE"][0]["word"] == "OHIO"
    assert "woe; sorrowful exclamation." in packet["surface_examples"]
    assert "woe; sorrowful exclamation." in group["surface_examples"]


def test_short_root_keeps_low_similarity_senses_separate(tmp_path: Path) -> None:
    db_path = tmp_path / "solo.sqlite3"
    _create_root_group_db(db_path)
    _insert_gloss(
        db_path,
        root="i",
        cluster_id=1,
        definition="A marker of identity and continuing existence.",
        semantic_core=["identity", "existence"],
        evidence=[
            {
                "word": "IA",
                "sense": "is; exists",
                "note": {
                    "role": "free",
                    "effect": "copular identity and existence",
                    "confidence": 0.8,
                    "sense_alignment": 0.82,
                },
            }
        ],
    )
    _insert_gloss(
        db_path,
        root="i",
        cluster_id=2,
        definition="A negative marker indicating inability or prohibition.",
        semantic_core=["negation", "prohibition"],
        evidence=[
            {
                "word": "IL",
                "sense": "not; cannot",
                "note": {
                    "role": "prefix",
                    "effect": "negation, inability, or prohibition",
                    "confidence": 0.78,
                    "sense_alignment": 0.76,
                },
            }
        ],
        attachment_prefix=0.8,
        attachment_free=0.1,
    )

    report = RootSenseGroupService(
        RootGroupOptions(
            variant_paths={"solo": db_path},
            variants=("solo",),
            detail="full",
        )
    ).build_report("i")

    assert report["diagnostics"]["packet_count"] == 2
    assert report["diagnostics"]["group_count"] == 2
    labels = {group["label"] for group in report["groups"]}
    assert any("identity" in label or "existence" in label for label in labels)
    assert any("negation" in label or "prohibition" in label for label in labels)


def test_compact_detail_omits_full_evidence_packets(tmp_path: Path) -> None:
    db_path = tmp_path / "solo.sqlite3"
    _create_root_group_db(db_path)
    _insert_gloss(
        db_path,
        root="d",
        cluster_id=10,
        definition="A marker of distinction.",
        semantic_core=["distinction"],
        evidence=[
            {
                "word": "DA",
                "sense": "different",
                "note": {
                    "role": "prefix",
                    "effect": "marks distinction",
                    "confidence": 0.86,
                    "sense_alignment": 0.8,
                },
            }
        ],
    )

    report = RootSenseGroupService(
        RootGroupOptions(
            variant_paths={"solo": db_path},
            variants=("solo",),
            detail="compact",
        )
    ).build_report("d")

    assert report["groups"][0]["evidence_packets"] == []
    assert report["groups"][0]["ranking"]["rank_score"] > 0


def test_missing_root_glosses_fails_with_clear_schema_error(tmp_path: Path) -> None:
    db_path = tmp_path / "missing.sqlite3"
    sqlite3.connect(db_path).close()

    service = RootSenseGroupService(
        RootGroupOptions(variant_paths={"solo": db_path}, variants=("solo",))
    )

    with pytest.raises(MissingRootGlossesViewError, match="root_glosses"):
        service.build_report("io")


def test_text_renderer_shows_effects_and_empty_reason(tmp_path: Path) -> None:
    db_path = tmp_path / "solo.sqlite3"
    _create_root_group_db(db_path)
    _insert_gloss(
        db_path,
        root="io",
        cluster_id=3069,
        definition="A state marker.",
        semantic_core=["manifestation"],
        evidence=[
            {
                "word": "OHIO",
                "sense": "woe",
                "note": {
                    "role": "suffix",
                    "effect": "negative manifestation",
                    "confidence": 0.9,
                    "sense_alignment": 0.9,
                },
            }
        ],
    )

    report = RootSenseGroupService(
        RootGroupOptions(variant_paths={"solo": db_path}, variants=("solo",))
    ).build_report("io")
    text = render_report_text(report)

    assert "Root: IO" in text
    assert "top_effects: negative manifestation" in text

    empty = RootSenseGroupService(
        RootGroupOptions(variant_paths={"solo": db_path}, variants=("solo",))
    ).build_report("zz")
    assert "no_accepted_root_glosses" in render_report_text(empty)
