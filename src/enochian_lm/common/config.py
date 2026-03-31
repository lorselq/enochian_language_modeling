from __future__ import annotations

from pathlib import Path


def _prefer_populated_path(primary: Path, fallback: Path) -> Path:
    """Choose the best on-disk artifact for a logical database role.

    Translation code should prefer the canonical filename when it is present and
    populated, but this project currently also carries a debate extraction-only
    database that contains the real usable data. Centralizing that choice here
    keeps the rest of the translation stack repo-truth-driven and avoids
    scattering file-name special cases across CLI and service layers.
    """
    if primary.exists() and primary.stat().st_size > 0:
        return primary
    if fallback.exists() and fallback.stat().st_size > 0:
        return fallback
    return primary


def get_config_paths() -> dict[str, Path]:
    """Return canonical on-disk locations for shared corpora and artifacts."""

    enochian_lm = Path(__file__).resolve().parents[1]
    root_extraction = enochian_lm / "root_extraction"
    data_dir = root_extraction / "data"
    tools_dir = root_extraction / "tools"
    interpretation_dir = root_extraction / "interpretation"
    debate_primary = interpretation_dir / "debate_derived_definitions.sqlite3"
    debate_extraction_only = (
        interpretation_dir / "debate_derived_definitions_(extraction_only).sqlite3"
    )

    return {
        "dictionary": data_dir / "dictionary.json",
        "substitution_map": data_dir / "substitution_map.json",
        "sequence_compressions": data_dir / "sequence_compressions.json",
        "model_output": tools_dir / "models" / "enochian_fasttext.model",
        "ngram_index": data_dir / "ngram_index.sqlite3",
        "root_word_insights": data_dir / "root_word_insights.json",
        "new_definitions": interpretation_dir / "new_definitions.sqlite3",
        "debate": _prefer_populated_path(debate_primary, debate_extraction_only),
        "debate_primary": debate_primary,
        "debate_extraction_only": debate_extraction_only,
        "solo": interpretation_dir / "solo_analysis_derived_definitions.sqlite3",
        "preanalysis_trusted": data_dir / "trusted_preanalysis_ngrams.json",
        "translation_memory": interpretation_dir / "translation_memory.sqlite3",
    }
