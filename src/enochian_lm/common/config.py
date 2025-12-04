from __future__ import annotations

from pathlib import Path


def get_config_paths() -> dict[str, Path]:
    """Return canonical on-disk locations for shared corpora and artifacts."""

    enochian_lm = Path(__file__).resolve().parents[1]
    root_extraction = enochian_lm / "root_extraction"
    data_dir = root_extraction / "data"
    tools_dir = root_extraction / "tools"
    interpretation_dir = root_extraction / "interpretation"

    return {
        "dictionary": data_dir / "dictionary.json",
        "substitution_map": data_dir / "substitution_map.json",
        "sequence_compressions": data_dir / "sequence_compressions.json",
        "model_output": tools_dir / "models" / "enochian_fasttext.model",
        "ngram_index": data_dir / "ngram_index.sqlite3",
        "root_word_insights": data_dir / "root_word_insights.json",
        "new_definitions": interpretation_dir / "new_definitions.sqlite3",
        "debate": interpretation_dir / "revised_debate_derived_definitions.sqlite3",
        "solo": interpretation_dir / "revised_solo_analysis_derived_definitions.sqlite3",
        "preanalysis_trusted": data_dir / "trusted_preanalysis_ngrams.json",
    }
