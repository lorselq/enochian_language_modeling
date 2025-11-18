from pathlib import Path

def get_config_paths():
    root = Path(__file__).resolve().parents[3]  # gets you to project root
    return {
        "dictionary": root / "src" / "enochian_translation_team" / "data" / "dictionary.json",
        "substitution_map": root / "src" / "enochian_translation_team" / "data" / "substitution_map.json",
        "sequence_compressions": root / "src" / "enochian_translation_team" / "data" / "sequence_compressions.json",
        "model_output": root / "src" / "enochian_translation_team" / "tools" / "models" / "enochian_fasttext.model",
        "ngram_index": root / "src" / "enochian_translation_team" / "data" / "ngram_index.sqlite3",
        "root_word_insights": root / "src" / "enochian_translation_team" / "data" / "root_word_insights.json",
        "new_definitions": root / "src" / "enochian_translation_team" / "data" / "new_definitions.sqlite3",
        "debate": root / "src" / "enochian_translation_team" / "data" / "debate_derived_definitions.sqlite3",
        "solo": root / "src" / "enochian_translation_team" / "data" / "solo_analysis_derived_definitions.sqlite3",
        "preanalysis_trusted": root
        / "src"
        / "enochian_translation_team"
        / "data"
        / "trusted_preanalysis_ngrams.json",
    }
