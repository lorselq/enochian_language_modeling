from pathlib import Path

def get_config_paths():
    root = Path(__file__).resolve().parents[3]  # gets you to project root
    return {
        "dictionary": root / "src" / "enochian_translation_team" / "data" / "dictionary.json",
        "substitution_map": root / "src" / "enochian_translation_team" / "data" / "substitution_map.json",
        "sequence_compressions": root / "src" / "enochian_translation_team" / "data" / "sequence_compressions.json",
        "model_output": root / "src" / "enochian_translation_team" / "tools" / "models" / "enochian_fasttext.model"
    }
