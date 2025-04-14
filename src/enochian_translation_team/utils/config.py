from pathlib import Path

def get_config_paths():
    root = Path(__file__).resolve().parents[3]  # gets you to project root
    return {
        "dictionary": root / "dictionary.json",
        "model_output": root / "src" / "enochian_translation_team" / "tools" / "models" / "enochian_fasttext.model"
    }
