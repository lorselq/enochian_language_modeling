from __future__ import annotations

import logging
from pathlib import Path

from enochian_lm.common.config import get_config_paths
from enochian_lm.common.sqlite_bootstrap import sqlite3
from enochian_lm.root_extraction.scripts.init_insights_db import init_db
from enochian_lm.root_extraction.tools.train_fasttext_model import (
    FastTextParams,
    train_fasttext_model,
)
from enochian_lm.root_extraction.utils.build_ngram_sidecar import build_sidecar
from enochian_lm.root_extraction.utils.dictionary_loader import (
    load_dictionary,
    load_dictionary_v2,
)
from enochian_lm.root_extraction.utils.preanalysis import execute_preanalysis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


def _load_dictionary_entries(dict_path: Path):
    try:
        entries = load_dictionary(str(dict_path))
        logger.info("Loaded dictionary entries via legacy loader.")
        return entries
    except Exception as exc:
        logger.info("Legacy loader failed (%s); using v2 adapter.", exc)
        return load_dictionary_v2(str(dict_path))


def _has_translation_runs(db_path: Path) -> bool:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM runs
            WHERE phase = 'translation'
            ORDER BY datetime(created_at) DESC
            LIMIT 1
            """
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _run_preanalysis(db_path: Path) -> None:
    logger.info("Running pre-analysis (initial) for %s", db_path)
    execute_preanalysis(db_path=db_path, stage="initial")

    if _has_translation_runs(db_path):
        logger.info("Running pre-analysis (subsequent) for %s", db_path)
        execute_preanalysis(db_path=db_path, stage="subsequent")
    else:
        logger.warning(
            "Skipping subsequent pre-analysis for %s (no translation runs found)",
            db_path,
        )


def main() -> None:
    paths = get_config_paths()
    dictionary_path = Path(paths["dictionary"])
    data_dir = dictionary_path.parent

    logger.info("Building n-gram index")
    build_sidecar(
        db=Path(paths["ngram_index"]),
        keys_txt=data_dir / "enochian_keys.txt",
        dict_json=dictionary_path,
        subst_json=Path(paths["substitution_map"]),
        compress_json=Path(paths["sequence_compressions"]),
        min_n=1,
        max_n=7,
        respect_pauses=True,
        variant_map_path=None,
        max_variants=None,
    )

    logger.info("Training FastText model")
    entries = _load_dictionary_entries(dictionary_path)
    params = FastTextParams()
    train_fasttext_model(entries, Path(paths["model_output"]), params)

    logger.info("Initializing insights databases")
    debate_db = Path(paths["debate"])
    solo_db = Path(paths["solo"])
    init_db(debate_db)
    init_db(solo_db)

    logger.info("Running pre-analysis for debate database")
    _run_preanalysis(debate_db)

    logger.info("Running pre-analysis for solo database")
    _run_preanalysis(solo_db)

    logger.info("Pipeline preparation complete")


if __name__ == "__main__":
    main()
