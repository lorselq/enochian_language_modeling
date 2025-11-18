# Project Structure

This repository now follows a four-package layout under `src/` so that modeling,
translation, textual ingestion, and model-training concerns stay separated.

## Top-level layout

- `pyproject.toml` / `poetry.lock` – dependency and packaging metadata shared by
  every package.
- `README.md` – overview, research goals, and the full operational checklist.
- `PROJECT_STRUCTURE.md` – (this file) quick orientation to the code layout.
- `src/` – houses all importable packages.

## Packages under `src/`

### `enochian_lm`
Modeling, analytics, and the root-extraction workflow live here.

- `root_extraction/` – the debate/solo crew, CLI entry point, and every utility
  used to build dictionaries, embeddings, and n-gram indices. Notable
  subfolders:
  - `data/` – JSON corpora such as `dictionary.json`, substitution maps, and the
    `enochian_keys.txt` corpus used across packages.
  - `interpretation/` – SQLite databases (`revised_*_derived_definitions.sqlite3`)
    plus cached n-gram reports consumed by both the CLI and the
    `translation` package.
  - `scripts/` – helpers like `init_insights_db.py`, migrations, and analytics
    patchers that operate directly on the SQLite insights DB.
  - `tools/` – FastText training, debate/solo orchestration, and ad-hoc
    inspection scripts (all wired into the Poetry CLI entry points).
  - `utils/` – shared helpers for configuration, dictionary ingestion,
    embeddings, analytics bridging, and other reusable primitives.
- `analysis/` – the `enlm` CLI plus attribution, collocation, factorization, and
  residual analytics. These modules read the same SQLite insights DB that the
  crew writes and populate secondary tables for downstream prompts.
- `common/` – infrastructure shared across packages, such as the sqlite
  bootstrapper that prefers `pysqlite3` when available.

### `translation`
Post-run interpretation tooling that stitches together dictionary entries,
insights databases, and candidate morphologies.

- `service.py` – the high-level `InterpretationService` used by the CLI and any
  downstream integrations. It loads the config paths from `enochian_lm` and
  coordinates candidate searches plus residual reconciliation.
- `repository.py` – read-only access to the solo/debate insights databases,
  exposing lightweight dataclasses (`ClusterRecord`, `ResidualDetail`).
- `tokenization.py` / `cli.py` – token expansion utilities and the user-facing
  command-line entry point wired to the `enochian-interpret` Poetry script.

### `loagaeth`
Digitization tooling for _Liber Loagaeth_.

- `liber_loagaeth_construction/` – JSON leaf data and the scripts that create or
  populate the `liber_loagaeth.sqlite3` database. The new default location for
  that database is `src/loagaeth/liber_loagaeth_construction/data/` so that
  ingestion and schema helpers share the same path.

### `training`
A placeholder package reserved for LLM-training experiments. The module exists
so that future training code can live under `src/training/` without additional
packaging work.

## Entry points and workflows

- `poetry run enochian-analysis` → `enochian_lm.root_extraction.main:main`
  launches the interactive root-extraction crew.
- `poetry run enochian-build-fasttext` / `enochian-build-ngram-index`
  orchestrate corpus prep (`tools/train_fasttext_model.py` and
  `utils/build_ngram_sidecar.py`).
- `poetry run enlm ...` commands live in `enochian_lm.analysis.cli` and operate
  on the SQLite insights databases found under
  `src/enochian_lm/root_extraction/interpretation/`.
- `poetry run enochian-interpret` exposes the translation CLI that consumes
  previously generated insights.

## Shared data

- All shared JSON corpora and dictionary artifacts live under
  `src/enochian_lm/root_extraction/data/`.
- Cross-package SQLite databases (solo/debate insights) live under
  `src/enochian_lm/root_extraction/interpretation/`.
- The Loagaeth database is created under
  `src/loagaeth/liber_loagaeth_construction/data/`.

This structure keeps reusable corpora and databases in predictable locations so
that scripts across packages can rely on `enochian_lm.root_extraction.utils.config`
for canonical paths.
