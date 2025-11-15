# `enochian_lm`: Analytical tooling for Enochian morpheme research

The `enochian_lm` package hosts the standalone analytics that consume the
conversation logs, embeddings, and morphological reconstructions produced by the
`enochian_translation_team` agents. It focuses on building secondary insights
such as attribution deltas, collocation scores, residual clusters, and
regression-based morph semantics. The module is distributed as part of the
shared Poetry project and exposes a CLI entry point named `enlm`.

## Relationship to `enochian_translation_team`

The analytics in this package piggy-back on the SQLite "insights" database that
is seeded by `enochian_translation_team.scripts.init_insights_db`. The shared
schema provides:

* **Core run metadata** (`runs`, `clusters`, `definitions`, ...), created by the
  translation agents.
* **Composite reconstruction data** (`composite_reconstruction`) capturing the
  morphemes, embeddings, and residuals per token that get ingested for analysis.
* **Morph semantic vectors** (`morph_semantic_vectors`) storing learned
  embedding vectors for each morpheme.
* **Residual definitions** (`residual_details`) capturing human/agent-written
  glosses for unexplained fragments.

`enochian_lm` adds its own tables (e.g. `attribution_marginals`,
`collocation_stats`, `residual_clusters`) via
`enochian_lm.utils.sql.ensure_analysis_tables`. The CLI automatically imports
`init_insights_db.init_db()` before every command to ensure the base schema is
present and then extends it with the analytics tables.

Because the analytics depend on tables populated by the translation pipeline,
run `enochian_translation_team` workflows first so that the composite parses,
morph vectors, and residual annotations exist. The analytics will otherwise
produce empty summaries.

## Directory layout

```
enochian_lm/
├── __init__.py
├── cli.py
├── analysis/
│   ├── __init__.py
│   ├── attribution.py
│   ├── colloc.py
│   ├── factorize.py
│   └── residuals.py
├── report/
│   ├── __init__.py
│   └── pipeline_summary.py
└── utils/
    ├── __init__.py
    ├── sql.py
    ├── stats.py
    └── text.py
```

The subsections below describe how each module fits into the analytics pipeline.

### `cli.py`

The CLI orchestrates database setup, ingestion, analytics, and reporting. It is
exposed by the Poetry script `enlm` and supports the following command groups:

* `attrib loo` — run leave-one-out attribution to quantify how much each morph
  contributes to a composite token. Results are written to the
  `attribution_marginals` table and summarized on stdout.
* `colloc` — compute PMI/LLR based collocation statistics for morph pairs using
  the attribution table as input. Writes `collocation_stats` and emits summary
  metrics.
* `residual cluster` — cluster residual spans (either precomputed in
  `residual_details` or reconstructed from morph vectors) to surface families of
  unexplained fragments.
* `morph factorize` — perform ridge-regression factorization that predicts gloss
  embeddings from morph participation, exporting CSV/JSON artifacts in an output
  directory.
* `report pipeline` — generate a static HTML/CSV bundle summarizing coverage,
  attribution, residual clustering, and factorization results. Optional baseline
  JSONL may be supplied for comparisons.
* `analyze all` — convenience driver that ingests fresh composite/morph JSONL
  files, reruns the attribution, collocation, residual clustering, and
  factorization steps, and exports ready-to-share CSV/JSON artifacts.

Each command ensures the insights database exists by calling
`init_insights_db.init_db()`, opens it through `enochian_lm.utils.sql.connect_sqlite`,
and guarantees that the analytics tables exist via `ensure_analysis_tables`.

### `analysis` subpackage

The analysis modules house the computation-heavy routines that persist their
outputs back into SQLite.

* `analysis.attribution` — Loads morph vectors and composite reconstructions to
  compute leave-one-out cosine deltas for every morph pair used within a token.
  Aggregates the deltas into `attribution_marginals` for later collocation work
  and returns summary statistics.
* `analysis.colloc` — Reuses the attribution marginals to produce PMI, log-
  likelihood ratio, and asymmetry measurements for frequent morph pairings.
  Uses helper utilities from `utils.stats` and writes results to
  `collocation_stats`.
* `analysis.residuals` — Clusters residual vectors either from the
  `residual_details` table or by reconstructing morph residuals using PMI
  neighbors. Produces `residual_clusters` plus membership tables and prints
  summaries, optionally falling back when scikit-learn is unavailable.
* `analysis.factorize` — Builds a design matrix from composite reconstructions,
  embeds gloss text using TF-IDF/Hashing vectorizers, and solves a ridge
  regression that approximates gloss semantics as additive morph vectors. Exports
  CSV/JSON artifacts (`morph_vectors.csv`, `reconstruction.csv`, `summary.json`,
  and optional `alignment.csv`) and updates the `morph_semantic_vectors` table.

### `utils` subpackage

Utility helpers provide shared services to the analysis modules:

* `utils.sql` — wraps SQLite connectivity, seeds or upgrades analytics tables,
  and exposes a bulk `upsert_rows` helper that the analytics modules rely on for
  persistence.
* `utils.text` — hosts lightweight normalization, reproducible seeding, and time
  helpers (`utcnow_iso`) used throughout the CLI and analytics.
* `utils.stats` — supplies numerical helpers for PMI and Dunning's log-likelihood
  calculations.

### `report` subpackage

`report.pipeline_summary` gathers analytics outputs into a publishable bundle.
It loads the various tables, optionally merges them with baseline JSONL metrics,
produces CSV extracts, and renders inline matplotlib visualizations when the
optional dependency is available. The CLI writes reports into timestamped
folders under `runs/` by default.

## Expected inputs and outputs

Most analytics operate on the SQLite database instead of raw text files. The
exception is `analyze all`, which uses JSONL sources:

* **Composite parses JSONL** — Each line should contain a token, predicted vector
  (list of floats), morpheme sequence, residual/error metrics, and optional gloss
  strings. The CLI ingests this into `composite_reconstruction`.
* **Morph inventory JSONL** — Each line should include a morph surface form and a
  semantic embedding vector. The CLI stores these as
  `morph_semantic_vectors` entries.

All commands emit structured outputs under user-specified directories:

* Attribution and collocation commands export CSV snapshots alongside their
  database tables.
* Residual clustering writes a JSON summary describing cluster assignments and
  prints a table of top clusters.
* Factorization writes morph embeddings, token reconstruction diagnostics, and an
  optional alignment-to-residuals CSV for cross-referencing cluster centroids.
* Pipeline reports generate an HTML dashboard plus CSV/JSON caches summarizing
  the intermediate data.

## Typical workflow

1. **Populate the insights database** by running `init_insights_db.init_db()`
   directly or simply executing any `enlm` command once the translation agents
   have produced new results. This ensures the shared schema exists.
2. **Ingest fresh composite and morph exports** (if you have new JSONL files) via
   `poetry run enlm analyze all --parses <path/to/parses.jsonl> --morphs <path/to/morphs.jsonl> ...`.
3. **Iterate on specific analytics**: rerun `attrib`, `colloc`, `residual`, or
   `morph factorize` subcommands with tuned parameters as needed. Each command
   reuses the existing database contents and overwrites stale analytics tables.
4. **Publish reports** with `poetry run enlm report pipeline --out runs/latest`
   to capture an HTML/CSV snapshot suitable for sharing with collaborators.

By grounding the analytics in the same database that powers the translation
agents, `enochian_lm` stays synchronized with evolving definitions and residual
notes while keeping the heavy numerical experimentation in an isolated package.
