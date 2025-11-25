# Analytics run checklist and troubleshooting

Use this checklist before calling `poetry run enlm analyze all` so the outputs are populated instead of empty CSV/JSON files.

## 1) Ensure the insights DB has real composites and morph vectors
- Run at least one translation pass (`poetry run enochian-analysis`) so the `composite_reconstruction` table has rows.
- Populate `morph_semantic_vectors` (e.g., `poetry run enlm morph factorize --db <db>`). `analyze all` aborts or produces zeros if either table is empty.
- Quick sanity check:
  ```bash
  sqlite3 <db> "SELECT 'composites', COUNT(*) FROM composite_reconstruction UNION ALL SELECT 'morph_vectors', COUNT(*) FROM morph_semantic_vectors;"
  ```
  Both counts should be >0 before proceeding.

## 2) Ingest parses or reuse database state explicitly
- If you have new parses JSONL, pass it directly: `poetry run enlm analyze all --parses <path/to/parses.jsonl> --db <db>`.
- If the DB already holds composites you want to reuse, add `--reuse-db-parses` **and** make sure the composites count from step 1 is non-zero; otherwise the command will fail or emit empty artifacts.

## 3) Order of operations for a clean run
1. `poetry run enochian-analysis` (debate or solo) → seeds composites/residuals.
2. `poetry run enlm morph factorize --db <db>` → fills `morph_semantic_vectors`.
3. `poetry run enlm analyze all --parses <parses.jsonl> --db <db>` (or `--reuse-db-parses` if reusing) → overwrites attribution, collocation, residual cluster tables and exports CSV/JSON summaries.

## 4) Verify analytics outputs look healthy
- Attribution: `sqlite3 <db> "SELECT COUNT(*) FROM attribution_marginals;"` should be >0.
- Collocations: `sqlite3 <db> "SELECT COUNT(*) FROM collocation_stats;"` should be >0.
- Residual clusters: check `residual_clusters.json` summary for non-zero `clusters` and `mean_sim`.

## 5) Optional pre-analysis seeds
Run `poetry run enlm preanalyze --db <db>` before translation sessions to seed trusted n-grams; this populates `preanalysis_seeds` and keeps prompts consistent.

If any check is zero, re-run the missing upstream step instead of relying on `analyze all` defaults. This prevents the empty-attribution and zero-cluster artifacts seen when the database lacks composites or morph vectors.
