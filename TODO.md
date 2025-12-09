## Interpretation Workflow TODO

Status snapshot:

- ✅ Solo-mode root extraction completed and persisted.
- ✅ Residual/remainder extraction logs remainders, residual semantics, and now LLM model names for audit trails.
- 🚧 Debate-mode passes still running; remaining clusters need to be adjudicated and merged back into the shared databases.
- 🚧 Translation CLI consumes the latest analytics tables; refresh attribution/collocation before major prompt runs.

Immediate priorities (ordered):

1. Finish debate-mode queue cycles and append accepted glosses to the insights DBs.
2. Run `poetry run enlm analyze all --db <db>` after each debate batch to keep residual and attribution priors current.
3. Perform remainder backfills (`poetry run enlm remainder backfill-remainders`) on legacy runs lacking the new model metadata.
4. Regenerate derived exports (JSONL/parquet) for downstream translation tests once debate data stabilizes.

Backlog / nice-to-have:

- Harden normalization/versioning into a tracked config artifact so repeated runs record the ruleset hash.
- Add automated lint/test hooks to CI; today the project relies on local checks.
- Expand segmentation experiments (lattice/beam search) once lexicon coverage is frozen.
