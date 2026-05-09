---
name: analytics-refresh-workflow
description: Use when running, changing, or troubleshooting analytics refresh commands that populate morph vectors, attribution, collocation, residual clusters, reports, or analytics-backed prompt priors. Do not use for unrelated source-only edits.
---

# Analytics Refresh Workflow

## Purpose

This Skill prevents empty or misleading analytics outputs by preserving the required prerequisite and refresh order for the project’s insights databases.

## When to use this Skill

- Running or editing `enlm analyze all`.
- Running or editing morph factorization, attribution, collocation, residual clustering, analytics reports, or analytics retrofit scripts.
- Troubleshooting empty attribution, collocation, residual cluster, or report outputs.
- Changing code that reads analytics tables into root-extraction prompts.

## When not to use this Skill

- Unit-only changes that do not touch analytics tables.
- Translation-only changes that only consume existing evidence.
- Dictionary enrichment work that does not touch insights databases.

## Project context

Analytics are computed against the same SQLite insights databases written by root extraction. Empty prerequisite tables can produce zeroed or useless analytics. The documented workflow expects composites and morph vectors before full analysis.

Important areas:

- `src/enochian_lm/analysis/cli.py`
- `src/enochian_lm/analysis/analysis/`
- `src/enochian_lm/root_extraction/utils/analytics_bridge.py`
- `src/enochian_lm/root_extraction/scripts/refresh_analytics_before_semantic_tests.py`
- `docs/analytics_run_checklist.md`

## Workflow

1. Identify the target insights DB and whether it is solo or debate.
2. Check prerequisites before full analytics:
   - `composite_reconstruction` should have rows.
   - `morph_semantic_vectors` should have rows.
3. Prefer the documented refresh order:
   - run or backfill translation/composite data,
   - run `morph factorize`,
   - run attribution, collocation, and residual clustering.
4. Use dry-run or preview modes when retrofitting accepted definitions.
5. Keep analytics table writes idempotent where practical.
6. Update docs only when commands are discoverable from project files.

## Required checks

Useful discovered commands include:

```bash
poetry run enlm morph factorize --db <db>
poetry run enlm attrib loo --db <db>
poetry run enlm colloc --db <db>
poetry run enlm residual cluster --db <db>
poetry run enlm analyze all --db <db> --reuse-db-parses
poetry run enochian-apply-analytics --db <db> --dry-run
PYTHONPATH=src python src/enochian_lm/root_extraction/scripts/refresh_analytics_before_semantic_tests.py --db <db>
```

Run targeted pytest tests if code changed. If checking live DB row counts, use read-only inspection when possible.

## Safety rules
- Do not modify generated databases unless the user explicitly requested a run that writes to them.
- Do not edit .env_local or .env_remote.
- Do not invent CLI commands.
- Do not run non-dry-run analytics retrofits unless requested.
- Do not treat empty analytics output as valid without checking prerequisites.

## Completion criteria
- The refresh order is correct for the target task.
- Prerequisite table state is known or uncertainty is reported.
- Any generated or DB-writing action was explicitly requested.
- Relevant tests or sanity checks passed, or blockers are documented.