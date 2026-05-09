---
name: dictionary-enrichment-change
description: Use when changing canonical dictionary enrichment, POS/domain inference, semantic domain config, dictionary schema documentation, or training-window generation. Do not use for root extraction or translation changes that only consume existing dictionary data.
---

# Dictionary Enrichment Change

## Purpose

This Skill protects the boundary between the canonical dictionary, generated enriched dictionary metadata, semantic-domain configuration, and downstream training examples.

## When to use this Skill

- Editing `src/training/datasets/enrich_dictionary_pos.py`.
- Editing `src/training/config/semantic_domains.yml`.
- Updating dictionary schema documentation.
- Changing training-window generation that consumes dictionary data.
- Adding tests for POS, semantic domains, citation evidence, spaCy, or WordNet behavior.

## When not to use this Skill

- Translation or root-extraction changes that only read existing dictionary artifacts.
- Analytics refresh work.
- Loagaeth ingestion changes.

## Project context

The canonical dictionary is `src/enochian_lm/root_extraction/data/dictionary.json`. The enriched output is `src/enochian_lm/root_extraction/data/dictionary_enriched.json`. Enrichment adds POS tags, semantic domains, phrase/copula flags, and notes. spaCy and WordNet behavior is optional and must degrade gracefully.

Important areas:

- `src/training/datasets/enrich_dictionary_pos.py`
- `src/training/config/semantic_domains.yml`
- `docs/dictionary_schema.md`
- `tests/training/datasets/test_enrich_dictionary_pos.py`
- `src/training/tools/generate_key_windows.py`

## Workflow

1. Confirm whether the task changes canonical data, enrichment logic, generated enriched output, or docs.
2. Prefer changing enrichment logic and tests before changing generated output.
3. Preserve existing dictionary fields and historical metadata.
4. Keep optional spaCy and WordNet paths optional.
5. Add tests for direct heuristics and optional dependency behavior.
6. If generated outputs need refreshing, confirm that the user wants generated artifacts updated.

## Required checks

Run targeted tests:

```bash
poetry run pytest tests/training/datasets/test_enrich_dictionary_pos.py
```

If training-window generation changes, add or run targeted tests if available. If no relevant tests exist, state that clearly.

Discovered enrichment-related commands include:

```bash
poetry run python src/training/datasets/enrich_dictionary_pos.py
python -m nltk.downloader wordnet omw-1.4
```
Only run external corpus/model downloads with explicit approval.

## Safety rules
- Do not edit .env_local or .env_remote.
- Do not delete dictionary fields.
- Do not overwrite generated dictionary output unless explicitly requested.
- Do not make spaCy or WordNet mandatory for baseline enrichment tests.
- Do not invent new semantic domain labels without checking semantic_domains.yml.

## Completion criteria
- Canonical and generated dictionary responsibilities are clear.
- POS/domain changes are tested.
- Optional dependency behavior remains graceful.
- Relevant tests pass, or blockers are documented.
