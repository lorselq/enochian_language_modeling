---
name: semantic-subtraction-pipeline-guard
description: Use when changing root/residual extraction orchestration, semantic subtraction, donor hierarchy, subtraction traces, or solo/debate extraction behavior. Do not use for unrelated translation, training, or docs-only work.
---

# Semantic Subtraction Pipeline Guard

## Purpose

This Skill helps preserve the semantic-subtraction pipeline that turns `HOST - ROOT = RESIDUAL` evidence into auditable prompts, traces, and SQLite records.

## When to use this Skill

- Editing `src/enochian_lm/root_extraction/pipeline/`.
- Editing semantic-subtraction utilities or engines.
- Changing donor hierarchy, word-break recursion, trace persistence, skip logic, queue handling, or solo/debate orchestration.
- Adding tests for residual evidence, hierarchy traversal, or semantic-subtraction DB writes.

## When not to use this Skill

- Pure translation CLI changes under `src/translation/`.
- Dictionary enrichment-only work.
- Docs-only edits that do not change pipeline behavior.

## Project context

The pipeline depends on dictionary evidence, accepted SQLite glosses, infix/fragment branching, recursive donor resolution, analytics priors, and durable trace tables. Solo and debate modes are both supported and should stay behaviorally aligned unless a task explicitly says otherwise.

Important areas:

- `src/enochian_lm/root_extraction/pipeline/run_residual_semantic_extraction.py`
- `src/enochian_lm/root_extraction/utils/residual_analysis.py`
- `src/enochian_lm/root_extraction/tools/*semantic_subtraction_engine.py`
- `tests/root_extraction/`

## Workflow

1. Read the relevant pipeline function and nearby tests before editing.
2. Identify whether the change affects solo mode, debate mode, or both.
3. Preserve the donor priority model: dictionary-attested, accepted SQLite roots, infix branching, then recursion.
4. Preserve multiple viable donor branches when ambiguity matters.
5. Keep trace rows auditable with root, host, residual, equation, source, recursion depth, and termination reason where available.
6. Prefer small, deterministic tests with lightweight stubs for optional heavy dependencies.
7. Do not directly mutate real research databases during tests.

## Required checks

- Run targeted tests for changed behavior, for example:
  - `poetry run pytest tests/root_extraction/test_word_break_subtractions.py`
  - `poetry run pytest tests/root_extraction/test_hierarchy_donor_resolution.py`
  - `poetry run pytest tests/root_extraction/test_hierarchy_multi_branch_guidance.py`
  - `poetry run pytest tests/root_extraction/test_pipeline_engine_guidance_contract.py`
  - `poetry run pytest tests/root_extraction/test_semantic_subtraction_trace_persistence.py`
- If the affected area is broad, run:
  - `poetry run pytest tests/root_extraction`

If a command is unavailable in the environment, report that clearly.

## Safety rules

- Do not edit `.env_local`, `.env_remote`, logs, generated DBs, or generated run artifacts without explicit approval.
- Do not delete files.
- Do not replace SQLite persistence with ad hoc JSON or text logs.
- Do not collapse solo/debate behavior unless the task explicitly requires it.
- Do not remove trace fields just because prompts do not currently display them.

## Completion criteria

- The pipeline behavior is covered by targeted tests.
- Solo/debate impact is stated.
- Traceability is preserved or improved.
- Relevant tests pass, or any blocked tests are reported with the exact blocker.
