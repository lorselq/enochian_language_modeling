---
name: translation-evidence-change
description: Use when changing single-word or phrase translation, evidence retrieval, decomposition, scoring, placeholder filtering, translation memory, or translation CLI behavior under src/translation. Do not use for root-extraction orchestration changes.
---

# Translation Evidence Change

## Purpose

This Skill helps preserve translation behavior that combines dictionary entries, solo/debate insights databases, candidate decomposition, residual evidence, scoring, and optional LLM synthesis.

## When to use this Skill

- Editing files under `src/translation/`.
- Changing `enochian-interpret` or `enlm translate-word` / `translate-phrase` behavior.
- Changing decomposition, scoring, placeholder gloss filtering, evidence selection, variant handling, or translation memory.
- Adding or updating translation tests.

## When not to use this Skill

- Root extraction or semantic-subtraction orchestration changes.
- Analytics refresh workflow changes.
- Dictionary enrichment-only work.

## Project context

Translation reads from canonical config paths and insights DBs. It should distinguish solo and debate evidence, avoid treating placeholder/numeric meta glosses as real meaning, and remain testable without requiring heavy optional dependencies.

Important areas:

- `src/translation/service.py`
- `src/translation/phrase_service.py`
- `src/translation/repository.py`
- `src/translation/decomposition.py`
- `src/translation/scoring.py`
- `src/translation/strategies.py`
- `src/translation/placeholder_glosses.py`
- `src/translation/cli.py`
- `tests/translation/`

## Workflow

1. Read the relevant service/repository/scoring code and nearest tests first.
2. Preserve variant-aware evidence: solo and debate results must stay distinguishable.
3. Preserve read-only expectations for repository lookups unless the task is explicitly about translation memory writes.
4. Keep candidate decomposition deterministic and explainable.
5. Keep placeholder and numeric meta gloss filtering intact.
6. Add tests for obvious, unusual, and edge cases.
7. Use lightweight stubs in tests when optional dependencies would make the test brittle.

## Required checks

Run targeted tests based on the changed area, for example:

```bash
poetry run pytest tests/translation/test_segmentation_rules.py
poetry run pytest tests/translation/test_coherence_scoring.py
poetry run pytest tests/translation/test_strategy_upgrade.py
poetry run pytest tests/translation
```

For CLI behavior, add or update tests rather than relying only on manual CLI output.

## Safety rules
- Do not write to insights DBs from translation lookup code.
- Do not hardcode DB paths outside get_config_paths() unless a test fixture needs a temporary path.
- Do not edit .env_local, .env_remote, generated DBs, logs, or run artifacts.
- Do not remove optional dependency stubs from tests unless replacing them with an equally lightweight approach.
- Do not treat unknown words as confidently translated without evidence.

## Completion criteria
- Translation output remains evidence-backed and variant-aware.
- Placeholder and weak evidence handling are covered by tests when affected.
- Relevant tests pass, or blockers are reported.
- CLI changes are reflected in parser tests or command-level tests where practical.
