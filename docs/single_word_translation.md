# Single-Word Translation CLI

This document explains how to translate a single Enochian word (or a proposed
Enochian word) using the stored solo/debate insights databases. The focus is on
human-readable output and transparent evidence tracing so you can inspect *why*
any given definition is suggested.

## Architecture overview

```text
Input Word (NAZPSAD)
    ↓
[Repository Layer] Query DB for clusters, residuals, morph_hypotheses
    ↓
[Decomposition Layer] Beam-search segmentation (NAZ + PSAD)
    ↓
[Filtering Layer] Hard filters + soft scoring
    ↓
[Ranking Layer] Apply strategy (prefer-fewer/prefer-known/prefer-balance)
    ↓
[Synthesis Layer] Optional LLM reconstitution of meanings
    ↓
Output: Candidate definitions with evidence
```

**Why this design?**
- We start with *evidence* before we propose meaning, so the output is anchored
  in what the database actually contains.
- We separate decomposition, filtering, and ranking to make each decision
  auditable and tunable.
- LLM synthesis is optional and placed last, so it never overrides the evidence;
  it only refines a top-ranked candidate into a more readable phrase.

## Evidence sources

The translation pipeline pulls from several databases tables and fallbacks:

- **Clusters** (`clusters` table): Direct n-gram clusters with adjudicated
  glosses. These are the strongest signals.
- **Residual semantics** (`root_residual_semantics`): Residual leftovers from
  prior runs that suggest weak or partial evidence.
- **Morph hypotheses** (`morph_hypotheses`, accepted only): Extra hints accepted
  during analysis, used at low weight.
- **FastText neighbors** (fallback): When nothing else exists, we surface the
  closest embedding neighbors as a heuristic so you can still explore related
  forms.

## Strategy selection

Strategies help break ties or guide ranking when multiple decompositions are
plausible:

- **prefer-fewer**: Favors fewer morphs; useful when you believe long splits are
  overfitting the data.
- **prefer-known**: Favors morphs with higher usage counts in the evidence.
- **prefer-balance**: Favors splits with similar-length morphs; often a neutral
  baseline.

## CLI usage

The single-word translation command is available via the main `enlm` CLI and the
translation CLI.

### `enlm` entry point

```bash
poetry run enlm translate-word NAZPSAD
```

```bash
poetry run enlm translate-word NAZPSAD \
  --variant both \
  --strategy prefer-known \
  --top-k 5
```

### Translation CLI entry point

```bash
poetry run enochian-interpret translate-word NAZPSAD
```

```bash
poetry run enochian-interpret translate-word NAZPSAD --format json --pretty
```

## Output formats

### Text format (default)

The text format emphasizes readability and wraps lines to 80 columns. It is
best when you want to quickly scan candidate senses in a terminal.

### JSON format

Use `--format json` for structured data, or `--format json --pretty` for a
human-readable JSON report.

#### JSON schema (simplified)

```json
{
  "word": "NAZPSAD",
  "variant": "solo",
  "variants_queried": ["solo"],
  "strategy": "prefer-balance",
  "timestamp": "2025-12-10T14:23:45Z",
  "llm_enabled": false,
  "llm_mode": "remote",
  "senses": [
    {
      "rank": 1,
      "variant": "solo",
      "morphs": ["NAZ", "PSAD"],
      "score": 7.82,
      "breakdown": {
        "coverage_ratio": 1.0,
        "residual_ratio": 0.0
      },
      "meanings": [
        {"morph": "NAZ", "definition": "rectangular prism", "provenance": "cluster"},
        {"morph": "PSAD", "definition": "sharp", "provenance": "residual"}
      ],
      "synthesized_definition": "sword, knife, cutting weapon",
      "concatenated_meanings": "rectangular prism + sharp",
      "confidence": 0.85,
      "warnings": []
    }
  ],
  "evidence": {}
}
```

## Edge cases & exit codes

- **No evidence**: The CLI returns a message that no direct evidence was found
  and shows FastText neighbors. Exit code is `1`.
- **Residual-only evidence**: The output includes
  `"provenance_note": "residual-only (observed as remainder)"` and the
  confidence is reduced by `0.2` to reflect weaker support.
- **Missing DBs / invalid input**: The CLI returns a clear error message and
  exits with code `2`.

## Tips for exploratory workflows

- Start with `--variant both` to compare solo and debate results side-by-side.
- Use `--strategy prefer-known` when exploring new or hypothetical compounds;
  it helps keep the output grounded in well-attested morphs.
- When using `--llm`, ensure `.env_remote` or `.env_local` exists so the
  synthesis step can initialize the chosen LLM backend.
