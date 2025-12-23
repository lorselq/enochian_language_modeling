## Architecture Overview

### Data Flow Pipeline
```
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

### Key Design Decisions
- **Two-stage filtering**: Hard filters (support requirements) → Soft scoring (composite metrics)
- **Three strategies**: `prefer-fewer` (minimize morphs), `prefer-known` (favor well-attested morphs), `prefer-balance` (similar morph lengths)
- **LLM synthesis**: Optional single-call semantic synthesis (no debate), with graceful fallback to concatenated meanings
- **Variant handling**: Solo/debate results kept separate; `--variant both` shows side-by-side comparison
- **FastText fallback**: When no direct evidence exists, show top-5 neighbors as heuristic
- **Morph hypotheses**: Accepted hypotheses from `morph_hypotheses` table used as supplementary low-weight evidence

### Data Schemas

**Evidence Bundle** (`WordEvidence`):
```python
{
    "word": str,                          # normalized uppercase
    "direct_clusters": list[ClusterRecord],
    "residual_semantics": list[dict],     # from root_residual_semantics table
    "morph_hypotheses": list[dict],       # from morph_hypotheses (accepted=1)
    "fasttext_neighbors": list[dict],     # fallback when no direct evidence
    "variants_queried": list[str]
}
```

**Decomposition**:
```python
{
    "morphs": list[str],                  # e.g., ["NAZ", "PSAD"]
    "beam_score": float,                  # TF-IDF prior from beam search
    "breakdown": {
        "segments": list[dict],           # per-segment metadata
        "uncovered": list[dict],          # unexplained spans
        "coverage_ratio": float,          # 0.0-1.0
        "residual_ratio": float           # 1.0 - coverage_ratio
    },
    "morph_support": dict[str, str]       # {morph: "cluster"|"residual"|"hypothesis"}
}
```

**Output Schema**:
```python
{
    "word": "NAZPSAD",
    "variants_queried": ["solo"],
    "strategy": "prefer-balance",
    "timestamp": "2025-12-10T14:23:45Z",
    "senses": [
        {
            "rank": 1,
            "variant": "solo",
            "morphs": ["NAZ", "PSAD"],
            "score": 7.82,
            "breakdown": {"coverage_ratio": 1.0, "residual_ratio": 0.0},
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
    "evidence": { }  
}
```

---

## PHASE 1: Data Access & Evidence Gathering

### Task 1.1: Extend Repository to Support Single-Word Lookup
**File**: `src/translation/repository.py`
**Estimated effort**: 4-6 hours

- [x] Add method `fetch_word_evidence(word: str, variants: list[str]) -> WordEvidence`
- [x] Query direct cluster matches: `clusters` table where `ngram = word`
- [x] Query residual semantic matches: `root_residual_semantics` where `residual = word`
- [x] Query morph hypothesis matches: `morph_hypotheses` where `morph = word` AND `accepted = 1`
- [x] Implement FastText neighbor fallback (top-5) when direct evidence is empty
- [x] Return `WordEvidence` dataclass with all evidence consolidated

**Testing**:
- Given "NAZ" with accepted clusters → returns `direct_clusters` populated
- Given "PSAD" only as residual → returns `residual_semantics` populated
- Given "XYZABC" with no evidence → returns `fasttext_neighbors` populated (top 5 neighbors)
- Given conflicting solo/debate variants → both returned separately in `direct_clusters`

---

### Task 1.2: Implement Morph Hypothesis Accessor
**File**: `src/translation/repository.py`
**Estimated effort**: 2-3 hours

- [x] Add method `fetch_accepted_morphs(variant: str) -> dict[str, dict]`
- [x] Query `morph_hypotheses WHERE accepted = 1`
- [x] Return map: `{morph: {gloss, rationale, delta_cosine, source_word, anchor}}`
- [x] Verify field mapping matches `morph_hypotheses` schema from `init_insights_db.py:356-370`

**Testing**:
- Query solo DB with 10 accepted hypotheses → returns 10-item dict
- Query debate DB with 0 accepted hypotheses → returns empty dict

---

## PHASE 2: Decomposition & Filtering

### Task 2.1: Implement Decomposition Engine
**File**: New file `src/translation/decomposition.py`
**Estimated effort**: 5-7 hours

- [x] Create wrapper around `MorphemeCandidateFinder` from `candidate_finder.py`
- [x] Implement method `generate_decompositions(word: str, evidence: WordEvidence) -> list[Decomposition]`
- [x] Use beam-search segmentation (`segment_target()`)
- [x] Return all plausible splits with metadata (morphs, beam_score, breakdown, morph_support)
- [x] Document `Decomposition` dataclass with full schema

**Testing**:
- "NAZPSAD" → returns [["NAZ", "PSAD"], ["NAZP", "SAD"], ...]
- "NAZ" → returns [["NAZ"]] (single morph)
- "XYZABC" → returns [] (no valid decomposition from index)

**Dependencies**: Task 1.1

---

### Task 2.2: Implement Hard Filters
**File**: `src/translation/decomposition.py`
**Estimated effort**: 4-5 hours

- [x] Implement method `apply_hard_filters(decompositions: list[Decomposition], evidence: WordEvidence, min_support_threshold: float = 0.2) -> list[Decomposition]`
- [x] **Filter 1**: Each morph must have support (in clusters OR residuals OR hypotheses with delta_cosine >= threshold)
- [x] **Filter 2**: Discard if residual_ratio > 0.5 when better alternatives exist
- [x] **Filter 3**: Prefer well-attested morphs (>3 uses) over singleton morphs when both cover the word
- [x] Log filtering decisions for debugging

**Testing**:
- Decomposition with all morphs supported → kept
- Decomposition with one unsupported morph → discarded
- Decomposition with residual_ratio=0.8 when alternative has 0.2 → discarded
- Two decompositions, one uses high-frequency morphs, other uses singletons → prefer high-frequency

**Dependencies**: Task 2.1

---

### Task 2.3: Implement Soft Scoring
**File**: New file `src/translation/scoring.py`
**Estimated effort**: 5-6 hours

- [x] Implement method `score_decomposition(decomp: Decomposition, evidence: WordEvidence, weights: ScoringWeights) -> float`
- [x] Composite score formula: `score = w1 * beam_prior + w2 * avg_cluster_quality + w3 * residual_coverage + w4 * acceptance_bonus`
- [x] **w1 (beam_prior)**: Normalized TF-IDF score from beam search (default: 0.3)
- [x] **w2 (avg_cluster_quality)**: Mean of (cohesion + semantic_coverage) / 2 for all morphs (default: 0.25)
- [x] **w3 (residual_coverage)**: 1.0 - residual_ratio (default: 0.25)
- [x] **w4 (acceptance_bonus)**: Count of accepted clusters + 0.5 * accepted residuals + 0.3 * accepted hypotheses (default: 0.2)
- [x] Create `ScoringWeights` dataclass with configurable weights

**Testing**:
- High beam_prior (5.0) + good cluster quality (0.8) + full coverage (1.0) + 3 accepted clusters → score ~7.5
- Low beam_prior (1.0) + poor cluster quality (0.3) + partial coverage (0.5) + 0 accepted → score ~1.8
- Verify weights sum to 1.0 (normalize internally if needed)

**Dependencies**: Task 2.2

---

## PHASE 3: Strategy & Ranking

### Task 3.1: Implement Strategy Selection
**File**: New file `src/translation/strategies.py`
**Estimated effort**: 4-6 hours

- [x] Implement `apply_strategy(decompositions: list[tuple[Decomposition, float]], strategy: str, evidence: WordEvidence) -> list[tuple[Decomposition, float]]`
- [x] **Strategy: prefer-fewer** - Add `bonus = -0.5 * len(morphs)` to score (favor fewer morphs)
- [x] **Strategy: prefer-known** - Add `bonus = 0.3 * (count of morphs with uses > 5)` (favor well-attested morphs)
- [x] **Strategy: prefer-balance** - Compute length variance, add `bonus = -0.2 * variance` (favor similar morph lengths)
- [x] Return re-ranked list sorted by final score

**Testing**:
- "prefer-fewer" with [["NAZ", "PSAD"], ["N", "A", "Z", "P", "S", "A", "D"]] → ranks 2-morph higher
- "prefer-known" with [["NAZ", "PSAD"], ["NAZP", "SAD"]] where NAZ/PSAD have 10 uses, NAZP/SAD have 1 → ranks first higher
- "prefer-balance" with [["NAZ", "PSAD"], ["NAZPSA", "D"]] → ranks first higher (lengths 3,5 vs 6,1)

**Dependencies**: Task 2.3

---

### Task 3.2: Implement Top-K Selection
**File**: `src/translation/strategies.py`
**Estimated effort**: 2-3 hours

- [x] Implement `select_top_k(ranked: list[tuple[Decomposition, float]], k: int = 3) -> list[dict]`
- [x] Return top-K decompositions with full metadata (rank, morphs, score, meanings, breakdown, warnings)
- [x] Include warning "alternate decomposition exists" if delta < 0.05 between top-2 scores
- [x] Extract meanings for each morph from evidence bundle

**Testing**:
- 5 candidates, k=3 → returns 3
- Top-2 scores are 7.8 and 7.79 → includes warning "alternate decomposition exists"
- Single candidate → returns list of 1 with no warnings

**Dependencies**: Task 3.1

---

## PHASE 4: LLM Synthesis

### Task 4.1: Implement LLM Synthesis Adapter
**File**: New file `src/translation/llm_synthesis.py`
**Estimated effort**: 6-8 hours

- [x] Implement `synthesize_definition(morphs: list[str], meanings: list[str], context: dict) -> dict`
- [x] Use existing LLM infrastructure from `enochian_lm.root_extraction.tools` (e.g., `solo_analysis_engine.py` patterns)
- [x] Build prompt including: morph decomposition, individual meanings, semantic synthesis request, context (coverage_ratio, residual_ratio, provenance)
- [x] Handle LLM failures gracefully (return concatenated_meanings as fallback)
- [x] Return dict with: `synthesized_definition`, `concatenated_meanings`, `confidence`, `reasoning`

**Testing**:
- Mock LLM response with test inputs → verify output schema
- Integration test with real LLM (manual, not automated)
- Handle LLM failures gracefully (return concatenated_meanings as fallback)

**Dependencies**: None

---

### Task 4.2: Implement LLM Toggle Logic
**File**: `src/translation/service.py` (new `SingleWordTranslationService`)
**Estimated effort**: 2-3 hours

- [x] Add CLI flag: `--llm` (enables synthesis), `--no-llm` (default, skip synthesis)
- [x] When enabled, call `llm_synthesis.synthesize_definition()` for top-ranked candidate
- [x] When disabled, return concatenated meanings directly
- [x] Create `SingleWordTranslationService` class to orchestrate the full pipeline

**Testing**:
- `--llm` flag set → `synthesized_definition` populated
- `--no-llm` flag set → `synthesized_definition` is None, `concatenated_meanings` shown instead
- LLM fails → gracefully degrade to concatenated_meanings with warning

**Dependencies**: Task 4.1, Task 3.2

---

## PHASE 5: CLI & Output Formatting

### Task 5.1: Implement CLI Entry Point
**File**: New file `src/translation/cli_word.py`
**Estimated effort**: 5-6 hours

- [ ] Create argument parser with all flags:
  - `word` (positional): Single word to translate
  - `--variant`: choices=["solo", "debate", "both"], default="both"
  - `--strategy`: choices=["prefer-fewer", "prefer-known", "prefer-balance"], default="prefer-balance"
  - `--llm`: Enable LLM synthesis
  - `--output`: JSON output file (default: stdout)
  - `--pretty`: Pretty-print JSON
  - `--top-k`: Number of candidate definitions (default: 3)
- [ ] Wire config using `get_config_paths()` to resolve DB paths
- [ ] Load solo DB if `--variant solo` or `--variant both`
- [ ] Load debate DB if `--variant debate` or `--variant both`
- [ ] Instantiate `SingleWordTranslationService` and execute translation

**Testing**:
- `enlm translate-word NAZPSAD` → returns JSON with 3 candidates
- `enlm translate-word NAZPSAD --variant solo --strategy prefer-fewer --top-k 1` → returns 1 candidate from solo DB
- `enlm translate-word XYZABC` → returns empty with "no evidence found" message

**Dependencies**: All previous tasks

---

### Task 5.2: Implement Text Output Formatter
**File**: `src/translation/cli_word.py`
**Estimated effort**: 3-4 hours

- [ ] Implement default text format for terminal display (80-char width wrapping)
- [ ] Add flag `--format json` for machine-readable output
- [ ] Text format shows: word, variant, ranked definitions with morphs/meanings/synthesized/coverage/warnings
- [ ] JSON format follows schema defined in Architecture Overview
- [ ] Pretty-print JSON when `--pretty` flag is set

**Testing**:
- Text format renders correctly in terminal (80-char width wrapping)
- JSON format is valid and pretty-prints with `--pretty`

**Dependencies**: Task 5.1

---

### Task 5.3: Handle Edge Cases
**File**: `src/translation/cli_word.py`
**Estimated effort**: 3-4 hours

- [ ] **No evidence**: Return message "No direct evidence found. Showing FastText neighbors as heuristic."
- [ ] **Residual-only**: Mark sense with `"provenance_note": "residual-only (observed as remainder)"`, lower confidence by 0.2
- [ ] **Conflicting variants**: When `--variant both`, show side-by-side: `[{"variant": "solo", ...}, {"variant": "debate", ...}]`
- [ ] **Exit codes**: 0 (success), 1 (no evidence but FastText fallback), 2 (error: DB missing, invalid word)
- [ ] Add helpful error messages for missing DB files

**Testing**:
- Word with no evidence → exit code 1, shows FastText neighbors
- Word only in residuals → sense marked "residual-only", confidence lowered by 0.2
- `--variant both` with conflicting definitions → both listed side-by-side
- Missing DB file → exit code 2, helpful error message

**Dependencies**: Task 5.2

---

## PHASE 6: Integration & Documentation

### Task 6.1: Wire CLI into `enlm` Entry Point
**File**: `src/enochian_lm/cli.py` or `pyproject.toml`
**Estimated effort**: 1-2 hours

- [ ] Add subcommand `translate-word` to main `enlm` CLI parser
- [ ] Ensure `poetry run enlm translate-word NAZPSAD` invokes `cli_word.py`
- [ ] Verify help text displays correctly: `poetry run enlm translate-word --help`

**Testing**:
- `poetry run enlm translate-word --help` → shows help text
- `poetry run enlm translate-word NAZPSAD` → executes successfully

**Dependencies**: Task 5.1

---

### Task 6.2: Update Documentation
**File**: `README.md` and new file `docs/single_word_translation.md`
**Estimated effort**: 3-4 hours

- [ ] Add usage examples to README under "Single-Word Translation" section
- [ ] Create `docs/single_word_translation.md` with:
  - Architecture overview (data flow diagram)
  - Evidence sources explanation (clusters, residuals, hypotheses, FastText)
  - Strategy descriptions (prefer-fewer, prefer-known, prefer-balance)
  - Example commands with expected outputs
  - JSON schema reference
  - Edge cases and error handling
- [ ] Ensure examples are copy-pasteable and functional

**Testing**:
- Documentation renders correctly in GitHub
- Examples are copy-pasteable and work

**Dependencies**: All previous tasks

---

## PHASE 7: Evaluation & Testing

### Task 7.1: Assemble Gold Test Set
**File**: New file `tests/translation/gold_words.json`
**Estimated effort**: 3-4 hours

- [ ] Create 10-15 curated test cases:
  - 3 well-attested words (e.g., NAZ, IAD, BLIOR)
  - 3 sparse evidence words (only residuals)
  - 3 ambiguous words (multiple valid decompositions)
  - 2 unknown words (no evidence, FastText fallback)
  - 2 complex compounds (3+ morphs)
- [ ] Document expected morphs, meaning keywords, and notes for each
- [ ] Manually validate that each word has known cluster/residual entries in DB

**Testing**:
- Manual validation against solo/debate DBs

**Dependencies**: None

---

### Task 7.2: Implement Unit Tests
**File**: New file `tests/translation/test_single_word_service.py`
**Estimated effort**: 6-8 hours

- [ ] Test `fetch_word_evidence()` with mocked DB connections
- [ ] Test hard filters with constructed decompositions
- [ ] Test soft scoring with known weights and expected scores
- [ ] Test strategy selection with edge cases (ties, empty lists, single candidate)
- [ ] Mock LLM adapter to avoid external API calls
- [ ] Achieve >80% code coverage for new files

**Testing**:
- `pytest tests/translation/test_single_word_service.py -v --cov=src/translation`

**Dependencies**: All Phase 2-4 tasks

---

### Task 7.3: Implement Integration Tests
**File**: New file `tests/translation/test_word_cli_integration.py`
**Estimated effort**: 4-6 hours

- [ ] Test CLI with gold words against real solo/debate DBs
- [ ] Verify JSON output schema matches specification
- [ ] Test `--no-llm`, `--strategy` flags
- [ ] Test exit codes (success, no evidence, error)
- [ ] Test `--variant both` side-by-side output

**Testing**:
- `pytest tests/translation/test_word_cli_integration.py -v --slow`

**Dependencies**: Task 7.1, Task 5.3

---

### Task 7.4: Performance Validation
**File**: Manual test checklist
**Estimated effort**: 2-3 hours

- [ ] Time typical word lookup: target <500ms
- [ ] Batch test 100 words: target <30s total
- [ ] Verify memory usage stays <500MB for typical corpus sizes
- [ ] Test with empty analytics tables (should warn but not crash)
- [ ] Profile hotspots: `python -m cProfile -o profile.stats -m translation.cli_word NAZPSAD`
- [ ] Analyze with `python -m pstats profile.stats`

**Testing**:
- Run profiler and identify any bottlenecks >100ms

**Dependencies**: All previous tasks

---

## Summary of Deliverables

### New Files (9 total)
1. `src/translation/cli_word.py` - CLI entry point (5-6 hours)
2. `src/translation/decomposition.py` - Decomposition & filtering (9-12 hours)
3. `src/translation/scoring.py` - Soft scoring logic (5-6 hours)
4. `src/translation/strategies.py` - Strategy selection (6-9 hours)
5. `src/translation/llm_synthesis.py` - LLM adapter (6-8 hours)
6. `tests/translation/gold_words.json` - Test data (3-4 hours)
7. `tests/translation/test_single_word_service.py` - Unit tests (6-8 hours)
8. `tests/translation/test_word_cli_integration.py` - Integration tests (4-6 hours)
9. `docs/single_word_translation.md` - Documentation (3-4 hours)

### Modified Files (4 total)
1. `src/translation/repository.py` - Add single-word methods (6-9 hours)
2. `src/translation/service.py` - Add `SingleWordTranslationService` (2-3 hours)
3. `src/enochian_lm/cli.py` or `pyproject.toml` - Wire new command (1-2 hours)
4. `README.md` - Add usage examples (included in Task 6.2)

### Dependencies Between Phases
- **Phase 1 → Phase 2**: Evidence needed for decomposition
- **Phase 2 → Phase 3**: Decompositions needed for ranking
- **Phase 3 → Phase 4**: Ranked candidates needed for LLM synthesis
- **Phase 4 → Phase 5**: Synthesis needed for output formatting
- **Phase 5 → Phase 6**: CLI needed for integration
- **Phase 6 → Phase 7**: Full system needed for testing

### Estimated Total Effort
- **Phase 1**: 6-9 hours (data access & evidence)
- **Phase 2**: 14-18 hours (decomposition & filtering)
- **Phase 3**: 6-9 hours (strategy & ranking)
- **Phase 4**: 8-11 hours (LLM synthesis)
- **Phase 5**: 11-14 hours (CLI & output)
- **Phase 6**: 4-6 hours (integration & docs)
- **Phase 7**: 15-21 hours (testing & validation)
- **TOTAL**: 64-88 hours

### Success Criteria
- [ ] CLI command `enlm translate-word NAZPSAD` returns 3 ranked definitions
- [ ] All gold test words pass with expected decompositions
- [ ] `--variant both` shows side-by-side solo/debate results
- [ ] `--llm` flag synthesizes coherent definitions
- [ ] FastText fallback works for unknown words
- [ ] Unit test coverage >80% for new code
- [ ] Performance: <500ms per word, <30s for 100 words
- [ ] Documentation complete, accurate, and copy-pasteable