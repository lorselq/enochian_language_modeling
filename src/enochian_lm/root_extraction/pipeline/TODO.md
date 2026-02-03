# TODO: Residual Semantic Subtraction Refactor (Word Break)

## Goal
Implement the semantic subtraction protocol so we can explicitly perform a **word break** by subtracting a known root from a host word and analyzing the remaining residual (e.g., `NAZPSAD - NAZ = PSAD`, meaning **SWORD - RECTANGLE = SHARPNESS**).

## Step-by-step plan

1. **Rename residual semantic engines to “semantic subtraction.”**
   - Rename `debate_residual_semantic_engine.py` → `debate_semantic_subtraction_engine.py` and update the primary entrypoint name (e.g., `debate_remainder` → `debate_semantic_subtraction`).
   - Rename `solo_residual_semantic_engine.py` → `solo_semantic_subtraction_engine.py` and update the main entrypoint name (e.g., `solo_analyze_remainder` → `solo_semantic_subtraction`).
   - Update imports in:
     - `root_extraction/pipeline/run_residual_semantic_extraction.py`
     - `root_extraction/main.py`
   - Update any tooling/documentation references (grep for old module names and update to new names).

2. **Define the “word break” subtraction helper(s).**
   - Add/centralize a helper that:
     - Finds all occurrences of a root in a host word (case-normalized),
     - Removes the root to compute the residual,
     - Returns structured results: `{host_word, root, residual, start, end}`.
   - Confirm it handles multiple occurrences (e.g., if the root appears twice, return all residual options or select the most plausible).
   - Validate the NAZ example: `NAZPSAD - NAZ = PSAD`.

3. **Update `run_residual_semantic_extraction.py` to use the word break helper.**
   - Replace/augment existing “residual fragments” logic with explicit `root + host → residual` word break outputs.
   - Ensure the residual statistics and counters reflect actual subtraction results rather than generic uncovered fragments.
   - Update the donor gloss collection flow (`_collect_donor_glosses_for_residual`) to use the same subtraction logic so residuals are grounded in `host - root`.
   - Pass the structured word break evidence into the downstream engines as part of the `residual_guidance` payload.

4. **Update the debate engine prompt for semantic subtraction.**
   - In the renamed `debate_semantic_subtraction_engine.py`, update the prompt/formatting so the LLM sees:
     - the host word,
     - the root being subtracted,
     - the computed residual,
     - the explicit subtraction equation (e.g., `NAZPSAD - NAZ = PSAD`),
     - the intended semantics (e.g., **SWORD − RECTANGLE = SHARPNESS**).
   - Confirm output schema includes residual-specific fields (residual gloss, evidence, confidence, and the subtraction explanation).

5. **Update the solo engine prompt for semantic subtraction.**
   - In the renamed `solo_semantic_subtraction_engine.py`, mirror the debate changes:
     - include the `host/root/residual` triple,
     - emphasize that the job is to interpret **residual** semantics via subtraction,
     - include the NAZ example in the prompt template.
   - Ensure JSON parsing / validation still works with the expanded schema.

6. **Update any shared residual analysis utilities to support explicit word breaks.**
   - If `exclude_root_segments` or residual summaries are used to infer residuals, ensure they now surface the explicit subtraction residual rather than generic uncovered spans.
   - Add helper(s) in `root_extraction/utils/residual_analysis.py` (or a new module) if needed to normalize and report subtraction results consistently.

7. **Audit & harden the decomposition/scoring pipeline against over-segmentation bias.**
   - Map the full pipeline (word → segmentations → candidate meanings → score → ranking output) with explicit code pointers.
   - Run a **static audit** for similarity pooling, normalization order, additive scoring, coverage interactions, and pruning that could reward “more pieces.” Capture yes/no answers with file/function references.
   - Add **instrumentation** to log per-candidate diagnostics: `piece_count`, `unknown_piece_count`, coverage/residual ratios, vector norms (pre/post pooling), each similarity contribution, final score, and weighting flags.
   - Run a **sampling experiment** (top-N candidates per word) and compute within-word correlations between `piece_count` and overall score + similarity components. Save a short report in `runs/` or `docs/` for traceability.
   - Propose at least 3 deterministic, explainable fixes (e.g., normalize similarity by piece_count, pooling standardization, evidence-weighted pooling) and decide on one to implement with tests.

8. **Add refactor-proof guardrails (tests + invariants).**
   - Add **2 unit tests** that assert: (a) adding a redundant split does not raise score unless it adds new evidence, (b) normalization/penalties keep piece_count-neutral scores for equivalent coverage.
   - Add **1 property-based or randomized regression test** to flag correlations between piece_count and score above a threshold.
   - Add a **diagnostic metric** (e.g., `piece_count_score_corr`) to verbose output/diagnostics so regressions are visible in runs.

9. **Add focused tests / fixtures (if feasible).**
   - Add a unit test for the word break helper to guarantee `NAZPSAD - NAZ = PSAD`.
   - Add an integration-ish test in `tests/` to validate that residual guidance passed to engines includes `host/root/residual` and the subtraction equation.

10. **Documentation & runbook updates.**
   - Update any docs or run instructions that mention “residual semantic” engines to “semantic subtraction.”
   - Mention the new word break behavior and include the NAZ example in the README or relevant docs.

---

### Notes / Acceptance Criteria
- The semantic subtraction flow should **always** be framed as `HOST - ROOT = RESIDUAL`.
- Downstream LLM prompts must explicitly spell out the subtraction result and its semantic interpretation.
- Output data should preserve the subtraction equation for traceability (e.g., stored in logs or database outputs).
