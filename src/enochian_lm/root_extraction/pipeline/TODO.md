# TODO: Residual Semantic Subtraction Refactor (Word Break)

## Goal
Implement the semantic subtraction protocol so we can explicitly perform a **word break** by subtracting a known root from a host word and analyzing the remaining residual (e.g., `NAZPSAD - NAZ = PSAD`, meaning **SWORD - RECTANGLE = SHARPNESS**).

## Step-by-step plan

1. **[x] Rename residual semantic engines to “semantic subtraction.”**
   - Rename `debate_residual_semantic_engine.py` → `debate_semantic_subtraction_engine.py` and update the primary entrypoint name (e.g., `debate_remainder` → `debate_semantic_subtraction`).
   - Rename `solo_residual_semantic_engine.py` → `solo_semantic_subtraction_engine.py` and update the main entrypoint name (e.g., `solo_analyze_remainder` → `solo_semantic_subtraction`).
   - Update imports in:
     - `root_extraction/pipeline/run_residual_semantic_extraction.py`
     - `root_extraction/main.py`
   - Update any tooling/documentation references (grep for old module names and update to new names).

2. **[x] Define the “word break” subtraction helper(s).**
   - Add/centralize a helper that:
     - Finds all occurrences of a root in a host word (case-normalized),
     - Removes the root to compute the residual,
     - Returns structured results: `{host_word, root, residual, start, end, residuals:[{artifact,start,end}]}`.
   - Confirm it handles multiple occurrences (e.g., if the root appears twice, return all residual options or select the most plausible).
   - For repeated-root hosts (e.g., `ANAZNAZ`), include ambiguity-aware options for both single-occurrence subtraction and all-occurrence subtraction so downstream steps can compare `ANAZ` vs `A`-style leftovers.
   - Confirm it handles split residual artifacts around the removed root (prefix/suffix remnants as distinct pieces) so partial explanatory segments can be chained in later passes (e.g., `ANAZNAZ - NAZ -> [A] + [NAZ]`).
   - Validate the NAZ example: `NAZPSAD - NAZ = PSAD`.

3. **[x] Update `run_residual_semantic_extraction.py` to use the word break helper.**
   - Replace/augment existing “residual fragments” logic with explicit `root + host → residual` word break outputs.
   - Ensure the residual statistics and counters reflect actual subtraction results rather than generic uncovered fragments.
   - Update the donor gloss collection flow (`_collect_donor_glosses_for_residual`) to use the same subtraction logic so residuals are grounded in `host - root`.
   - Pass the structured word break evidence into the downstream engines as part of the `residual_guidance` payload.

4. **[x] Update the debate engine prompt for semantic subtraction.**
   - In the renamed `debate_semantic_subtraction_engine.py`, update the prompt/formatting so the LLM sees:
     - the host word,
     - the root being subtracted,
     - the computed residual,
     - the explicit subtraction equation (e.g., `NAZPSAD - NAZ = PSAD`),
     - the intended semantics (e.g., **SWORD − RECTANGLE = SHARPNESS**).
   - Confirm output schema includes residual-specific fields (residual gloss, evidence, confidence, and the subtraction explanation).

5. **[x] Update the solo engine prompt for semantic subtraction.**
   - In the renamed `solo_semantic_subtraction_engine.py`, mirror the debate changes:
     - include the `host/root/residual` triple,
     - emphasize that the job is to interpret **residual** semantics via subtraction,
     - include the NAZ example in the prompt template.
   - Ensure JSON parsing / validation still works with the expanded schema.

6. **[x] Update any shared residual analysis utilities to support explicit word breaks + hierarchy recursion.**
   - Ensure `exclude_root_segments`/residual summaries surface explicit subtraction artifacts (`host/root/residual`, artifact spans), not only generic uncovered spans.
   - Add helper(s) that operationalize hierarchy traversal: dictionary-first donor selection, then largest accepted SQLite donor, then infix artifact branching, then recursive unresolved-artifact passes.
   - Standardize machine-readable payloads for each recursion step (selected donor source, subtraction equation, remaining artifacts, depth, termination reason).

7. **[x] Audit & harden the decomposition/scoring pipeline against over-segmentation bias (hierarchy-aware).**
   - Map the full pipeline with hierarchy stages explicitly separated (host discovery, donor selection order, infix branching, recursion).
   - Run a static audit for scoring bias that may over-reward additional fragments/recursion depth (`piece_count`, `depth`, and branch-count effects).
   - Add instrumentation per candidate and per recursion node: `piece_count`, `unknown_piece_count`, `recursion_depth`, `donor_source` (dictionary/sqlite), coverage/residual ratios, similarity contributions, and final score.
   - Run sampling experiments computing within-word correlations between score and (a) piece_count, (b) recursion_depth, (c) branch count; publish traceable report (see `docs/semantic_subtraction_step7_audit.md` for the current static audit + diagnostics baseline).
   - Implement one deterministic fix ensuring hierarchy-respecting neutrality (extra splits should not win unless they add evidence).

8. **[x] Add refactor-proof guardrails (tests + invariants) for hierarchy behavior.**
   - Add unit tests for donor-priority order: dictionary donor selected before SQLite donor when both are viable.
   - Add unit tests for infix branching + recursion termination (no infinite loops; deterministic termination reason recorded).
   - Keep scoring invariants: redundant splits/extra depth do not increase score without new evidence.
   - Add property/randomized regression checks for correlations between score and `piece_count`/`recursion_depth` above threshold.
   - Emit diagnostic metrics in verbose output: `piece_count_score_corr`, `depth_score_corr`, and donor-source usage ratios.

9. **[x] Add focused tests / fixtures (if feasible) for real hierarchy scenarios.**
   - Keep helper tests for `NAZPSAD - NAZ = PSAD`, repeated roots, and all-occurrence ambiguity.
   - Add integration-ish tests asserting residual guidance includes hierarchy traces: selected donor source, recursion depth, host/root/residual triples, subtraction equations, and remaining artifacts.
   - Add fixture-driven scenario for mixed known/unknown artifacts (e.g., ALNAZPSAD-like decomposition chain) validating dictionary-first then SQLite fallback behavior.

10. **[x] Documentation & runbook updates (hierarchy-first).**
   - Update docs/run instructions to describe the ordered hierarchy (dictionary → SQLite-largest → infix branching → recursion) and how to interpret ambiguous outputs.
   - Include worked examples for NAZPSAD and at least one recursive artifact chain (ALNAZPSAD-style) showing how residual semantics are inferred.
   - Document persisted fields/traceability expectations so downstream analysis can audit every subtraction decision.

---


### Word-break methodology (agreed hierarchy)
- Start with a hypothetical root (ngram).
- Find host word(s) containing the ngram (typically one or fewer).
- Use host-word definition(s) as semantic targets.
- Select subtractable known roots in this order:
  1. Dictionary-attested roots present in host and shorter than host (not identical host token).
  2. Largest accepted root from SQLite (style-specific table) that yields clean subtraction with no leftover letters.
  3. If the target root appears infix-wise, repeat (1) and (2) on left/right residual artifacts.
  4. Recurse on remaining unresolved artifacts, always preferring dictionary-attested pieces before SQLite-only accepted roots.
- Fill prompt templates with host/root/residual triples and explicit equations.
- Send to the selected engine (solo/debate), parse response JSON, and persist outputs to insights tables.
- Iterate until requested residual candidates are processed.


## Addendum (Post-Step-10 Execution Requirements)

### Addendum A — Hierarchy helpers exist, but hierarchy execution is not yet wired into donor-selection recursion
**Intent:** Move from helper availability to actual hierarchy-driven execution in the residual orchestration path.

**Required implementation scope:**
- In `src/enochian_lm/root_extraction/pipeline/run_residual_semantic_extraction.py`, refactor `_collect_donor_glosses_for_residual` and all related residual-evidence assembly so donor candidates are built explicitly per node and then ranked through `prioritize_donor_candidates(...)`.
- Apply subtraction in the agreed hierarchy order at every recursion node:
  1. dictionary-attested donor roots,
  2. largest accepted SQLite donor root,
  3. infix artifact branching,
  4. recursive resolution of unresolved artifacts.

**Traceability requirements:**
- For each recursion node, emit normalized trace rows via `build_subtraction_evidence(...)` with non-placeholder values for:
  - `donor_source`,
  - `recursion_depth`,
  - `termination_reason`.
- Replace hardcoded/constant trace metadata with real per-node values from hierarchy traversal.

**Safety/termination requirements:**
- Add recursion protections:
  - visited-state tracking (e.g., `(host, residual, donor)` tuples),
  - maximum recursion depth,
  - deterministic termination reasons (`resolved`, `no_viable_donor`, `cycle_detected`, `max_depth_reached`, etc.).

**Persistence requirements:**
- Persist full per-step traces into `analytics_summary` so downstream prompts (solo/debate) consume actual hierarchy paths instead of flattened/partial summaries.

---

### Addendum B — Tests are mostly utility-level; add true pipeline-to-engine integration tests for guidance contract
**Intent:** Verify the orchestration layer passes the full guidance contract to engines before real runs.

**Required implementation scope:**
- Add an integration-ish test module in `tests/root_extraction/` that:
  - mocks/patches `solo_semantic_subtraction` and `debate_semantic_subtraction`,
  - executes a minimal pipeline path for one ngram in each mode,
  - asserts the exact `residual_guidance` payload sent to engines.

**Contract assertions (minimum):**
- `word_breaks` contains structured `host/root/residual` rows.
- `subtraction_equations` is present and consistent with `word_breaks`.
- Hierarchy trace metadata is present when hierarchy recursion is enabled:
  - `donor_source`,
  - `recursion_depth`,
  - `termination_reason`.

**Coverage requirements:**
- Include five northstar-style cases (`NAZPSAD`, `NAZ`, `PSAD`, with dictionary-backed host and donor roots).
- Include five infix/fragment cases to validate artifact-branch handling.

---

### Addendum C — Guardrail coverage can silently degrade when optional deps are missing
**Intent:** Ensure key score-guardrail tests run in constrained environments (including CI) without optional heavy deps.

**Required implementation scope:**
- Refactor `tests/root_extraction/test_candidate_finder_guardrails.py` into two strata:
  1. dependency-free tests (must always run),
  2. optional dependency tests (may be skipped with explicit marker).

**Dependency-free subset must include:**
- `_score_with_bonus` invariants (redundant split neutrality / no bonus on incomplete coverage).
- Correlation helper behavior sanity checks (`piece_count_score_corr` style checks with deterministic synthetic data).

**CI requirement:**
- Ensure CI executes the dependency-free subset unconditionally (no global module skip should suppress core guardrail assertions).

---

### Addendum D — Orchestration comments/docs must explain decision logic, not just utilities
**Intent:** Make orchestration behavior auditable and maintainable without relying on chat context.

**Required implementation scope:**
- In `src/enochian_lm/root_extraction/pipeline/run_residual_semantic_extraction.py`, add concise docstrings/comments at donor discovery and residual evidence assembly points, especially:
  - `_collect_donor_glosses_for_residual`,
  - the word-break evidence assembly block.

**Documentation content requirements:**
- Explicit hierarchy order and rationale.
- Ambiguity handling policy (single-occurrence vs all-occurrence cases).
- Recursion termination criteria and cycle protection notes.
- Persisted fields required for traceability (what is stored and why).

**Style requirement:**
- Keep comments colocated with branching logic and avoid broad, detached file-level prose.

---

### Addendum E — Build richer dictionary-driven test cases now
**Intent:** Validate subtraction behavior on real corpus-like inputs, not only synthetic strings.

**Required implementation scope:**
- Add a test module that reads `src/enochian_lm/root_extraction/data/dictionary.json` and creates a deterministic sample of host/root pairs where root is a strict substring of host (`host != root`).

**Invariant assertions (minimum):**
- At least one subtraction candidate exists for sampled pairs.
- Equations and artifact spans are internally consistent.
- No malformed indices (`start/end` bounds, ordering).
- Repeated-root hosts expose ambiguity options (including all-occurrence alternatives).

**Stability requirement:**
- Include pinned cases (`NAZPSAD` plus 2–3 additional explicit dictionary-backed pairs) so behavior remains stable even if sampling logic evolves.

---

### Addendum F — Unblock debate-mode orchestration in the main pipeline entrypoint
**Intent:** Ensure semantic-subtraction processing can run end-to-end in both supported modes (`solo` and `debate`) from the same orchestration path.

**Background context for implementers:**
- `evaluate_ngram(...)` already contains branching logic for both `solo_semantic_subtraction(...)` and `debate_semantic_subtraction(...)`.
- `process_ngrams(...)` still has a temporary guard that raises unless `style == "solo"`, which prevents debate-mode batch runs and makes debate support partially unreachable.
- This mismatch creates false confidence in tests that call `evaluate_ngram(...)` directly but do not exercise full queue-based orchestration.

**Required implementation scope:**
- In `src/enochian_lm/root_extraction/pipeline/run_residual_semantic_extraction.py`, remove/replace the temporary solo-only style guard in `process_ngrams(...)`.
- Validate style values explicitly (`solo`/`debate`) and raise only on unsupported styles.
- Update stale comments/messages that claim debate is inaccessible so operational text matches real behavior.

**Validation requirements:**
- Add a regression test under `tests/root_extraction/` that runs a minimal mocked `process_ngrams(..., style="debate")` flow.
- Assert it no longer raises the solo-only `ValueError` and that debate mode reaches evaluation dispatch.

---

### Addendum G — Preserve multiple viable donor branches per recursion node
**Intent:** Capture ambiguity instead of collapsing too early, especially for repeated-root and infix-heavy host words.

**Background context for implementers:**
- Current donor recursion ranks candidates but returns after the first successful subtraction path.
- This behavior can hide legitimate alternatives (`A`, `ANAZ`, `A+NAZ` style ambiguity) that should remain available for downstream semantic reasoning.
- The project northstar favors auditable subtraction alternatives over single-path guesses.

**Required implementation scope:**
- Refactor `_resolve_donor_hierarchy(...)` in `run_residual_semantic_extraction.py` to evaluate more than one viable ranked donor branch per node.
- Introduce a configurable branch cap (e.g., `MAX_DONOR_BRANCHES_PER_NODE`) to bound cost while preserving ambiguity.
- For each retained branch:
  - emit its own `build_subtraction_evidence(...)` trace row,
  - recurse independently using branch-local `visited` state,
  - preserve donor-source provenance and depth metadata.

**Safety requirements:**
- Keep deterministic ordering so repeated runs produce stable traces.
- Ensure branch expansion still honors recursion-depth and cycle protections.

**Validation requirements:**
- Add tests proving at least two competing donor paths survive for an ambiguous host case.
- Assert both branches appear in `analytics_summary["word_breaks"]` and `analytics_summary["hierarchy_traces"]`.

---

### Addendum H — Emit explicit terminal traces for non-success recursion exits
**Intent:** Make every recursion stop condition auditable so investigators can explain why traversal ended.

**Background context for implementers:**
- Some recursion exits currently return silently (`empty token`, `max depth`, `no candidates`, cycle skip).
- Without terminal rows, post-hoc diagnostics cannot distinguish “resolved cleanly” from “stopped early due to guardrails.”

**Required implementation scope:**
- In `_resolve_donor_hierarchy(...)`, emit a terminal trace row for each non-success exit condition.
- If `build_subtraction_evidence(...)` cannot represent terminal-only events cleanly, add a companion helper for terminal trace payload normalization and keep fields aligned with existing evidence schema.

**Terminal reasons that must be represented:**
- `empty_token`
- `no_viable_donor`
- `cycle_detected`
- `max_depth_reached`

**Trace schema requirements (minimum):**
- include `donor_source` (or `none` when not applicable),
- include `recursion_depth`,
- include concrete `termination_reason`,
- include host/token identifiers sufficient to locate the failing node.

**Validation requirements:**
- Extend `tests/root_extraction/test_hierarchy_donor_resolution.py` with focused cases for each termination reason.
- Assert a terminal trace row is present for each case.

---

### Addendum I — Add true hierarchy integration coverage (not only payload contract stubs)
**Intent:** Verify real hierarchy execution (dictionary → sqlite → infix → recursion) produces guidance, rather than only validating prebuilt payload wiring.

**Background context for implementers:**
- `test_pipeline_engine_guidance_contract.py` intentionally stubs deep internals, including donor collection logic.
- That test is useful for API-shape contracts, but it does not prove real traversal decisions are functioning.

**Required implementation scope:**
- Create a new integration-style test in `tests/root_extraction/` that does **not** stub `_collect_donor_glosses_for_residual`.
- Use a controlled mini fixture setup:
  - mock dictionary lookups for selected tokens,
  - mock sqlite accepted-gloss loading,
  - configure one host with infix/fragment residual artifacts.
- Run a minimal `evaluate_ngram(...)` path and inspect emitted `residual_guidance`.

**Assertions required:**
- donor-source ordering follows project hierarchy (`dictionary` before `sqlite` when both viable),
- recursion metadata is present and non-placeholder,
- resulting guidance rows are produced by real resolver execution (not injected rows).

---

### Addendum J — Persist semantic-subtraction evidence into insights DB tables
**Intent:** Close the gap between analysis-time evidence and durable records used for audits, downstream scripts, and reruns.

**Background context for implementers:**
- Current pipeline assembles rich `analytics_summary` evidence (`word_breaks`, `subtraction_equations`, `hierarchy_traces`) but large insert blocks are commented out.
- Committing transactions without storing this structured evidence weakens reproducibility and violates the methodology requirement to record results in DB structures.

**Required implementation scope:**
- In `run_residual_semantic_extraction.py`, restore/implement DB writes for semantic-subtraction outputs.
- Define a stable mapping from `analytics_summary` fields into normalized DB rows (host/root/residual/equation/source/depth/termination).
- Use existing schema modules/scripts in `src/enochian_lm/root_extraction/scripts/` as the source of truth; if schema extensions are required, update migrations/initialization accordingly.

**Data-integrity requirements:**
- Deduplicate trace inserts where appropriate (idempotent reruns).
- Preserve linkage keys (run id, ngram/root token, cluster id where applicable).
- Handle malformed/partial analytics rows defensively without crashing the run.

**Validation requirements:**
- Add a DB-focused test that executes one ngram path and verifies persisted rows include:
  - `host_word`, `root`, `residual`, `equation`,
  - `donor_source`, `recursion_depth`, `termination_reason`.

---

### Addendum K — Document ambiguity policy at subtraction decision points
**Intent:** Make local decision rules explicit where ambiguity is actually resolved, so future maintainers and LLM agents do not misinterpret policy.

**Background context for implementers:**
- High-level docs mention ambiguity, but key selection points in orchestration do not always state why one option is chosen first.
- Cases like `ANAZNAZ` require clarity on:
  - single-occurrence subtraction selection,
  - retention of `remove_all_occurrences` alternatives,
  - where competing interpretations remain available downstream.

**Required implementation scope:**
- In `run_residual_semantic_extraction.py`, add concise comments/docstrings exactly where subtraction candidates are ranked/selected and where ambiguity payloads are forwarded.
- Explicitly document:
  1. what option is selected for immediate recursion,
  2. why that option is preferred (determinism/performance/coverage rationale),
  3. where alternative interpretations are preserved in guidance.

**Validation requirements:**
- Add a focused test (or extend existing guidance tests) asserting:
  - current primary-selection policy,
  - continued presence of alternative interpretations in guidance payload fields (`remove_all_occurrences`, additional traces, or equivalent).

---

### Recommended Processing Order for Addendums F–K
Process these in dependency-aware order:
1. **F** — unblock debate orchestration entrypoint.
2. **G** — preserve multi-branch donor recursion.
3. **H** — emit terminal traces for all non-success exits.
4. **K** — document ambiguity policy at exact decision points (after branching behavior stabilizes).
5. **I** — add integration test that exercises real hierarchy execution.
6. **J** — finalize DB persistence mapping and persistence tests.

**Why this order:** F/G/H/K define runtime behavior and observability; I should verify the finalized behavior; J should persist stable, already-validated payload structures.

---

### Notes / Acceptance Criteria
- The semantic subtraction flow should **always** be framed as `HOST - ROOT = RESIDUAL`.
- Downstream LLM prompts must explicitly spell out the subtraction result and its semantic interpretation.
- Output data should preserve the subtraction equation for traceability (e.g., stored in logs or database outputs).
