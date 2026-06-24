# Translation and Interpretation Knobs to Tweak

This document lists the main parameters that can be tuned when improving root
group diagnostics, single-word translation, and phrase interpretation. Treat
these as experiment knobs: change one family at a time, keep before/after JSON,
and preserve tests for both improvements and guarded regressions.

## Root Group Report Knobs

Source: `src/translation/root_groups.py`

### Grouping thresholds

- `merge_similarity_threshold` default `0.78`
  - Raise to split broad short-root bundles more aggressively.
  - Lower to merge near-duplicate groups.
- `strong_merge_similarity_threshold` default `0.86`
  - Raise to let fewer conflicted pairs remain in one group.
  - Lower to allow very similar pairs to survive conflicts as
    `needs_review`/`split_recommended`.
- `max_groups_default` / CLI `--max-groups` default `12`
  - Lower for compact inspection.
  - Raise when studying short roots such as `A`, `I`, `D`, or `IO`.

### Conflict thresholds

- `attachment_axis_divergence_threshold` default `0.35`
  - Raise if prefix/suffix/free differences are over-splitting groups.
  - Lower if positional behavior is being blurred.
- `observed_attachment_ratio_divergence_threshold` default `0.45`
  - Tune when observed counts disagree with likelihoods.
- `exception_density_needs_review_threshold` default `0.25`
  - Lower if risky clusters should be flagged sooner.
  - Raise if nearly every short-root group becomes `needs_review`.
- `low_confidence_threshold` default `0.65`
  - Raise for stricter stability.
  - Lower if useful low-confidence groups are too noisy in diagnostics.

### Semantic text weights

- `semantic_core`: `3.0`
- `nested_effect`: `3.0`
- `nested_sense`: `2.0`
- `definition`: `2.0`
- `decoding_guide`: `1.0`
- `examples`: `1.0`

Raise `nested_effect` or `nested_sense` if cases like `IO`/`OHIO` need stronger
protection from generic definitions. Raise `definition` if group labels become
too evidence-fragment driven.

## Root Group Alignment Knobs

Source: `DEFAULT_ALIGNMENT_PARAMETERS` in `src/translation/root_groups.py`

- `source_cluster_id_match_weight` default `0.45`
  - Highest-value signal. Lower only if source cluster traces are known to be
    noisy.
- `nested_evidence_word_match_weight` default `0.25`
  - Raise when target words in nested evidence should dominate local alignment.
- `attachment_role_match_weight` default `0.15`
  - Raise if prefix/suffix/free behavior should drive disambiguation.
- `semantic_overlap_weight` default `0.10`
  - Keep small. This is a tie-breaker, not a provenance replacement.
- `group_rank_weight` default `0.05`
  - Small prior from the group’s diagnostic rank.
- `needs_review_penalty` default `0.08`
- `split_recommended_penalty` default `0.12`
- `provisional_penalty` default `0.02`
  - Raise penalties if risky groups are becoming too influential.

## Word Translation Knobs

Source: `src/translation/service.py`, `src/translation/scoring.py`,
`src/translation/strategies.py`

### CLI flags

- `--strategy prefer-fewer|prefer-known|prefer-balance`
  - `prefer-fewer`: favors chunkier decompositions.
  - `prefer-known`: favors morphs with stronger evidence counts.
  - `prefer-balance`: default balance of chunkiness and variance.
- `--evidence-mode all|clusters-only|residuals-only`
  - Use `clusters-only` to test stricter accepted-cluster behavior.
  - Use `residuals-only` to inspect remainder semantics separately.
- `--top-k`
  - Controls returned candidates and Phase 3 root-group diagnostics.
- `--fallback-top-n`
  - Controls provisional fallback breadth when hard filters remove candidates.
- `--allow-whole-word` / `--no-whole-word`
  - Blind retranslation mode is useful for forcing decomposition.
- `--with-root-groups`
  - Adds diagnostics only.
- `--use-root-groups-for-ranking`
  - Experimental. Adds the bounded Phase 4 alignment component.

### Decision weights

- `DECISION_SEMANTIC_WEIGHT` default `0.22`
- `DECISION_ATTESTATION_WEIGHT` default `0.14`
- `FNP_DECISION_WEIGHT` default `0.08`
- `DECISION_SINGLETON_BURDEN_WEIGHT` default `0.08`
- `ROOT_GROUP_DECISION_WEIGHT` default `0.06`

Raise `ROOT_GROUP_DECISION_WEIGHT` only after fixtures show that alignment
improves ranking without allowing short roots to explain everything. Keep it
small relative to baseline evidence signals.

### Tie-breaks and blind mode

- `DECISION_FEWER_MORPHS_TIE_MARGIN` default `0.10`
- `DECISION_SINGLETON_CLEAR_WIN_MARGIN` default `0.18`
- `BLIND_RETRANSLATION_SHORT_ROOT_MAX_LEN` default `4`

These protect against singleton-heavy decompositions and whole-word anchors
dominating blind decomposition experiments.

## Phrase Translation Knobs

Source: `src/translation/phrase_service.py`

### CLI flags

- `--top-k`
  - Controls token candidates and parse candidates retained.
- `--with-root-groups`
  - Adds chosen-parse root group diagnostics after parse selection.
- `--use-root-groups-for-ranking`
  - Passes the ranking experiment through token-level word translation.
- `--llm`
  - Enables phrase rendering; root groups enter render payloads only when
    `--with-root-groups` is also enabled.
- `--llm-unknown-context`
  - Adds the unknown-token refinement pass after parse selection.
- `--no-memory-update`
  - Disables provisional translation-memory writes.

### Parse scoring families

Phrase parse ranking combines token candidate score, confidence, semantic bundle
coherence, adjacent-token relations, FNP grammar sequence, and penalties for
weak/provisional readings. Tune phrase ranking only after saving parse candidate
diagnostics for the same phrase before and after the change.

## Experiment Discipline

- Prefer JSON output with `--pretty` and `--verbose` for before/after snapshots.
- For root groups, compare `diagnostics.rejected_merges`, group statuses, and
  `source_cluster_ids`.
- For word ranking, compare `diagnostics.root_group_ranking.baseline_rows` to
  `diagnostics.decision_rows`.
- For phrases, compare `chosen_parse`, `translation_skeleton`, and
  `chosen_parse.token_choices[*].root_group_alignments`.
- Add at least one positive fixture and one guarded regression fixture before
  changing defaults.
