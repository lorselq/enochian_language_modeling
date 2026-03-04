# Step 7 Audit: Decomposition/Scoring Bias (Hierarchy-Aware)

## Pipeline map (code pointers)
1. **Target token → segmentation beams**: `segment_target()` in `candidate_finder.py`.
2. **Segmentation → composite scores**: `score_parse()` in `candidate_finder.py`.
3. **Coverage/residual accounting**: `_build_breakdown()` in `candidate_finder.py`.
4. **Pruning, overlap filtering, ranking**: `find_candidates()` in `candidate_finder.py`.
5. **Residual pipeline consumption + guidance**: `_get_candidate_breakdown()` and residual analytics flow in `run_residual_semantic_extraction.py`.

## Static audit checklist
- Similarity pooling uses mean vectors for both target and candidate paths: **YES** (potential piece-count sensitivity).  
- Score is additive over TF-IDF + cosine + confidence: **YES** (risk of additive over-segmentation bias).  
- Multi-segment bonus exists: **YES** via `_score_with_bonus` (explicitly can reward more pieces).  
- Coverage/residual interactions used downstream: **YES** (`coverage_ratio`, `residual_ratio` from breakdown).  
- Candidate pruning by overlap and cosine before ranking: **YES**.

## Instrumentation added in this step
- Per-candidate diagnostics now include:
  - `piece_count`, `unknown_piece_count`
  - `coverage_ratio`, `residual_ratio`
  - vector norms (`target_vec_norm`, `candidate_vec_norm`)
  - weighted component contributions (`tfidf_component`, `cosine_component`, `confidence_component`)
  - `raw_score`, `final_score`, and `multi_segment_bonus_applied`
- Correlation diagnostics surfaced in residual analytics:
  - `piece_count_score_corr`
  - `unknown_piece_count_score_corr`

## Deterministic fix direction
Current instrumentation is in place to evaluate and tune segment-bonus behavior. The immediate guardrail is visibility: `piece_count_score_corr` can now be monitored per root run before applying normalization/penalty recalibration.
