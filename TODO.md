# Interpretation Workflow TODO

This checklist tracks the new post-run interpretation pipeline that consumes the
solo/debate insights databases.

- [x] CLI entry point for programmatic interpretation (`poetry run enochian-interpret`).
- [x] Insights repository helper to query clusters, residuals, and definitions.
- [x] Tokenization helpers for sentence-to-ngrams expansion (1–7 characters).
- [x] Candidate reconciliation logic that merges database hits with
      `MorphemeCandidateFinder` residual breakdowns.
- [ ] Sentence-level reconciliation loop that iteratively swaps alternate
      definitions to minimize residual coverage gaps.
- [ ] Agent hand-off surface for optional qualitative review of reconciled
      definitions.

Each commit should tick the relevant box as functionality lands.
