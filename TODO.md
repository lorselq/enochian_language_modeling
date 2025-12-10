## Interpretation Workflow TODO


NOTE: what follows is what *used* to be in this section; consolidating later.

# enochian-lm — Next Steps (Translator/Tokenizer/LM) ✅

> Purpose: build a tiny, repeatable loop that segments text with your **dictionary-driven defs**, measures improvement with a **baseline LM**, and produces **run artifacts** (receipts). No Loagaeth here.

# THINGS TO CAPTURE from the current pipeline (so the LM loop benefits)

You already have most of this—just make sure it writes to disk in stable schemas.

## A. Definitions & senses

- Table: `sense(sense_id, lemma_id, gloss, tags, confidence, source, evidence_json, is_canon, created_at)`
- For `evidence_json`, include:
  - `examples` (short contexts), `ngram_origins`, `cluster_id`, `provenance` (hand, model, rule)

## B. N-gram ↔ sense linkages

- Table: `ngram_index(ngram, n, form_norm, candidate_sense_ids[], sim_scores[], freq, contexts_sample[])`
- This lets the segmentation engine quickly propose candidates.

## C. Variant spellings & normalization

- Table: `lemma_variants(lemma_id, variants[], rules_used)`
- Lets you explain strings with alternate paths when strict normalization fails.

## D. Residual reports (baseline)

- File: `baseline_residuals.jsonl` with per-sentence results from your current best attempt.
- Summary metrics:
  - `coverage%`, `residual_rate%`, `avg_confidence`, `top_conflicting_pairs` (sense A vs sense B co-occurrence conflicts)

## E. Prompt/response cache (if LLMs involved)

- Table: `llm_job(job_id, prompt_hash, request_json, response_json, model, tokens_in, tokens_out, cost_usd, status, created_at)`
- Never re-spend on the same prompt; always log costs.

## F. Context windows per lemma (for future audits)

- Table: `lemma_contexts(lemma_id, text_id, left_context, lemma_form, right_context, idx)`
- Keeps short snippets to inspect or to feed a model later.

---

# PRE-STEPS: Prep the knowledge base before the LM loop

Goal: lock in the dictionary/defs + indexes so the segmentation + LM loop has solid inputs, with receipts you can reproduce later.

## P0. Freeze normalization + settings

- [ ] Decide & version normalization (`norm_v1`): strip diacritics; v/w→u; j/y→i; k→c; case policy.
- [ ] Write a single YAML/TOML with these rules.
- [ ] Record the hash of the normalization config in all downstream artifacts.

## P1. Refresh dictionary + clusters

- [ ] Load `dictionary.json` → normalize all forms → dedupe lemmas.
- [ ] Rebuild **clusters** of raw defs (if you cluster senses): write to `clusters` table with:
  - `cluster_id, lemma_id, sense_gloss, confidence, sources[], examples[], created_at`
- [ ] Mark `is_canon` where you’re confident; keep alternates as `candidate`.

## P2. Build the n-gram index (this is your superpower)

- [ ] From your corpus, generate n-grams (1..6 or 1..7).
- [ ] For each n-gram, link candidate senses (same normalized form or high semantic sim).
- [ ] Persist to `ngram_index.sqlite3` (or Parquet) with:
  - `ngram, n, form_norm, candidate_sense_ids[], sim_scores[], freq, contexts_sample[]`
- [ ] Ensure there’s an index on `form_norm` for fast lookups.

## P3. Embed things (optional but useful)

- [ ] (If you already have FastText or similar) embed lemmas and n-grams; persist vectors.
- [ ] Store top-k neighbors per item to speed up “semantic similarity” lookups later.

## P4. Capture residuals NOW (so you can measure improvement later)

- [ ] Using your current pipeline, run a coarse segmentation pass over the corpus.
- [ ] For each sentence/line, store:
  - `residual_spans`, `coverage%`, `chosen_senses[]`, `confidences[]`
- [ ] Write a “baseline residual report” you can compare against after improvements.

## P5. Provenance + cost hygiene

- [ ] If any LLM was used to create/clean defs, log:
  - `model, prompt_hash, tokens_in/out, cost, response_json, acceptance_decision`
- [ ] Keep this separate from canon senses (don’t silently overwrite).

**Output of pre-steps (artifacts to check in or export):**

- `norm_v1.yaml` (or TOML)
- `clusters.parquet` / `clusters.sqlite3`
- `ngram_index.sqlite3` (with fast lookup)
- `embeddings/` (optional)
- `baseline_residuals.jsonl` (+ summary HTML)

---

## 0) Project foundation

- [ ] **Create project skeleton**

  - [ ] `src/enochian_lm/{data,tokenize,segment,models,eval,cli,utils,report}/`
  - [ ] `datasets/{raw,processed}`
  - [ ] `runs/` (one subfolder per experiment)
  - [ ] `configs/` (TOML/JSON config files)

- [ ] **Tooling / hygiene**

  - [ ] `pre-commit` with: black, ruff, isort
  - [ ] mypy (strict on core libs; permissive elsewhere)
  - [ ] pytest + coverage gate (~80% on core)
  - [ ] GitHub Actions: lint + tests on PRs
  - [ ] Seed control (set and record `random`, `numpy`, `torch` seeds)

- [ ] **Config pattern**
  - [ ] Use TOML or OmegaConf YAML
  - [ ] Always write the final, resolved config to `runs/<run_id>/config.json`

---

## 1) Data preparation (normalize → tokenize)

- [ ] **Normalization module** `data/normalize.py`

  - [ ] Rules: strip diacritics; v/w→u; j/y→i; k→c; lowercase/uppercase policy (choose one)
  - [ ] Preserve **raw** vs **normalized** forms separately
  - [ ] Version the normalization (e.g., `"norm_v1"`) and store in config/metadata

- [ ] **Tokenizer** `tokenize/char.py`

  - [ ] Character-level tokenizer (letters only; no whitespace/punct)
  - [ ] Deterministic output (no incidental re-chunking)
  - [ ] Save to `datasets/processed/translated_char.jsonl` (one JSON per line with fields: `{"text_id":..., "chars":[...]}`)

- [ ] **Split** `data/split.py`
  - [ ] With tiny data (~5.3k chars), implement **K-fold CV** (default K=10)
  - [ ] Generate `folds.json` mapping each contiguous chunk to a fold index
  - [ ] Write metadata with seed/time/version

**Acceptance criteria:** rerunning with same inputs + seed yields identical tokenization and fold splits.

---

## 2) Dictionary & n-gram inventory (you already have most of this)

- [ ] **Dictionary loader** `data/dictionary.py`

  - [ ] Load JSON into SQLite/Parquet tables:
    - `lemma(lemma_id, form_raw, form_norm, variant_json, created_at)`
    - `sense(sense_id, lemma_id, gloss, tags, confidence, source, evidence_json, is_canon)`
  - [ ] Keep provenance and confidence scores from your clustering pipeline

- [ ] **N-gram inventory** `tokenize/ngram_index.py`
  - [ ] Build/all load index for n=1..6 (or 7), normalized forms
  - [ ] Provide `get_candidates(ngram_norm)` → list of `(sense_id, confidence, features...)`
  - [ ] (Optional) expose semantic sim score if you have FastText embeddings

**Acceptance criteria:** for an input string, you can list all substrings and return candidate senses (with confidences) for each.

---

## 3) Segmentation engine (symbolic decoder)

- [ ] **Lattice builder** `segment/lattice.py`

  - [ ] For a given text (chars), create edges for all substrings length 1..N (N<=7)
  - [ ] For each substring, attach candidate senses from dictionary (or a NULL/unknown edge)

- [ ] **Scoring** `segment/score.py`

  - [ ] Define a simple, tunable scoring:
    ```
    score(Segmentation) =
        + w1 * Coverage(S)              # fraction of chars covered by segments with defs
        + w2 * AvgDefConfidence(S)      # mean confidence of chosen senses
        + w3 * Coherence(S)             # neighbors don't clash semantically
        - w4 * ResidualRate(S)          # leftover chars
        - w5 * OverlapPenalty(S)        # weird overlaps, backtracks
    ```
  - [ ] Start weights: w1=3, w2=2, w3=1, w4=3, w5=1 (configurable)

- [ ] **Search / decoding** `segment/decode.py`

  - [ ] Implement **Viterbi / DP** over the lattice to find best segmentation + sense choices
  - [ ] Allow a **beam** size (default small like 5) if combinatorics get large
  - [ ] Output per sentence:
    - chosen segments with `(start,end,ngram, sense_id, confidence)`
    - residual spans (if any)
    - lattice stats (#paths explored, pruned)

- [ ] **Interfaces**
  - [ ] `segment_text(text) -> SegmentationResult`
  - [ ] Batch mode over a corpus with progress & logging

**Acceptance criteria:** For any input string, you get a segmentation + senses + residuals, with a numeric score and reproducible results.

---

## 4) Baseline LM (predictive sanity check)

- [ ] **Character 5-gram with Kneser–Ney** `models/ngram_char.py`

  - [ ] Train/eval via **K-fold CV** (K=10 default) over your 5.3k chars
  - [ ] Metrics per fold:
    - Perplexity (PPL) on validation fold
    - Top-1 and Top-5 next-char accuracy
  - [ ] Aggregate mean/std across folds
  - [ ] Save minimal artifacts: counts/probs (JSON or lightweight binary)

- [ ] **Evaluation CLI** `eval/ngram_eval.py`
  - [ ] `run_eval_char_5gram(data, folds, order=5, smoothing='KN') -> fold_metrics, summary`
  - [ ] Compute and return:
    - `ppl_mean`, `ppl_std`, `top1_mean`, `top1_std`, `top5_mean`, `top5_std`

**Acceptance criteria:** Running on the same folds yields identical metrics; lower PPL and higher top-k should reflect improvements after better defs/segmentation.

---

## 5) Morpheme-sequence LM (optional but recommended)

- [ ] **Morpheme mapping** `segment/to_morphemes.py`

  - [ ] Convert best segmentation to a **sequence of morpheme IDs**
  - [ ] Save `datasets/processed/morphemes.jsonl`

- [ ] **Morpheme n-gram** `models/ngram_morph.py`
  - [ ] Train a small n-gram (e.g., order=3 or 4) over morpheme IDs
  - [ ] Report PPL via K-fold CV on morpheme sequences

**Acceptance criteria:** PPL is computed for morpheme-level sequences; often more stable than char PPL on tiny corpora.

---

## 6) Run artifacts (receipts)

- [ ] **Run folder spec**

```
runs/<run_id>/
config.json # frozen settings, seeds, versions, paths
segments.jsonl # one line per input text with segmentation & senses
fold_metrics.csv # per-fold PPL/top-k metrics
metrics.json # summary: ppl_mean/std, top1/5 mean/std, coverage, residual_rate
vocab.json # coverage stats, type counts
report.html # human-readable summary (see below)
stdout.txt # (optional) captured logs
```

- [ ] **Coverage metrics**
- [ ] `token_coverage` = % of characters covered by chosen segments with defs
- [ ] `sense_coverage` = % of lemma occurrences with ≥1 selected sense
- [ ] `residual_rate` = residual chars / total chars

- [ ] **Determinism**
- [ ] Run ID can include timestamp + short hash of config
- [ ] Re-run with same seed → identical `segments.jsonl` & metrics

**Acceptance criteria:** Every experiment leaves a self-contained folder you can compare later.

---

## 7) Reporting

- [ ] **Report generator** `report/generate.py`
- [ ] Inputs: `runs/<run_id>/metrics.json`, `segments.jsonl`, `fold_metrics.csv`, `vocab.json`
- [ ] Produce `report.html` with:
  - [ ] Overview: config, data size, seeds
  - [ ] Metrics table: PPL/top-k (mean ± std), coverage, residual rate
  - [ ] Top n-grams (1–6) with selected senses and example contexts
  - [ ] Residuals dashboard: worst sentences; residual spans highlighted
  - [ ] Morpheme PPL (if used)
  - [ ] Diff vs a previous run (if provided): ▲/▼ for key metrics

**Acceptance criteria:** Opening `report.html` gives a clear, human summary. If defs improve, you can **see** coverage↑ and PPL↓.

---

## 8) CLI entrypoints (Typer or argparse)

- [ ] `enlm ingest --src datasets/raw --out datasets/processed --norm norm_v1`
- [ ] `enlm tokenize --in datasets/processed --scheme char --out datasets/processed/translated_char.jsonl`
- [ ] `enlm split --in datasets/processed/translated_char.jsonl --folds 10 --seed 42`
- [ ] `enlm segment --in datasets/processed/translated_char.jsonl --dict path/to/dictionary.json --max_n 7 --out runs/<run_id>/segments.jsonl`
- [ ] `enlm train ngram --order 5 --cv 10 --data datasets/processed/translated_char.jsonl --out runs/<run_id>`
- [ ] `enlm eval --run runs/<run_id> --data datasets/processed/translated_char.jsonl`
- [ ] `enlm morphize --segments runs/<run_id>/segments.jsonl --out datasets/processed/morphemes.jsonl`
- [ ] `enlm train morphgram --order 3 --cv 10 --data datasets/processed/morphemes.jsonl --out runs/<run_id>`
- [ ] `enlm report --run runs/<run_id> [--baseline runs/<prev_run_id>]`

**Acceptance criteria:** Each command writes/updates the expected files under `runs/<run_id>/` or `datasets/processed/`.

---

## 9) LLM-assisted definitions (optional & gated)

- [ ] **When to call LLM**
- [ ] High residual or low-confidence segments only
- [ ] Frequency ≥ threshold OR entropy across contexts high

- [ ] **Batching**
- [ ] 25–50 lemmas per request; include 2–3 short contexts each
- [ ] Strict JSON schema:
  ```json
  {
    "lemma": "...",
    "candidates": [
      {"gloss":"...", "confidence":0.0, "tags":["..."], "evidence":["..."]},
      ...
    ]
  }
  ```
- [ ] Store responses in `raw_defs` with `source=model_name`, `prompt_hash`, token counts, cost

- [ ] **Caching & retries**
- [ ] Skip if `prompt_hash` exists
- [ ] One retry for format fix; else mark `status=failed`

- [ ] **Promotion rules**
- [ ] Don’t mark as `is_canon` until:
  - coverage improves **and/or**
  - PPL/top-k improves relative to prior run

**Acceptance criteria:** Money spent = measurable gains (coverage↑/PPL↓). All calls are auditable via stored prompts/responses.

---

## 10) Evaluation extras (robust on tiny data)

- [ ] **Contrastive next-char test**
- [ ] For each position, present true next char + 4 decoys (same class)
- [ ] 5-way accuracy → add to `metrics.json` & `report.html`

- [ ] **Ablations**
- [ ] Compare with vs without normalization
- [ ] Compare with vs without residual-sense promotion
- [ ] Plot metric deltas

**Acceptance criteria:** Extras run fast and give clearer signals when corpus is small.

---

## 11) Testing strategy

- [ ] **Unit tests**
- [ ] Normalizer rules (gold cases)
- [ ] Lattice construction (edge counts for toy strings)
- [ ] Scoring (hand-crafted segmentations with expected ranks)
- [ ] Viterbi/DP returns expected path on toy inputs
- [ ] KN 5-gram probabilities sum sanity checks

- [ ] **Property tests (Hypothesis)**
- [ ] Random strings: segmentation never produces overlapping segments without penalty
- [ ] Determinism: same seed → same segments & scores

- [ ] **Golden tests**
- [ ] A tiny fixed corpus where PPL/top-k are known, alert if regress

**Acceptance criteria:** CI blocks merges on failing tests; golden metrics protect against accidental behavior changes.

---

## 12) Milestones (timeboxed)

- **Milestone 1 (1–2 days):**
- [ ] Normalizer + tokenizer + K-fold split
- [ ] Char 5-gram CV with PPL/top-k → `runs/<id>/metrics.json`
- [ ] Minimal `report.html` with metrics table

- **Milestone 2 (2–4 days):**
- [ ] Segmentation lattice + DP + scoring
- [ ] `segments.jsonl` + coverage/residual stats in `metrics.json`
- [ ] Report shows top segments and residuals

- **Milestone 3 (2–3 days):**
- [ ] Morpheme mapping + morpheme n-gram PPL
- [ ] Contrastive next-char test
- [ ] Ablation comparisons in report

- **Milestone 4 (optional, after results look sane):**
- [ ] LLM-assisted defs for worst residual cases (pilot 100 lemmas)
- [ ] Re-run; confirm coverage↑ / PPL↓ before scaling

---

## 13) Glossary (plain)

- **Baseline LM:** the tiny model (char 5-gram) you use to sanity-check progress.
- **Perplexity (PPL):** lower = less surprised by the text (better).
- **Top-k next-char accuracy:** does the model include the correct next letter in its top 1 or top 5 guesses?
- **Segmentation:** breaking text into n-grams with meanings to explain all characters.
- **Residuals:** leftover characters not explained by chosen segments.
- **Run artifact:** the folder of receipts (config, metrics, outputs) for one experiment.
