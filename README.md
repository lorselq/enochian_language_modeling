# Enochian Language Modeling: Experimental Computational Philology

## Overview

This project explores computational approaches to deciphering _Liber Loagaeth_, a 16th-century manuscript associated with John Dee and Edward Kelley. Rather than pursuing esoteric interpretations of the undeciphered corpus, this research treats it as a **synthetic, low-resource linguistic corpus**—an experimental setting for semantic structure discovery, root-word extraction, and AI-driven linguistic modeling.

The precise linguistic status of _Liber Loagaeth_ remains uncertain. Its linguistic flavor differs from that of the better-known Enochian Keys, and scholarly consensus generally regards the _Liber Loagaeth_ as either impossible cryptography or nonsensical glossolalia.

This project does not seek to contest this interpretation. Instead, _Liber Loagaeth_ provides a challenging, bounded dataset for exploring how AI systems can infer meaningful semantic relationships and construct speculative lexicons from irregular, anomalous text data. Such experiments offer methodological insights into computational semantic analysis, low-resource language modeling, and broader digital humanities applications.

## Project Goals

**Current phase**

- ✅ Solo-mode root extraction completed and persisted.
- ✅ Residual/remainder extraction logs remainders, residual semantics, and now LLM model names for audit trails.
- 🚧 Debate-mode passes still running (they take time because what could be one pass ends out being easily upwards of ten); remaining clusters need to be adjudicated and merged back into the shared databases.
- 🚧 Translation CLI in progress 

**Upcoming work**

- Finish the remaining debate-mode cycles and merge their accepted glosses.
- Expand the speculative lexicon with cross-run normalization and residual backfills.
- Use the speculative lexicon to pilot a blind translation pass over the Enochian Keys
- Identify proper strategy to apply prior work to _Liber Loagaeth_.

## Current Project Status and Next Steps

Status snapshot:

Immediate priorities (ordered):

1. Finish debate-mode queue cycles and append accepted glosses to the insights DBs.
2. Run `poetry run enlm analyze all --db <db>` after each debate batch to keep residual and attribution priors current.
3. Perform remainder backfills (`poetry run enlm remainder backfill-remainders`) on legacy runs lacking the new model metadata.
4. Regenerate derived exports (JSONL/parquet) for downstream translation tests once debate data stabilizes.

Backlog / nice-to-have:

- Harden normalization/versioning into a tracked config artifact so repeated runs record the ruleset hash.
- Add automated lint/test hooks to CI; today the project relies on local checks.
- Expand segmentation experiments (lattice/beam search) once lexicon coverage is frozen.

## Core Components

### Embedding and Semantic Modeling (Foundational Stage)

- FastText embeddings generated from a limited glossary (~800–1300 entries) based on _Angelical Language Vol. II_ by Aaron Leitch, an adjacent corpus with rough translations.
- Calculation of semantic similarities via dynamic clustering methods—including k-nearest neighbors (kNN), agglomerative clustering, fuzzy clustering, and others—selected through automatic parameter tuning.
- Clustering of potential morphemes to serve as input for agentic debates and solo analysis.

### Agent-Based Root Extraction (Root Inference and Validation)

A simulated AI team operates in two distinct modes:

**Debate Mode:**

- **Linguist Agent:** Proposes candidate roots based on embeddings, heuristics, and semantic proximity.
- **Skeptic Agent:** Critically evaluates proposed roots, presenting counterarguments and alternate hypotheses.
- **Adjudicator Agent:** Finalizes judgments based on linguistic plausibility, semantic consistency, and evidence provided.
- **Glossator Agent:** Synthesizes and records official root definitions approved by the Adjudicator.

**Solo Mode:**

- A single-agent configuration where a designated linguistic expert reviews and directly adjudicates root candidates without debate, intended as a methodological contrast to the Debate Mode.

## Methodology

- **Bottom-Up Semantic Modeling**, where semantic structures are inferred from recurring substrings (n-grams) and semantic proximity metrics without presupposing fixed linguistic rules.
- **Pattern Recognition and Morphological Analysis** through identifying stems, derivatives, and root candidates by analyzing morphological and semantic regularities across clusters.
- **Semantic Clustering (Dynamic Tuning):** Dynamically adjusting and selecting clustering methods (kNN, agglomerative, fuzzy) to propose robust candidate root groups.
- **SQLite Database for Data Recording:** Systematic logging of agent deliberations, accepted/rejected definitions, and clustering metadata for reproducibility and further analysis.

## Current Accomplishments

- **Dictionary Foundation:** Established initial lexicon from Aaron Leitch's _Angelical Language Vol. II_.
- **Ngram Generation:** Robust n-gram indexing including morphological variants derived from John Dee's irregular spellings.
- **Embedding Infrastructure:** Functional FastText embedding pipelines producing promising initial semantic analyses.
- **Dynamic Clustering:** Implementation and testing of multiple clustering methods with automatic parameter tuning, selecting optimal methods based on internal cluster consistency metrics.
- **Agent Modes:** Both Debate and Solo modes implemented; solo results are fully persisted, and debate runs are in progress.
- **Analytics + Residuals:** Attribution, collocation, and residual clustering outputs are linked into the crew prompts; residual and remainder tables now keep the originating LLM model for auditability.

### Canonical dictionary location

The authoritative dictionary source checked into this repo lives at
`src/enochian_lm/root_extraction/data/dictionary.json`. Running the enrichment
script (`python src/training/datasets/enrich_dictionary_pos.py`) produces
`dictionary_enriched.json` in the same folder, adding POS tags, semantic
domains, and guard-rail metadata that downstream datasets rely upon. The
script also inspects any emphasized tokens inside each sense's citations
(`*highlighted words*`) and funnels them through a lightweight English POS
tagger so citation evidence can outweigh ambiguous gloss heuristics when the
two disagree. By default it expects [`spaCy`](https://spacy.io/) with the
`en_core_web_sm` model available (e.g., `pip install spacy` then `python -m
spacy download en_core_web_sm`). Pass `--spacy-model` to switch models or
`--citation-tagger none` to bypass the dependency entirely.

Semantic domain coverage now also benefits from an optional WordNet lookup.
When a gloss headword is not explicitly listed inside
`src/training/config/semantic_domains.yml`, the enrichment script can query
WordNet for synonyms, lemma names, and nearby hypernyms, then reuse any of the
project's curated headword mappings or fall back to a configurable
`wordnet_lexname_to_domains` table. Install the extra dependency with `poetry
install -E wordnet` (or add `wordnet` to `poetry install --with ...`) and make
sure the corpus itself exists via `python -m nltk.downloader wordnet omw-1.4`.
Results are cached in-memory per run so repeated lookups stay fast.

### Pre-analysis safeguards

Run `poetry run enlm preanalyze --db <path>` before the
main analytics loop. The command writes an `initial` row to the new
`preanalysis_runs` table the first time it executes, capturing a trusted seed
list (the default JSON includes established Enochian words like `NAZ`, `BLIOR`,
`IAD`, which would be on the list of ngrams evaluated during a run) and the
accompanying light diagnostics it derives straight from the n-gram index.

Subsequent calls with `--stage subsequent --run-id <translation-run-id>`
append additional rows that tie the safeguards to a specific translation
session while reusing or refreshing the same trusted inventory. Each
sweep performs quick pre-analytics—occurrence counts and gloss snippets
for the trusted n-grams—so those hints populate the `preanalysis_seeds`
records. When the translation crew starts an LLM pass and the heavier
analytics tables are still empty, it now pulls the stored seed payloads
into the prompt, then marks them as consumed once that run processes
the n-grams, closing the loop back into the full translation → analytics →
translation workflow.

## Analytics integration workflow

Recent updates wire the standalone analytics package (`enlm`) back into the
interactive translation crew so that debate/solo prompts surface attribution,
collocation, and residual-cluster priors automatically. To make use of those
signals:

1. **Run at least one translation session.** Use `poetry run enochian-analysis`
   and choose whether the crew works in debate or solo mode. This seeds the
   SQLite insights database at
   `src/enochian_lm/root_extraction/interpretation/revised_debate_derived_definitions.sqlite3` or
   `src/enochian_lm/root_extraction/interpretation/revised_solo_analysis_derived_definitions.sqlite3`
   depending on the mode (see `utils.config.get_config_paths()` for the exact
   filenames).
2. **Populate analytics tables against the same database.** Execute the
   following commands, swapping the `--db` argument for the debate/solo file you
   are iterating on:

   ```bash
   poetry run enlm attrib loo --db src/enochian_lm/root_extraction/interpretation/revised_debate_derived_definitions.sqlite3
   poetry run enlm colloc --db src/enochian_lm/root_extraction/interpretation/revised_debate_derived_definitions.sqlite3
   poetry run enlm residual cluster --db src/enochian_lm/root_extraction/interpretation/revised_debate_derived_definitions.sqlite3
   ```

   These create and fill `attribution_marginals`, `collocation_stats`, and
   `residual_cluster_*` tables that the crew consults when it prepares prompts.
   If you already have JSONL exports, you can also run
   `poetry run enlm analyze all ...` to ingest them and perform the same steps in
   one pass. The attribution and collocation commands require populated
   `composite_reconstruction` and `morph_semantic_vectors` tables; ingest those
   sources first if they are empty.

3. **Restart the translation crew.** Subsequent `poetry run enochian-analysis`
   sessions will automatically display “Analytics priors” sections and weave the
   evidence into the residual-focus prompts.
4. **Retrofit accepted definitions when analytics shift semantics.** After
   computing analytics, run `poetry run enochian-apply-analytics --db <same db>`
   (add `--dry-run` to preview changes). The helper appends an
   `ANALYTICS_NOTES` block to any accepted definition whose paired morpheme is
   clearly carrying the semantics, preventing future sessions from repeating the
   same mistake.

Following this workflow ensures that morphemes like `PRG` receive the fiery
attribution instead of letting shared tokens skew toward `IAL`, and the team can
course-correct existing glosses without manual table edits.

> Need a quick preflight? See `docs/analytics_run_checklist.md` for a
> short, ordered checklist that prevents empty attribution/collocation outputs
> when running `poetry run enlm analyze all`.

## Single-word translation CLI

The translation pipeline includes a dedicated single-word CLI so you can test
proposed Enochian roots or compounds directly against the stored insights
databases.

```bash
poetry run enlm translate-word NAZPSAD
```

```bash
poetry run enlm translate-word NAZPSAD --variant both --strategy prefer-known
```

You can also access the same functionality through the translation-specific
entry point:

```bash
poetry run enochian-interpret translate-word NAZPSAD --format json --pretty
```

See `docs/single_word_translation.md` for architecture details, evidence
sources, and full output schemas.

## Future Work and Phased Development

### Near-Term Development Goals

- Comprehensive execution until all viable n-grams are processed and defined.
- Enhancement of data collection routines, including further improvements to the SQLite-based definition logging system and supporting metadata capture.

### Phase Two: Lexicon Refinement and Semantic Reconstruction

With a complete speculative root-word lexicon in place, subsequent efforts will focus on refining and standardizing dictionary entries. Human-in-the-loop semantic validation and curation will ensure consistent lexical quality and facilitate practical linguistic reconstruction tasks.

A central methodological validation will involve a "blind retranslation" exercise, reinterpreting the previously translated Enochian Keys exclusively using newly derived root meanings, without recourse to historical translations. This test case aims to empirically validate the viability and internal consistency of the speculative lexicon.

#### Anticipated Side-Projects and Additional Experiments

- **Solo vs. Debate Comparative Analysis:** Investigate whether multi-agent debates yield superior semantic quality compared to single-agent evaluations.
- **Root Structure and Syllabic Patterns:** Explore relationships between extracted roots and their phonological and morphological structures, offering potential insights into the linguistic design principles underlying the corpus.

### Phase Three: Full Corpus Application

Following successful methodological validation, the agentic semantic extraction system will be extended to the entire _Liber Loagaeth_ corpus. This larger-scale analysis will assess the scalability and robustness of the developed computational methods, potentially uncovering new semantic regularities, lexicon structures, and insights into the underlying linguistic—or deliberately artificial—nature of the corpus.

## Broader Impact

- Demonstrates computational methods for low-resource linguistic corpora, contributing methodological frameworks applicable to other historically ambiguous or synthetic languages.
- Provides interdisciplinary insights connecting computational linguistics, semantic modeling, and digital humanities, serving as an illustrative case study of AI-assisted linguistic reconstruction in low-resource contexts.

## Note on Setup

This project is currently configured for a highly customized local development environment (involving LM Studio, WSL2, Python poetry-based dependencies, and various advanced tooling). Due to this complexity, detailed setup instructions are currently omitted but may be provided in simplified form in future updates.

---

_This project remains experimental and actively evolving. All analyses, definitions, and methodological approaches are speculative, aimed at computational and theoretical exploration rather than definitive linguistic reconstruction._

### Step-by-step setup and run checklist

The commands below assume `.env_local` / `.env_remote` already exist and that you
are starting from a clean checkout.

1. **Install dependencies.** Run `poetry install` once to create the virtual
   environment and pull the shared dependencies for the `enochian_lm`,
   `translation`, `loagaeth`, and `training` packages.
2. **Train the FastText embeddings.** Execute `poetry run
enochian-build-fasttext` to generate
   `src/enochian_lm/root_extraction/tools/models/enochian_fasttext.model`
   from the dictionary corpus. The script hashes dictionary entries and reuses
   cached vectors when nothing changed, so you can rerun it safely when the
   lexicon is updated.【F:src/enochian_lm/root_extraction/tools/train_fasttext_model.py†L1-L132】【F:src/enochian_lm/root_extraction/tools/train_fasttext_model.py†L182-L233】
3. **Rebuild the n-gram sidecar.** Populate `data/ngram_index.sqlite3` with
   canonical entries and variant mappings by running `poetry run python
src/enochian_lm/root_extraction/utils/build_ngram_sidecar.py --db
src/enochian_lm/root_extraction/data/ngram_index.sqlite3 --keys
src/enochian_lm/root_extraction/data/enochian_keys.txt`. The remaining
   arguments default to the dictionary, substitution, and compression JSON files
   that ship with the repo.【F:src/enochian_lm/root_extraction/utils/build_ngram_sidecar.py†L1-L120】【F:src/enochian_lm/root_extraction/utils/build_ngram_sidecar.py†L600-L620】
4. **Export fine-tuning windows.** Produce sliding 5-gram slices of the Keys
   paired with dictionary gloss tokens and citation contexts via

   ```bash
   poetry run python tools/generate_key_windows.py \
     --keys src/enochian_lm/root_extraction/data/enochian_keys.txt \
     --dictionary src/enochian_lm/root_extraction/data/dictionary_enriched.json \
     --window-size 5 --stride 1 --format jsonl --output keys_windows.jsonl \
     --max-windows 200
   ```

   Each record includes parallel Enochian, definition, and citation token
   sequences. Use `--no-lowercase`, `--keep-punctuation`, `--max-windows`, or
   `--format csv` to tweak the emitted examples.【F:tools/generate_key_windows.py†L1-L234】

5. **Initialize the insights databases.** Seed both the debate and solo SQLite
   files by running `poetry run python
src/enochian_lm/root_extraction/scripts/init_insights_db.py`. The script
   creates or migrates the shared schema (runs, clusters, definitions, residual
   tables, analytics scaffolding) and may be rerun at any time.【F:src/enochian_lm/root_extraction/scripts/init_insights_db.py†L1-L609】
6. **Run a translation session.** Launch `poetry run enochian-analysis` and pick
   solo or debate mode. `RootExtractionCrew` orchestrates the agents, writes run
   metadata into the selected insights database, and streams progress to the
   console.【F:src/enochian_lm/root_extraction/main.py†L1-L86】【F:src/enochian_lm/root_extraction/pipeline/run_root_extraction.py†L34-L115】
7. **Refresh analytics priors.** After each batch of accepted definitions,
   compute the supporting tables that the crew will read on the next pass:

   ```bash
   poetry run enlm morph factorize --db <path-to-solo-or-debate-db> --out artifacts/morph_factorize
   poetry run enlm attrib loo --db <path-to-solo-or-debate-db>
   poetry run enlm colloc --db <path-to-solo-or-debate-db>
   poetry run enlm residual cluster --db <path-to-solo-or-debate-db>
   ```

   The factorization step updates `morph_semantic_vectors`, the attribution step
   populates `attribution_marginals`, the collocation step fills
   `collocation_stats`, and the residual command builds
   `residual_cluster_*`—all tables that `RootExtractionCrew` consults through
   `gather_morph_evidence`. Each subcommand invokes
   `init_insights_db.init_db()` and upgrades the schema if necessary.

8. **Optionally export reports or retrofit glosses.**

   - `poetry run enlm report pipeline --db <db> --out artifacts/report` produces
     an HTML/CSV digest of coverage, attribution, residual, and factorization
     metrics for archival reference.【F:src/enochian_lm/report/pipeline_summary.py†L752-L836】【F:src/enochian_lm/cli.py†L680-L747】
   - `poetry run enochian-apply-analytics --db <db> --dry-run` appends the
     analytics notes block to existing definitions so future sessions inherit the
     latest priors.【F:src/enochian_lm/root_extraction/utils/analytics_bridge.py†L218-L320】【F:src/enochian_lm/analysis/README.md†L63-L87】

9. **Rerun `poetry run enochian-analysis`.** The crew reads the refreshed tables
   on startup and surfaces “Analytics priors” inside each prompt, letting the
   agents leverage attribution deltas, strong collocations, and residual hot
   spots during the next debate/solo cycle.

You can replace steps 6–7 with a single `poetry run enlm analyze all ...` pass if
you maintain JSONL exports of composite reconstructions and morph vectors; the
driver ingests both files, truncates the analytics tables, reruns every stage,
and writes attribution/collocation/residual artifacts in one sweep.【F:src/enochian_lm/cli.py†L680-L748】
