# Enochian Language Modeling: Experimental Computational Philology

## Overview
This project explores computational approaches to deciphering *Liber Loagaeth*, a 16th-century manuscript associated with John Dee and Edward Kelley. Rather than pursuing esoteric interpretations of the undeciphered corpus, this research treats it as a **synthetic, low-resource linguistic corpus**—an experimental setting for semantic structure discovery, root-word extraction, and AI-driven linguistic modeling.

The precise linguistic status of *Liber Loagaeth* remains uncertain. Its linguistic flavor differs from that of the better-known Enochian Keys, and scholarly consensus generally regards the *Liber Loagaeth* as either impossible cryptography or nonsensical glossolalia. 

This project does not seek to contest this interpretation. Instead, *Liber Loagaeth* provides a challenging, bounded dataset for exploring how AI systems can infer meaningful semantic relationships and construct speculative lexicons from irregular, anomalous text data. Such experiments offer methodological insights into computational semantic analysis, low-resource language modeling, and broader digital humanities applications.

The repository is organized as a monorepo built with Poetry. Two cooperating packages live under `src/`:

- `enochian_translation_team` — the interactive, agent-driven pipeline that proposes and adjudicates speculative morpheme glosses.
- `enochian_lm` — the analytics companion that consumes the agents' SQLite logs, recomputes embeddings, and exports attribution/collocation/residual statistics.

All CLIs described below (e.g., `enochian-analysis`, `enochian-build-fasttext`, `enlm`) are registered via `pyproject.toml` and can be executed with `poetry run <command>` once dependencies have been installed.

## Project Goals
**Current phase**:
- Identify plausible root morphemes via examination of semantic clusters within the linguistic corpus.
- Evaluate whether single-agent analysis or multi-agent debates yields better proposals for candidate root structures.
- Construct an AI-assisted speculative lexicon through pattern recognition, semantic proximity, and agentic validation.
- Develop and evaluate computational methods suitable for texts with limited ground-truth data, irregular morphological patterns, and historical ambiguity.

Goals in the current phase are mostly complete, e.g., solo analysis has finished (but requires data wrangling), but analysis via debate is still underway and may take a while due to resource constraints.

**Future phases**:
- Construct an expanded dictionary of Enochian morphemes and words.
- Use the expanded dictionary as part of an AI-driven attempt to provide a "blind" translation of the Enochian Keys.
- Upon successful translation, aim the translation program at Liber Loagaeth to see if translation of any amount cannot be performed.

Goals in the future phases section are covered in more detail later in this document.

## Core Components

### Embedding and Semantic Modeling (Foundational Stage)
- FastText embeddings generated from a limited glossary (~800–1300 entries) based on *Angelical Language Vol. II* by Aaron Leitch, an adjacent corpus with rough translations.
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
- **Dictionary Foundation:** Established initial lexicon from Aaron Leitch's *Angelical Language Vol. II*.
- **Ngram Generation:** Robust n-gram indexing including morphological variants derived from John Dee's irregular spellings.
- **Embedding Infrastructure:** Functional FastText embedding pipelines producing promising initial semantic analyses.
- **Dynamic Clustering:** Implementation and testing of multiple clustering methods with automatic parameter tuning, selecting optimal methods based on internal cluster consistency metrics.
- **Agent Modes:** Both Debate and Solo modes fully implemented; comparative analyses planned as future research.
- **CLI Enhancements:** Improved terminal interface for smoother operation and enhanced readability.

## Analytics integration workflow

Recent updates wire the standalone analytics package (`enlm`) back into the
interactive translation crew so that debate/solo prompts surface attribution,
collocation, and residual-cluster priors automatically. To make use of those
signals:

1. **Run at least one translation session.** Use `poetry run enochian-analysis`
   and choose whether the crew works in debate or solo mode. This seeds the
   SQLite insights database at
   `src/enochian_translation_team/data/debate_derived_definitions.sqlite3` or
   `src/enochian_translation_team/data/solo_analysis_derived_definitions.sqlite3`
   depending on the mode (see `utils.config.get_config_paths()` for the exact
   filenames).
2. **Populate analytics tables against the same database.** Execute the
   following commands, swapping the `--db` argument for the debate/solo file you
   are iterating on:

   ```bash
   poetry run enlm attrib loo --db src/enochian_translation_team/data/debate_derived_definitions.sqlite3
   poetry run enlm colloc --db src/enochian_translation_team/data/debate_derived_definitions.sqlite3
   poetry run enlm residual cluster --db src/enochian_translation_team/data/debate_derived_definitions.sqlite3
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

### Pre-analysis safeguards

> "We keep the translation → analytics → translation flow tight so each pass inherits better priors."

To honor that cadence, run `poetry run enlm preanalyze --db <path>` before the
main analytics loop. The command writes an `initial` row to the new
`preanalysis_runs` table the first time it executes, capturing a trusted seed
list (the default JSON includes `NAZ`) and the accompanying light diagnostics it
derives straight from the n-gram index. Subsequent calls with
`--stage subsequent --run-id <translation-run-id>` append additional rows that
tie the safeguards to a specific translation session while reusing or refreshing
the same trusted inventory. Each sweep performs quick pre-analytics—occurrence
counts and gloss snippets for the trusted n-grams—so those hints populate the
`preanalysis_seeds` records. When the translation crew starts an LLM pass and
the heavier analytics tables are still empty, it now pulls the stored seed
payloads into the prompt, then marks them as consumed once that run processes
the n-grams, closing the loop back into the full translation → analytics →
translation workflow.

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
Following successful methodological validation, the agentic semantic extraction system will be extended to the entire *Liber Loagaeth* corpus. This larger-scale analysis will assess the scalability and robustness of the developed computational methods, potentially uncovering new semantic regularities, lexicon structures, and insights into the underlying linguistic—or deliberately artificial—nature of the corpus.

## Broader Impact
- Demonstrates computational methods for low-resource linguistic corpora, contributing methodological frameworks applicable to other historically ambiguous or synthetic languages.
- Provides interdisciplinary insights connecting computational linguistics, semantic modeling, and digital humanities, serving as an illustrative case study of AI-assisted linguistic reconstruction in low-resource contexts.

## Note on Setup
This project is currently configured for a highly customized local development environment (involving LM Studio, WSL2, Python poetry-based dependencies, and various advanced tooling). Due to this complexity, detailed setup instructions are currently omitted but may be provided in simplified form in future updates.

---

*This project remains experimental and actively evolving. All analyses, definitions, and methodological approaches are speculative, aimed at computational and theoretical exploration rather than definitive linguistic reconstruction.*

## Initialization order for databases, caches, and analytics

Because so many scripts share the same SQLite artifacts, tasks must be executed in a predictable order. The sequence below guarantees that every downstream command finds the prerequisites it expects:

1. **Install dependencies** – Run `poetry install` to materialize the virtual environment and install both `enochian_translation_team` and `enochian_lm` dependencies.
2. **Train FastText embeddings** – Execute `poetry run enochian-build-fasttext` to create `tools/models/enochian_fasttext.model`. The rest of the pipeline assumes these vectors exist when clustering n-grams or reconstructing composites.
3. **Build the n-gram sidecar** – Populate `data/ngram_index.sqlite3` via `poetry run python src/enochian_translation_team/utils/build_ngram_sidecar.py ...`. This database feeds canonical/variant lookups whenever the crew prepares work queues.
4. **Initialize the insights databases** – Seed the solo and debate SQLite files by running `poetry run python src/enochian_translation_team/scripts/init_insights_db.py`. This ensures the shared schema (`runs`, `clusters`, `definitions`, `composite_reconstruction`, etc.) exists before any translation or analytics work begins.
5. **Run at least one translation session** – Launch `poetry run enochian-analysis` in solo or debate mode. Doing so writes the first batches of accepted/rejected glosses plus the composite reconstructions that analytics depend on.
6. **Refresh analytics tables** – Invoke `poetry run enlm ...` (details in the checklist below) so that attribution, collocation, residual, and factorization tables exist before the next translation run. These tables become the “analytics priors” surfaced in prompts.
7. **Optionally retrofit glosses** – `poetry run enochian-apply-analytics --db <db>` adds `ANALYTICS_NOTES` to existing definitions, keeping historical glosses synchronized with the latest priors.
8. **Iterate** – Alternate between translation sessions and analytics refreshes. Every cycle enriches the evidence stored in SQLite, and new analytics output immediately influences subsequent agent debates.

If a step is skipped, later commands typically fail with missing-table errors or silently run without any context, so following this order prevents subtle data gaps.

### Step-by-step setup and run checklist

The commands below assume `.env_local` / `.env_remote` already exist and that you
are starting from a clean checkout.

1. **Install dependencies.** Run `poetry install` once to create the virtual
   environment and pull the shared dependencies for both
   `enochian_translation_team` and `enochian_lm`.
2. **Train the FastText embeddings.** Execute `poetry run
   enochian-build-fasttext` to generate `tools/models/enochian_fasttext.model`
   from the dictionary corpus. The script hashes dictionary entries and reuses
   cached vectors when nothing changed, so you can rerun it safely when the
   lexicon is updated.【F:src/enochian_translation_team/tools/train_fasttext_model.py†L1-L132】【F:src/enochian_translation_team/tools/train_fasttext_model.py†L182-L233】
3. **Rebuild the n-gram sidecar.** Populate `data/ngram_index.sqlite3` with
   canonical entries and variant mappings by running `poetry run python
   src/enochian_translation_team/utils/build_ngram_sidecar.py --db
   src/enochian_translation_team/data/ngram_index.sqlite3 --keys
   src/enochian_translation_team/data/enochian_keys.txt`. The remaining
   arguments default to the dictionary, substitution, and compression JSON files
   that ship with the repo.【F:src/enochian_translation_team/utils/build_ngram_sidecar.py†L1-L120】【F:src/enochian_translation_team/utils/build_ngram_sidecar.py†L600-L620】
4. **Initialize the insights databases.** Seed both the debate and solo SQLite
   files by running `poetry run python
   src/enochian_translation_team/scripts/init_insights_db.py`. The script
   creates or migrates the shared schema (runs, clusters, definitions, residual
   tables, analytics scaffolding) and may be rerun at any time.【F:src/enochian_translation_team/scripts/init_insights_db.py†L1-L609】
5. **Run a translation session.** Launch `poetry run enochian-analysis` and pick
   solo or debate mode. `RootExtractionCrew` orchestrates the agents, writes run
   metadata into the selected insights database, and streams progress to the
   console.【F:src/enochian_translation_team/main.py†L1-L86】【F:src/enochian_translation_team/crew/root_extraction_crew.py†L34-L115】
6. **Refresh analytics priors.** After each batch of accepted definitions,
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
   `init_insights_db.init_db()` and upgrades the schema if necessary.【F:src/enochian_lm/analysis/factorize.py†L420-L488】【F:src/enochian_lm/analysis/attribution.py†L113-L206】【F:src/enochian_lm/analysis/colloc.py†L1-L200】【F:src/enochian_lm/cli.py†L760-L938】【F:src/enochian_translation_team/utils/analytics_bridge.py†L218-L320】
7. **Optionally export reports or retrofit glosses.**
   - `poetry run enlm report pipeline --db <db> --out artifacts/report` produces
     an HTML/CSV digest of coverage, attribution, residual, and factorization
     metrics for archival reference.【F:src/enochian_lm/report/pipeline_summary.py†L752-L836】【F:src/enochian_lm/cli.py†L680-L747】
   - `poetry run enochian-apply-analytics --db <db> --dry-run` appends the
     analytics notes block to existing definitions so future sessions inherit the
     latest priors.【F:src/enochian_translation_team/utils/analytics_bridge.py†L218-L320】【F:src/enochian_lm/README.md†L63-L87】

8. **Rerun `poetry run enochian-analysis`.** The crew reads the refreshed tables
   on startup and surfaces “Analytics priors” inside each prompt, letting the
   agents leverage attribution deltas, strong collocations, and residual hot
   spots during the next debate/solo cycle.【F:src/enochian_translation_team/crew/root_extraction_crew.py†L600-L676】【F:src/enochian_translation_team/utils/analytics_bridge.py†L218-L320】

You can replace steps 6–7 with a single `poetry run enlm analyze all ...` pass if
you maintain JSONL exports of composite reconstructions and morph vectors; the
driver ingests both files, truncates the analytics tables, reruns every stage,
and writes attribution/collocation/residual artifacts in one sweep.【F:src/enochian_lm/cli.py†L680-L748】
