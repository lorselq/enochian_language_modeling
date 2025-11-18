# TODO: Enochian Agent / LLM Training Pipeline

This is the end-to-end plan:

- enrich the dictionary with POS + semantic constraints  
- build word-level + phrase/sentence-level training data  
- train models (embeddings + small LLM)  
- wrap them into something agent-like

---

## Phase 0 · Repo & scaffolding

- [x] **0.1 Create training package structure**
  - [ ] Under `src/`, create:
    - [x] `training/__init__.py`
    - [x] `training/datasets/` (for scripts that *create* data)
    - [x] `training/models/` (for training/eval code)
    - [x] `training/config/` (YAML/TOML configs later)
- [ ] **0.2 Wire up editable install**
  - [ ] Confirm `pyproject.toml` exposes `training` as the main package
  - [ ] In repo root: `pip install -e .` works
- [ ] **0.3 Decide where dictionary lives**
  - [ ] Confirm single source-of-truth dictionary path, e.g.
        `src/enochian_lm/root_extraction/data/dictionary.json`
  - [x] Add a note in the top-level `README.md` pointing to it

---

## Phase 1 · Enrich `dictionary.json` (POS + semantic domains)

### 1A. Design schema additions

- [x] **1.1 Define new fields for each sense**
  - [x] `parts_of_speech`: `List[str]` (e.g. `["NOUN"]`, `["AUX", "PRON"]`)
  - [x] `semantic_domains`: `List[str]` (e.g. `["PLANT"]`, `["DIVINE", "ABSTRACT"]`)
  - [x] `is_copula`: `bool` (true for “to be” / “which are”-style lemmas)
  - [x] `is_compound_standing_for_phrase`: `bool` (e.g. DSCHIS = “which are”)
  - [x] (optional) `notes_pos`: `str` for human comments
- [x] **1.2 Document schema**
  - [x] Create `docs/dictionary_schema.md` describing each field and examples

### 1B. Implement gloss-based POS heuristics (in Python)

- [x] **1.3 Create script** `training/datasets/enrich_dictionary_pos.py`
- [x] **1.4 Load dictionary.json**
  - [x] Read the existing JSON into Python as a list of entries
- [x] **1.5 Implement basic POS heuristics**
  - [x] If gloss starts with `"to be "` or equals forms like `"am"`, `"are"`, `"is"`
          → `parts_of_speech` includes `"AUX"`, `is_copula = true`
  - [x] If gloss starts with `"to <verb>"` (regex: `^to\s+\w+`)
          → `parts_of_speech` includes `"VERB"`
  - [x] If gloss is a single preposition (`"of"`, `"from"`, `"in"`, `"amongst"` etc)
          → `parts_of_speech` includes `"ADP"`
  - [x] If gloss is a coordinator (`"and"`, `"or"`, `"but"`)
          → `parts_of_speech` includes `"CCONJ"`
  - [x] If gloss includes `","` or `" / "` and clearly stands for multiple words (e.g. `"which are"`, `"that which is"`)
          → set `is_compound_standing_for_phrase = true`, allow multiple POS tags
  - [x] Fallback: default to `"NOUN"` if no other rule hits (to be reviewed later)
- [ ] **1.6 (Optional) Hook in an English POS tagger** *(deferred—heuristics currently sufficient and avoid extra runtime deps)*
  - [ ] Add a function `infer_pos_with_tagger(gloss)` that:
    - [ ] Takes the last content word (head)
    - [ ] Maps its POS to one of: `NOUN`, `VERB`, `ADJ`, `ADV`, `ADP`, `PRON`, etc.
  - [ ] Use this only when heuristics return `None`
- [x] **1.7 Write out enriched dictionary**
  - [x] Save as `dictionary_enriched.json`
  - [x] Add CLI flag or guard so you don't clobber the original by accident

### 1C. Add semantic domains

- [x] **1.8 Create mapping file** `training/config/semantic_domains.yml`
    - [x] Define a small set of domain labels:
      - [x] `DIVINE`
      - [x] `CELESTIAL`
      - [x] `MORAL`
      - [x] `MENTAL`
      - [x] `SPEECH`
      - [x] `SOCIAL`
      - [x] `ACTION`
      - [x] `QUALITY`
          - [x] `QUANTITY`
          - [x] `SPACE`
          - [x] `TIME`
      - [x] `RELATION`
      - [x] `PHYSICAL`
    - [x] Map common gloss headwords to these domains
- [x] **1.9 Implement domain assignment** in `enrich_dictionary_pos.py`
    - [x] Extract gloss “head” word (e.g. from `"voice of God"` → `"voice"`)
    - [x] Look up in `semantic_domains.yml`
    - [x] If unknown, leave `semantic_domains = []` for later manual tagging
- [x] **1.10 Manual review pass**
  - [x] Write a quick script to print:
    - [x] all lemmas with `is_copula = true`
    - [x] all lemmas with multiple POS tags
    - [x] all lemmas with no `semantic_domains`
  - [x] Manually adjust the worst offenders and re-run enrichment if needed *(see `runs/dictionary_enrichment_report.txt` for triage list)*

---

## Phase 2 · Single-word training data (embeddings & classifier-friendly)

Goal: create a clean, word-level dataset that encodes lemma ↔ gloss ↔ POS ↔ semantics.

- [ ] **2.1 Create script** `training/datasets/build_word_dataset.py`
- [ ] **2.2 Load `dictionary_enriched.json`**
- [ ] **2.3 For each sense, build a record like:**
  ```json
  {
    "lemma": "LORSELQ",
    "en_text": "flowers",         // main gloss or short phrase
    "pos": ["NOUN"],
    "semantic_domains": ["PLANT"],
    "is_copula": false
  }
  ```

* [ ] **2.4 Save as JSONL**

  * [ ] Path: `data/word_dataset.jsonl`
  * [ ] One record per line
* [ ] **2.5 (Optional) Train simple embeddings / classifier**

  * [ ] Later: use this data to:

    * [ ] train a small embedding model or
    * [ ] train a simple classifier that predicts POS/domain from gloss text

---

## Phase 3 · Phrase & sentence template design (English side)

Goal: define *structures* you’ll use to generate synthetic sentences, constrained by POS + semantics.

* [ ] **3.1 Create config file** `training/config/templates.yml`

  * [ ] Define templates as sequences of slots, e.g.:

    ```yaml
    - name: "simple_noun"
      slots:
        - { role: "NOUN", domain_any_of: ["PLANT", "ANIMAL", "ABSTRACT"] }

    - name: "noun_of_noun"
      slots:
        - { role: "NOUN", domain_any_of: ["ABSTRACT", "DIVINE", "EVENT"] }
        - { literal: "of" }
        - { role: "NOUN", domain_any_of: ["PLANT", "BODY_PART", "DIVINE", "PLACE"] }

    - name: "amongst_plants"
      slots:
        - { role: "NOUN", domain_any_of: ["PLANT"] }
        - { literal: "amongst" }
        - { role: "NOUN", domain_any_of: ["PLANT"] }

    - name: "copula_noun"
      slots:
        - { literal_pronoun: "I" }
        - { role: "COPULA" }
        - { role: "NOUN", domain_any_of: ["PLANT", "ABSTRACT", "DIVINE"] }

    - name: "copula_adj"
      slots:
        - { literal_pronoun: "I" }
        - { role: "COPULA" }
        - { role: "ADJ", domain_any_of: ["EMOTION", "ABSTRACT"] }
    ```
* [ ] **3.2 Decide allowed POS → slot mappings**

  * [ ] Map `COPULA` to lemmas with `is_copula = true`
  * [ ] Map `NOUN`, `ADJ`, `VERB`, etc. to `parts_of_speech` entries
* [ ] **3.3 Document design**

  * [ ] Create `docs/template_design.md` explaining slot types and domain rules

---

## Phase 4 · Generating synthetic phrase/sentence data (parallel EN ↔ Enochian)

* [ ] **4.1 Create generator script** `training/datasets/generate_synthetic_parallel.py`
* [ ] **4.2 Load:**

  * [ ] `dictionary_enriched.json`
  * [ ] `templates.yml`
* [ ] **4.3 Build in-memory index:**

  * [ ] Lemmas grouped by POS & semantic domain:

    * e.g. `index["NOUN"]["PLANT"] = [list of lemmas]`
* [ ] **4.4 Implement template expansion**

  * [ ] For each template:

    * [ ] Sample multiple realizations (e.g. 100–1000 per template, adjustable)
    * [ ] For each slot:

      * [ ] If `literal`: use the literal English word (`"of"`, `"amongst"`, `"I"`)
      * [ ] If `role: "COPULA"`: choose lemma with `is_copula = true`
      * [ ] If `role: "NOUN"` etc:

        * [ ] Filter lemmas by POS & semantic_domains
        * [ ] Randomly sample one
  * [ ] Build:

    * [ ] English-side sentence (using gloss or short English label for each lemma)
    * [ ] Enochian-side sentence (using lemma strings)
* [ ] **4.5 Save as JSONL**

  * [ ] Path: `data/synthetic_parallel.jsonl`
  * [ ] Schema:

    ```json
    {
      "id": "templateName-000123",
      "template": "noun_of_noun",
      "src_lang": "eno",
      "tgt_lang": "en",
      "src": "LORSELQ NANBA",
      "tgt": "flowers of thorns"
    }
    ```
* [ ] **4.6 Add command-line hooks**

  * [ ] Add CLI entry in `cli.py` to run:

    * [ ] `enochian-lm generate-synthetic --num-per-template 500` (or similar)

---

## Phase 5 · Incorporate real corpus (Keys, Loagaeth if/when)

* [ ] **5.1 Create corpus loader scripts**

  * [ ] `training/datasets/load_keys_corpus.py`
* [ ] **5.2 Tokenize real Enochian text**

  * [ ] Use your existing tokenization rules
  * [ ] Store as sequences of lemmas
* [ ] **5.3 Align to gloss-side where possible**

  * [ ] For Keys sections where you have known translations / glosses:

    * [ ] Build JSONL records similar to synthetic ones, but mark:

      * [ ] `"source": "corpus_keys"` or similar
* [ ] **5.4 Merge synthetic + real**

  * [ ] `training/datasets/merge_datasets.py`:

    * [ ] Combine:

      * [ ] `synthetic_parallel.jsonl`
      * [ ] `keys_parallel.jsonl`
    * [ ] Add a `dataset_split` field: `"train"`, `"val"`, `"test"`
    * [ ] Ensure some real corpus is held out for evaluation

---

## Phase 6 · Tokenizer & model choice

* [ ] **6.1 Decide on model family**

  * [ ] Start with a small, accessible architecture:

    * [ ] Option A: seq2seq (e.g. T5-style) for `eno → en` and maybe `en → eno`
    * [ ] Option B: decoder-only LM that learns Enochian alone (for generation)
* [ ] **6.2 Decide tokenization strategy**

  * [ ] For Enochian side:

    * [ ] Likely tokenize at **lemma level** (each lemma = one token)
      so you can control the vocabulary and avoid hallucinated lemmas
  * [ ] For English side:

    * [ ] Use a standard BPE tokenizer (or reuse existing HuggingFace tokenizer)
* [ ] **6.3 Implement tokenizer builder**

  * [ ] `training/models/build_tokenizers.py`:

    * [ ] Collect vocab list of all Enochian lemmas used in datasets
    * [ ] Build a simple mapping: `token → integer id`
    * [ ] Save as `eno_vocab.json`
  * [ ] For English:

    * [ ] Choose an off-the-shelf tokenizer, or train a tiny one on `tgt` texts

---

## Phase 7 · Model training loop

* [ ] **7.1 Create config for experiments**

  * [ ] `training/config/experiment_default.yml`:

    * [ ] model type, hidden size, num layers
    * [ ] batch size, learning rate, num epochs
    * [ ] paths to JSONL datasets
* [ ] **7.2 Implement dataset readers**

  * [ ] `training/models/dataset.py`:

    * [ ] Class that:

      * [ ] streams JSONL
      * [ ] tokenizes `src` and `tgt`
      * [ ] returns tensors for model input
* [ ] **7.3 Implement training script**

  * [ ] `training/models/train_seq2seq.py` (or similar):

    * [ ] load config
    * [ ] instantiate model & tokenizers
    * [ ] create dataloaders for train/val
    * [ ] run training loop (or use a framework like HF `Trainer` if you go that route)
* [ ] **7.4 Add evaluation metrics**

  * [ ] Basic:

    * [ ] perplexity on Enochian-only generation
    * [ ] token-level accuracy on `eno → en` pairs
  * [ ] Later:

    * [ ] more interesting eval: does it pick correct lemma given gloss?

---

## Phase 8 · “Agent” wrapping / interactive usage

* [ ] **8.1 Implement small inference CLI**

  * [ ] `training/models/infer.py`:

    * [ ] `eno → en` mode: given Enochian sentence, output gloss-ish translation
    * [ ] `en → eno` mode: given gloss-like English, output Enochian lemma sequence
* [ ] **8.2 Hook into main `cli.py`**

  * [ ] Add commands:

    * [ ] `enochian-lm translate-enochian "..."`
    * [ ] `enochian-lm generate-enochian "..."` (constrained generation)
* [ ] **8.3 Optional: add agent logic**

  * [ ] Small “agent” wrapper that:

    * [ ] uses the LM for proposals
    * [ ] queries the dictionary for extra info
    * [ ] surfaces multiple alternative translations / readings

---

## Phase 9 · Iteration & refinement

* [ ] **9.1 Tighten POS & semantic domains**

  * [ ] Use model errors / weird generations to find mis-tagged lexemes
* [ ] **9.2 Expand templates**

  * [ ] Add templates that mirror known syntactic patterns in Keys
* [ ] **9.3 Add Loagaeth-specific stages**

  * [ ] Under `enochian_lm/loagaeth/`, add:

    * [ ] corpus loader
    * [ ] structural analysis tools
    * [ ] later: generators / hypotheses based on the trained models