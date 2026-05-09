# Translation and Interpretation Strategy Specification

This document specifies how the current translation pipeline works and how to
add a conservative diagnostic layer for root-level sense grouping. The guiding
principle is evidence preservation: deterministic evidence selects and explains
candidate readings; LLMs may render, label, or adjudicate only after provenance
and uncertainty are retained.

The proposed third layer is called **surviving cluster groups**. It groups
accepted `root_glosses` rows into provisional sense bundles for inspection. It
is a read-only diagnostic read model first, an optional prompt context second,
and a ranking signal only after separate tests prove that it improves results.

## Resolved Implementation Defaults

This section removes choices that would otherwise be made ad hoc during
implementation.

- Canonical Phase 1/2 CLI shape is
  `enlm report root-groups --root <ROOT> --variant <solo|debate|both>`.
- Default variant is `both`, matching the translation commands.
- Root input is case-insensitive. SQLite lookups normalize roots with
  `LOWER(TRIM(root))`; public JSON and text output render roots uppercase.
- JSON field names in this document are the public contract. Python internals
  may use dataclasses or typed dictionaries, but serialized output must use the
  names documented here.
- Phase 1 JSON includes full `evidence_packets` by default. Use
  `--detail compact` only if a compact mode is explicitly implemented.
- Phase 3 translation diagnostics use compact group summaries by default and do
  not include full `evidence_packets` unless verbose/debug output explicitly
  requests them.
- Grouping uses deterministic connected components over pairwise similarity,
  followed by conflict checks. Do not choose a different clustering algorithm
  without updating this spec and tests.
- Embedding preference order is:
  1. local `paraphrase-MiniLM-L6-v2` sentence-transformer if available;
  2. deterministic TF-IDF fallback with word unigrams/bigrams;
  3. empty/single-packet grouping when neither vectorizer path can run.
- No LLM calls are allowed in Phases 1 or 2.
- Phase 3 may pass compact summaries to an already-enabled LLM renderer, but it
  must not add new LLM calls by itself.
- Phase 4 is the only phase that may change ranking, and only behind
  `--use-root-groups-for-ranking`.

No open implementation decisions remain for v1. If an implementer wants to
change command shape, clustering algorithm, output field names, default detail
level, embedding fallback, ranking formula, or translation integration behavior,
they must update this specification and the corresponding tests first.

## Goals and Non-Goals

### Goals

- Document the existing word and phrase translation contracts well enough that
  future changes can preserve current behavior.
- Define an implementable schema for root-level diagnostic sense groups.
- Preserve row-level provenance from `root_glosses`, including nested
  `raw_glossator_json.EVIDENCE` effects and examples.
- Prevent short roots such as `A`, `I`, `D`, and `IO` from being collapsed into
  a single over-broad denotation.
- Make grouped evidence inspectable through the Phase 1 diagnostic command
  before it affects translation output.
- Provide explicit defaults, thresholds, and warning rules so implementation
  decisions are not left implicit.

### Non-Goals

- Do not declare a single true meaning for any root.
- Do not replace `root_glosses`, `semantic_core`, existing candidate bundles, or
  FNP profiles.
- Do not merge solo and debate evidence without preserving variant provenance.
- Do not feed surviving cluster groups into LLM prompts by default.
- Do not change `translate-word` or `translate-phrase` ranking until a separate
  group-aware mode is implemented and tested.
- Do not treat examples, exceptions, or LLM-generated explanations as
  authoritative definitions without supporting provenance.

## Evidence Sources

The translation stack reads several kinds of evidence, each with a different
trust level and diagnostic purpose.

- **Cluster evidence** comes from accepted `clusters` rows and is the strongest
  direct signal for a word or morph. The `root_glosses` view exposes accepted
  cluster payloads as normalized fields such as `definition`, `semantic_core`,
  `negative_contrast`, POS bias, attachment likelihoods, confidence, examples,
  and raw glossator JSON.
- **Residual semantics** come from `root_residual_semantics` and represent
  remainder meanings from semantic subtraction. They are useful but weaker than
  direct cluster evidence because they describe what remains after a host/root
  subtraction.
- **Morph hypotheses** come from accepted `morph_hypotheses` and are low-weight
  hints for possible morph meanings.
- **Dictionary morphs** come from the canonical dictionary data and supply exact
  whole-word anchors plus substring hints. Blind retranslation can suppress
  whole-word dictionary anchors while still allowing evidence-backed roots.
- **FNP profiles** are deterministic functional-nature profiles derived from
  accepted root-gloss metadata. They summarize soft noun/modifier/verb behavior,
  attachment behavior, semantic core terms, definitions, and uncertainty.
- **Translation memory** stores cautious observations for unknown or provisional
  phrase tokens so repeated phrase translation can accumulate weak context
  without pretending it is authoritative lexical evidence.
- **FastText neighbors** are fallback diagnostics when no direct evidence is
  found. They are exploratory, not a confident translation source.

All evidence must remain variant-aware. Solo and debate results may agree, but
they must not be silently merged without preserving their provenance.

## Current Word Translation Contract

`enochian-interpret translate-word` and `enlm translate-word` share the same
entry point and service behavior.

### Inputs

The word translator accepts a normalized target word plus these strategy
controls:

- `--variant solo|debate|both` chooses which insights databases to query.
- `--strategy prefer-fewer|prefer-known|prefer-balance` adjusts reranking.
- `--evidence-mode all|clusters-only|residuals-only` determines which evidence
  types can support decomposition, filtering, scoring, and definition lookup.
- `--weight` / `--no-weight` toggles weighted scoring.
- `--allow-whole-word` / `--no-whole-word` controls blind retranslation behavior.
- `--no-dictionary` suppresses exact dictionary matches while preserving DB
  evidence.
- `--llm` enables optional synthesis after deterministic candidate selection.

### Processing Flow

1. Fetch direct word evidence: clusters, residual semantics, morph hypotheses,
   attested definitions, dictionary morphs, rejected morph flags, and FastText
   fallback diagnostics when nothing else exists.
2. Enumerate all substrings for the word, including singleton candidates, then
   fetch substring support, substitution aliases, accepted definition counts,
   accepted definition glosses, rejected morphs, and FNP profiles.
3. Apply evidence mode early so generation, support labeling, hard filtering,
   and definition lookup agree on the same evidence surface.
4. Build decompositions. The normal path enumerates attested full-cover
   segmentations and then applies beam scores for comparison. Phrase translation
   may request beam-search generation directly with `use_beam_search=True`.
5. Hard-filter unsupported decompositions, preserving diagnostics for dropped
   candidates and fallback paths.
6. Score and rank the surviving decompositions with an explicit decision
   function. The current decision combines base decomposition score, semantic
   coherence, attestation strength/specificity, FNP/attachment fit, singleton
   burden penalty, and fewer-morph tie-breaks.
7. Convert selected decompositions into candidate payloads. Each candidate keeps
   ordered `meanings`, per-morph provenance, definition traces, score
   breakdowns, warnings, and diagnostics.
8. Compose a candidate-level `semantic_bundle` from the surviving morph
   meanings. The bundle chooses a `bundle_head_gloss`,
   `bundle_surface_gloss`, function profile, coherence score, surface
   candidates, and selection reason for later phrase assembly.
9. Optionally synthesize a readable definition with an LLM. The LLM receives the
   selected morphs, meanings, provenance, coverage, strategy hint, and context;
   it must not invent external etymology or override deterministic evidence.

### Output Shape

The word result should preserve:

- target metadata: `word`, `variants_queried`, `strategy`, `evidence_mode`,
  dictionary and weighting flags, LLM flags, timestamp;
- `candidates`, each with rank, morphs, canonicals, score, score breakdown,
  meanings, semantic bundle fields, definition traces, confidence, warnings,
  and optional LLM synthesis;
- `evidence`, a summary of retrieved evidence;
- `diagnostics`, including repository/lookup state, substring support,
  decomposition counts, hard filters, decision rows, FNP profile count, and
  fallback details.

## Current Phrase Translation Contract

`enochian-interpret translate-phrase` and `enlm translate-phrase` build on the
word translator. Phrase translation should not maintain a separate lexical
truth source; it consumes word-service payloads and adds phrase-level search,
relations, grammar, and rendering.

### Processing Flow

1. Split the phrase into independent clauses on current hard boundaries:
   semicolons, colons, and commas.
2. Tokenize each clause into alphabetic Enochian tokens.
3. Prewarm repository caches for every token and token substring. This includes
   cluster support, residual support, hypotheses, rejected morph flags,
   accepted definition counts/glosses, clustered definition counts, and FNP
   observations.
4. For each token, call `translate_word(..., llm=False,
   use_beam_search=True)` with the active phrase flags.
5. Convert each word candidate into a phrase token candidate. The conversion
   keeps the primary definition, raw definition, alternates, morphs, sources,
   definition trace, decision trace, semantic bundle, bundle function profile,
   bundle coherence, FNP profile, inferred function, and warnings.
6. Beam-search whole-phrase parse candidates over the token candidate matrix.
   Parse scoring combines token candidate score, confidence, bundle coherence,
   adjacent-token relation scores, FNP grammar sequence score, and penalties for
   unresolved or weak provisional readings.
7. Infer lightweight relations between adjacent tokens, including
   coordination/apposition, relational attachment, head-modifier, and
   predicate-argument.
8. Build a deterministic translation skeleton from the chosen parse. The
   skeleton uses bundle heads and normalized function profiles such as
   conjunction, relative marker, locative marker, imperative existential, and
   feminine locative possessive.
9. Optionally run LLM rendering. The bundled renderer can produce technical,
   lay, poetic, contextual, interpretive, footnoted, and confidence-bearing
   renderings from the chosen parse and top contextual alternatives.
10. Record provisional memory observations for unknown/provisional token
    readings when memory updates are enabled.

### Output Shape

The phrase result should preserve:

- phrase metadata: `phrase`, `tokens`, variants, strategy, evidence mode,
  weighting/dictionary flags, LLM flags, timestamp;
- `token_analyses`, including each token's word result and phrase candidates;
- `parse_candidates` and `chosen_parse`, including token choices, relations,
  grammar evidence, grammar warnings, function sequence, score, and skeleton;
- rendered outputs: deterministic skeleton, optional technical render, lay
  translation, poetic/contextual/interpretive translations, footnoted
  translation, footnotes, warnings, and memory updates.

## Surviving Cluster Group Layer

The surviving cluster group layer is a read-only diagnostic/reporting layer that
groups accepted `root_glosses` rows by root and provisional sense bundle. The
layer must be implemented as a separate service/report path before any
translation integration.

### Code Boundaries

Implement the layer in translation/reporting code, not in root-extraction
orchestration.

- Put the grouping service in `src/translation/root_groups.py`.
- Put CLI parser wiring under the existing `enlm report` command family.
- Add tests under `tests/translation/test_root_groups.py`.
- Reuse existing repository/config helpers for insights DB paths. Do not
  hardcode the DB path used during exploratory analysis.
- Keep the service read-only. It may open SQLite connections and read views; it
  must not create tables, write cache rows, or mutate insight databases.

### Data Contracts

The names below are public JSON shape names. Python implementation may use
dataclasses, typed dictionaries, or plain dicts, but serialized output fields
must remain stable.

Use these JSON conventions everywhere:

- Missing scalar values are `null`.
- Missing list values are `[]`.
- Missing map/object values are `{}`.
- Unknown categorical values use `"unknown"`.
- Numeric values are floats unless explicitly described as counts.
- Public roots are uppercase; variant names remain lowercase.

#### `NestedEvidenceEffect`

Represents one item from `raw_glossator_json.EVIDENCE[*]`.

Required fields:

- `word`: source word from the evidence item, normalized to uppercase when
  present. Use `null` if missing.
- `sense`: evidence-item sense text, preserving usage snippets when present.
- `role`: normalized `note.role` when available. Allowed values are `prefix`,
  `suffix`, `infix`, `free`, `unknown`, or the original lowercased string if a
  new role appears.
- `effect`: `note.effect`; this is first-class semantic evidence.
- `confidence`: numeric `note.confidence` when available.
- `sense_alignment`: numeric `note.sense_alignment` when available.

Optional fields:

- `loc`: source location string.
- `note`: raw note payload for fields not promoted above.

#### `AttachmentSummary`

Represents row-level and observed positional behavior.

Required fields:

- `prefix_likelihood`
- `suffix_likelihood`
- `infix_likelihood`
- `free_likelihood`
- `productivity`
- `estimated_profile`
- `observed_prefix_count`
- `observed_suffix_count`
- `observed_infix_count`
- `observed_free_count`

Defaults:

- Missing likelihoods become `null`, not `0.0`.
- Missing observed counts become `0`.
- `estimated_profile` is `unknown` when no profile can be derived.
- `infix_likelihood` is usually `null` because current root-gloss rows do not
  expose it directly; observed infix counts still appear in
  `observed_infix_count`.

#### `RootEvidencePacket`

Represents one accepted `root_glosses` row plus attached positional evidence.

Required fields:

- `root`
- `variant`
- `evaluation`
- `source_cluster_id`
- `definition`
- `semantic_core`
- `decoding_guide`
- `negative_contrast`
- `examples`
- `nested_evidence`
- `confidence_score`
- `examples_in_cluster`
- `attachment`
- `raw_glossator_json`

Optional fields:

- `reason`
- `contribution`
- `confidence_drivers`
- `confidence_risks`
- `attachment_exceptions`
- `pos_bias`
- `source_run_id`, when available from surrounding cluster metadata.

Derived fields:

- `semantic_text`: normalized text used for embedding and lexical fallback.
- `surface_examples`: compact examples extracted from `examples` and
  `nested_evidence[*].sense`.
- `exception_terms`: terms extracted from attachment exceptions and risks.
- `packet_warnings`: row-level warning strings.

Packet IDs:

- `packet_id` must be `{variant}:{ROOT}:{source_cluster_id}`.
- If `source_cluster_id` is missing, use `{variant}:{ROOT}:row-{ordinal}` and
  add a `missing_source_cluster_id` packet warning.

#### `RootSenseGroupMember`

Represents a packet assigned to a group.

Required fields:

- `packet_id`: stable local identifier using the packet ID rules above.
- `source_cluster_id`
- `variant`
- `assignment_score`
- `assignment_reasons`
- `split_flags`

#### `RootSenseGroup`

Represents one provisional root sense group.

Required fields:

- `group_id`
- `root`
- `label`
- `status`
- `semantic_terms`
- `surface_examples`
- `source_cluster_ids`
- `variants`
- `members`
- `attachment_profile`
- `ranking`
- `warnings`
- `evidence_packets`

Allowed `status` values:

- `provisional`: default for generated groups.
- `needs_review`: group is plausible but has high ambiguity, exception density,
  or positional conflict.
- `stable`: group has strong support, consistent examples, and low conflict.
- `split_recommended`: group was formed but contains strong internal conflicts.
- `rejected_merge`: group is shown as a failed merge candidate for diagnostics.

#### `RootSenseGroupReport`

Represents the top-level diagnostic output.

Required fields:

- `root`
- `variants_queried`
- `generated_at`
- `grouping_version`
- `parameters`
- `groups`
- `ungrouped_packets`
- `diagnostics`

Report diagnostics must include:

- `packet_count`
- `group_count`
- `ungrouped_count`
- `rejected_merges`
- `embedding_backend`
- `warnings`
- `empty_reason`

`empty_reason` is `null` when groups exist. For a successful empty report, use
`"no_accepted_root_glosses"`.

#### `RootSenseGroupSummary`

Represents the compact form used by Phase 3 translation diagnostics and LLM
context.

Required fields:

- `root`
- `group_id`
- `label`
- `status`
- `rank_score`
- `semantic_terms`
- `surface_examples`
- `source_cluster_ids`
- `warnings`

Forbidden fields in compact summaries:

- `raw_glossator_json`
- full `evidence_packets`
- full nested evidence note payloads

This summary is the only group shape allowed in normal translation output unless
verbose/debug output explicitly requests the full report.

### Evidence Packet Construction

Implementation should construct packets in this order:

1. Normalize the requested root to both forms:
   - `root_lookup = lower(trim(root))` for SQLite queries;
   - `root_public = upper(trim(root))` for public output.
2. Query accepted `root_glosses` rows for `root_lookup` and the selected
   variants. If the `root_glosses` view is missing, fail with a clear
   missing-schema error. Do not create or migrate the DB from this command.
3. Attach `root_attachment_profile` rows by `(root, source_cluster_id)` when
   available, falling back to root-level aggregate counts if necessary.
4. Parse JSON columns into typed values:
   - `semantic_core`, `negative_contrast`, `examples_json`,
     `confidence_drivers`, `confidence_risks`, `attachment_exceptions`;
   - `contribution_json`;
   - nested `raw_glossator_json.EVIDENCE`.
5. Normalize only for comparison fields. Keep original strings in output.
6. Build `semantic_text` from weighted components:
   - `semantic_core`: weight `3.0`;
   - nested evidence `effect`: weight `3.0`;
   - nested evidence `sense`: weight `2.0`;
   - `definition`: weight `2.0`;
   - `decoding_guide`: weight `1.0`;
   - `examples_json`: weight `1.0`;
   - `attachment_exceptions` and `confidence_risks`: warning text, not primary
     similarity text.

Weight implementation:

- For sentence-transformer embeddings, embed each component separately and
  compute a weighted average vector using the weights above.
- For TF-IDF fallback, repeat each component text `ceil(weight)` times inside
  `semantic_text`.
- Expose these weights in report `parameters`.

JSON parsing rules:

- If a JSON column is already a JSON array/object string, parse it.
- If parsing fails, keep the original string under the relevant raw field when
  possible, return an empty parsed value, and add a packet warning.
- If `raw_glossator_json.EVIDENCE` is missing or malformed, set
  `nested_evidence` to `[]` and add `missing_nested_evidence`.
- Do not discard a packet only because optional JSON fields are malformed.

Text normalization rules for comparison:

- lowercase;
- strip punctuation except apostrophes inside words;
- collapse whitespace;
- remove empty tokens;
- do not stem or lemmatize in v1;
- keep original unnormalized strings in public output.

### Grouping Algorithm

The initial implementation should be deterministic and inspectable.

1. Embed `semantic_text` for every packet using the backend selected by the
   preference order in `Resolved Implementation Defaults`.
2. For the sentence-transformer backend, compute cosine similarity over sentence
   embeddings.
3. For TF-IDF fallback, use `TfidfVectorizer(lowercase=True,
   analyzer="word", ngram_range=(1, 2), min_df=1)` and cosine similarity.
4. Build initial groups as connected components where an edge exists when
   pairwise similarity is greater than or equal to
   `merge_similarity_threshold`.
5. Compute pairwise conflict checks for every proposed merge:
   - attachment divergence,
   - negative-contrast conflict,
   - exception/risk conflict,
   - role/effect divergence,
   - low support or low confidence.
6. Split or flag groups after initial clustering. Do not silently force a merge
   when conflict checks exceed thresholds.
7. Label each group from its strongest terms and examples. Labels are compact
   diagnostics, not canonical definitions.
8. Assign statuses and ranking metadata.
9. Preserve ungrouped packets when support is insufficient or all candidate
   merges are rejected.

Connected-component behavior:

- If packets A-B and B-C meet the similarity threshold but A-C has a strong
  conflict, keep all three in a provisional component only if the component
  status is `split_recommended`; otherwise split by removing the lowest
  similarity conflicting edge and recomputing components.
- Single-packet components are valid groups.
- Component ordering is deterministic: sort by `rank_score` descending, then
  label ascending, then smallest `source_cluster_id` ascending.

### Default Thresholds

These defaults are intentionally conservative.

- `merge_similarity_threshold`: `0.78`
- `strong_merge_similarity_threshold`: `0.86`
- `attachment_axis_divergence_threshold`: `0.35`
- `observed_attachment_ratio_divergence_threshold`: `0.45`
- `negative_contrast_overlap_threshold`: `1` overlapping normalized term
- `exception_density_needs_review_threshold`: `0.25`
- `low_confidence_threshold`: `0.65`
- `stable_min_packets`: `2`
- `stable_min_avg_confidence`: `0.78`
- `stable_max_attachment_divergence`: `0.25`
- `max_groups_default`: `12`

Conflict policy:

- If similarity is below `merge_similarity_threshold`, do not merge.
- If similarity is between `merge_similarity_threshold` and
  `strong_merge_similarity_threshold`, any strong conflict should prevent merge
  and create a `rejected_merge` diagnostic.
- If similarity is above `strong_merge_similarity_threshold`, strong conflicts
  may remain in one group only if the group status becomes `needs_review` or
  `split_recommended`.

### Split and Flag Rules

Attachment divergence:

- Compare `prefix_likelihood`, `suffix_likelihood`, and `free_likelihood` when
  both packets have values.
- Also compare observed attachment counts after normalizing each row to ratios.
- If the dominant axis differs and the difference is at least `0.35`, mark
  `attachment_divergence`.

Negative contrast conflict:

- Normalize contrast terms by lowercasing, removing punctuation, and splitting
  short comma/semicolon lists.
- If one packet's positive semantic terms overlap another packet's negative
  contrast terms, mark `negative_contrast_conflict`.
- Positive semantic terms are drawn from parsed `semantic_core`, contribution
  keys, nested evidence `effect` tokens, and non-meta definition spans.
- A conflict flag does not automatically split a group when the overlap comes
  from generic words: `state`, `being`, `existence`, `relation`, `marker`, or
  `morpheme`. These generic words should be ignored for this check.

Exception/risk conflict:

- Treat `attachment_exceptions` and `confidence_risks` as warning sources.
- Exception text does not define a group by itself.
- If at least 25 percent of packets in a group carry exception/risk warnings,
  set group status to `needs_review`.

Nested evidence divergence:

- Role divergence is expected for productive roots, but should be visible.
- If the same semantic group has both mostly prefix and mostly suffix evidence,
  keep the group but add `mixed_role_evidence`.
- If nested `effect` terms indicate incompatible domains, such as comfort-state
  versus negation/prohibition, prefer split unless similarity is very high and
  examples support the bridge.
- In v1, incompatible domains are detected deterministically, not by LLM. Mark
  `effect_domain_divergence` when the average similarity between nested
  evidence `effect` texts across two packets is below `0.45` and their
  `semantic_core` term overlap is empty after generic terms are removed.
- If nested evidence is missing for one packet, do not mark divergence solely
  because of missing data; add `missing_nested_evidence` instead.

Low support:

- Single-packet groups are allowed, but their status remains `provisional`
  unless confidence is high and no warnings are present.
- A single-packet group may be `stable` only if `confidence_score` is greater
  than or equal to `0.85`, nested evidence exists, and no exception/risk flags
  are present.

### Ranking Strategy

Group ranking is advisory and diagnostic. It should not decide translation
output until a separate group-aware mode exists.

Compute these ranking fields:

- `packet_count`
- `avg_confidence`
- `max_confidence`
- `examples_total`
- `nested_evidence_count`
- `avg_nested_evidence_confidence`
- `avg_nested_sense_alignment`
- `semantic_coherence`
- `attachment_consistency`
- `exception_density`
- `risk_density`
- `review_penalty`
- `rank_score`

Default rank formula:

```text
rank_score =
  0.25 * avg_confidence
+ 0.20 * semantic_coherence
+ 0.15 * attachment_consistency
+ 0.15 * avg_nested_evidence_confidence
+ 0.10 * avg_nested_sense_alignment
+ 0.10 * support_factor
+ 0.05 * examples_factor
- 0.20 * review_penalty
```

`support_factor` must use
`min(1.0, log1p(packet_count) / log1p(8))`.

`examples_factor` must use
`min(1.0, log1p(examples_total) / log1p(24))`.

Ranking field definitions:

- `avg_confidence`: average non-null packet `confidence_score`, default `0.0`
  if no packet has confidence.
- `max_confidence`: maximum non-null packet `confidence_score`, default `0.0`.
- `examples_total`: sum of non-null `examples_in_cluster` values plus count of
  parsed `examples` when `examples_in_cluster` is missing.
- `avg_nested_evidence_confidence`: average non-null nested evidence
  confidence, default `0.0`.
- `avg_nested_sense_alignment`: average non-null nested sense alignment,
  default `0.0`.
- `semantic_coherence`: average pairwise semantic similarity inside the group;
  single-packet groups use `1.0`.
- `attachment_consistency`: `1.0 - max_attachment_divergence`, clamped to
  `[0.0, 1.0]`.
- `exception_density`: packets with attachment exceptions divided by packet
  count.
- `risk_density`: packets with confidence risks divided by packet count.
- `review_penalty`: maximum of exception density, risk density, attachment
  divergence, and `1.0` when status is `split_recommended`.

Status assignment:

- `stable`: group meets `stable_min_packets`, `stable_min_avg_confidence`, and
  `stable_max_attachment_divergence`, with no strong conflict flags.
- `needs_review`: group has exception/risk density above threshold, low average
  confidence, mixed role evidence, or moderate conflicts.
- `split_recommended`: group has strong conflicts that remain after edge
  removal or contains a connected-component bridge with incompatible endpoints.
- `provisional`: default when no stronger status applies.
- `rejected_merge`: only appears inside diagnostics for attempted merges that
  conflict policy rejected; it should not appear in normal `groups`.

### Labeling Strategy

Labels should be compact and evidence-derived.

1. Prefer repeated `semantic_core` terms and nested evidence `effect` terms.
2. Prefer concrete evidence senses when the summary terms are generic.
3. Include positional/function words only when they are actually supported by
   POS and attachment evidence.
4. Avoid labels that are merely scaffolding, such as `state of being`, when
   concrete subdomains are available.
5. If an LLM is later used to label groups, it must receive the packet evidence
   and return a label plus reasoning. The deterministic label remains available
   as fallback.

Examples:

- `IO`: prefer `manifestation/state marker` with visible subdomains such as
  `woe`, `comfort`, `widowhood`, `temporal moment`, and `sound`.
- `I`: do not collapse all rows into `existence`; keep separate visible groups
  for existential/copular, negation/prohibition, comfort/solace,
  divine/flame/power, division/separation, repetition/succession, and other
  supported outliers.
- `A` or `D`: expect function-like groups such as locative, deictic,
  relational, ordinal, and connective meanings to overlap; preserve positional
  and example evidence before naming a canonical sense.

### Diagnostic Output Shape

The Phase 1 JSON report must emit this top-level shape. The example omits full
packet internals for readability, but `--detail full` must include populated
`evidence_packets`.

```json
{
  "root": "IO",
  "variants_queried": ["debate"],
  "grouping_version": "root-sense-groups-v1",
  "parameters": {
    "merge_similarity_threshold": 0.78,
    "strong_merge_similarity_threshold": 0.86
  },
  "groups": [
    {
      "group_id": "io:manifestation-state:1",
      "root": "IO",
      "label": "manifestation/state marker",
      "status": "needs_review",
      "semantic_terms": ["emergence", "manifestation", "existence"],
      "surface_examples": ["bring forth", "woe", "comfort", "widowhood"],
      "source_cluster_ids": [3069],
      "variants": ["debate"],
      "attachment_profile": {
        "prefix_likelihood": 0.7,
        "suffix_likelihood": 0.6,
        "free_likelihood": 0.2,
        "estimated_profile": "mixed"
      },
      "ranking": {
        "packet_count": 1,
        "avg_confidence": 0.85,
        "semantic_coherence": 1.0,
        "rank_score": 0.73
      },
      "warnings": [
        "OHIO is marked as negative manifestation rather than standard emergence."
      ],
      "members": [
        {
          "packet_id": "debate:IO:3069",
          "source_cluster_id": 3069,
          "variant": "debate",
          "assignment_score": 1.0,
          "assignment_reasons": ["single-packet group"],
          "split_flags": ["exception_present"]
        }
      ],
      "evidence_packets": []
    }
  ],
  "ungrouped_packets": [],
  "diagnostics": {
    "packet_count": 1,
    "rejected_merges": []
  }
}
```

## Implementation Phases

The phases are ordered by how much they affect existing behavior. Each phase has
a clear artifact and consumption story. Phase 1 creates the grouping engine and
JSON report. Phase 2 adds a human-readable view over the same engine. Phase 3
lets translation compute and attach group summaries on demand. Phase 4 is the
first phase allowed to influence ranking.

### Phase 1: Read-Only Diagnostic Report

Goal: create the grouping engine and expose its raw JSON output for inspection.

Trigger:

- User invokes the root-group command for exactly one root.

Inputs:

- root string;
- variant selection: `solo`, `debate`, or `both`;
- optional grouping parameters, defaulting to the thresholds in this document;
- active insights database paths from the existing config.

Artifact produced:

- `RootSenseGroupReport` JSON for one requested root and selected variants.

Storage and caching:

- The command writes only to stdout or an explicit `--output` path if provided.
- It must not write to SQLite, caches, generated artifacts, or repo-tracked
  files unless the user explicitly chooses an output path.
- Saved JSON is an export, not a canonical cache.

Consumption story:

- The report is a diagnostic artifact for humans and tests.
- It is not a prerequisite file for translation.
- It may be saved by redirecting CLI output, but no later phase should require
  saved JSON as the normal path.

Implementation shape:

- Add a read-only service that builds `RootSenseGroupReport` directly from the
  selected insights databases.
- Add a CLI/report command that calls that service for one root.
- Return a nonzero exit code only for invalid input, missing database paths, or
  unrecoverable query/JSON parsing failures. A root with no accepted rows should
  return an empty report and a successful exit.

Canonical command shape:

```bash
enlm report root-groups --root IO --variant debate --format json --pretty
```

Parser requirements:

- Add `root-groups` under the existing `enlm report` command family.
- Require `--root`.
- `--variant` choices are `solo`, `debate`, and `both`; default is `both`.
- `--format` choices are `json` and `text`; default is `json`.
- `--pretty` applies only to JSON.
- `--output <path>` writes the rendered output to that path; otherwise write to
  stdout.
- `--detail full|compact` controls JSON evidence packet inclusion; default is
  `full` for Phase 1/2 reports.
- `--max-groups <n>` defaults to `12` and caps rendered groups after ranking;
  diagnostics must still include the total pre-cap group count.

Requirements:

- read only from insights databases;
- support `solo`, `debate`, and `both`;
- support JSON output first;
- include all parameters in output;
- never write back to SQLite;
- never alter word or phrase translation outputs.

### Phase 2: Human-Facing Diagnostics

Goal: make the Phase 1 report easy to inspect without reading raw JSON.

Trigger:

- User invokes the same root-group command with text output, or invokes a text
  report wrapper over the Phase 1 service.

Inputs:

- Same inputs as Phase 1.
- No saved JSON input is required.
- `--detail` controls how much evidence appears in JSON only; text output is
  always compact and human-facing.

Artifact produced:

- Text rendering of the same `RootSenseGroupReport` produced in Phase 1.

Storage and caching:

- Text output follows normal CLI output rules: stdout by default, optional
  explicit output path if the command supports it.
- It must not create or update any persistent report cache.

Consumption story:

- The text report is for human review only.
- It does not create a new data source and does not feed translation.

Implementation shape:

- Reuse the Phase 1 grouping service.
- Add `--format text` or a report renderer that formats the JSON report.
- Text output should be derived from the same in-memory report object used for
  JSON output so text and JSON cannot drift.

The text report should show:

- root and variants queried;
- group rank, label, status, score, and warnings;
- semantic terms and surface examples;
- source cluster IDs;
- attachment summary;
- top nested evidence effects;
- rejected merge notes when available.

### Phase 3: Optional Translation Context

Goal: let translation attach compact root-group summaries as optional context
without changing deterministic candidate selection.

Trigger:

- User invokes `translate-word` or `translate-phrase` with
  `--with-root-groups`.

Inputs:

- The ordinary word/phrase translation inputs.
- The selected variants and insights DB connections already opened by the
  translation service.
- The unique morphs from selected candidates after ordinary deterministic
  candidate selection:
  - for `translate-word`, use all morphs from returned candidates up to
    `top_k`;
  - for `translate-phrase`, use all morphs from token choices in the chosen
    parse only by default.
- Phase 3 v1 has no external group-report input. It computes live only.

Artifact produced:

- Additional diagnostic/context fields inside `translate-word` and
  `translate-phrase` output payloads.
- Optional compact group summaries inside phrase LLM render payloads when LLM
  rendering is enabled.

Storage and caching:

- Default behavior computes groups live in memory.
- Request-level memoization is allowed to avoid regrouping the same root during
  one command invocation.
- No persistent cache is required or created.
- `--root-groups-json <path>` is explicitly out of scope for Phase 3 v1. If it
  is later added, it is a validated override or replay input for debugging, not
  the default data path.

Consumption story:

- `--with-root-groups` must compute root groups live from the same insights
  database connections already used by translation.
- Translation must not require a previously generated
  `enlm report root-groups` JSON file.
- A later `--root-groups-json <path>` debug/caching option must be specified in
  a separate doc update before implementation. It must validate root, variant,
  grouping version, parameters, and source cluster IDs against the active
  translation inputs.

Implementation shape:

- Add the explicit opt-in flag `--with-root-groups` for word and phrase
  translation.
- For `translate-word`, compute group reports for the unique morphs from
  returned candidates up to `top_k`. Attach compact summaries under
  diagnostics, for example `diagnostics.root_groups`.
- For `translate-phrase`, compute group reports for chosen token candidate
  morphs after parse selection. Attach compact summaries under token
  diagnostics and, when `--llm` is enabled, include a bounded summary in the LLM
  render context.
- Use a request-level cache keyed by `(variant_set, root, grouping_version,
  parameters)` so repeated roots in a phrase are grouped once.
- Do not compute groups for every possible substring by default; that is too
  expensive and too noisy. Broader substring grouping is out of scope for Phase
  3 v1 and requires a separate flag/spec update.

Requirements:

- group evidence must appear under diagnostics or LLM context metadata, not as
  a replacement for selected candidate meanings;
- deterministic ranking must remain unchanged unless a separate
  `--use-root-groups-for-ranking` flag exists;
- `--with-root-groups` without `--llm` must still add diagnostic group
  summaries to output, but must not affect rendered definitions;
- phrase LLM payloads must receive group summaries only when
  `--with-root-groups` and `--llm` are both enabled;
- group summaries must include source IDs and warnings.
- group summaries must be compact by default: label, status, rank score,
  semantic terms, surface examples, source cluster IDs, and warnings.
- full evidence packets must remain available in Phase 1 reports and must
  not be shoved wholesale into normal translation output unless verbose/debug
  output explicitly asks for them.
- LLM context summaries must include at most the top 3 groups per root and at
  most 5 surface examples per group.

### Phase 4: Optional Ranking Experiment

Goal: evaluate whether surviving cluster groups improve deterministic
translation ranking.

Trigger:

- User invokes `translate-word` or `translate-phrase` with the explicit
  experimental ranking flag `--use-root-groups-for-ranking`.
- The flag implies `--with-root-groups`; group summaries are computed if not
  already present.

Inputs:

- Ordinary translation candidate pool and decision rows.
- Live `RootSenseGroupReport` summaries for candidate morphs.
- Phase 4 v1 has no cached report input. It computes live only.

Artifact produced:

- Experimental ranking diagnostics showing baseline rank, group-aware rank, and
  the group-derived features that changed the score.

Storage and caching:

- Default behavior computes live in memory and writes no persistent state.
- Before/after diagnostics are emitted in the translation payload.
- Any experiment export must be explicitly requested with an output path.

Consumption story:

- This phase consumes live group summaries from the Phase 1 service.
- Cached reports for reproducible experiments are out of scope for Phase 4 v1
  and require a separate spec update.
- Existing translation ranking must remain unchanged unless the experimental flag is
  present.

Implementation shape:

- Add the explicit experimental flag `--use-root-groups-for-ranking`.
- Compute group-derived features only after ordinary candidates are built.
- Apply group-derived scoring as a small additive component, never as a hard
  override.
- Emit before/after decision rows so regressions are inspectable.

Requirements:

- gated behind an explicit flag;
- default off;
- deterministic baseline decision rows must still be emitted;
- emits before/after ranking diagnostics;
- has fixtures showing improvements and regressions;
- can be disabled without changing existing outputs.

## Failure Modes and Guardrails

- **LLM gloss laundering:** Previous LLM definitions may cluster because they
  share wording, not because corpus evidence supports one sense. Mitigation:
  weight nested evidence effects and examples; preserve source rows.
- **Over-abstracting grammatical roots:** Embeddings may merge `being`,
  `state`, `identity`, and `presence` too broadly. Mitigation: keep positional,
  role, contrast, and examples visible.
- **Example-as-definition drift:** Example text may include context that does
  not belong to the root. Mitigation: examples support warnings and labels but
  do not define a group alone.
- **Exception-as-primary-meaning drift:** `attachment_exceptions` can identify
  critical sub-senses but should not become the main group meaning by itself.
- **Position loss:** Prefix, suffix, infix, and free behavior may represent
  different functions. Mitigation: attachment divergence creates split flags or
  review status.
- **Homograph collapse:** Same surface root may represent distinct morphemes.
  Mitigation: keep runner-up groups and rejected merge diagnostics.
- **Variant flattening:** Solo and debate may disagree. Mitigation: preserve
  variant on every packet, member, and group.
- **Prompt bloat:** Feeding full packet evidence into LLM rendering may become
  expensive. Mitigation: expose compact group summaries and require explicit
  opt-in.

## Interpretation Policy

Deterministic evidence is authoritative. The pipeline may use LLMs for readable
rendering, bundle labeling, or adjudicating close alternatives, but LLM output
must not erase evidence provenance or introduce unsupported etymology.

The practical policy is:

- preserve raw evidence before summarizing it;
- prefer `semantic_core` and cleaned lexical definitions for human-facing
  surfaces, but consult examples and nested evidence effects when semantics are
  underspecified;
- treat short roots as potentially polysemous or homographic until grouped
  evidence proves otherwise;
- keep diagnostics visible whenever a root has multiple plausible groups;
- wire any future group-aware scoring behind an explicit flag and test it
  separately from the diagnostic implementation.

## Acceptance Criteria

### Documentation Acceptance

- The specification names every public data shape required for implementation.
- Defaults and thresholds are explicit.
- Non-goals prevent accidental ranking or prompt changes.
- Failure modes and guardrails are visible to future implementers.

### Phase 1 Acceptance: JSON Diagnostic Report

- A root-sense grouping service produces a `RootSenseGroupReport` JSON object
  for one requested root.
- The command accepts `solo`, `debate`, and `both` variants and records the
  selected variants in output.
- The command returns a successful empty report when the root has no accepted
  `root_glosses` rows.
- Reports preserve `root`, variant, `source_cluster_id`, definitions,
  `semantic_core`, examples, nested evidence effects, attachment data,
  confidence, risks, and raw provenance.
- Missing optional fields are represented as `null`, empty arrays, or explicit
  `unknown` values; missing data does not crash grouping.
- Grouping parameters and thresholds are included in every JSON report.
- The command is read-only and does not mutate SQLite, caches, generated
  artifacts, or repo-tracked files.
- Existing `translate-word` and `translate-phrase` outputs are unchanged.

### Required Tests for Phase 1

- Packet extraction preserves nested `EVIDENCE[*].note.effect`.
- `IO` cluster `3069` keeps `negative manifestation` / `woe` visible as a
  warning or surface example rather than flattening it into generic existence.
- A short-root fixture with multiple accepted rows returns multiple groups or
  at least `needs_review` / `split_recommended` status rather than one silent
  merge.
- Attachment divergence creates a split flag or review status.
- Solo and debate provenance remain distinguishable when `--variant both` is
  used.
- Fallback lexical grouping works when the sentence-transformer embedder is
  unavailable.
- No accepted root rows returns successful JSON with empty `groups` and an
  explanatory diagnostic.

### Phase 2 Acceptance: Text Diagnostics

- Text output is generated from the same `RootSenseGroupReport` object as JSON
  output.
- Text output includes root, variants, group labels, statuses, rank scores,
  warnings, semantic terms, surface examples, source cluster IDs, attachment
  summary, top nested evidence effects, and rejected merge notes.
- Text output does not require or consume a saved JSON report.
- Text output does not write persistent state.
- JSON output remains unchanged after adding text rendering.

### Required Tests for Phase 2

- Text rendering of a fixture report includes the top group label, status,
  source cluster ID, and at least one nested evidence effect.
- Text rendering of an empty report clearly states that no accepted groups were
  found.
- JSON output for the same fixture remains byte-for-byte or structurally
  unchanged except for expected timestamp fields.

### Phase 3 Acceptance: Optional Translation Context

- `translate-word --with-root-groups` computes group summaries live for selected
  candidate morphs and places compact summaries under diagnostics.
- `translate-phrase --with-root-groups` computes group summaries live for chosen
  token candidate morphs and places compact summaries under token or phrase
  diagnostics.
- `--with-root-groups` without `--llm` does not change candidate ranking,
  selected definitions, deterministic skeletons, or rendered output fields.
- `--with-root-groups --llm` may include compact group summaries in LLM context,
  but selected morph meanings remain present and authoritative.
- Translation does not require a prior `enlm report root-groups` JSON export.
- Request-level memoization prevents repeated roots in one phrase from being
  regrouped more than once.
- Phase 3 v1 has no `--root-groups-json` input.

### Required Tests for Phase 3

- Phrase LLM payloads receive grouped evidence only when explicitly enabled.
- Group-aware translation context does not replace selected morph meanings.
- `translate-word` output without `--with-root-groups` is unchanged from the
  baseline.
- `translate-word --with-root-groups --no-llm` has the same candidate order and
  definitions as baseline output, plus root-group diagnostics.
- `translate-phrase --with-root-groups --no-llm` has the same chosen parse and
  deterministic skeleton as baseline output, plus root-group diagnostics.
- Repeated root grouping in a phrase uses request-level memoization.

### Phase 4 Acceptance: Optional Ranking Experiment

- `--use-root-groups-for-ranking` is explicit, default off, and implies or
  requires `--with-root-groups`.
- Baseline decision rows and group-aware decision rows are both emitted.
- Group-derived score components are visible and bounded.
- Group-derived scoring is additive and cannot hard-override hard filters or
  unsupported decompositions.
- Disabling the flag restores baseline candidate order and output.
- Fixtures include at least one intended improvement and one guarded
  regression/neutral case.

### Required Tests for Phase 4

- Experimental ranking mode can be toggled off and leaves baseline output
  unchanged.
- Before/after diagnostics show any ranking change caused by group-aware
  scoring.
- Group-derived scoring does not resurrect hard-filtered candidates.
- A regression fixture proves that exception-heavy or split-recommended groups
  do not automatically outrank cleaner baseline evidence.
