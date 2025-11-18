# Dictionary Schema

This document records the JSON structure used by `dictionary.json` and the derived file `dictionary_enriched.json`. Each entry describes one lemma and contains metadata plus one or more sense objects.

## Entry-level fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `word` | `str` | Canonical lemma spelling from the source dictionary. |
| `normalized` | `str` | Normalized form (lowercased and simplified) used for matching. |
| `canon_word` | `bool` | Indicates whether the lemma appears in the core Dee/Kelley corpus. |
| `definition` | `str` | Primary English gloss or summary provided by the original dictionary. |
| `key_citations` | `List[Dict[str, str]]` | Optional supporting citations with `location` and `context`. |
| `senses` | `List[Sense]` | Individual senses for the lemma (see below). |

Other historical metadata from the upstream sources are preserved verbatim.

## Sense fields

Each item in `senses` is an object with at least the following keys:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `sense_id` | `int` | Local identifier scoped to the lemma. |
| `definition` | `str` | English gloss text for the sense. |
| `parts_of_speech` | `List[str]` | POS tags inferred by heuristics. Values follow the UD-style tags (`NOUN`, `VERB`, `ADJ`, `ADV`, `ADP`, `PRON`, `AUX`, `CCONJ`). A fallback `"NOUN"` is inserted if no other rule applies. |
| `semantic_domains` | `List[str]` | Conceptual domains tied to the gloss headword. Domains are defined in `src/training/config/semantic_domains.yml`. Multiple labels may be assigned when a gloss overlaps categories. |
| `is_copula` | `bool` | True when the gloss indicates copular usage (`"to be"`, `"is"`, etc.). |
| `is_compound_standing_for_phrase` | `bool` | True when the gloss clearly represents a multi-word English phrase (commas, slashes, long paraphrases). |
| `notes_pos` | `Optional[str]` | Human note describing why a POS choice was made. The enrichment script only sets this when a fallback POS is applied. |

Future pipelines can add further annotations while keeping this baseline compatible.

## Example

```
{
  "word": "AAI",
  "normalized": "aai",
  "canon_word": true,
  "definition": "amongst, amongst you",
  "senses": [
    {
      "sense_id": 1,
      "definition": "amongst, amongst you",
      "parts_of_speech": ["ADP"],
      "semantic_domains": ["SOCIAL"],
      "is_copula": false,
      "is_compound_standing_for_phrase": true,
      "notes_pos": null
    }
  ]
}
```

The enriched dictionary is written to `src/enochian_lm/root_extraction/data/dictionary_enriched.json` by the script `training/datasets/enrich_dictionary_pos.py`.
