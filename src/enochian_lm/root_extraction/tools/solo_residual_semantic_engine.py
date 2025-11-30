"""Subtractive semantics for singleton residual fragments.

This module performs a secondary pass over residual fragments that only ever
appear as uncovered pieces of longer words. Given a root with an accepted
gloss and a composite word that contains it, we ask the LLM to speculate about
the leftover substring once the root's contribution is subtracted.

Example (conceptual)
--------------------
>>> engine.process_run("demo-run")  # doctest: +SKIP
root=NAZ, word=NAZPSAD, residual=PSAD → propose gloss for PSAD
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Optional, Any

from crewai import Task

from enochian_lm.root_extraction.tools.query_model_tool import QueryModelTool
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord
from enochian_lm.root_extraction.utils.embeddings import (
    get_sentence_transformer,
    select_definitions,
    stream_text,
)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.api_requestor").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

embedder = get_sentence_transformer("all-MiniLM-L6-v2")


def _get_field(item, field, default=""):
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def solo_analyze_remainder(
    root: str,
    candidates: list[EntryRecord],
    stats_summary: str,
    stream_callback=None,
    root_entry: Optional[EntryRecord] = None,
    use_remote: bool = True,
    residual_prompt: str | None = None,
    residual_guidance: dict | None = None,
    query_db: Any | None = None,
    query_run_id: Any | None = None,
):
    """
    Solo-style analysis for *residual*-leaning roots, reusing the same JSON
    schema as solo_analysis_engine but allowing extra guidance about
    remainders / uncovered fragments.
    """
    joined_defs: list[str] = []
    evidence_prompt_portion: list[str] = []

    candidate_list = ", ".join(_get_field(c, "word", "").upper() for c in candidates)

    for c in candidates:
        word = _get_field(c, "word", "")
        definition = _get_field(c, "enhanced_definition", "")
        fasttext = round(float(_get_field(c, "fasttext", "0.0")), 3)
        semantic = round(float(_get_field(c, "semantic", "0.0")), 3)
        tier = _get_field(c, "tier", "Untiered")

        if word and definition:
            line = (
                f"{word.strip()} — {definition.strip()} "
                f"<fasttext:{fasttext}, semantic similarity:{semantic}, tier:{tier}>"
                if fasttext > 0 or semantic > 0 or tier != "Untiered"
                else ""
            )
            joined_defs.append(line)

        if word:
            evidence_prompt_portion.append(
                json.dumps(
                    {
                        "word": word,
                        "sense": definition,
                        "loc": "dict/corpus",
                        "note": {
                            "role": 'must be one of: "prefix", "suffix", "free", "infix" (choose exactly one)',
                            "effect": f"effect of {root.upper()} on {word}",
                            "sense_alignment": (
                                "cosine-ish semantic alignment between the sense of "
                                f"{word} and {root.upper()}'s proposed semantics"
                            ),
                            "confidence": "must be a float between 0.0 and 1.0 (e.g., 0.75, 0.92)",
                            "note": (
                                f"breakdown for how {root.upper()} contributes to the sense of {word}"
                            ),
                        },
                    },
                    ensure_ascii=False,
                )
            )

    evidence_prompt_text = (
        ",\n    ".join(evidence_prompt_portion) if evidence_prompt_portion else ""
    )

    if root_entry is None:
        root_entry = next(
            (
                c
                for c in candidates
                if _get_field(c, "normalized", "").lower() == root.lower()
            ),
            None,
        )

    # === ROOT-LEVEL CONTEXT ===
    stats_section = f"ROOT-LEVEL STATS\n{stats_summary}\n\n"

    candidate_defs = (
        "\n".join(joined_defs) if joined_defs else "(no definitions available)"
    )

    lexicon_section = textwrap.dedent(
        f"""
        LEXICON CONTEXT
        The candidate residual root is: {root.upper()}.

        The following words contain or may be morphologically related to {root.upper()}:

        {candidate_defs}
        """
    ).strip()

    residual_section = ""
    if residual_prompt or residual_guidance:
        residual_bits: list[str] = []
        if residual_prompt:
            residual_bits.append(f"FOCUS NOTE (residual): {residual_prompt}")
        if residual_guidance:
            try:
                pretty = json.dumps(residual_guidance, ensure_ascii=False, indent=2)
            except TypeError:
                pretty = str(residual_guidance)
            residual_bits.append("RESIDUAL GUIDANCE (analytics):\n" + pretty)
        residual_section = "\n\n".join(residual_bits)

    about_task = textwrap.dedent(
        f"""
        You are analyzing a *residual-style* root candidate: {root.upper()}.

        Treat {root.upper()} exactly like any other hypothesized root, but be
        especially attentive to how it behaves as a *remainder* or *leftover*
        segment when other, stronger roots are subtracted.

        Use ONLY the evidence given here. Do not import external etymology,
        theology, or conlang lore.
        """
    ).strip()

    about_metrics = (
        "The metrics are as follows:\n"
        "- FastText Score—measures surface-level similarity based on character n-grams; "
        "ranges 0.0 to 1.0, with higher being more morphologically similar.\n"
        "- Semantic Similarity: Compares word definitions using sentence embeddings; "
        "ranges 0.0 to 1.0, with the higher the number the more conceptually aligned.\n"
        "- Tier: a very strong connection begins/ends with the root and has a high "
        "combined score and should be taken into special consideration; from there, "
        "possible connection > somewhat possible connection > weak or no connection.\n\n"
        "For residual-style roots, pay extra attention to whether the leftover segment "
        "seems to add a coherent semantic twist across its host words, or whether it "
        "behaves like noise."
    )

    default_evidence_entry = json.dumps(
        {
            "word": "<word>",
            "sense": "<definition>",
            "loc": "dict/corpus",
            "note": {
                "role": 'must be one of: "prefix", "suffix", "free", "infix" (choose exactly one)',
                "effect": f"effect of {root.upper()} on <word>",
                "sense_alignment": (
                    "cosine-ish semantic alignment between the sense of <word> and "
                    f"{root.upper()}'s proposed semantics"
                ),
                "confidence": "must be a float between 0.0 and 1.0 (e.g., 0.75, 0.92)",
                "note": f"breakdown for how {root.upper()} contributes to the sense of <word>",
            },
        },
        ensure_ascii=False,
    )
    evidence_entries = evidence_prompt_text or default_evidence_entry
    indented_evidence = textwrap.indent(evidence_entries, "    ")

    output_template = textwrap.dedent(
        f"""
        {{
          "ROOT": "{root.upper()}",
          "EVALUATION": "accepted or rejected (choose exactly one)",
          "REASON": "1-3 sentences explaining the reason for the evaluation selected",
          "DEFINITION": "1-3 sentences of core semantics; no negatives, be as concrete as possible and not vague",
          "EXAMPLE": "give 1-3 short example sentences of how its English equivalence would be used, marking it in each sentence",
          "DECODING_GUIDE": "concrete rules to resolve compound words, <=25 words",
          "SEMANTIC_CORE": ["up to three nouns or gerunds that captures the semantics of the root {root.upper()}"],
          "NEGATIVE_CONTRAST": ["max 4 phrases (e.g., 'non-temporal', 'non-agentive')"],
          "CONTRIBUTION": {{"lemmas describing ontology of {root.upper()}, accompanied by a rating of semantic composition": 0.0}},
          "POS_BIAS": {{"nounness": 0.0, "modifier": 0.0, "verbness": 0.0}},
          "ATTACHMENT": {{
            "prefix": {{"prob": 0.0}},
            "suffix": {{"prob": 0.0}},
            "free": {{"prob": 0.0}},
            "productivity": 0.0,
            "exceptions": ["descriptions of exceptions found with one string per exception"]
          }},
          "RESIDUAL_IMPACT": {{
            "coverage_gain_mean": 0.0,
            "residual_drop_mean": 0.0,
            "n_examples": 0
          }},
          "EVIDENCE": [
            {indented_evidence}
          ],
          "CONFIDENCE": {{
            "score": 0.0,
            "drivers": ["list of one to three short phrases that explain why you are confident in this analysis"],
            "risks": ["list of one to three short phrases that explain where your reservations are in this analysis"]
          }}
        }}
        """
    ).strip()

    # === ABLOV / SUBABLOV EXAMPLE ===
    example_output = textwrap.dedent(
        """
        {
          "ROOT": "ABLOV",
          "EVALUATION": "accepted",
          "REASON": "The root 'ABLOV' consistently appears in words describing affective or emotional states with coherent, aligned semantics.",
          "DEFINITION": "Open, expressive affection or heartfelt emotional presence that radiates outward toward others.",
          "EXAMPLE": [
            "Their ablov toward one another was obvious to everyone in the room.",
            "She carried an ablov for her community that shaped every decision she made."
          ],
          "DECODING_GUIDE": "Treat ABLOV as the core 'felt affection / emotional presence'. Affixes modulate where that affection sits (underneath, beyond, intensified) or how it manifests.",
          "SEMANTIC_CORE": ["affection", "emotional-presence", "warmth"],
          "NEGATIVE_CONTRAST": ["apathy", "emotional-withdrawal", "indifference", "coldness"],
          "CONTRIBUTION": {
            "affection": 0.95,
            "warmth": 0.9,
            "emotional-presence": 0.85,
            "expression": 0.75
          },
          "POS_BIAS": {
            "nounness": 0.9,
            "modifier": 0.4,
            "verbness": 0.2
          },
          "ATTACHMENT": {
            "prefix": {
              "prob": 0.7
            },
            "suffix": {
              "prob": 0.6
            },
            "free": {
              "prob": 0.3
            },
            "productivity": 0.7,
            "exceptions": [
              "Some idiomatic ABLOV-like forms behave more like full clauses than simple modifiers."
            ]
          },
          "RESIDUAL_IMPACT": {
            "coverage_gain_mean": 0.75,
            "residual_drop_mean": 0.2,
            "n_examples": 3
          },
          "EVIDENCE": [
            {
              "word": "SUBABLOV",
              "sense": "The underlying layers of emotion beneath an ABLOV-like outward expression.",
              "loc": "dict/corpus",
              "note": {
                "role": "prefix",
                "effect": "SUB- highlights what lies underneath or beneath the visible ABLOV state.",
                "sense_alignment": 0.9,
                "confidence": 0.9,
                "note": "Interpreted as the 'underbelly' or submerged portion of the same affectionate/emotional field."
              }
            }
          ],
          "CONFIDENCE": {
            "score": 0.9,
            "drivers": [
              "consistent emotional/affective readings across examples",
              "clear modulation of ABLOV semantics in compounds like SUBABLOV"
            ],
            "risks": [
              "possible conflation with more generic 'love' semantics in neighboring vocabulary"
            ]
          }
        }
        """
    ).strip()

    task_description = textwrap.dedent(
        f"""


        {stats_section}

        {lexicon_section}

        {residual_section}

        {about_task}

        {about_metrics}

        TASK
        ----
        1. Decide whether {root.upper()} should be accepted as a coherent root in this residual-ish space.
        2. If accepted, propose a concrete, non-negative definition and decoding guide.
        3. Summarize its semantic core and contrasts.
        4. Describe how it attaches (prefix/suffix/free) and how productive it seems.
        5. Provide structured evidence entries for the most informative words that contain it.
        6. Estimate an overall confidence score with drivers and risks.

        OUTPUT (JSON ONLY; use double quotes; no trailing commas)

        - The JSON MUST follow the schema shown in the template below.
        - If "EVALUATION" is "rejected", fill "REASON" and keep other fields minimal (empty strings/arrays).
        - Numeric scores are floats 0.00–1.00.

        TEMPLATE
        --------
        {output_template}

        CONSTRAINTS
        -----------
        - Use only data in INPUT; no external etymologies or languages.
        - Do not cite or invent Enochian items beyond {candidate_list}.
        - Be concise; no hedging. If any required field cannot be confidently filled, set "EVALUATION":"rejected".
        """
    ).strip()

    do_it_all = Task(
        description=task_description,
        expected_output=example_output,
    )

    # === Direct Tool Access with Streaming ===
    GRAY = "\033[38;5;250m"
    RESET = "\033[0m"

    lexicographer_cb = (
        (lambda _role, content: stream_callback("Lexicographer", content))
        if stream_callback
        else None
    )

    # separator between words
    print(
        f"\n==={(len('Now examining the possible residual root ') + len(f'<{root.upper()}>')) * '='}==="
    )
    print(f"===Now examining the possible residual root '{root.upper()}'===")
    print(
        f"==={(len('Now examining the possible residual root ') + len(f'<{root.upper()}>')) * '='}==="
    )

    stream_text(do_it_all.description)
    print(f"\n{RESET}\n")

    lexicographer = QueryModelTool(
        system_prompt=f"""
You are a **disciplined and insightful computational linguist** specializing in the Enochian language—a constructed system with irregular morphology, cryptic derivations, and unknown origin.

⚠️ DO NOT reference natural language etymologies (e.g., English, Greek, Latin, Hebrew). No speculative outside sources.
All reasoning must rely exclusively on **internal evidence**—relationships and patterns among the Enochian words themselves.

Your tone must be confident, scholarly, and analytical.

Be thorough, avoid vague generalizations, and always back claims with observed data.""",
        name="Lexicographer",
        description="",
        use_remote=use_remote,
    )

    if query_db is not None and query_run_id is not None:
        lexicographer.attach_logging(query_db, query_run_id)

    raw_response = lexicographer._run(
        prompt=do_it_all.description,
        stream_callback=lexicographer_cb,
        print_chunks=True,
        role_name="👩‍🏫\tLexicographer",
    )

    response = raw_response["response_text"]
    model = raw_response.get("gloss_model", "")

    archivist = [
        "\n\n\n========================\n====== TRANSCRIPT ======\n========================\n\n"
    ]
    archivist.append("=== 📖 PROMPT FOR LEXICOGRAPHER ===\n")
    archivist.append(do_it_all.description)
    archivist.append("\n\n")
    archivist.append("=== 👩‍🏫 LEXICOGRAPHER ===\n")
    archivist.append(response)
    archivist_recording = "\n".join(archivist)

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = {"RAW_TEXT": response}

    return {
        "Glossator": parsed,
        "Model": model,
        "Glossator_Prompt": do_it_all.description,
        "Archivist": archivist_recording,
        "raw_output": {
            "Glossator": response,
            "Glossator_Prompt": do_it_all.description,
            "Model": model,
            "Archivist": archivist_recording,
        },
    }
