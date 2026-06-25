# FILE FOR ARCHIVAL PURPOSES ONLY
"""Shared prompt wording for concrete root glosses.

The root extraction engines build several prompt variants, but they all write
into the same persisted gloss schema. This module centralizes the wording that
keeps definitions concrete and evidence-valenced so solo and debate modes do
not drift into generic abstractions for hard evidence such as OHIO = woe.
"""

from __future__ import annotations

import textwrap


OPINIONATED_GLOSS_STYLE_MARKER = "OPINIONATED GLOSS STYLE"


def opinionated_gloss_style_directive() -> str:
    """Return the shared glossing policy used by solo and debate prompts.

    Why: root glosses are downstream evidence for translation and residual
    analysis, so vague definitions like "state of being" leave too little for
    later systems to work with.
    How: each engine injects this block near the task or gloss schema, where it
    can steer DEFINITION, SEMANTIC_CORE, CONTRIBUTION, examples, and evidence.
    Responsibility: preserve affective or hardship valence when the provided
    internal evidence actually carries it, without changing the JSON schema.
    """

    return textwrap.dedent(
        f"""
        {OPINIONATED_GLOSS_STYLE_MARKER}
        - Write affirmative dictionary-style glosses, but preserve negative,
          painful, judgmental, affective, or hardship valence when the evidence
          carries it.
        - Do not launder evidence into generic abstractions. If a source sense
          is "woe", "suffering", "distress", "judgment", "curse", or another
          forceful condition, keep that force visible in DEFINITION,
          SEMANTIC_CORE, CONTRIBUTION, EXAMPLE, DECODING_GUIDE, and EVIDENCE.
        - Abstractions such as "state", "condition", "presence", "existence",
          or "manifestation" are allowed only when paired with the concrete
          force they carry, such as "manifested distress", "afflicted
          condition", or "judgment-laden proclamation".
        - For OHIO-style evidence, prefer concrete readings like distress,
          affliction, woe-manifestation, vocalized judgment, or suffered
          condition over bare phrases like "state of being".
        """
    ).strip()


def glossator_system_prompt() -> str:
    """Return the shared system prompt for final dictionary-style glossing.

    Why: debate and residual-debate glossators previously had identical prose
    copied inline, which made it easy for prompt policy to diverge.
    How: debate engines pass this system prompt to the final Glossator tool.
    Responsibility: define the final glossator persona and include the shared
    anti-laundering directive without altering model I/O contracts.
    """

    return (
        "You are a highly precise Enochian glossator. Your role is to propose "
        "a single, clear, authoritative dictionary-style definition for a root "
        "word that has been approved by an adjudicator. Use the prior "
        "linguistic analysis to distill the core conceptual meaning of the "
        "word, based solely on its internal usage patterns, morphology, and "
        "semantic range across the cited examples. Avoid descriptive "
        "summaries. Instead, craft a definition that would be suitable for "
        "formal inclusion in a lexicon. This definition should be concise "
        "(1-2 lines), but maximally informative. You must not reference "
        "English or natural-language etymology. Write in an academic tone, as "
        "if submitting this to a linguistic corpus project.\n\n"
        f"{opinionated_gloss_style_directive()}"
    )


def definition_instruction() -> str:
    """Return schema wording for the DEFINITION field.

    Why: the old "no negatives" phrase discouraged preserving negative
    evidence, which erased meanings like woe or suffering.
    How: prompt templates use this instruction as the DEFINITION placeholder.
    Responsibility: ask for affirmative, concrete prose while preserving
    evidence-backed valence.
    """

    return (
        "1-3 sentences of core semantics; use affirmative dictionary-style "
        "phrasing, but preserve evidence-backed hardship, distress, judgment, "
        "or affective force; avoid vague abstractions by themselves"
    )


def example_instruction() -> str:
    """Return schema wording for the EXAMPLE field.

    Why: examples are a quick human check that the gloss has operational force.
    How: prompt templates use this instruction without adding new JSON fields.
    Responsibility: make examples show the concrete English force of the root.
    """

    return (
        "give 1-3 short English examples that show the concrete force of the "
        "gloss, including hardship or judgment when the evidence supports it"
    )


def decoding_guide_instruction() -> str:
    """Return schema wording for the DECODING_GUIDE field.

    Why: downstream translation needs a compact rule that says what the root
    contributes, not just an abstract label.
    How: prompt templates use this instruction inside the existing schema.
    Responsibility: require concrete compound-resolution guidance.
    """

    return (
        "concrete rules to resolve compound words, <=25 words; name the "
        "semantic force the root adds, not only an abstract category"
    )


def semantic_core_instruction(root: str) -> str:
    """Return schema wording for SEMANTIC_CORE.

    Why: semantic cores feed clustering and reports, so generic nouns alone
    make roots harder to use.
    How: prompt templates pass the current root into this instruction.
    Responsibility: push compact labels toward concrete domains or forces.
    """

    root_name = root.upper()
    return (
        "up to three concrete nouns or gerunds that capture the semantic force "
        f"of {root_name}; pair abstractions like state/presence with the "
        "specific affect, judgment, hardship, or action when present"
    )


def contribution_instruction(root: str) -> str:
    """Return schema wording for CONTRIBUTION.

    Why: contribution weights are later consumed as semantic priors, so their
    keys should be usable lemmas rather than bland ontology labels.
    How: prompt templates pass the current root into this instruction.
    Responsibility: request weighted concrete meanings in the existing object.
    """

    root_name = root.upper()
    return (
        f"concrete lemmas describing what {root_name} contributes, including "
        "valence-bearing lemmas when supported, accompanied by semantic "
        "composition ratings"
    )


def evidence_effect_instruction(root: str, word: str) -> str:
    """Return schema wording for an EVIDENCE note.effect value.

    Why: evidence rows should explain how a root changes a host word rather
    than repeat that it has an "effect".
    How: candidate evidence templates use this value for each attested word.
    Responsibility: make the LLM name concrete semantic contribution, including
    negative or affective force when the host sense has it.
    """

    root_name = root.upper()
    word_name = word.upper() if word != "<word>" else word
    return (
        f"concrete effect of {root_name} on {word_name}; preserve hardship, "
        "distress, judgment, affective force, or other valence if the host "
        "sense shows it"
    )


def evidence_note_instruction(root: str, word: str) -> str:
    """Return schema wording for an EVIDENCE note.note value.

    Why: note.note is the place where humans can audit whether a gloss follows
    the actual host meaning.
    How: candidate evidence templates use this wording for each attested word.
    Responsibility: prevent abstraction-only explanations for concrete senses.
    """

    root_name = root.upper()
    word_name = word.upper() if word != "<word>" else word
    return (
        f"break down how {root_name} contributes to {word_name}; do not reduce "
        "woe, suffering, judgment, or other concrete force to a bare state or "
        "condition"
    )
