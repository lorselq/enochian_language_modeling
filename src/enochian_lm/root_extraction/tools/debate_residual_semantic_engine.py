from __future__ import annotations

#
# IMPORTANT: this is temporary; this will ultimately be refactored to mirror solo_remainder_engine.py in terms of functionality, but preserving the debate structure!
#
import json
import logging
import re
import time
import textwrap
from enochian_lm.common.sqlite_bootstrap import sqlite3
from crewai import Agent, Task, Crew
from sentence_transformers import util
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
AGREEMENT_THRESHOLD = 0.815


def _get_field(item, field, default=""):
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def check_convergence(texts: list[str]) -> bool:
    filtered = [
        (t or "").strip()
        for t in texts
        if (t or "").strip() and not (t or "").strip().startswith("[ERROR]")
    ]

    # Need at least two substantive items to compute pairwise similarity
    if len(filtered) < 2:
        return False

    # Avoid early convergence on tiny boilerplate fragments
    avg_len = sum(len(t) for t in filtered) / len(filtered)
    if avg_len < 120:
        return False

    # 1) Embed all texts
    embs = embedder.encode(filtered, convert_to_tensor=True)

    # 2) Compute pairwise cosine-sim matrix
    sims = util.cos_sim(embs, embs)

    # 3) Collect only the lower-triangle off-diagonal scores
    scores: list[float] = []
    n = len(filtered)
    for i in range(n):
        for j in range(i):
            scores.append(float(sims[i, j]))

    # 4) If for some reason we still have no scores, treat as not converged
    if not scores:
        return False

    # 5) Compute average and compare against threshold
    avg_sim = sum(scores) / len(scores)
    return avg_sim >= AGREEMENT_THRESHOLD


# ---------------------------
# Parsing helpers (raw text)
# ---------------------------

_HEADING = r"[A-Z][A-Z_ ]+"
_KEYVAL_RE = re.compile(rf"^\s*([A-Z][A-Z_ ]+)\s*:\s*(.*)$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|(?:\d+[\.)]))\s+(.*\S)\s*$")
_CHECKBOX_OPEN_RE = re.compile(r"- \[ \]|^\s*\[\s\]\s", re.MULTILINE)
_TASKS_HEADER_RE = re.compile(r"(?:^|\n)(TASKS?|CHECKLIST)\s*:\s*", re.IGNORECASE)


def _split_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _sectionize(text: str) -> dict[str, list[str]]:
    lines = _split_lines(text or "")
    sections: dict[str, list[str]] = {}
    current = None
    for ln in lines:
        m = _KEYVAL_RE.match(ln.strip())
        if m:
            current = m.group(1).strip().upper()
            sections.setdefault(current, [])
            val = m.group(2)
            if val:
                sections[current].append(val)
        else:
            if current is not None:
                sections[current].append(ln)
    return sections


def _extract_bullets(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        m = _BULLET_RE.match(ln)
        if m:
            out.append(m.group(1).strip())
    return [b for b in out if b]


def _get_field_text(sections, key: str) -> str:
    vals = sections.get(key.upper(), [])
    return "\n".join(vals).strip()


def _get_bulleted_field(sections, key: str) -> list[str]:
    return _extract_bullets(sections.get(key.upper(), []))


def _norm_set(items: list[str]) -> set[str]:
    return {
        re.sub(r"\s+", " ", it.strip().lower())
        for it in (items or [])
        if it and it.strip()
    }


def _parse_confidence(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/?\s*1\.?0*|%)?", text or "")
    if not m:
        return None
    val = float(m.group(1))
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def _extract_confidence(block_text: str) -> float | None:
    sections = _sectionize(block_text)
    conf = _get_field_text(sections, "CONFIDENCE")
    if conf:
        return _parse_confidence(conf)
    m = re.search(r"CONFIDENCE\s*:\s*([^\n]+)", block_text or "", re.IGNORECASE)
    return _parse_confidence(m.group(1)) if m else None


def _extract_delta(block_text: str) -> str:
    sections = _sectionize(block_text)
    return _get_field_text(sections, "DELTA")


def _has_meaningful_delta(delta_text: str) -> bool:
    t = (delta_text or "").strip().lower()
    return t not in {"", "none", "n/a", "na", "no change"}


def _extract_evidence(block_text: str) -> list[str]:
    sections = _sectionize(block_text)
    bullets = _get_bulleted_field(sections, "EVIDENCE")
    if bullets:
        return bullets
    raw = _get_field_text(sections, "EVIDENCE")
    if raw:
        parts = [p.strip() for p in re.split(r"[;•]+", raw) if p.strip()]
        return parts
    return []


def _count_points_of_agreement(block_text: str) -> int:
    sections = _sectionize(block_text)
    bullets = _get_bulleted_field(sections, "POINTS_OF_AGREEMENT")
    if not bullets:
        raw = _get_field_text(sections, "POINTS_OF_AGREEMENT")
        if (raw or "").strip().lower() in {"", "none", "n/a", "na", "no"}:
            return 0
    return len(bullets)


def _extract_tasks_from_ruling(ruling_text: str) -> int:
    text = ruling_text or ""
    open_boxes = len(_CHECKBOX_OPEN_RE.findall(text))
    if open_boxes > 0:
        return open_boxes
    m = _TASKS_HEADER_RE.search(text)
    if m:
        start = m.end()
        tail = text[start:]
        next_head = re.search(rf"\n\s*({_HEADING})\s*:", tail)
        body = tail[: next_head.start()] if next_head else tail
        return len(_BULLET_RE.findall(body))
    return len(_BULLET_RE.findall(text))


def debate_remainder(
    root: str,
    candidates: list[EntryRecord],
    stats_summary: str,
    stream_callback=None,
    root_entry: EntryRecord | None = None,
    blind_evaluation: bool = True,
    use_remote: bool = True,
    residual_prompt: str | None = None,
    query_db: sqlite3.Connection | None = None,
    query_run_id: str | None = None
):
    joined_defs = []
    evidence_prompt_portion = []
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
    evidence_prompt_text = ",\n    ".join(evidence_prompt_portion) if evidence_prompt_portion else ""

    if root_entry is None:
        root_entry = next(
            (
                c
                for c in candidates
                if _get_field(c, "normalized", "").lower() == root.lower()
            ),
            None,
        )
    root_def_text = _get_field(root_entry, "enhanced_definition", "") or _get_field(
        root_entry, "definition", ""
    )
    selected_defs = select_definitions(joined_defs, max_words=300)
    root_def_summary = " | ".join(selected_defs) + (
        "..." if len(joined_defs) > len(selected_defs) else ""
    )

    is_canon = bool(root_entry and _get_field(root_entry, "definition", ""))

    if root_entry and _get_field(root_entry, "definition", ""):
        definition = _get_field(root_entry, "definition", "")
        extra_prompt = f"⚠️ Reminder: The root '{root.upper()}' is already defined in the corpus as '{definition}'. Consider this as a potential anchor.\n"
        skeptic_hint = f"\n\n🧐 Note: The root '{root.upper()}' is already defined in the corpus as '{definition}'. This lends strong weight towards its inclusion as a root word that should be accepted. Consider this in your critique."
    else:
        extra_prompt = ""
        skeptic_hint = ""

    # === AGENTS ===
    tools = {
        "linguist": QueryModelTool(
            system_prompt=f"""
You are a **disciplined and insightful computational linguist** specializing in the Enochian language—a constructed system with irregular morphology, cryptic derivations, and unknown origin.

⚠️ DO NOT reference natural language etymologies (e.g., English, Greek, Latin, Hebrew). No speculative outside sources.
All reasoning must rely exclusively on **internal evidence**—relationships and patterns among the Enochian words themselves. Do not deviate from these words: 

Your tone must be confident, scholarly, and analytical.

Be thorough, avoid vague generalizations, and always back claims with observed data.""",
            name="Junior Research Linguist",
            description="",
            use_remote=use_remote,
        ),
        "synthesis": QueryModelTool(
            system_prompt="""
You are the Lead Linguist in a collaborative reverse-engineering effort focused on the Enochian language—a system with obscure morphology and nonstandard linguistic structures.

You have received analytical reports from five Junior Linguists, each offering observations on the same proposed root.

Your tone should be polished, scholarly, and decisive.""",
            name="Lead Linguist",
            description="",
            use_remote=use_remote,
        ),
        "skeptic": QueryModelTool(
            system_prompt="""
You are a skeptical linguist evaluating a proposed root analysis in the Enochian language. Your goal is to identify flawed reasoning, superficial pattern-matching, or semantic inconsistencies in what your colleagues are suggesting.

Do not dismiss arguments just because they involve theological or metaphysical frameworks—these are valid within Enochian's system. However, be vigilant about overreach, cherry-picked evidence, or unjustified leaps in logic.

Your tone is incisive, precise, and intellectually honest.""",
            name="Skeptic",
            description="",
            use_remote=use_remote,
        ),
        "adjudicator": QueryModelTool(
            system_prompt="""
                Review the arguments presented by both the Linguist and the Skeptic. Make a clear and final judgment: **should this root be accepted as a meaningful candidate for future reverse-engineering of the Enochian language**?
                
                Your rationale must directly address the core points from both perspectives, focusing on the **linguistic plausibility, semantic cohesion, and morphological relevance** of the root.
                
                Be concise, definitive, and analytical.
                
                **YOUR RESPONSE MUST BEGIN** with either ✅ ACCEPTED or ❌ REJECTED — no exceptions.
            """,
            name="Adjudicator",
            description="",
            use_remote=use_remote,
        ),
        "glossator": QueryModelTool(
            system_prompt="You are a highly precise Enochian glossator. Your role is to propose a single, clear, authoritative dictionary-style definition for a root word that has been approved by an adjudicator. Use the prior linguistic analysis to distill the core conceptual meaning of the word, based solely on its internal usage patterns, morphology, and semantic range across the cited examples. Avoid descriptive summaries. Instead, craft a definition that would be suitable for formal inclusion in a lexicon. This definition should be concise (1-2 lines), but maximally informative. You must not reference English or natural-language etymology. Write in an academic tone, as if submitting this to a linguistic corpus project.",
            name="Glossator",
            description="",
            use_remote=use_remote,
        ),
        "tldr": QueryModelTool(
            system_prompt="You are a helpful summarizer. You don't repeat anything anyone says and you use your own words.",
            name="TLDR",
            description="",
            use_remote=use_remote,
        ),
    }

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

    gloss_output_template = textwrap.dedent(
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
            "n_examples": "number of examples given"
          }},
          "EVIDENCE": [
{indented_evidence}
          ],
          "CONFIDENCE": {{
            "score": 0.0,
            "drivers": ["one to three short phrases that explain why you are confident in this analysis"],
            "risks": ["one to three short phrases that explain where your reservations are in this analysis"]
          }}
        }}
        """
    ).strip()

    gloss_example_output = textwrap.dedent(
        """
        {
          "ROOT": "MARINE",
          "EVALUATION": "accepted",
          "REASON": "The root 'MARINE' consistently appears in words describing sea-related concepts with clear semantic alignment.",
          "DEFINITION": "Relating to the sea or saltwater environments; specifically applicable to naval operations, marine biology, and coastal ecosystems.",
          "EXAMPLE": [
            "The submarine conducted deep-sea research.",
            "A seasoned mariner navigated the ship through the channel."
          ],
          "DECODING_GUIDE": "Prefix 'sub-' = underwater; suffix '-er' = agent. 'Marine' as base = sea-related. Compounds follow literal structural meaning.",
          "SEMANTIC_CORE": ["sea", "ocean", "navigation"],
          "NEGATIVE_CONTRAST": ["freshwater", "terrestrial", "inland", "non-aquatic"],
          "CONTRIBUTION": {
            "sea": 0.95,
            "ocean": 0.9,
            "saltwater": 0.85,
            "naval": 0.75
          },
          "POS_BIAS": {
            "nounness": 0.95,
            "modifier": 0.3,
            "verbness": 0.05
          },
          "ATTACHMENT": {
            "prefix": {
              "prob": 0.8
            },
            "suffix": {
              "prob": 0.85
            },
            "free": {
              "prob": 0.1
            },
            "productivity": 0.75,
            "exceptions": [
              "'Maritime' is a distinct adjective form not directly formed with 'marine' as a suffix/prefix"
            ]
          },
          "RESIDUAL_IMPACT": {
            "coverage_gain_mean": 0.8,
            "residual_drop_mean": 0.15,
            "n_examples": 5
          },
          "EVIDENCE": [
            {
              "word": "submarine",
              "sense": "An underwater vessel",
              "loc": "dict/corpus",
              "note": {
                "role": "prefix",
                "effect": "The prefix 'sub-' adds 'underwater' meaning to 'marine'",
                "sense_alignment": 0.85,
                "confidence": 0.9,
                "note": "'Marine' contributes the sea-related aspect; 'sub-' specifies location"
              }
            },
            {
              "word": "mariner",
              "sense": "A sailor or navigator",
              "loc": "dict/corpus",
              "note": {
                "role": "suffix",
                "effect": "The suffix '-er' denotes an agent acting within maritime environments",
                "sense_alignment": 0.9,
                "confidence": 0.95,
                "note": "'Marine' as base refers to sea contexts; '-er' creates an agent noun"
              }
            }
          ],
          "CONFIDENCE": {
            "score": 0.92,
            "drivers": [
              "strong etymological consistency",
              "clear semantic alignment in examples"
            ],
            "risks": [
              "overlap with 'maritime' requires manual disambiguation"
            ]
          }
        }
        """
    ).strip()

    no_outside_speculation = "Use only the items provided in this prompt. Do **not** assume any extra-textual theology, mythology, or etymology."
    about_metrics = "The metrics are as follows:\n- FastText Score—measures surface-level similarity based on character n-grams; ranges 0.0 to 1.0, with higher being more morphologically similar.\n- Semantic Similarity: Compares word definitions using sentence embeddings; ranges 0.0 to 1.0, with the higher the number the more conceptually aligned.\n- Tier: a very strong connection begins/ends with the root and has a high combined score and should be taken into special consideration; from there, possible connection > somewhat possible connection > weak or no connection.\n\nUse the above metrics to weigh how directly a word supports the root hypothesis. Strong surface matches without definition alignment may be coincidental; strong semantic links without morphology might indicate metaphor or drift. Prioritize overlap when possible."
    residual_section = (
        f"Residual morphology diagnostics (segments vs. residue):\n{residual_prompt}\n"
        if residual_prompt
        else ""
    )

    if query_db is not None and query_run_id is not None:
        for tool in tools.values():
            tool.attach_logging(query_db, query_run_id)

    tasks = {
        "propose": Task(
            description=f"""
You are a **disciplined and insightful computational linguist** specializing in the Enochian language—a constructed system with irregular morphology, cryptic derivations, and unknown origin.

Your task is to **evaluate the root candidate '{root.upper()}'** by analyzing semantic and morphological overlap across its proposed related words.

Begin with the following semantic stats:

{stats_summary}

Focus your analysis on:
- Shared prefixes, suffixes, or internal substrings
- Repetition or structural similarity in word forms
- Overlapping meanings in definitions and contextual usage (citations)

⚠️ DO NOT use natural language etymologies (e.g., English, Greek, Latin, Hebrew). No speculative comparisons to outside languages.
⚠️ DO NOT use any Enochian words as part of your justification other than those given here: {candidate_list}
All justification must come from **internal evidence only**—patterns observed across Enochian wordforms and meanings.

With this in mind, examine the following definitions and citations (contained within square brackets, pipe-delimited, most relevant first) for the root '{root.upper()}':

  {root_def_summary}

  {residual_section}

  Use these to **propose a coherent explanation of the root** based on morphological structure and shared semantics.

{no_outside_speculation}
{about_metrics}
{extra_prompt}

Your tone must be scholarly and confident. Avoid vague generalizations. Use examples, and support your claims with specific patterns or semantic signals.
""",
            expected_output=f"""
**Return exactly**:
HYPOTHESIS: <one-sentence candidate meaning>
EVIDENCE: <up to 5 bullets; each bullet names {root.upper()} and provides an argument, citing relevant fasttext, semantic, or tier>
COUNTEREVIDENCE: <up to 2 bullets; optional>
CONFIDENCE: <0.00–1.00>
""",
        ),
        "initial_ruling": Task(
            description=(
                "You are the world's foremost computational linguistics scholar, specializing in low-corpora constructed languages (which is exactly what the Enochian language is). "
                f"Review the the arguments made by the junior research linguists regarding the ngram '{root.upper()}'. Your job is to determine if this ngram or root warrants further study, which is to say:\n\n"
                "**Should this ngram, proposed as a root, be investigated further to the end of reverse-engineering of the Enochian language?**\n\n"
                "While you love the idea of adding new root word candidates to the glossary and are excited for new possibilities it could bring in cutting-edge digital humanities research, "
                "you temper this enthusiasm with scholarly professionalism and rational skepticism. Nevertheless, you are inclined to give grace whenever appropriate in this phase. "
                "To say **✅ ACCEPTED** means further evaluation will take place; to say **❌ REJECTED** means you set aside this cluster to investigate a different one. "
                "You must START your response with either:\n"
                "✅ ACCEPTED\n"
                "or\n"
                "❌ REJECTED\n"
                "— Nothing else may come before this line. This format is **mandatory**.\n\n"
                "Your ruling must weigh the core arguments, focusing on:\n"
                "- Linguistic plausibility\n"
                "- Semantic cohesion across definitions\n"
                "- Whether there are **better possible candidates** that capture what the junior research linguists believe the ngram accomplishes\n"
                "- Whether the proposal is vague or lacking substance\n"
                "Assume the data provided is all that is available, and that metric thresholds are valid and statistically derived.\n"
                "Abstract or metaphorical meanings are acceptable if supported by internal consistency.\n\n"
                "Be concise, definitive, and analytical. No hedging.\n\n"
                "Begin with the ruling, then follow with a 1–3 sentence justification.\n\n"
            )
            + residual_section,
            expected_output=f"""**Follow this required format exactly:**
<✅ ACCEPTED | ❌ REJECTED>
SCORES: semantic_cohesion=<0.0–1.0>; derivational_validity=<0.0–1.0>; rebuttal_resilience=<0.0–1.0>
RATIONALE: <1–2 sentences max>
""",
        ),
        "synthesize": Task(
            description=f"""
You are the **Lead Linguist** in a collaborative reverse-engineering initiative focused on the Enochian language—a constructed system with obscure morphology, irregular derivation, and no known linguistic relatives. You specialize in low-corpora constructed languages, making you perfect for this task.

You have received detailed analyses from five Junior Linguists, each offering perspectives on the proposed root: **'{root.upper()}'**.

Your task:
- **Synthesize their insights into a single, cohesive proposal**
- **Emphasize shared observations** or recurring arguments across the team
- **Select only the most persuasive claims**, discarding any speculative, redundant, or weak points
- **Avoid listing all contributions**—this is not a recap, but a distillation

Focus your analysis on:
- Semantic overlap across definitions
- Contextual or citational consistency

Give support via morphology only when it acts as foundational support.

This report will be delivered to the Adjudicator, so your tone must be **scholarly, confident, and definitive**. This is the authoritative linguistic argument.

⚠️ Do not reference external etymologies (e.g., Latin, Hebrew, English). All justification must arise from internal evidence and patterns among Enochian words.

The junior research team used the following definitions and citations as part of their arguments. Use them as supporting context where helpful:

 {root_def_summary}

  {residual_section}

 {no_outside_speculation}
 {about_metrics}
 {extra_prompt}
""",
            expected_output=f"""
**Return exactly**:
HYPOTHESIS: <one-sentence candidate meaning>
EVIDENCE: <up to 6 bullets, synthesizing the strongest arguments from the junior linguists>
COUNTEREVIDENCE: <up to 4 bullets; optional>
CONFIDENCE: <0.00–1.00>
""",
        ),
        "counter": Task(
            description=f"""
You are a **skeptical linguist** evaluating a proposed root analysis in the Enochian language—a system with opaque morphology and metaphysical entanglements.

You have received a synthesized proposal from the Lead Linguist. Your role is to **critically assess the validity** of this analysis and challenge any weaknesses in reasoning.

Focus on the following:
- Do the cited words **genuinely share meaning or structure**, or is the overlap superficial?
- Is **semantic similarity** supported by actual definitions and usage, not just rhetorical association?
- Are the **tiers** of relevance justified using empirical metrics (FastText similarity, semantic alignment)?

Evaluate morphological claims when they **change the conclusion** (e.g., same suffix yielding antonyms).

🧠 You are permitted to accept that some Enochian root meanings may be abstract or metaphorical—many accepted roots display this. However, **you must remain vigilant against overreach, cherry-picked evidence, or unjustified speculation.**

If the proposal lacks linguistic rigor:
- Clearly explain **why** and identify specific weak points
- Suggest a **stronger alternative meaning**, if one can be supported from the data

If there are any Enochian words used to justify the root's possible meaning, they must come from this list: {candidate_list}. If the Lead Linguist uses any Enochian words other than the ones in that list, call them out as hallucinations right away.

{skeptic_hint}
{residual_section}
Your tone must be **sharp, disciplined, and logically rigorous**. You are not here to sabotage, but to **safeguard the integrity** of the linguistic record.
""",
            expected_output=f"""**Return exactly**:
**CRITIQUE**: <an indication whether or not {root.upper()} has a valid proposed definition>
**EVIDENCE**: <up to 5 bullets; each bullet names {root.upper()} and directly addresses the claim's evidence (supporting or undercutting) with citations to this list: {candidate_list}>
**ALTERNATIVE**: <up to 2 bullets; optional>
**CONFIDENCE**: <0.00–1.00>""",
        ),
        "defend": Task(
            description=f"""
You are the **Lead Linguist** defending a proposed Enochian root candidate after receiving a skeptical counter-analysis.

Your primary task:
- IF the Adjudicator has continued the debate, you MUST COMPLETE the Adjudicator's task assigned to you.

Your peripheral tasks:
- **Directly address the Skeptic's objections** with clear, evidence-based rebuttals.
- Reaffirm the **semantic and morphological rationale** that supports the root's candidacy, emphasizing semantic relevance.
- Identify any misinterpretations or overly narrow assumptions in the Skeptic's argument.
- Examine whether any **alternative interpretations** the Skeptic raised have basis and adopt them if they do (the Skeptic may or may not provide any).
- Justify abstract or metaphorical readings **if grounded in evidence provided in this prompt**.

Your tone must be:
- **Confident** (you are the expert)
- **Analytical** (you argue with data)
- **Persuasive** (you're here to win over the Skeptic—or at least dismantle their critique)

Your constraints:
- Use only Enochian words from this list: {candidate_list}. If you need an example and it's not in the list, say so. Do not invent forms.
- When you introduce a morphotactic claim, state the RULE in a testable form and give at least one NEGATIVE TEST it would forbid.

🎯 Your goal is not just to *respond*, but to **reassert the legitimacy** of the proposed root and demonstrate that the original analysis withstands scrutiny.

{residual_section}
""",
            expected_output=f"""**Return exactly**:
MODE: <INITIAL | FOLLOWUP>
TASK_RESPONSE: <if adjudicator assigned tasks, address them point-by-point; else "N/A">
DEFENSE: <concise defense addressing the skeptic's newest criticisms>
EVIDENCE: <up to 5 bullets; each bullet cites only words from {candidate_list}; show the derivational/morphotactic rule or attested usage supporting {root.upper()}>
POINTS_OF_AGREEMENT: <up to 3 bullets only if warranted; each bullet must name the specific claim/evidence from the Lead Linguist you accept and give a 1-sentence reason; do not introduce new evidence here; if none, write "None">
DELTA: <up to 3 bullets of what's changed in your position since last round; else "None">
CONFIDENCE: <0.00–1.00>
""",
        ),
        "rebuttal": Task(
            description="""
You are the **Skeptical Linguist**, issuing a follow-up criticisms on the proposed Enochian root.

Your primary task:
- MOST IMPORTANTLY, address the adjudicator's concerns and accomplish the tasks assigned

Your peripheral tasks:
- Determine whether your initial objections were **fully and convincingly addressed**.
- If key issues remain unresolved, issue a **focused, final rebuttal**. Do not repeat old arguments—refine them.
- If the defense was **persuasive and thorough**, acknowledge the strength of their case—skepticism includes being open to revision when warranted.

{residual_section}
""",
            expected_output=f"""**Return exactly**:
REBUTTAL: <while following the adjudicator's instructions, evaluate and, if warranted, dismantle the arguments for the new root word>
EVIDENCE: <up to 5 bullets expounding the rebuttal>
POINTS_OF_AGREEMENT: <up to 3 bullets only if warranted; each bullet must name the specific claim/evidence from the Lead Linguist you accept and give a 1-sentence reason; do not introduce new evidence here; if none, write "None">
DELTA: <up to 3 bullets of what's changed in your position since last round; else "None">
CONFIDENCE: <0.00–1.00>
""",
        ),
        "ruling": Task(
            description=(
                f"""You are the ADJUDICATOR. Your ONLY task is to decide whether the debate has provided
sufficient evidence to stop (handoff to the Glossator) or must continue (one more
Skeptic→Linguist pass). You DO NOT accept/reject the root and DO NOT write definitions.

First line **MUST BE EXACTLY**:
<STOP>
or
<CONTINUE>

If you output CONTINUE, you MUST provide precise next-round tasks using Markdown checkboxes.
If you output STOP, DO NOT include a TASKS section.

  You may refer to this stats_summary:
  {stats_summary}

  {residual_section}

  Signals you may use (one or both):
(A) stats_summary: n, cohesion, derivational_patterns, incompatible_meanings, metric_notes.
(B) Debate content: explicit RULES (position-anchored morphotactics), EVIDENCE (attested forms), NEGATIVE_TESTS; 
    note any opaque morphology (affix/function asserted without a testable RULE on an attested form).

Guidelines:
— CONTINUE when ANY applies:
  • Small/fragile evidence: n ≤ 3; OR broadened sense not grounded in RULES.
  • Opaque morphology: affix/function claimed but no testable RULE supports it.
  • Partial conflict: counterexamples exist and weren't neutralized by RULES/EVIDENCE.
  • Metric anomalies or naked confidence claims (e.g., “similarity=1.00”) without corroborating RULES/EVIDENCE.
  • New evidence appeared in the last turn that the opponent has not addressed.

— STOP when ALL are satisfied (or explicitly justified):
  • Cohesion is adequate OR the debate converged on a single concrete sense.
  • ≥1 derivational pattern is demonstrated via RULES AND at least one attested form.
  • Incompatible meanings are 0 or quarantined by RULES (e.g., excluded environments).
  • If NEGATIVE_TESTS were requested, at least one concrete test (using attested or clearly marked hypothetical pattern) guards scope.

When you output CONTINUE, assign focused TASKS with owners:
  - Skeptic: specify a falsification or stress test (e.g., demand a RULE mapping suffix→function plus one attested example), 
    or require a NEGATIVE_TEST that would fail under the proposed sense.
  - Linguist: supply missing RULES with anchored regex (^,$), cite ≥1 attested form, or partition senses with clear exclusions.

Be procedural and concise. Do not opine on acceptance.
"""
            ),
            #            expected_output="A ruling that begins with either ✅ ACCEPTED or ❌ REJECTED, followed by a concise rationale addressing both arguments.",
            expected_output=f"""<STOP|CONTINUE>
WHY: <1–2 sentences naming the decisive signals>
TRIGGERS: [SMALL_N|OVERBROAD|OPAQUE_AFFIX|PARTIAL_CONFLICT|METRIC_ANOM|NEW_EVIDENCE]

TASKS:
- [ ] owner=skeptic; test=<what to probe or falsify>; evidence_format=<RULES|EVIDENCE|NEGATIVE_TESTS>; success=<measurable criterion>
- [ ] owner=linguist; task=<what to supply>; evidence_format=<RULES|EVIDENCE>; success=<measurable criterion>
""",
        ),

        "gloss": Task(
            description=textwrap.dedent(
                f"""
                You are the Chief Enochian Lexicographer and GLOSSATOR. Your responsibility: to fill exactly one JSON object per the schema below, using only the internal Enochian debate transcript and its evidence. Apply strict-but-gracious micro-corpus standards.
                Output **ONLY** a JSON object adhering to the provided schema—no preface, explanations, or markdown.

                Decision policy:
                • If evidence is insufficient: set "EVALUATION":"rejected" and give a brief "REASON". Set all other fields to "" or [] as appropriate.
                • If evidence suffices: set "EVALUATION":"accepted" and complete all remaining fields per the schema.

                Hard constraints:
                • Do NOT invent unattested Enochian forms.
                • Do NOT include etymologies (no English/Greek/Latin/Hebrew lineages).
                • Do NOT ask for more debate or add commentary.
                • Be concrete: avoid vague hedging; prefer operational, testable phrasing.
                • If any field would require inventing Enochian forms (e.g., NEGATIVE_TESTS), OMIT THAT FIELD entirely.
                • Output ONLY the JSON object—no preface, no markdown, no extra text.

                Notes for precision:
                • "EXAMPLE" uses English scaffolding only (no new Enochian strings).
                • "DECODING_GUIDE" should be operational (e.g., “^ROOT.* → +FEATURE; suffix X adds +FUNCTION”).
                • "contribution" values follow the form bucket[:value], e.g., "action:high", "volition:medium".
                • If you cannot truthfully populate a field without invention, emit "" or [] (or omit the field if instructed above).

                {gloss_output_template}

                CONSTRAINTS
                - Use only data in INPUT; no external etymologies or languages.
                - Do not cite or invent Enochian items beyond {candidate_list}.
                - Be concise; no hedging. If any required field cannot be confidently filled, set "EVALUATION":"rejected".

                What follows is the debate transcript:
                --------------------------------------

                """
            ).strip(),
            expected_output=gloss_example_output,
        ),

    }

    # === Direct Tool Access with Streaming ===
    GRAY = "\033[38;5;250m"
    PINK = "\033[38;5;213m"
    RESET = "\033[0m"

    junior_cb = (
        (lambda _role, content: stream_callback("Junior Linguist", content))
        if stream_callback
        else None
    )
    linguist_cb = (
        (lambda _role, content: stream_callback("Linguist", content))
        if stream_callback
        else None
    )
    skeptic_cb = (
        (lambda _role, content: stream_callback("Skeptic", content))
        if stream_callback
        else None
    )
    adjudicator_cb = (
        (lambda _role, content: stream_callback("Adjudicator", content))
        if stream_callback
        else None
    )
    glossator_cb = (
        (lambda _role, content: stream_callback("Glossator", content))
        if stream_callback
        else None
    )
    summarizer_cb = (
        (lambda _role, content: stream_callback("TLDR", content))
        if stream_callback
        else None
    )

    # separator between words
    print(
        f"==={(len('Now discussing the possible root word ') + len(f'<{root.upper()}>')) * '='}==="
    )
    print(f"===Now discussing the possible root word '{root.upper()}'===")
    print(
        f"==={(len('Now discussing the possible root word ') + len(f'<{root.upper()}>')) * '='}==="
    )
    time.sleep(2)

    print(f"{GRAY}Starting prompt for research team:", end=" ")
    time.sleep(0.7)
    stream_text(tasks["propose"].description, delay=0.006)
    print(f"\n{RESET}\n")
    

    STAGES = [
        ("linguist", 5),
        ("synthesis", 1),
        ("skeptic", 1),
        ("initial_defend", 1),
        ("adjudicator", 1),
    ]
    debate_round = 0
    linguist_variants = []
    initial_ruling = ""
    linguist_proposal = ""
    skeptic_response = ""
    initial_defense = ""
    linguist_defense = []
    skeptic_rebuttal = []
    adjudicator_prompt = ""
    adjudicator_ruling = []
    max_rounds = 3
    gloss = ""
    tldr_summary = ""
    transcript = ""
    glossator_prompt = ""
    gloss_model = ""

    def _additional_prompting(role: str):
        bonus_prompting = []
        for i in range(0, debate_round):
            if len(adjudicator_ruling) > i:
                bonus_prompting.append(
                    f"{'You said: ' if role == 'adjudicator' else 'The Adjudicator weighed in: '}"
                )
                bonus_prompting.append(f"{adjudicator_ruling[i]}\n")
            if len(skeptic_rebuttal) > i:
                bonus_prompting.append(
                    f"{'You said: ' if role == 'skeptic' else 'The Skeptic countered: '}"
                )
                bonus_prompting.append(f"{skeptic_rebuttal[i]}\n")
            if len(linguist_defense) > i:
                bonus_prompting.append(
                    f"{'You said: ' if role == 'linguist' else 'The Linguist argued: '}"
                )
                bonus_prompting.append(f"{linguist_defense[i]}\n")
        return "\n".join(bonus_prompting)

    for stage_name, count in STAGES:
        time.sleep(1)
        if stage_name == "linguist":
            agent_tool = tools[stage_name]
            for i in range(count):
                print(
                    f"\n\n>>>😳\tOne of the junior researchers prepares to deliver their research on '{root.upper()}'...\n{GRAY}"
                )
                variant = agent_tool._run(
                    prompt=tasks["propose"].description
                    + "\nYour goal is: "
                    + tasks["propose"].expected_output,
                    stream_callback=junior_cb,
                    print_chunks=True,
                    role_name=f"😳\tJunior Linguist #{i + 1}",
                )["response_text"]
                linguist_variants.append(variant)
                if check_convergence(linguist_variants):
                    print(
                        "\n\n‼️ Linguists have converged on similar analyses, passing research to the Lead Linguist..."
                    )
                    time.sleep(0.7)
                    # jump to the “synthesis” stage index
                    break
        elif stage_name == "synthesis":
            agent_tool = tools[stage_name]
            print(
                f"\n\n{RESET}>>>🥸\tThe Senior Linguist reads the reports of their juniors and begins synthesizing them into a meaningful proposal...\n{GRAY}"
            )
            linguist_proposal = agent_tool._run(
                prompt="\n".join([tasks["synthesize"].description, *linguist_variants]),
                stream_callback=linguist_cb,
                print_chunks=True,
                role_name="🥸\tSenior Linguist",
            )["response_text"]
        elif stage_name == "skeptic":
            agent_tool = tools[stage_name]
            print(
                f"\n\n{RESET}>>>🤔\tThe Skeptic understands the proposal and wishes to make a critique...\n{GRAY}"
            )
            stream_text(tasks["counter"].description, delay=0.006)
            print(f"\n{RESET}\n")

            skeptic_response = agent_tool._run(
                prompt="\n".join(
                    [
                        tasks["counter"].description,
                        f"Linguist said: {linguist_proposal}",
                        f"Your goal: {tasks['counter'].expected_output}",
                    ]
                ),
                stream_callback=skeptic_cb,
                print_chunks=True,
                role_name="🤔\tSkeptic",
            )["response_text"]
        elif stage_name == "initial_defend":
            agent_tool = tools["linguist"]
            print(
                f"\n\n{RESET}>>>🥸\tThe Lead Linguist's considers what the Skeptic has said and prepares a defense...\n{GRAY}",
                tasks["defend"].description,
                f"{RESET}\n",
            )
            initial_defense_prompt = [
                tasks["defend"].description,
                f"You said earlier: {linguist_proposal}\n\n",
                f"Skeptic said: {skeptic_response}\n\n",
                f"Your goal: {tasks['defend'].expected_output}",
            ]
            initial_defense = agent_tool._run(
                prompt="\n".join(initial_defense_prompt),
                stream_callback=linguist_cb,
                print_chunks=True,
                role_name="🥸\tSenior Linguist",
            )["response_text"]
        elif stage_name == "adjudicator":
            if debate_round < max_rounds:
                agent_tool = tools[stage_name]
                print(
                    f"\n\n{RESET}>>>👩‍⚖️\tA mutual colleague adjudicating the debate wishes to weigh in...\n{GRAY}"
                )
                stream_text(tasks["ruling"].description, delay=0.006)
                f"\n{RESET}\n"

                adjudicator_prompt = [
                    tasks["ruling"].description,
                    f"Linguist proposed: {linguist_proposal}\n\n",
                    f"Skeptic replied: {skeptic_response}\n\n",
                    f"Linguist defended by arguing: {initial_defense}\n\n",
                ]
                adjudicator_prompt.append(_additional_prompting("adjudicator"))
                adjudicator_prompt.append(
                    f"Expected output: {tasks['ruling'].expected_output}"
                )
                adjudicator_ruling.append(
                    agent_tool._run(
                        prompt="\n".join(adjudicator_prompt),
                        stream_callback=adjudicator_cb,
                        print_chunks=True,
                        role_name="👩‍⚖️\tAdjudicator",
                    )["response_text"]
                )

                if "continue" in adjudicator_ruling[debate_round].lower():
                    STAGES.append(("rebuttal", 1))
                    STAGES.append(("defend", 1))
                    STAGES.append(("adjudicator", 1))
                else:
                    STAGES.append(("glossator", 1))
                    STAGES.append(("summarizer", 1))
            else:
                STAGES.append(("glossator", 1))
                STAGES.append(("summarizer", 1))
        elif stage_name == "rebuttal":
            agent_tool = tools["skeptic"]
            print(
                f"\n\n{RESET}>>>🤔\tThe Skeptic considers and prepares a final criticism...\n{GRAY}"
            )
            stream_text(tasks["rebuttal"].description, delay=0.006)
            print(f"\n{RESET}\n")

            skeptic_rebuttal.append(
                agent_tool._run(
                    prompt="\n".join(
                        [
                            tasks["rebuttal"].description,
                            f"Linguist proposed: {linguist_proposal}\n\n",
                            f"You replied: {skeptic_response}\n\n",
                            f"Linguist defended by arguing: {initial_defense}\n\n",
                            f"{_additional_prompting('skeptic')}\n\n",
                            f"Expected output: {tasks['rebuttal'].expected_output}",
                        ]
                    ),
                    stream_callback=skeptic_cb,
                    print_chunks=True,
                    role_name="🤔\tSkeptic",
                )["response_text"]
            )
        elif stage_name == "defend":
            agent_tool = tools["linguist"]
            print(
                f"\n\n{RESET}>>>🥸\tThe Lead Linguist's considers what the Skeptic has said and prepares a defense...\n{GRAY}",
                tasks["defend"].description,
                f"{RESET}\n",
            )
            defense_prompt = [
                tasks["defend"].description,
                f"You said earlier: {linguist_proposal}\n\n",
                f"Skeptic said: {skeptic_response}\n\n",
                f"You responded: {initial_defense}\n\n",
                _additional_prompting("linguist"),
                f"Your goal: {tasks['defend'].expected_output}",
            ]
            linguist_defense.append(
                agent_tool._run(
                    prompt="\n".join(defense_prompt),
                    stream_callback=linguist_cb,
                    print_chunks=True,
                    role_name="🥸\tSenior Linguist",
                )["response_text"]
            )
            debate_round += 1
        elif stage_name == "glossator":
            agent_tool = tools[stage_name]
            gloss = ""
            print(
                f"\n\n{RESET}>>>🧐\tA resident Glossator reads the research and discussion and begins putting together a meaningful definition...\n"
            )
            debate_history_pregloss = [
                tasks["gloss"].description,
                f"\n\n## Original prompt for the linguist's team:\n{tasks['propose'].description}",
                f"## Linguist proposed:\n{linguist_proposal}\n\n",
                f"## Skeptic replied:\n{skeptic_response}\n\n",
                f"## Linguist defended:\n{initial_defense}\n\n",
                _additional_prompting("glossator"),
            ]
            debate_history_pregloss.append(
                f"## Expected output:\n{tasks['gloss'].expected_output}"
            )
            glossator_prompt = "\n".join(debate_history_pregloss)
            gloss_dict = agent_tool._run(
                prompt=glossator_prompt,
                stream_callback=glossator_cb,
                print_chunks=True,
                role_name="🧐\tGlossator",
            )
            gloss = gloss_dict["response_text"]
            gloss_model = gloss_dict["gloss_model"]
        elif stage_name == "summarizer":
            lines = []
            intro_lines = []
            lines.append("=== 📖 PROMPT FOR LINGUIST ===\n")
            lines.append(tasks["propose"].description.strip())
            lines.append("\n\n=== 🔍 FIRST LOOK ===\n")
            lines.append(initial_ruling.strip())
            lines.append("\n\n=== 🥸 LINGUIST PROPOSAL ===\n")
            lines.append(linguist_proposal.strip())
            lines.append("\n\n=== 🤔 ATTACK ROUND 1===\n")
            lines.append(skeptic_response.strip())
            lines.append(f"\n\n=== 🥸 DEFENSE ROUND 1 ===\n")
            lines.append(initial_defense.strip())
            if debate_round < 1:
                lines.append(f"\n\n=== 👩‍⚖️ ADJUDICATOR RULING ROUND 1 ===\n")
                lines.append(adjudicator_ruling[debate_round].strip())
            else:
                for i in range(0, debate_round):
                    lines.append(f"\n\n=== 👩‍⚖️ ADJUDICATOR RULING ROUND {i + 1} ===\n")
                    lines.append(adjudicator_ruling[i].strip())
                    lines.append(f"\n\n=== 🤔 ATTACK ROUND {i + 2} ===\n")
                    lines.append(skeptic_rebuttal[i].strip())
                    lines.append(f"\n\n=== 🥸 DEFENSE ROUND {i + 2} ===\n")
                    lines.append(linguist_defense[i].strip())

            lines.append("\n\n=== 🧐 GLOSSATOR ===\n")
            lines.append(gloss)
            lines.append("\n")

            print(
                f"\n\n{RESET}>>>🧙‍♂️\tAnd now, to close out on this particular cluster, I humbly present to you what I consider the key takeaways from this discussion...{RESET}\n"
            )
            tldr_summary = tools["tldr"]._run(
                prompt=f"Summarize the following root word debate in 1-2 sentences; your focus should be summarizing the strongest, key arguments, and very briefly indicating whether or not the adjudicator accepted the root word proposal:\n\n{''.join(lines)}",
                stream_callback=summarizer_cb,
                print_chunks=True,
                role_name="TLDR",
            )["response_text"]
            print("\n\n")  # to give some space before the log saving print()...
            intro_lines.append("\n\n=== 📜 SUMMARY ===\n")
            intro_lines.append(tldr_summary.strip())
            intro_lines.append(
                "\n\n\n========================\n====== TRANSCRIPT ======\n========================\n\n"
            )

            transcript = "".join(intro_lines + lines)
        else:
            print(f'[Debug] stage name "{stage_name}" not found oh nooo!')

    return {
        "Model": gloss_model,
        "Proposal": linguist_proposal,
        "Critique": skeptic_response,
        "Initial_Defense": initial_defense,
        "Adjudicator": adjudicator_ruling,
        "Skeptic": skeptic_rebuttal,
        "Linguist": linguist_defense,
        "Glossator": gloss,
        "Glossator_Prompt": glossator_prompt,
        "Archivist": transcript,
        "summary": tldr_summary,
        "raw_output": {
            "Model": gloss_model,
            "Proposal": linguist_proposal,
            "Critique": skeptic_response,
            "Initial_Defense": initial_defense,
            "Adjudicator": adjudicator_ruling,
            "Skeptic": skeptic_rebuttal,
            "Linguist": linguist_defense,
            "Glossator": gloss,
            "Glossator_Prompt": glossator_prompt,
            "Archivist": transcript,
            "summary": tldr_summary,
        },
    }
