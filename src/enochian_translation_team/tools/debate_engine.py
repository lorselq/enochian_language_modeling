import logging
import time
import sys
from typing import Optional
from crewai import Agent, Task, Crew
from sentence_transformers import SentenceTransformer, util
from enochian_translation_team.tools.query_model_tool import QueryModelTool
from enochian_translation_team.utils.dictionary_loader import Entry

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.api_requestor").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
AGREEMENT_THRESHOLD = 0.815


def _get_field(item, field, default=""):
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def stream_text(text: str, delay: float = 0.006):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            # if you really need to interrupt, break cleanly
            break


def check_convergence(texts: list[str]) -> bool:
    # Need at least two items to compute pairwise similarity
    if len(texts) < 2:
        return False

    # 1) Embed all texts
    embs = embedder.encode(texts, convert_to_tensor=True)

    # 2) Compute pairwise cosine-sim matrix
    sims = util.cos_sim(embs, embs)

    # 3) Collect only the lower-triangle off-diagonal scores
    scores: list[float] = []
    n = len(texts)
    for i in range(n):
        for j in range(i):
            scores.append(float(sims[i, j]))

    # 4) If for some reason we still have no scores, treat as not converged
    if not scores:
        return False

    # 5) Compute average and compare against threshold
    avg_sim = sum(scores) / len(scores)
    return avg_sim >= AGREEMENT_THRESHOLD


def select_definitions(def_list, max_words=75):
    selected = []
    total_words = 0

    for d in def_list:
        # Only count words before the first citation bracket
        bracket_index = d.find(" [")
        if bracket_index != -1:
            word_slice = d[:bracket_index]
        else:
            word_slice = d
        word_count = len(word_slice.split())

        if total_words + word_count > max_words:
            break

        selected.append(d)
        total_words += word_count

    return selected


def safe_output(crew_output) -> dict:
    if not crew_output:
        return {}

    try:
        return getattr(crew_output, "raw_output", {})
    except Exception as e:
        print(f"[!] Failed to extract output: {e}")
        return {}


def debate_ngram(
    root: str,
    candidates: list[Entry],
    stats_summary: str,
    stream_callback=None,
    root_entry: Optional[Entry] = None,
    blind_evaluation: bool = True,
    use_remote: bool = True,
):
    joined_defs = []
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
            use_remote=use_remote
        ),
        "initial_ruling": QueryModelTool(
            system_prompt=f"You are the world's foremost computational linguistics scholar, specializing in low-corpora constructed languages (which is exactly what the Enochian language is).",
            name="Adjudicator",
            description="",
            use_remote=use_remote
        ),
        "synthesis": QueryModelTool(
            system_prompt="""
You are the Lead Linguist in a collaborative reverse-engineering effort focused on the Enochian language—a system with obscure morphology and nonstandard linguistic structures.

You have received analytical reports from five Junior Linguists, each offering observations on the same proposed root.

Your tone should be polished, scholarly, and decisive.""",
            name="Lead Linguist",
            description="",
            use_remote=use_remote
        ),
        "skeptic": QueryModelTool(
            system_prompt="""
You are a skeptical linguist evaluating a proposed root analysis in the Enochian language. Your goal is to identify flawed reasoning, superficial pattern-matching, or semantic inconsistencies in what your colleagues are suggesting.

Do not dismiss arguments just because they involve theological or metaphysical frameworks—these are valid within Enochian's system. However, be vigilant about overreach, cherry-picked evidence, or unjustified leaps in logic.

Your tone is incisive, precise, and intellectually honest.""",
            name="Skeptic",
            description="",
            use_remote=use_remote
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
            use_remote=use_remote
        ),
        "glossator": QueryModelTool(
            system_prompt="You are a highly precise Enochian glossator. Your role is to propose a single, clear, authoritative dictionary-style definition for a root word that has been approved by an adjudicator. Use the prior linguistic analysis to distill the core conceptual meaning of the word, based solely on its internal usage patterns, morphology, and semantic range across the cited examples. Avoid descriptive summaries. Instead, craft a definition that would be suitable for formal inclusion in a lexicon. This definition should be concise (1-2 lines), but maximally informative. You must not reference English or natural-language etymology. Write in an academic tone, as if submitting this to a linguistic corpus project.",
            name="Glossator",
            description="",
            use_remote=use_remote
        ),
        "tldr": QueryModelTool(
            system_prompt="You are a helpful summarizer. You don't repeat anything anyone says and you use your own words.",
            name="TLDR",
            description="",
            use_remote=use_remote
        ),
    }

    no_outside_speculation = "Use only the items provided in this prompt. Do **not** assume any extra-textual theology, mythology, or etymology."
    about_metrics = "The metrics are as follows:\n- FastText Score—measures surface-level similarity based on character n-grams; ranges 0.0 to 1.0, with higher being more morphologically similar.\n- Semantic Similarity: Compares word definitions using sentence embeddings; ranges 0.0 to 1.0, with the higher the number the more conceptually aligned.\n- Tier: a very strong connection begins/ends with the root and has a high combined score and should be taken into special consideration; from there, possible connection > somewhat possible connection > weak or no connection.\n\nUse the above metrics to weigh how directly a word supports the root hypothesis. Strong surface matches without definition alignment may be coincidental; strong semantic links without morphology might indicate metaphor or drift. Prioritize overlap when possible."

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
            ),
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
- Use only Enochian words from this list: {candidate_list}. If you need an example and it’s not in the list, say so. Do not invent forms.
- When you introduce a morphotactic claim, state the RULE in a testable form and give at least one NEGATIVE TEST it would forbid.

🎯 Your goal is not just to *respond*, but to **reassert the legitimacy** of the proposed root and demonstrate that the original analysis withstands scrutiny.
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
""",
            expected_output=f"""**Return exactly**:
REBUTTAL: <while following the adjudicator's instructions, evaluate and, if warranted, dismantle the arguments for the new root word>
EVIDENCE: <up to 5 bullets expounding the rebuttal>
POINTS_OF_AGREEMENT: <up to 3 bullets only if warranted; each bullet must name the specific claim/evidence from the Lead Linguist you accept and give a 1-sentence reason; do not introduce new evidence here; if none, write "None">
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

If you output CONTINUE, you MUST provide precise tasks for the Skeptic and Linguist.

You may refer to this **stats_summary**: 
{stats_summary}

Signals you may use (one or both):
(A) The stats_summary: n, cohesion, derivational_patterns, incompatible_meanings, metric_notes.
(B) Debate content: look for explicit RULES (position-anchored morphotactics), EVIDENCE (attested forms),
    and any explicit NEGATIVE_TESTS. Also note if any claim relies on opaque morphology (affix/function
    asserted without a RULE that can be tested on an attested form).

Guidelines:
— CONTINUE when ANY of the following applies:
  • Small/fragile evidence: n ≤ 3; OR broadened sense not grounded in RULES.
  • Opaque morphology: an affix/function is claimed but no testable RULE supports it.
  • Partial conflict: some counterexamples exist and were not neutralized by RULES/EVIDENCE.
  • Metric anomalies or missing confidence cited to justify claims (e.g., “similarity=1.00”), without corroborating RULES/EVIDENCE.
  • New evidence appeared in the last turn that the opponent has not addressed.

— STOP when ALL of the following are satisfied (or justified explicitly):
  • Cohesion is adequate OR, if cohesion not provided, the debate converged on a single concrete sense.
  • ≥1 derivational pattern is demonstrated via RULES **and** at least one attested form.
  • Incompatible meanings are either 0, or quarantined by RULES (e.g., excluded environments).
  • If NEGATIVE_TESTS were requested, at least one concrete test (using attested or clearly marked hypothetical pattern)
    has been proposed to guard scope.

When you output CONTINUE, assign focused NEXT_ROUND_TASKS with owners:
  - Skeptic: specify a falsification or stress test (e.g., demand a RULE that maps suffix → function and one attested example),
    or require a NEGATIVE_TEST that would fail under the proposed sense.
  - Linguist: supply missing RULES with anchored regex (^,$), cite ≥1 attested form, or partition senses with clear exclusions.

Do not opine on acceptance. Be procedural and concise.
"""
            ),
            #            expected_output="A ruling that begins with either ✅ ACCEPTED or ❌ REJECTED, followed by a concise rationale addressing both arguments.",
            expected_output=f"""<STOP|CONTINUE>
WHY: <1–2 sentences naming the decisive signals>
TRIGGERS: [tokens like: SMALL_N, OVERBROAD, OPAQUE_AFFIX, PARTIAL_CONFLICT, METRIC_ANOM, NEW_EVIDENCE]
IF CONTINUE, NEXT_ROUND_TASKS:
- owner=skeptic; test=<what to probe or falsify>; evidence_format=<RULES|EVIDENCE|NEGATIVE_TESTS>; success=<measurable criterion>
- owner=linguist; task=<what to supply>; evidence_format=<RULES|EVIDENCE>; success=<measurable criterion>
""",
        ),
        "gloss": Task(
            description=(
                f"""You are the Chief Enochian Lexicographer and GLOSSATOR. Your responsibility: to fill exactly one JSON object per the schema below, using only the internal Enochian debate transcript and its evidence. Apply strict-but-gracious micro-corpus standards.
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

Schema to emit (keys and casing must match):"""
                "{\n"
                f'  "ROOT": "{root.upper()}",'
                """
  "EVALUATION": "<accepted/rejected>",
  "REASON": "<1-3 sentences explaining the reason for the evaluation selected>",
  "DEFINITION": "<1-3 sentences of core semantics; no negatives, be as concrete as possible and not vague>",  
  "EXAMPLE": "<give 1-3 short example sentences of how its English equivalence would be used, marking it in each sentence.">,  
  "DECODING_GUIDE": "<concrete rules to resolve compound words, <=25 words>",
  "SEMANTIC_CORE": ["<noun/gerund>", "<noun/gerund>", "(optional)"],
  "SIGNATURE": {
    "position": "prefix|infix|suffix|root|particle|variable",
    "boundness": "bound|clitic|free|unknown",
    "slot": "initial|medial|final|mixed",
    "contribution": ["bucket[:value]", "bucket[:value]", "bucket[:value]"],
    "ontology": ["≤3 lemmas, e.g., 'motion','boundary','light'"]
  },
  "NEGATIVE_CONTRAST": ["max 4 phrases (e.g., 'non-temporal', 'non-agentive')"]
}

What follows is the debate transcript:
--------------------------------------

"""
            ),
            expected_output=(
                "{\n"
                f'    "ROOT": "{root.upper()}",'
                """
    "EVALUATION": "<accepted/rejected>",
    "REASON": "<1-3 sentences explaining the reason for the evaluation selected; reasoning should be based on an evaluation of the debate>",
    "DEFINITION": "<1-3 sentences of core semantics; no negatives, be as concrete as possible and not vague>",
    "EXAMPLE": "<give 1-3 short example sentences of how its English equivalence would be used, marking it in each sentence.>",
    "DECODING_GUIDE": "<concrete rules to resolve compound words, <=25 words>",
    "SEMANTIC_CORE": ["<noun/gerund>", "<noun/gerund>", "(optional)"],
    "SIGNATURE": {
        "position": "prefix|infix|suffix|root|particle|variable",
        "boundness": "bound|clitic|free|unknown",
        "slot": "initial|medial|final|mixed",
        "contribution": ["bucket[:value]", "bucket[:value]", "bucket[:value]"],
        "ontology": ["≤3 lemmas, e.g., 'motion','boundary','light'"]
    },
    "RULES": "Return an ARRAY (not a string) of 2–5 strings. Each string MUST follow: REGEX → +FEATURE[, +FEATURE]. Anchor REGEX to position (^, $). Use <ROOT> to stand for the candidate where applicable (e.g., ^<ROOT>.*). Encode only testable morphotactics and core semantic contributions (no metaphors/speculation). Omit any affix/function you cannot justify with evidence instead of guessing.",
    "NEGATIVE_CONTRAST": ["max 4 phrases (e.g., 'non-temporal', 'non-agentive')"]
}"""
            ),
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
    stream_text(tasks["propose"].description)
    print(f"\n{RESET}\n")

    STAGES = [
        ("linguist", 3),
        ("synthesis", 1),
        ("skeptic", 1),
        ("defend", 1),
        ("adjudicator", 1),
    ]
    debate_round = 0
    linguist_variants = []
    initial_ruling = ""
    linguist_proposal = ""
    skeptic_response = ""
    linguist_defense = []
    skeptic_rebuttal = []
    adjudicator_prompt = ""
    adjudicator_ruling = []
    gloss = ""
    tldr_summary = ""
    transcript = ""
    glossator_prompt = ""
    gloss_model = ""
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
        # elif stage_name == "initial_ruling":
        #     agent_tool = tools[stage_name]
        #     print(
        #         f"\n\n{RESET}>>>👩‍⚖️\tAn expert in the field takes an initial look at the research and decides whether it makes sense to deliberate on the material or move on to other ngram candidates...\n\n{GRAY}"
        #     )
        #     stream_text(tasks["initial_ruling"].description)
        #     print(f"\n{RESET}\n")

        #     if is_canon and blind_evaluation:
        #         initial_ruling = (
        #             f"✅ ACCEPTED\n"
        #             f"The proposed root '{root.upper()}' is already a canon entry defined as '{root_def_text}'. "
        #             "This existing definition provides sufficient internal linguistic evidence for approval.\n"
        #             "The following debate is preserved for insight and extended justification:"
        #         )
        #         print(initial_ruling)
        #         continue
        #     else:
        #         if linguist_variants and len(linguist_variants) > 0:
        #             initial_ruling = agent_tool._run(
        #                 prompt="\n".join(
        #                     [tasks["initial_ruling"].description, *linguist_variants]
        #                 ),
        #                 stream_callback=adjudicator_cb,
        #                 print_chunks=True,
        #                 role_name="👩‍⚖️\tAdjudicator",
        #             )["response_text"]
        #         else:
        #             print(f"[Error] linguist_variants are empty")

        #         txt = initial_ruling.strip().lower()
        #         initial_ruling_verdict = (
        #             txt.startswith("✅ accepted")
        #             or txt.startswith("accepted")
        #             or "✅" in initial_ruling
        #             or ("accepted" in txt and "not accepted" not in txt)
        #         )

        #         if initial_ruling and initial_ruling_verdict:
        #             continue
        #         else:
        #             print("\n\n")
        #             stream_text(initial_rejection)
        #             break
        elif stage_name == "synthesis":
            agent_tool = tools[stage_name]
            print(
                f"\n\n{RESET}>>>🥸\tThe Senior Linguist reads the reports of their juniors and begins synthesizing them into a meaningful proposal...\n{GRAY}"
            )
            if linguist_variants and len(linguist_variants) > 0:
                linguist_proposal = agent_tool._run(
                    prompt="\n".join(
                        [tasks["synthesize"].description, *linguist_variants]
                    ),
                    stream_callback=linguist_cb,
                    print_chunks=True,
                    role_name="🥸\tSenior Linguist",
                )["response_text"]
            else:
                print(f"[Error] linguist_variants are empty")
                break
        elif stage_name == "skeptic":
            agent_tool = tools[stage_name]
            print(
                f"\n\n{RESET}>>>🤔\tThe Skeptic understands the proposal and wishes to make a critique...\n{GRAY}"
            )
            stream_text(tasks["counter"].description)
            print(f"\n{RESET}\n")

            if linguist_proposal and len(linguist_proposal) > 0:
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
            else:
                print("[Error] linguist_proposal defined as ''")
                break
        elif stage_name == "defend":
            agent_tool = tools["linguist"]
            print(
                f"\n\n{RESET}>>>🥸\tThe Lead Linguist's considers what the Skeptic has said and prepares a defense...\n{GRAY}",
                tasks["defend"].description,
                f"{RESET}\n",
            )
            if (
                len(skeptic_rebuttal) > 0 and len(skeptic_rebuttal[debate_round]) > 0
            ) or (skeptic_response and len(skeptic_response) > 0):
                defense_prompt_data = [
                    tasks["defend"].description,
                    f"You said earlier: {linguist_proposal}\n\n",
                    f"Skeptic said: {skeptic_response}\n\n",
                ]
                if (
                    len(skeptic_rebuttal) > 0
                    and len(skeptic_rebuttal[debate_round]) > 0
                ):
                    # this is sometimes supposed to be range(0, 0) in how it turns out
                    for i in range(0, debate_round):
                        defense_prompt_data.append(
                            f"Then you defended by saying: {linguist_defense[i]}\n\n"
                        )
                        defense_prompt_data.append(
                            f"Then the adjudicator declared: {adjudicator_ruling[i]}\n\n"
                        )
                        defense_prompt_data.append(
                            f"Then the skeptic rebuttaled: {skeptic_rebuttal[i]}\n\n"
                        )
                    debate_round += 1
                defense_prompt_data.append(
                    f"Your goal: {tasks['defend'].expected_output}"
                )
                linguist_defense.append(
                    agent_tool._run(
                        prompt="\n".join(defense_prompt_data),
                        stream_callback=linguist_cb,
                        print_chunks=True,
                        role_name="🥸\tSenior Linguist",
                    )["response_text"]
                )
            else:
                print("[Error] skeptic_response defined as ''")
                break
        elif stage_name == "adjudicator":
            agent_tool = tools[stage_name]
            if len(linguist_defense) > 0 and len(linguist_defense[debate_round]) > 0:
                print(
                    f"\n\n{RESET}>>>👩‍⚖️\tA mutual colleague adjudicating the debate wishes to weigh in...\n{GRAY}"
                )
                stream_text(tasks["ruling"].description)
                f"\n{RESET}\n"

                # if is_canon:
                #     adjudicator_ruling = (
                #         f"✅ ACCEPTED\n"
                #         f"The proposed root '{root.upper()}' is already a canon entry defined as '{root_def_text}'. "
                #         "This existing definition provides sufficient internal linguistic evidence for approval.\n"
                #         "The prior debate is preserved for insight and extended justification."
                #     )
                #     stream_text(adjudicator_ruling)
                #     print()

                adjudicator_prompt = [
                    tasks["ruling"].description,
                    f"Linguist proposed: {linguist_proposal}\n\n",
                    f"Skeptic replied: {skeptic_response}\n\n",
                    f"Linguist defended by arguing: {linguist_defense[0]}\n\n",
                ]
                # intentionally range(0, 0) sometimes
                for i in range(0, debate_round):
                    adjudicator_prompt.append(
                        f"You said: {adjudicator_ruling[i]}\n\n"
                    )
                    adjudicator_prompt.append(
                        f"Then the skeptic rebuttaled: {skeptic_rebuttal[i]}\n\n"
                    )
                    adjudicator_prompt.append(
                        f"Then the linguist defended by saying: {linguist_defense[i + 1]}\n\n"
                    )
                adjudicator_prompt.append(
                    f"Expected output: {tasks['ruling'].expected_output}"
                )
                adjudicator_ruling.append(agent_tool._run(
                    prompt="\n".join(adjudicator_prompt),
                    stream_callback=adjudicator_cb,
                    print_chunks=True,
                    role_name="👩‍⚖️\tAdjudicator",
                )["response_text"])

                if "CONTINUE" in adjudicator_ruling[debate_round] and len(STAGES) < 10:
                    STAGES.append(("rebuttal", 1))
                    STAGES.append(("defend", 1))
                    STAGES.append(("adjudicator", 1))
                else:
                    STAGES.append(("glossator", 1))
                    STAGES.append(("summarizer", 1))

            else:
                print("[Error] Linguist's Defense is an empty array!")
                break
        elif stage_name == "rebuttal":
            if len(linguist_defense) > 0 and len(linguist_defense[debate_round]) > 0:
                agent_tool = tools["skeptic"]
                print(
                    f"\n\n{RESET}>>>🤔\tThe Skeptic considers and prepares a final criticism...\n{GRAY}"
                )
                stream_text(tasks["rebuttal"].description)
                print(f"\n{RESET}\n")

                # this is intentionally potentially range(0, 0) sometimes
                for i in range(0, debate_round):
                    defense_prompt_data.append(
                        f"You rebuttaled: {skeptic_rebuttal[i]}\n\n"
                    )
                    defense_prompt_data.append(
                        f"Then the linguist defended by saying: {linguist_defense[i]}\n\n"
                    )
                skeptic_rebuttal.append(
                    agent_tool._run(
                        prompt="\n".join(
                            [
                                tasks["rebuttal"].description,
                                f"Linguist proposed: {linguist_proposal}\n\n",
                                f"You replied: {skeptic_response}\n\n",
                                f"Linguist defended by arguing: {linguist_defense}\n\n",
                                f"Adjudicator decided the debate should continue. Do what they are asking you, the skeptic, to do: {adjudicator_ruling}\n\n"
                                f"Expected output: {tasks['rebuttal'].expected_output}",
                            ]
                        ),
                        stream_callback=skeptic_cb,
                        print_chunks=True,
                        role_name="🤔\tSkeptic",
                    )["response_text"]
                )
            else:
                print("[Error] linguist_defense defined as ''")
                break
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
                f"## Linguist defended:\n{linguist_defense[0]}\n\n",
            ]
            # again, sometimes this is supposed to be range(0, 0)
            for i in range(0, debate_round):
                debate_history_pregloss.append(
                    f"## Adjudicator decided:\n{adjudicator_ruling[i]}\n\n"
                )
                debate_history_pregloss.append(
                    f"## Skeptic rebuttaled:\n{skeptic_rebuttal[i]}\n\n"
                )
                debate_history_pregloss.append(
                    f"## Linguist defended:\n{linguist_defense[i + 1]}\n\n"
                )
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
            lines.append(linguist_defense[0].strip())
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
        "Linguist": linguist_proposal,
        "Skeptic": skeptic_response,
        "Defense": linguist_defense,
        "Rebuttal": skeptic_rebuttal,
        "Adjudicator": adjudicator_ruling,
        "Adjudicator_Prompt": adjudicator_prompt,
        "Glossator": gloss,
        "Glossator_Prompt": glossator_prompt,
        "Glossator_Model": gloss_model,
        "Archivist": transcript,
        "summary": tldr_summary,
        "raw_output": {
            "Linguist": linguist_proposal,
            "Skeptic": skeptic_response,
            "Defense": linguist_defense,
            "Rebuttal": skeptic_rebuttal,
            "Adjudicator": adjudicator_ruling,
            "Adjudicator_Prompt": adjudicator_prompt,
            "Glossator": gloss,
            "Glossator_Prompt": glossator_prompt,
            "Glossator_Model": gloss_model,
            "Archivist": transcript,
            "summary": tldr_summary,
        },
    }
