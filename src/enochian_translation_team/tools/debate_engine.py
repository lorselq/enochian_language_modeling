from typing import Optional
from crewai import Agent, Task, Crew
from enochian_translation_team.tools.query_model_tool import QueryModelTool


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
    candidates: list[dict],
    stats_summary: str,
    stream_callback=None,
    root_entry: Optional[dict] = None,
):
    joined_defs = []
    candidate_list = '"' + ", ".join(c["word"].upper() for c in candidates) + '"'

    for c in candidates:
        word = c.get("word", "")
        definition = c.get("definition", "")
        fasttext = round(c.get("fasttext", 0.0), 3)
        semantic = round(c.get("semantic", 0.0), 3)
        tier = c.get("tier", "Untiered")

        if word and definition:
            line = (
                f"{word.strip()} — {definition.strip()} "
                f"<fasttext:{fasttext}, semantic similarity:{semantic}, tier:{tier}>"
            )
            joined_defs.append(line)
    if root_entry is None:
        root_entry = next(
            (c for c in candidates if c.get("word", "").lower() == root.lower()), None
        )
    selected_defs = select_definitions(joined_defs, max_words=75)
    root_def_summary = " | ".join(selected_defs) + (
        "..." if len(joined_defs) > len(selected_defs) else ""
    )

    is_canon = bool(root_entry and root_entry.get("definition"))

    if root_entry and root_entry.get("definition"):
        extra_prompt = f"⚠️ Reminder: The root '{root}' is already defined in the corpus as '{root_entry.get('definition')}'. Consider this as a potential anchor.\n"
        skeptic_hint = f"\n\n🧐 Note: The root '{root}' is already defined in the corpus as '{root_entry.get('definition')}'. This lends strong weight towards its inclusion as a root word that should be accepted. Consider this in your critique."
    else:
        extra_prompt = ""
        skeptic_hint = ""

    # === AGENTS ===
    tools = {
        "linguist": QueryModelTool(
            system_prompt="""
                You are a **disciplined and insightful computational linguist** specializing in the Enochian language—a constructed system with irregular morphology, cryptic derivations, and unknown origin.

                Your task is to **evaluate a proposed root** by identifying **semantic and morphological overlaps** across a set of candidate words. Focus your attention on:
                - Shared prefixes, suffixes, or internal substrings
                - Repetition or structural similarity in word forms
                - Overlapping definitions and contextual meanings from citations

                ⚠️ DO NOT reference natural language etymologies (e.g., English, Greek, Latin, Hebrew). No speculative outside sources.
                All reasoning must rely exclusively on **internal evidence**—relationships and patterns among the Enochian words themselves.

                Your tone must be confident, scholarly, and analytical.
                Use specific examples. Clearly explain why any connections you observe are linguistically plausible, not merely coincidental.

                Be thorough, avoid vague generalizations, and always back claims with observed data.
                """,
            name="Junior Research Linguist",
            description="",
        ),
        "synthesis": QueryModelTool(
            system_prompt="""
                You are the Lead Linguist in a collaborative reverse-engineering effort focused on the Enochian language—a system with obscure morphology and nonstandard linguistic structures.

                You have received analytical reports from five Junior Linguists, each offering observations on the same proposed root.

                Your task:
                - Synthesize their insights into a **single, cohesive proposal**.
                - Prioritize **common patterns or shared conclusions** across the responses.
                - Highlight **strong arguments**, discarding speculation or weak/unsubstantiated claims.
                - Do **not repeat all points**—only include the most persuasive and consistent ideas.

                Focus on:
                - Morphological regularities (shared prefixes, suffixes, or structures)
                - Semantic overlap across definitions
                - Citational or contextual clues that reinforce connections

                Your tone should be polished, scholarly, and decisive—this is the authoritative linguistic interpretation that will be presented to the Adjudicator.

                Do not hedge. Present the best possible case for this root as a meaningful candidate, based solely on internal linguistic evidence from the Enochian data.
                """,
            name="Lead Linguist",
            description="",
        ),
        "skeptic": QueryModelTool(
            system_prompt="""
                You are a skeptical linguist evaluating a proposed root analysis in the Enochian language. Your goal is to identify flawed reasoning, superficial pattern-matching, or semantic inconsistencies.

                You have received a synthesized proposal from the Lead Linguist. Critically evaluate whether:
                - The words cited genuinely share meaning or structure
                - The claimed morphological patterns are consistent and non-coincidental
                - Semantic overlap is significant, not just rhetorical
                - Tiering is justified based on empirical thresholds (e.g., FastText similarity, semantic alignment)

                Do not dismiss arguments just because they involve theological or metaphysical frameworks—these are valid within Enochian's system. However, be vigilant about overreach, cherry-picked evidence, or unjustified leaps in logic.

                If the root hypothesis lacks rigor, clearly explain why. Offer specific counterpoints. If you believe a stronger candidate or cluster exists, propose it concisely—but only if the evidence supports it.

                Your tone is incisive, precise, and intellectually honest. Your aim is not to destroy for its own sake, but to ensure that only the most robust linguistic hypotheses move forward.
                """,
            name="Skeptic",
            description="",
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
        ),
        "glossator": QueryModelTool(
            system_prompt="You are a highly precise Enochian glossator. Your role is to propose a single, clear, authoritative dictionary-style definition for a root word that has been approved by an adjudicator. Use the prior linguistic analysis to distill the core conceptual meaning of the word, based solely on its internal usage patterns, morphology, and semantic range across the cited examples. Avoid descriptive summaries. Instead, craft a definition that would be suitable for formal inclusion in a lexicon. This definition should be concise (1-2 lines), but maximally informative. You must not reference English or natural-language etymology. Write in an academic tone, as if submitting this to a linguistic corpus project.",
            name="Glossator",
            description="",
        ),
    }

    # === AGENTS (not needed at present) ===
    # agents = {
    #     "linguist": Agent(
    #         role="Linguist",
    #         goal="Analyze the semantic relationships between these Enochian words and their definitions. Identify shared morphemes or root candidates based on: shared character substrings (especially prefixes/suffixes); overlapping definitions or conceptual meanings; patterns in usage across citations. Justify why these words might be related, but never use English, Greek, Latin, or Hebrew etymology to substantiate a proposed Enochian root word. Never use hypothetical Enochian words you've made up to justify your arguments; use only the ones provided in the prompts. Reference actual text segments or gloss overlaps.",
    #         backstory="A creative, inventive, and excited linguist with deep pattern recognition skills, always hopeful to discover something new and great.",
    #         tools=[tools["linguist"], tools["synthesis"]],
    #         verbose=True,
    #         callbacks=[stream_callback] if stream_callback else [],
    #     ),
    #     "skeptic": Agent(
    #         role="Skeptic",
    #         goal="You are a skeptical linguist. Critically examine the Linguist's proposed root and semantic analysis. Look for: weak reasoning or overgeneralization; words grouped together without strong definition overlap; n-grams that appear coincidentally (e.g., short, common patterns). Propose stronger alternatives if you have them, or explain why a proposal lacks rigor.",
    #         backstory="A cynical critic with linguistic expertise and a grudge against bad etymology.",
    #         tools=[tools["skeptic"]],
    #         verbose=True,
    #         callbacks=[stream_callback] if stream_callback else [],
    #     ),
    #     "adjudicator": Agent(
    #         role="Adjudicator",
    #         goal="Make a judgment call based on both sides.",
    #         backstory="A supervising linguist who wants solid reasoning before accepting newly proposed root words as part of the lexicon.",
    #         tools=[tools["adjudicator"]],
    #         verbose=True,
    #         callbacks=[stream_callback] if stream_callback else [],
    #     ),
    #     "glossator": Agent(
    #         role="Glossator",
    #         goal="Write a clear, precise, and lexicon-ready definition for an approved Enochian root word, based on all the prior linguistic reasoning and citation patterns.",
    #         backstory="A senior philologist responsible for converting root hypotheses into formal dictionary entries for the Enochian corpus. Deeply meticulous and wary of poetic overreach.",
    #         tools=[tools["glossator"]],
    #         verbose=True,
    #         callbacks=[stream_callback] if stream_callback else [],
    #     ),
    # }
    about_enochiana = "As a bit of context about the Enochian language: the root words are derived from Enochian, the language Adam spoke (from the Biblical Adam and Eve), and is allegedly used as a form of celestial speech by angels and other divine entities; there are many Christian (and Gnostic) undertones in the language, and the known words' main focus is divine cosmology, theology, and human action and government."
    about_metrics = "The metrics are as follows:\n- FastText Score—measures surface-level similarity based on character n-grams; ranges 0.0 to 1.0, with higher being more morphologically similar.\n- Semantic Similarity: Compares word definitions using sentence embeddings; ranges 0.0 to 1.0, with the higher the number the more conceptually aligned.\n- Tier: a very strong connection begins/ends with the root and has a high combined score and should be taken into special consideration; from there, possible connection > somewhat possible connection > weak or no connection.\n\nUse the above metrics to weigh how directly a word supports the root hypothesis. Strong surface matches without definition alignment may be coincidental; strong semantic links without morphology might indicate metaphor or drift. Prioritize overlap when possible."

    tasks = {
        "propose": Task(
            description=f"""
You are a **disciplined and insightful computational linguist** specializing in the Enochian language—a constructed system with irregular morphology, cryptic derivations, and unknown origin.

Your task is to **evaluate the root candidate '{root}'** by analyzing semantic and morphological overlap across its proposed related words.

Begin with the following semantic stats:

{stats_summary}

Focus your analysis on:
- Shared prefixes, suffixes, or internal substrings
- Repetition or structural similarity in word forms
- Overlapping meanings in definitions and contextual usage (citations)

⚠️ DO NOT use natural language etymologies (e.g., English, Greek, Latin, Hebrew). No speculative comparisons to outside languages.
⚠️ DO NOT use any Enochian words, real or imagined, as part of your justification other than those given here: {candidate_list}
All justification must come from **internal evidence only**—patterns observed across Enochian wordforms and meanings.

With this in mind, examine the following definitions and citations (contained within square brackets, pipe-delimited, most relevant first) for the root '{root}':

{root_def_summary}

Use these to **propose a coherent explanation of the root** based on morphological structure and shared semantics.

{about_enochiana}
{about_metrics}
{extra_prompt}

Your tone must be scholarly and confident. Avoid vague generalizations. Use examples, and support your claims with specific patterns or semantic signals.
""",
            expected_output="A strong case for the root, citing semantic and morphological evidence.",
        ),
        "synthesize": Task(
            description=f"""
You are the **Lead Linguist** in a collaborative reverse-engineering initiative focused on the Enochian language—a constructed system with obscure morphology, irregular derivation, and no known linguistic relatives.

You have received detailed analyses from five Junior Linguists, each offering perspectives on the proposed root: **'{root}'**.

Your task:
- **Synthesize their insights into a single, cohesive proposal**
- **Emphasize shared observations** or recurring arguments across the team
- **Select only the most persuasive claims**, discarding any speculative, redundant, or weak points
- **Avoid listing all contributions**—this is not a recap, but a distillation

Focus your analysis on:
- Morphological structure (prefixes, suffixes, repeated substrings)
- Semantic overlap across definitions
- Contextual or citational consistency

This report will be delivered to the Adjudicator, so your tone must be **scholarly, confident, and definitive**. This is the authoritative linguistic argument.

⚠️ Do not reference external etymologies (e.g., Latin, Hebrew, English). All justification must arise from internal evidence and patterns among Enochian words.

The junior research team used the following definitions and citations as part of their arguments. Use them as supporting context where helpful:

{root_def_summary}

{about_enochiana}
{about_metrics}
{extra_prompt}
""",
            expected_output="A definitive and well-argued proposal for the root, based on internal semantic and morphological evidence, synthesizing the strongest arguments from the junior linguists.",
        ),
        "counter": Task(
            description=f"""
You are a **skeptical linguist** evaluating a proposed root analysis in the Enochian language—a system with opaque morphology and metaphysical entanglements.

You have received a synthesized proposal from the Lead Linguist. Your role is to **critically assess the validity** of this analysis and challenge any weaknesses in reasoning.

Focus on the following:
- Do the cited words **genuinely share meaning or structure**, or is the overlap superficial?
- Are the **morphological patterns** consistent and non-coincidental?
- Is **semantic similarity** supported by actual definitions and usage, not just rhetorical association?
- Are the **tiers** of relevance justified using empirical metrics (FastText similarity, semantic alignment)?

🧠 You are permitted to accept that some Enochian root meanings may be abstract or metaphorical—many accepted roots display this. However, **you must remain vigilant against overreach, cherry-picked evidence, or unjustified speculation.**

If the proposal lacks linguistic rigor:
- Clearly explain **why** and identify specific weak points
- Suggest a **stronger candidate or cluster**, if one can be supported from the data

If there are any Enochian words used to justify the root's possible meaning, they must come from this list: {candidate_list}. If the Lead Linguist uses any Enochian words other than the ones in that list, call them out as hallucinations right away.

{skeptic_hint}
Your tone must be **sharp, disciplined, and logically rigorous**. You are not here to sabotage, but to **safeguard the integrity** of the linguistic record.
""",
            expected_output="A focused, evidence-based rebuttal to the proposed root word—highlighting flawed logic, semantic gaps, or alternative interpretations, when supported.",
        ),
        "defend": Task(
            description="""
You are the **Lead Linguist** defending a proposed Enochian root candidate after receiving a skeptical counter-analysis.

Your task:
- **Directly address the Skeptic's objections** with clear, evidence-based rebuttals.
- Reaffirm the **morphological and semantic rationale** that supports the root's candidacy.
- Identify any misinterpretations or overly narrow assumptions in the Skeptic's argument.
- Defend the use of empirical metrics (e.g., FastText similarity, semantic cohesion) as legitimate support for root analysis.
- Justify abstract or metaphorical readings **if grounded in internal Enochian evidence**.

Your tone must be:
- **Confident** (you are the expert)
- **Analytical** (you argue with data)
- **Persuasive** (you're here to win over the Skeptic—or at least dismantle their critique)

🎯 Your goal is not just to *respond*, but to **reassert the legitimacy** of the proposed root and demonstrate that the original analysis withstands scrutiny.
""",
            expected_output="A confident, evidence-driven defense of the root hypothesis that refutes the Skeptic's critique and re-establishes the root as a serious candidate for inclusion.",
        ),
        "rebuttal": Task(
            description="""
You are the **Skeptical Linguist**, issuing your **final response** after reviewing the Lead Linguist's defense of a proposed Enochian root.

Your task:
- Determine whether your initial objections were **fully and convincingly addressed**.
- If key issues remain unresolved, issue a **focused, final rebuttal**. Do not repeat old arguments—refine them.
- If the defense was **persuasive and thorough**, acknowledge the strength of their case—skepticism includes being open to revision when warranted.

You must:
- Pinpoint any remaining **logical inconsistencies**, unconvincing assumptions, or semantic leaps.
- Avoid vague dismissals—only critique if you can articulate **specific remaining weaknesses**.
- If the defense meaningfully strengthens the hypothesis, say so—but make it clear *why*.

🎯 This is your last chance to weigh in before the adjudication. Be precise, fair, and intellectually rigorous.
""",
            expected_output="A conclusive rebuttal that either challenges unresolved flaws in the Linguist's defense or concedes that the root candidate now appears valid.",
        ),
        "ruling": Task(
            description=(
                "Review the full exchange between the Linguist and the Skeptic. Your job is to make a clear and final determination:\n\n"
                "**Should this proposed root be accepted as a meaningful candidate for future reverse-engineering of the Enochian language?**\n\n"
                "You must START your response with either:\n"
                "✅ ACCEPTED\n"
                "or\n"
                "❌ REJECTED\n"
                "— Nothing else may come before this line. This format is **mandatory**.\n\n"
                "Your ruling must weigh the core arguments on both sides, focusing on:\n"
                "- Linguistic plausibility\n"
                "- Semantic cohesion across definitions\n"
                "- Morphological consistency or structure\n"
                "- Whether the defense meaningfully addressed the skeptic’s objections\n"
                "- Use of empirical metrics (e.g., FastText similarity, semantic alignment)\n\n"
                "Assume the data provided is all that is available, and that metric thresholds are valid and statistically derived.\n"
                "Abstract or metaphorical meanings are acceptable if supported by internal consistency.\n\n"
                "Be concise, definitive, and analytical. No hedging.\n\n"
                "Begin with the ruling, then follow with a 1–3 sentence justification.\n\n"
                "+++\n\n"
            ),
            expected_output="A ruling that begins with either ✅ ACCEPTED or ❌ REJECTED, followed by a concise rationale addressing both arguments.",
        ),
        "gloss": Task(
            description=(
                f'The adjudicator has approved the root "{root}". Your responsibility is to respond with a precise and practical dictionary-style entry.\n\n'
                "Your definition must:\n"
                "- Describe the **core semantic meaning** of the root\n"
                "- Indicate how it functions **morphologically** (e.g., prefix, infix, suffix)\n"
                "- Explain its **role** in compound or derived words (e.g., what kind of meaning it adds and how it functions)\n"
                "- Provide **guidance** on how this root could help decode other, currently unknown words\n\n"
                "Format your output as:\n"
                f"{root} - [Definition including both meaning and morphological/functional guidance.]\n\n"
                "Focus entirely on internal linguistic evidence and patterns observed across related Enochian words. DO NOT reference English, Greek, Latin, or Hebrew etymologies.\n"
                "Below is a summary of the debate and root data. Use it to guide your construction of the definition:\n\n"
            ),
            expected_output=(
                f"{root.upper()} - A linguistically precise and practically useful definition that reflects both the semantic meaning and usage potential of the root in compound forms."
            ),
        ),
    }

    # === Direct Tool Access with Streaming ===
    GRAY = "\033[90m"
    RESET = "\033[0m"

    linguist_cb = (
        (lambda r, m: stream_callback("Linguist", m)) if stream_callback else None
    )
    skeptic_cb = (
        (lambda r, m: stream_callback("Skeptic", m)) if stream_callback else None
    )
    adjudicator_cb = (
        (lambda r, m: stream_callback("Adjudicator", m)) if stream_callback else None
    )
    glossator_cb = (
        (lambda r, m: stream_callback("Glossator", m)) if stream_callback else None
    )

    # === LINGUISTS ===
    if stream_callback:
        stream_callback("Linguist", "**Linguist:**")

    # separator between words
    print("\n\n\n")
    print(
        f"==={(len('Now discussing the possible root word ') + len(f'<{root}>')) * '='}==="
    )
    print(f"===Now discussing the possible root word '{root}'===")
    print(
        f"==={(len('Now discussing the possible root word ') + len(f'<{root}>')) * '='}==="
    )

    # === RESEARCH TEAM ===
    linguist_variants = []
    for i in range(5):
        if i == 0:
            print(
                f"{GRAY}Starting prompt for research team: {tasks['propose'].description}"
            )
        print(
            f"\n\n>>>👩‍🎓\tLinguist {i + 1}'s research on the root word...\n{GRAY}Research:"
        )
        variant = tools["linguist"]._run(
            prompt=tasks["propose"].description
            + "\nYour goal is: "
            + tasks["propose"].expected_output.lower(),
            stream_callback=linguist_cb,
            print_chunks=True,
            role_name=f"👩‍🎓 Junior Linguist Researcher #{i+1}",
        )
        linguist_variants.append(variant)

    # === LEAD LINGUIST: 1 ===
    print(
        f"\n\n{RESET}>>>🥸\tLead Linguist's turn to propose a new root word...\nSynthesizing junior linguist's input into a meaningful argument:"
    )

    linguist_proposal = tools["synthesis"]._run(
        prompt="\n".join([tasks["synthesize"].description, *linguist_variants]),
        stream_callback=linguist_cb,
        print_chunks=True,
        role_name="🥸\tLead Linguist",
    )

    # === SKEPTIC: 1 ===
    if stream_callback:
        stream_callback("Skeptic", "**Skeptic:**")

    print(
        f"\n\n{RESET}>>>🤔\tSkeptic's turn to refute...\nRefutation prompt:{GRAY}",
        tasks["counter"].description,
        f"\n{RESET}",
    )

    skeptic_response = tools["skeptic"]._run(
        prompt="\n".join(
            [
                tasks["counter"].description,
                f"Linguist said: {linguist_proposal}",
                f"Your goal: {tasks['counter'].expected_output.lower()}",
            ]
        ),
        stream_callback=skeptic_cb,
        print_chunks=True,
        role_name="🤔\tSkeptic",
    )

    # === LEAD LINGUIST: 2 ===
    if stream_callback:
        stream_callback("Linguist", "**Linguist (Defense):**")

    print(
        f"\n\n{RESET}>>>🥸\tLead Linguist's turn to defend...\nDefense prompt:{GRAY}",
        tasks["defend"].description,
        f"{RESET}\n",
    )

    linguist_defense = tools["linguist"]._run(
        prompt="\n".join(
            [
                tasks["defend"].description,
                f"Skeptic said: {skeptic_response}",
                f"Your goal: {tasks['defend'].expected_output.lower()}",
            ]
        ),
        stream_callback=linguist_cb,
        print_chunks=True,
        role_name="🥸\tLead Linguist",
    )

    # === SKEPTIC: 2 ===
    if stream_callback:
        stream_callback("Skeptic", "**Skeptic (Rebuttal):**")

    print(
        f"\n\n{RESET}>>>🤔\tSkeptic's turn to rebuttal...\nFinal word:{GRAY}",
        tasks["rebuttal"].description,
        f"{RESET}\n",
    )
    skeptic_rebuttal = tools["skeptic"]._run(
        prompt="\n".join(
            [
                tasks["rebuttal"].description,
                f"Linguist proposed: {linguist_proposal}",
                f"You replied: {skeptic_response}",
                f"Linguist defended by arguing: {linguist_defense}",
                f"Your goal: {tasks['rebuttal'].expected_output.lower()}",
            ]
        ),
        stream_callback=skeptic_cb,
        print_chunks=True,
        role_name="🤔\tSkeptic",
    )

    # === ADJUDICATOR ===
    if stream_callback:
        stream_callback("Adjudicator", "**Adjudicator:**")

    print(
        f"\n\n{RESET}>>>👩‍⚖️\tAdjudicator's turn to pass their ruling...\nRuling:{GRAY}",
        tasks["ruling"].description,
        f"{RESET}\n",
    )

    if is_canon:
        adjudicator_ruling = (
            f"✅ ACCEPTED\n"
            f"The proposed root '{root.upper()}' is already a canon entry defined as '{root_entry.get('definition') if root_entry else ''}'. "
            "This existing definition provides sufficient internal linguistic evidence for approval.\n"
            "The following debate is preserved for insight and extended justification:"
        )
        print(adjudicator_ruling)
    else:
        adjudicator_ruling = tools["adjudicator"]._run(
            prompt="\n".join(
                [
                    tasks["ruling"].description,
                    f"Linguist proposed: {linguist_proposal}",
                    f"Skeptic replied: {skeptic_response}",
                    f"Linguist defended by arguing for: {linguist_defense}",
                    f"Skeptic made their final argument against: {skeptic_rebuttal}"
                    f"Your goal: {tasks['ruling'].expected_output.lower()}",
                ]
            ),
            stream_callback=adjudicator_cb,
            print_chunks=True,
            role_name="👩‍⚖️\tAdjudicator",
        )

    # === GLOSSATOR ===
    gloss = f"<there is no (new) definition for '{root.upper()}'>"
    if (
        adjudicator_ruling.strip().lower().startswith("✅ accepted")
        or adjudicator_ruling.strip().lower().startswith("accepted")
        or "✅" in adjudicator_ruling
    ):
        if stream_callback:
            stream_callback("Glossator", "**Glossator:**")

        print(
            f"\n\n{RESET}>>>🧐\tGlossator's turn to provide a definition...\nGenerating definition...\n"
        )

        gloss = tools["glossator"]._run(
            prompt="\n".join(
                [
                    tasks["gloss"].description,
                    f"Linguist proposed: {linguist_proposal}",
                    f"Skeptic replied: {skeptic_response}",
                    f"Linguist defended by arguing for: {linguist_defense}",
                    f"Skeptic made their final argument against: {skeptic_rebuttal}"
                    f"Adjudicator decided: {adjudicator_ruling}",
                ]
            ),
            stream_callback=glossator_cb,
            print_chunks=True,
            role_name="🧐\tGlossator",
        )

    tldr_tool = QueryModelTool(
        system_prompt="You are a helpful summarizer. You don't repeat anything anyone says and you use your own words."
    )

    archivist_summary_formatted = (
        "=== 📖 PROMPT FOR LINGUIST ===\n"
        + tasks["propose"].description.strip()
        + "\n\n=== 🥸 LINGUIST PROPOSAL ===\n"
        + linguist_proposal.strip()
        + "\n\n=== 🤔 SKEPTIC ===\n"
        + skeptic_response.strip()
        + "\n\n=== 🥸 DEFENSE ===\n"
        + linguist_defense.strip()
        + "\n\n=== 🤔 REBUTTAL ===\n"
        + skeptic_rebuttal.strip()
        + "\n\n=== 👩‍⚖️ ADJUDICATOR ===\n"
        + adjudicator_ruling.strip()
    )

    if gloss:
        archivist_summary_formatted += "\n\n=== 🧐 GLOSSATOR ===\n" + gloss

    tldr_summary = tldr_tool._run(
        prompt="Summarize the following root word debate in 1-2 sentences; your focus should be summarizing the strongest, key arguments, and very briefly indicating whether or not the adjudicator accepted the root word proposal:\n\n"
        + archivist_summary_formatted,
        stream_callback=None,
    )

    archivist_summary_formatted = (
        "\n\n=== 📜 SUMMARY ===\n"
        + tldr_summary.strip()
        + "\n\n\n========================\n====== TRANSCRIPT ======\n========================\n\n"
        + archivist_summary_formatted
    )

    return {
        "Linguist": linguist_proposal,
        "Skeptic": skeptic_response,
        "Defense": linguist_defense,
        "Rebuttal": skeptic_rebuttal,
        "Adjudicator": adjudicator_ruling,
        "Glossator": gloss,
        "Archivist": archivist_summary_formatted,
        "summary": tldr_summary,
        "raw_output": {
            "Linguist": linguist_proposal,
            "Skeptic": skeptic_response,
            "Defense": linguist_defense,
            "Rebuttal": skeptic_rebuttal,
            "Adjudicator": adjudicator_ruling,
            "Glossator": gloss,
            "Archivist": archivist_summary_formatted,
            "summary": tldr_summary,
        },
    }
