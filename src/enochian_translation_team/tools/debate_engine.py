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

    if root_entry and root_entry.get("definition"):
        extra_prompt = f"⚠️ Reminder: The root '{root}' is already defined in the corpus as '{root_entry.get('definition')}'. Consider this as a potential anchor.\n"
        skeptic_hint = f"\n\n🧐 Note: The root '{root}' is already defined in the corpus as '{root_entry.get('definition')}'. This lends strong weight towards its inclusion as a root word that should be accepted. Consider this in your critique."
    else:
        extra_prompt = ""
        skeptic_hint = ""

    # === AGENTS ===
    tools = {
        "linguist": QueryModelTool(
            system_prompt="You are a bold and insightful computational linguist specializing in the Enochian language—a constructed system with irregular morphology and uncertain origins. Your job is to analyze a proposed root by examining semantic and morphological overlap across multiple words. Identify patterns in prefixes, suffixes, or repeated substrings that suggest shared structure. Support your hypothesis by referencing similarities in definitions, glosses, or contextual usage from citations. Do not use natural language etymologies (e.g., English, Greek, Hebrew, or Latin roots). Justify relationships based solely on internal evidence across Enochian terms. Your tone should be confident and scholarly. Provide specific examples and explain why the connection is more than coincidental. Absolutely be thorough in your justifications.",
            name="Junior Research Linguist",
            description="",
        ),
        "synthesis": QueryModelTool(
            system_prompt="You're the lead linguist. Given multiple root analyses by junior linguists, synthesize them into one strong, cohesive proposal with the best arguments only, giving preference to common ideas.",
            name="Lead Linguist",
            description="",
        ),
        "skeptic": QueryModelTool(
            system_prompt="You are a skeptical linguist reviewing a proposed root analysis in the Enochian language. Your goal is to uncover weak reasoning, accidental pattern-matching, or semantic mismatches. Examine whether the proposed words actually share meaningful definitions. Challenge vague or speculative claims. Look for missing evidence or inconsistent logic. If the root hypothesis is flawed, explain why. If you believe a stronger candidate exists—which is something you would like—make a concise counterproposal. You are sharp, analytical, and unafraid to criticize overreach.",
            name="Skeptic",
            description="",
        ),
        "adjudicator": QueryModelTool(
            system_prompt="Review the arguments from both the Linguist and the Skeptic. Make a final determination: should this root be accepted as a meaningful candidate? Be concise and definitive. A short rationale is fine, but it must address key reasoning on both sides. Your response must begin with either ✅ ACCEPTED or ❌ REJECTED, no exceptions!",
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
            description=f"""{extra_prompt}Analyze the root candidate '{root}' using the following semantic stats:\n\n{stats_summary}\n\nBreak down shared semantics or patterns. Propose a coherent explanation of the root. Do not use English, Greek, Hebrew, or Latin etymological justifications; the proposal must come from the candidate root word's letter composition and possible meanings based on its and related word's definitions. {about_enochiana} {about_metrics}\n\nWith the above in mind, consider the following definitions and citations contained within [] (they are pipe-delimited and strongly ordered from most to least relevant) in your evaluation of '{root}':\n{root_def_summary}""",
            expected_output="A strong case for the root, citing semantic and morphological evidence.",
        ),
        "synthesize": Task(
            description=f"You're the lead linguist. Given multiple root analyses by junior linguists, synthesize them into one strong, cohesive proposal with the best arguments only, giving preference to common ideas. {about_enochiana}\nThe basis of the research team's arguments utilize metrics and definitions for the potential root, '{root}'. {about_metrics}\n\nThe research team used the following definitions with accompanying citations as part of their arguments, which are included here to provide further context:\n\n{root_def_summary}",
            expected_output="A strong case for the root, citing semantic and morphological evidence, synthesizing the strongest points from the junior linguist team's research.",
        ),
        "counter": Task(
            description="Respond to the Linguist's analysis. Challenge weak points, semantic gaps, or coincidences. However, you should allow for somewhat generalized, abstract, and metaphorical meanings; established root words found previously tend to be abstract in nature, so it follows these will be too."
            + skeptic_hint,
            expected_output="A thorough and convincing rebuttal to the Linguist's proposal to add the new root word to the records.",
        ),
        "defend": Task(
            description="Do your absolute best to defend the original linguistic hypothesis. Respond to the Skeptic's objections directly. Try to convince the Skeptic to see things your way.",
            expected_output="A solid defense and doubling down on the original linguistic hypothesis; a direct response to the Skeptic's criticisms; an attempt to sway the Skeptic to accept the new root word.",
        ),
        "rebuttal": Task(
            description="Issue a final rebuttal if the defense failed to address key concerns.",
            expected_output="Either a final rebuttal that reiterates criticisms that were not addressed by the Linguist's defense or an acknowledgement that the Linguist may be onto something.",
        ),
        "ruling": Task(
            description=(
                "Make a ruling based on the discussion after '+++'. You must START your response with either:\n"
                "✅ ACCEPTED\n"
                "or\n"
                "❌ REJECTED\n"
                "— Nothing else should come before this line.\n\n"
                "Then provide a very brief justification in 1–3 sentences.\n"
                "Your ruling should come from a neutral perspective, but should allow for overgeneral and metaphorical meanings; established root words found previously are abstract in nature, so it follows these will be too. "
                "Assume the data provided is all the data available to work with, and than the metrics provided are also derived from acceptable, rigorous processes.\n"
                "Your role is to make a ruling based on the arguments provided.\n"
                "Be sure the verdict is the first line. This format is mandatory.\n\n+++\n\n"
            ),
            expected_output=(
                "A ruling that begins with either ✅ ACCEPTED or ❌ REJECTED, followed by a short rationale."
            ),
        ),
        "gloss": Task(
            description=f'The adjudicator has approved the root "{root}". Your responsibility is to respond ONLY with a dictionary-style definition for the root word. For your definition, focus on the semantics rather than how the word functions in the language. Your response must take the form of "[root] - [definition]". Again, provide ONLY the word and its definition. What follows is information you can use to base your definition on; again, your definition must be of the form "[root] - [definition]".\n\n',
            expected_output="A thorough and meaningful dictionary-style definition.",
        ),
    }

    # === Direct Tool Access with Streaming ===
    GRAY = "\033[90m"

    linguist_cb = (
        (lambda r, m: stream_callback("Linguist", m)) if stream_callback else None
    )
    skeptic_cb = (
        (lambda r, m: stream_callback("Skeptic", m)) if stream_callback else None
    )
    adjudicator_cb = (
        (lambda r, m: stream_callback("Adjudicator", m)) if stream_callback else None
    )
    # archivist_cb = (
    #     (lambda r, m: stream_callback("Archivist", m)) if stream_callback else None
    # )

    glossator_cb = (
        (lambda r, m: stream_callback("Glossator", m)) if stream_callback else None
    )

    # === LINGUISTS ===
    if stream_callback:
        stream_callback("Linguist", "**Linguist:**")

    # separator between words
    print("\n\n\n")
    print(f"==={(len('Now discussing the possible root word ') + len(f'<{root}>')) * '='}===")
    print(f"===Now discussing the possible root word '{root}'===")
    print(f"==={(len('Now discussing the possible root word ') + len(f'<{root}>')) * '='}===")
    
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
        f"\n\n>>>🥸\tLead Linguist's turn to propose a new root word...\nSynthesizing junior linguist's input into a meaningful argument:"
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
        f"\n\n>>>🤔\tSkeptic's turn to refute...\n{GRAY}Refutation prompt:",
        tasks["counter"].description,
        "\n",
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
        f"\n\n>>>🥸\tLead Linguist's turn to defend...\n{GRAY}Defense prompt:",
        tasks["defend"].description,
        "\n",
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
        f"\n\n>>>🤔\tSkeptic's turn to rebuttal...\n{GRAY}Final word:",
        tasks["rebuttal"].description,
        "\n",
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
        f"\n\n>>>👩‍⚖️\tAdjudicator's turn to pass their ruling...\n{GRAY}Ruling:",
        tasks["ruling"].description,
        "\n",
    )

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
    gloss = None
    if (
        adjudicator_ruling.strip().lower().startswith("✅ accepted")
        or adjudicator_ruling.strip().lower().startswith("accepted")
        or "✅" in adjudicator_ruling
    ):
        if stream_callback:
            stream_callback("Glossator", "**Glossator:**")

        print(
            f"\n\n>>>🧐\tGlossator's turn to provide a definition...\n{GRAY}Generating definition...\n"
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

    # === ARCHIVIST ===
    # NOTE: ARCHIVIST SHELVED FOR NOW
    # if stream_callback:
    #     stream_callback("Archivist", "**Archivist:**")

    # print(
    #     f"\n\n>>>📜\tArchivist's turn to record...\n{GRAY}Record>>\n",
    #     record.description,
    #     "\n",
    # )

    # archivist_summary = archivist_tool._run(
    #     prompt=record.description
    #     + f"\n\nLinguist: {linguist_proposal}\n\nSkeptic: {skeptic_response}\n\nLinguist: {linguist_defense}\n\nSkeptic: {skeptic_rebuttal}\n\nAdjudicator: {adjudicator_ruling}",
    #     stream_callback=None,
    #     print_chunks=True,
    #     role_name="📜\tArchivist",
    # )

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
        "Archivist": archivist_summary_formatted,
        "summary": tldr_summary,
        "raw_output": {
            "Linguist": linguist_proposal,
            "Skeptic": skeptic_response,
            "Defense": linguist_defense,
            "Rebuttal": skeptic_rebuttal,
            "Adjudicator": adjudicator_ruling,
            "Archivist": archivist_summary_formatted,
            "summary": tldr_summary,
        },
    }
