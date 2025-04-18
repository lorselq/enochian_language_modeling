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
    def_list = [(c.get("word", ""), c.get("definition", "")) for c in candidates if c]
    joined_defs = [
        f"{word.strip()} — {definition.strip()}"
        for word, definition in def_list
        if word and definition
    ]
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
    linguist_tool = QueryModelTool(
        system_prompt="You are a bold and insightful computational linguist specializing in the Enochian language—a constructed system with irregular morphology and uncertain origins. Your job is to analyze a proposed root by examining semantic and morphological overlap across multiple words. Identify patterns in prefixes, suffixes, or repeated substrings that suggest shared structure. Support your hypothesis by referencing similarities in definitions, glosses, or contextual usage from citations. Do not use natural language etymologies (e.g., English or Latin roots). Justify relationships based solely on internal evidence across Enochian terms. Your tone should be confident and scholarly. Provide specific examples and explain why the connection is more than coincidental. Absolutely be thorough in your justifications."
    )
    linguist = Agent(
        role="Linguist",
        goal="Analyze the semantic relationships between these Enochian words and their definitions. Identify shared morphemes or root candidates based on: shared character substrings (especially prefixes/suffixes); overlapping definitions or conceptual meanings; patterns in usage across citations. Justify why these words might be related, but never use English, Greek, Latin, or Hebrew etymology to substantiate a proposed Enochian root word. Never use hypothetical Enochian words you've made up to justify your arguments; use only the ones provided in the prompts. Reference actual text segments or gloss overlaps.",
        backstory="A creative, inventive, and excited linguist with deep pattern recognition skills, always hopeful to discover something new and great.",
        tools=[linguist_tool],
        verbose=True,
        callbacks=[stream_callback] if stream_callback else [],
    )

    skeptic_tool = QueryModelTool(
        system_prompt="You are a skeptical linguist reviewing a proposed root analysis in the Enochian language. Your goal is to uncover weak reasoning, accidental pattern-matching, or semantic mismatches. Examine whether the proposed words actually share meaningful definitions. Challenge vague or speculative claims. Look for missing evidence or inconsistent logic. If the root hypothesis is flawed, explain why. If you believe a stronger candidate exists—which is something you would like—make a concise counterproposal. You are sharp, analytical, and unafraid to criticize overreach."
    )
    skeptic = Agent(
        role="Skeptic",
        goal="You are a skeptical linguist. Critically examine the Linguist's proposed root and semantic analysis. Look for: weak reasoning or overgeneralization; words grouped together without strong definition overlap; n-grams that appear coincidentally (e.g., short, common patterns). Propose stronger alternatives if you have them, or explain why a proposal lacks rigor.",
        backstory="A cynical critic with linguistic expertise and a grudge against bad etymology.",
        tools=[skeptic_tool],
        verbose=True,
        callbacks=[stream_callback] if stream_callback else [],
    )

    adjudicator_tool = QueryModelTool(
        system_prompt="Review the arguments from both the Linguist and the Skeptic. Make a final determination: should this root be accepted as a meaningful candidate? Be concise and definitive. A short rationale is fine, but it must address key reasoning on both sides."
    )
    adjudicator = Agent(
        role="Adjudicator",
        goal="Make a judgment call based on both sides.",
        backstory="A supervising linguist who wants solid reasoning before accepting newly proposed root words as part of the lexicon.",
        tools=[adjudicator_tool],
        verbose=True,
        callbacks=[stream_callback] if stream_callback else [],
    )

    archivist_tool = QueryModelTool(
        system_prompt="Summarize the root debate. Include the root, who said what, and the final verdict."
    )
    archivist = Agent(
        role="Archivist",
        goal="Summarize the outcome of the debate for recordkeeping.",
        backstory="A meticulous historian recording linguistic discoveries.",
        tools=[archivist_tool],
        verbose=True,
        callbacks=[stream_callback] if stream_callback else [],
    )

    # === TASKS ===
    propose = Task(
        description=f"""{extra_prompt}Analyze the root candidate '{root}' using the following semantic stats:\n\n{stats_summary}\n\nBreak down shared semantics or patterns. Propose a coherent explanation of the root. Do not use English, Greek, Hebrew, or Latin etymological justifications; the proposal must come from the candidate root word's letter composition and possible meanings based on its and related word's definitions.\n\nDefinitions and citations contained in [] to consider (they are pipe-delimited and strongly ordered from most to least relevant):\n{root_def_summary}
        """,
        expected_output="A strong case for the root, citing semantic and morphological evidence.",
        agent=linguist,
    )

    counter = Task(
        description="Respond to the Linguist's analysis. Challenge weak points, semantic gaps, or coincidences."
        + skeptic_hint,
        expected_output="A thorough and convincing rebuttal to the Linguist's proposal to add the new root word to the records.",
        agent=skeptic,
        context=[propose],
    )

    defense = Task(
        description="Do your absolute best to defend the original linguistic hypothesis. Respond to the Skeptic's objections directly. Try to convince the Skeptic to see things your way.",
        expected_output="A solid defense and doubling down on the original linguistic hypothesis; a direct response to the Skeptic's criticisms; an attempt to sway the Skeptic to accept the new root word.",
        agent=linguist,
        context=[propose, counter],
    )

    counter2 = Task(
        description="Issue a final rebuttal if the defense failed to address key concerns.",
        expected_output="Either a final rebuttal that reiterates criticisms that were not addressed by the Linguist's defense or an acknowledgement that the Linguist may be onto something.",
        agent=skeptic,
        context=[propose, counter, defense],
    )

    ruling = Task(
        description="Make a ruling. Accept or reject the root proposal. Justify briefly.",
        expected_output="A ruling as to whether or not to accept or reject the proposed root word. Justify the decision briefly.",
        agent=adjudicator,
        context=[propose, counter, defense, counter2],
    )

    record = Task(
        description="Summarize the debate and the final outcome in 3-4 sentences. Be sure to cover the strongest arguments for an against.",
        expected_output="A brief summarization of the debate and its outcome.",
        agent=archivist,
        context=[propose, counter, defense, counter2, ruling],
    )

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
    archivist_cb = (
        (lambda r, m: stream_callback("Archivist", m)) if stream_callback else None
    )

    # === LINGUIST 1 ===
    if stream_callback:
        stream_callback("Linguist", "**Linguist:**")

    print(
        f"\n\n>>>🥸\tLinguist's turn to propose a new root word...\n{GRAY}Proposal prompt:\n",
        propose.description,
        "\n",
    )

    linguist_response = linguist_tool._run(
        prompt=propose.description,
        stream_callback=linguist_cb,
        print_chunks=True,
        role_name="🥸\tLinguist",
    )

    # === SKEPTIC 1 ===
    if stream_callback:
        stream_callback("Skeptic", "**Skeptic:**")

    print(
        f"\n\n>>>🤔\tSkeptic's turn to refute...\n{GRAY}Refutation prompt:\n",
        counter.description,
        "\n",
    )

    skeptic_response = skeptic_tool._run(
        prompt=counter.description + f"\n\nLinguist said: {linguist_response}",
        stream_callback=skeptic_cb,
        print_chunks=True,
        role_name="🤔\tSkeptic",
    )

    # === LINGUIST 2 ===
    if stream_callback:
        stream_callback("Linguist", "**Linguist (Defense):**")

    print(
        f"\n\n>>>🥸\tLinguist's turn to defend...\n{GRAY}Defense prompt:\n",
        defense.description,
        "\n",
    )

    linguist_defense = linguist_tool._run(
        prompt=defense.description + f"\n\nSkeptic said: {skeptic_response}",
        stream_callback=linguist_cb,
        print_chunks=True,
        role_name="🥸\tLinguist",
    )

    # === SKEPTIC 2 ===
    if stream_callback:
        stream_callback("Skeptic", "**Skeptic (Rebuttal):**")

    print(
        f"\n\n>>>🤔\tSkeptic's turn to rebuttal...\n{GRAY}Final word:\n",
        counter2.description,
        "\n",
    )
    skeptic_rebuttal = skeptic_tool._run(
        prompt=counter2.description + f"\n\nLinguist said: {linguist_defense}",
        stream_callback=skeptic_cb,
        print_chunks=True,
        role_name="🤔\tSkeptic",
    )

    # === ADJUDICATOR ===

    if stream_callback:
        stream_callback("Adjudicator", "**Adjudicator:**")

    print(
        f"\n\n>>>👩‍⚖️\tAdjudicator's turn to pass their ruling...\n{GRAY}Ruling:\n",
        ruling.description,
        "\n",
    )

    adjudicator_response = adjudicator_tool._run(
        prompt=ruling.description
        + f"\n\nLinguist: {linguist_response}\n\nSkeptic: {skeptic_response}\n\nDefense: {linguist_defense}\n\nFinal Skeptic: {skeptic_rebuttal}",
        stream_callback=adjudicator_cb,
        print_chunks=True,
        role_name="👩‍⚖️\tAdjudicator",
    )

    # === ARCHIVIST ===
    if stream_callback:
        stream_callback("Archivist", "**Archivist:**")

    print(
        f"\n\n>>>📜\tArchivist's turn to record...\n{GRAY}Record>>\n",
        record.description,
        "\n",
    )

    archivist_summary = archivist_tool._run(
        prompt=record.description
        + f"\n\nLinguist: {linguist_response}\n\nSkeptic: {skeptic_response}\n\nLinguist: {linguist_defense}\n\nSkeptic: {skeptic_rebuttal}\n\nAdjudicator: {adjudicator_response}",
        stream_callback=archivist_cb,
        print_chunks=True,
        role_name="📜\tArchivist",
    )

    archivist_summary_formatted = (
        "=== LINGUIST ===\n"
        + linguist_response.strip()
        + "\n\n=== SKEPTIC ===\n"
        + skeptic_response.strip()
        + "\n\n=== DEFENSE ===\n"
        + linguist_defense.strip()
        + "\n\n=== REBUTTAL ===\n"
        + skeptic_rebuttal.strip()
        + "\n\n=== ADJUDICATOR ===\n"
        + adjudicator_response.strip()
        + "\n\n=== ARCHIVIST ===\n"
        + archivist_summary.strip()
    )

    tldr_tool = QueryModelTool(
        system_prompt="You are a helpful summarizer. You don't repeat anything anyone says and you use your own words."
    )

    tldr_summary = tldr_tool._run(
        prompt="Summarize the following root word debate in 1-2 sentences; your focus should be summarizing the strongest, key arguments and takeaways, and briefly indicating whether or not the root word proposal was accepted:\n\n"
        + archivist_summary_formatted,
        stream_callback=None,
    )

    return {
        "Linguist": linguist_response,
        "Skeptic": skeptic_response,
        "Defense": linguist_defense,
        "Rebuttal": skeptic_rebuttal,
        "Adjudicator": adjudicator_response,
        "Archivist": archivist_summary_formatted,
        "summary": tldr_summary,
        "raw_output": {
            "Linguist": linguist_response,
            "Skeptic": skeptic_response,
            "Defense": linguist_defense,
            "Rebuttal": skeptic_rebuttal,
            "Adjudicator": adjudicator_response,
            "Archivist": archivist_summary_formatted,
            "summary": tldr_summary,
        },
    }
