import os
from typing import Callable, Optional, Union
from crewai import Agent, Task, Crew, CrewOutput
from enochian_translation_team.tools.query_model_tool import QueryModelTool

def run_crew(word: str, definition: str, stream_callback: Optional[Callable[[str, str], None]] = None) -> dict[str, Union[str, CrewOutput]]:
    from dotenv import load_dotenv
    load_dotenv(override=True)

    local_model_name = os.environ["MODEL_NAME"]

    # --- TOOL SETUP ---
    # Tool for the Linguist: smart, eager, deluded
    linguist_tool = QueryModelTool(system_prompt="""
    You are a computational linguist specializing in ancient, obscure, and constructed languages.
    Your job is to analyze Enochian words, break them down into possible roots, affixes, or morphemes, and propose likely components of the word.
    Be confident and use linguistic terminology where appropriate.
    """)

    # Tool for the adjudicator: neutral and judgy
    adjudicator_tool = QueryModelTool(system_prompt="""
    You are a senior linguistics project manager.
    Your job is to evaluate proposals for root words in Enochian from other linguists.
    Review their reasoning and make a decision about whether the proposal is likely valid.
    Be concise, analytical, and slightly skeptical.
    """)
    
    # --- PROMPT SETUP ---
    linguist_prompt = f"""
    Analyze the Enochian word '{word}', which means '{definition}'.
    Break it down into likely root components or affixes.
    Provide reasoning for your breakdown.
    """
    
    adjudicator_prompt = f"""
    Review the linguistic analysis of the word '{word}'.
    Evaluate whether the proposed components make sense and decide whether to accept the proposed roots.
    """

    # --- AGENT SETUP ---
    linguist = Agent(
        role="Linguist",
        goal="Propose plausible root structures for Enochian words.",
        backstory="An expert in theoretical and comparative linguistics with an obsession for lost languages.",
        tools=[linguist_tool],
        verbose=True,
        llm=local_model_name,
        callbacks = [stream_callback] if stream_callback else []
    )

    adjudicator = Agent(
        role="Adjudicator",
        goal="Evaluate linguistic analyses and accept or reject proposed root words.",
        backstory="An experienced linguist and editor who oversees complex language decoding projects.",
        tools=[adjudicator_tool],
        verbose=True,
        llm=local_model_name,
        callbacks = [stream_callback] if stream_callback else []
    )

    # === TASKS ===
    linguist_task = Task(
        description=(
            f"Analyze the Enochian word '{word}', which means '{definition}'. "
            "Break it down into likely root components or affixes. Provide reasoning for your breakdown."
        ),
        expected_output="A proposed list of components (e.g., root, affix) and reasoning for each.",
        agent=linguist
    )

    adjudicator_task = Task(
        description=(
            f"Review the linguistic analysis of the word '{word}'. "
            "Evaluate whether the proposed components make sense and decide whether to accept the proposed roots."
        ),
        expected_output="A clear decision: accept or reject the proposed root breakdown. Justify your response.",
        agent=adjudicator
    )

    # === CREW ===
    crew = Crew(
        agents=[linguist, adjudicator],
        tasks=[linguist_task, adjudicator_task],
        verbose=True
    )

    if stream_callback:
        stream_callback("Linguist", "**Linguist:**")

    if stream_callback:
        linguist_callback = (lambda _, msg: stream_callback("Linguist", msg)) if stream_callback else None
    else:
        linguist_callback = None

    linguist_response = linguist_tool.run(
        prompt=linguist_prompt,
        stream_callback=linguist_callback
    )

    if stream_callback:
        stream_callback("Adjudicator", "**Adjudicator:**")

    if stream_callback:
        adjudicator_callback = (lambda _, msg: stream_callback("Adjudicator", msg)) if stream_callback else None
    else:
        adjudicator_callback = None

    adjudicator_response = adjudicator_tool.run(
        prompt=adjudicator_prompt,
        stream_callback=adjudicator_callback
    )

    return {
        "Linguist": linguist_response,
        "Adjudicator": adjudicator_response
    }
