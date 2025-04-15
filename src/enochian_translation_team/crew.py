import os
from typing import Callable, Optional
from crewai import Agent, Task, Crew, CrewOutput
from enochian_translation_team.tools.query_model_tool import QueryModelTool

def run_crew(word: str, definition: str, stream_callback: Optional[Callable[[str, str], None]] = None) -> CrewOutput:
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # --- TOOL SETUP ---
    local_model_name = os.environ["MODEL_NAME"]

    # Tool for the Linguist: smart, eager, deluded
    linguist_tool = QueryModelTool(system_prompt="""
    You are a computational linguist specializing in ancient, obscure, and constructed languages.
    Your job is to analyze Enochian words, break them down into possible roots, affixes, or morphemes, and propose likely components of the word.
    Be confident and use linguistic terminology where appropriate.
    """)

    # Tool for the Orchestrator: neutral and judgy
    orchestrator_tool = QueryModelTool(system_prompt="""
    You are a senior linguistics project manager.
    Your job is to evaluate proposals for root words in Enochian from other linguists.
    Review their reasoning and make a decision about whether the proposal is likely valid.
    Be concise, analytical, and slightly skeptical.
    """)

    # --- AGENT SETUP ---
    linguist = Agent(
        role="Computational Linguist",
        goal="Propose plausible root structures for Enochian words.",
        backstory="An expert in theoretical and comparative linguistics with an obsession for lost languages.",
        tools=[linguist_tool],
        verbose=True,
        llm=local_model_name,
        callbacks = [stream_callback] if stream_callback else []
    )

    orchestrator = Agent(
        role="Orchestrator",
        goal="Evaluate linguistic analyses and accept or reject proposed root words.",
        backstory="An experienced linguist and editor who oversees complex language decoding projects.",
        tools=[orchestrator_tool],
        verbose=True,
        llm=local_model_name,
        callbacks = [stream_callback] if stream_callback else []
    )

    # --- TASK SETUP ---
    word = "AAI"
    definition = "amongst"

    linguist_task = Task(
        description=(
            f"Analyze the Enochian word '{word}', which means '{definition}'. "
            f"Break it down into likely root components or affixes. Provide reasoning for your breakdown."
        ),
        expected_output="A proposed list of components (e.g., root, affix) and reasoning for each.",
        agent=linguist
    )

    orchestrator_task = Task(
        description=(
            f"Review the linguistic analysis of the word '{word}'. "
            "Evaluate whether the proposed components make sense and decide whether to accept the proposed roots."
        ),
        expected_output="A clear decision: accept or reject the proposed root breakdown. Justify your response.",
        agent=orchestrator
    )
    
    # --- CREW SETUP ---
    crew = Crew(
        agents=[linguist, orchestrator],
        tasks=[linguist_task, orchestrator_task],
        verbose=True
    )

    if stream_callback:
        stream_callback("orchestrator", "_Initializing semantic tribunal..._")

    result = crew.kickoff()

    if stream_callback:
        stream_callback("orchestrator", f"**Final verdict:**\n\n{result}")

    return result
