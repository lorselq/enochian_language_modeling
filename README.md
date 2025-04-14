# Enochian Translation Team
Hacking the Enochian language and Liber Loagaeth

## notes

`dictionary.json` is, at present, not final—I am still refining the dictionary. A lot of this is under development—very early stages.

## Agentic team plan (for phase one)

1. Orchestrator Agent
    - Role: Overseer and final arbiter
    - Input: word + linguist + skeptic reports
    - Output: accepts/rejects root proposal and adds them to the roots.json via tool
2. Linguist Agent
    - Role: Root extractor
    - Tools: FastText, heuristics of various kinds, maybe LLM help
    - Output: proposed root word, reasoning, likelihood score
3. Skeptic Agent
    - Role: "Prove it" gremlin
    - Tools: access to same data, challenges proposed roots with counterexamples, alternate etymologies, etc.
    - Output: objections, counterpoints, uncertainty measures
    