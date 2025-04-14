# Enochian Translation Team
Hacking the Enochian language and Liber Loagaeth

## notes

`dictionary.json` is, at present, not final—I am still refining the dictionary. A lot of this is under development—very early stages.

You also will want to run `poetry run python -m src/enochian_translation_team/tools/train_fasttext_model.py` to get the models going before running `main.py`. The models are absolutely necessary for this to work.

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
    