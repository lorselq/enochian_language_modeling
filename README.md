# Enochian Language Modeling
Hacking the Enochian language and Liber Loagaeth—because why not look at glossolalia with a fresh set of AIs? 🥸

## notes

`dictionary.json` is, at present, not final—I am still refining the dictionary. A lot of this is under development—very early stages.

You also will want to run `poetry run python -m src/enochian_translation_team/tools/train_fasttext_model.py` to get the models going before running `main.py`. The models are absolutely necessary for this to work.

## Applied computational linguistics (for... pre-phase one)

1. Throw FastText at it the `dictionary.json`. See the above mentioned `train_fasttext_model.py` for some sense as to what I'm going for.
2. Once the `dictionary.json` is complete (I still have about eight to eleven letters to get through before it's done at this point, 4/14/2025), I can craft something that extracts possible roots based on semantic similarities.

## Agentic team plan (for actually phase one)

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
    