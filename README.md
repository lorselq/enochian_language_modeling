# Enochian Language Modeling: Experimental Computational Philology

## Overview
This project explores computational approaches to deciphering *Liber Loagaeth*, a 16th-century manuscript associated with John Dee and Edward Kelley. Rather than pursuing esoteric interpretations of the undeciphered corpus, this research treats it as a **synthetic, low-resource corpus**—an experimental ground for semantic structure discovery, root-word extraction, and agent-driven linguistic modeling.

It is unclear exactly what language *Liber Loagaeth* is, as the purported language of the text does not wholly resemble that which appears in the Enochian Keys. The majority of scholars believe it to be nonsense—likely an example of glossolalia or even simply an alphabet soup of the English language—and this project does not seek to prove them wrong. Rather, *Liber Loagaeth* serves as a white whale for the project and the inspiration towards a broader, more realistic goal: to investigate how AI systems can infer semantic relationships and construct speculative lexicons from anomalous, non-standardized text data, offering insights into language emergence, semantic clustering, and machine learning applications in historical and undeciphered language systems.

## Project Goals
- **Semantic Structure Discovery**: Identify plausible root morphemes and semantic clusters within the linguistic corpus.
- **Experimental Lexicon Building**: Construct a bottom-up, AI-assisted speculative dictionary based on pattern recognition and semantic proximity.
- **Agent-Based Linguistic Modeling**: Simulate multi-agent debates where linguist and skeptic agents propose, refine, and critique possible root structures.
- **Low-Resource Language Methods**: Explore AI methods applicable to texts with limited ground-truth data, irregular structure, and historical ambiguity.

## Core Components

### Embedding and Semantic Modeling (Foundational Stage)
- Development of FastText embeddings trained on a limited glossary (~800–1300 entries) compiled from an adjacent corpus that comes with rough translations for each word.
- Semantic proximity analysis to generate hypotheses about potential morphemes.
- Data preparation for agent-driven root extraction based on lexical clustering methodologies.

### Agent-Based Root Extraction (Root Inference and Validation)
A simulated AI team, consisting of multiple agent roles:
- **Linguist Agent**: Proposes candidate roots using FastText embeddings, heuristic analysis, and LLM-assisted reasoning.
- **Skeptic Agent**: Challenges proposed roots by presenting counterexamples, alternate derivations, and uncertainty evaluations.
- **Adjudicator Agent**: Reviews debates and finalizes judgments on proposed root candidates.
- **Glossator Agent**: Synthesizes and records an official root definition if the Adjudicator approves the proposal.

### Methodology
- **Bottom-Up Semantic Modeling**: Inferring semantic structures from recurring substrings (n-grams) and contextual proximity without presupposing a fixed linguistic framework.
- **Pattern Recognition and Subtractive Morphology**: Identifying potential stems and derivatives by analyzing morphological patterns across related word forms.
- **Semantic Clustering**: Grouping lexical items based on model-inferred proximity to propose and validate candidate roots.

## Current Status
- **Starting Dictionary Construction**: Completed and integrated—words drawn from *Angelical Language Vol. II* by Aaron Leitch as a starting point.
- **Ngram Generation**: Configured such that it generates ngrams based on words as they appear in the corpus and possible variants created from letter substitution rules derived Dee's irregular spellings.
- **Embedding Infrastructure**: Initial FastText models functional and generating promising proximity data.
- **Semantic Clustering**: A variety of clustering methods and parameters are trialed and a final clustering selected based on the degree to which the clusters resemble an effective clustering.
- **Agent Design**: Agents are not orchestrated by a supervisory agent presently; presently, prompts and their context follow a fixed pattern of stages, making the process more like prompt chaining.
- **CLI Presentation**: UI implementation is console-based and tolerable.

## Future Work and Phased Development

### Near-Term Development Goals
- Creating a process to record new definitions to a sqlite database with accompanying relevant data.
- Running the program until all viable ngrams processed.

### Phase Two: Lexicon Refinement and Semantic Reconstruction
Upon completing the speculative root-word lexicon, the next stage will involve standardizing and refining dictionary entries. Due to inconsistencies in AI-generated definitions (particularly from the Glossator agent), additional data wrangling and human-in-the-loop curation will be necessary to ensure semantic coherence across the lexicon.

With a stabilized root-word dictionary, the project will undertake a reconstruction exercise: attempting to reinterpret the *Angelic Keys*—a previously translated portion of the Enochian corpus—exclusively using the newly derived root meanings, without reference to historical translations. This "blind retranslation" will serve as a proof of concept for the methodology's validity and provide data for further semantic calibration.

### Phase Three: Full Corpus Application
Following validation through the *Angelic Keys*, the final phase envisions applying the agentic semantic extraction system to the broader corpus of *Liber Loagaeth*. This full application would test the scalability of the methods on a much larger, less structured text body, potentially revealing new semantic patterns, emergent lexicons, and deeper insights into the structure—or deliberate artifice—of the Enochian system.


## Broader Impact
- Provides a speculative case study in applying AI methodologies to anomalous, undeciphered linguistic corpora.
- Demonstrates techniques potentially transferable to the analysis of low-resource, synthetic, or historically ambiguous languages.
- Contributes to interdisciplinary dialogues at the intersection of computational linguistics, semantic inference, and digital humanities.

## Note on Setup
This project is currently configured for a highly customized local development environment involving LM Studio integration, WSL2, and Python poetry-based dependency management. Due to complexity, detailed setup instructions are not provided here. Future updates may include simplified deployment options.

---

*This project is experimental and under active development. All analyses and conclusions are speculative and intended for methodological exploration rather than definitive linguistic reconstruction.*
