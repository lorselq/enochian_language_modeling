# Enochian Language Modeling: Experimental Computational Philology

## Overview
This project explores computational approaches to modeling *Liber Loagaeth*, a 16th-century manuscript associated with John Dee and Edward Kelley. Rather than pursuing esoteric interpretations of the undeciphered corpus, this research treats it as a **synthetic, low-resource corpus**—an experimental ground for semantic structure discovery, root-word extraction, and agent-driven linguistic modeling.

The broader goal is to investigate how AI systems can infer semantic relationships and construct speculative lexicons from anomalous, non-standardized text data, offering insights into language emergence, semantic clustering, and machine learning applications in historical and undeciphered language systems.

## Project Goals
- **Semantic Structure Discovery**: Identify plausible root morphemes and semantic clusters within the linguistic corpus.
- **Experimental Lexicon Building**: Construct a bottom-up, AI-assisted speculative dictionary based on pattern recognition and semantic proximity.
- **Agent-Based Linguistic Modeling**: Simulate multi-agent debates where linguist and skeptic agents propose, refine, and critique possible root structures.
- **Low-Resource Language Methods**: Explore AI methods applicable to texts with limited ground-truth data, irregular structure, and historical ambiguity.

## Core Components

### Embedding and Semantic Modeling (Foundational Stage)
- Development of FastText embeddings trained on a limited glossary (~800–1300 entries) compiled from an adjacent, partially deciphered corpus.
- Semantic proximity analysis to generate hypotheses about potential morphemes.
- Data preparation for agent-driven root extraction based on lexical clustering methodologies.

### Agent-Based Root Extraction (Root Inference and Validation)
A simulated AI team, consisting of multiple agent roles:
- **Adjudicator Agent**: Reviews debates and finalizes judgments on proposed root candidates.
- **Linguist Agent**: Proposes candidate roots using FastText embeddings, heuristic analysis, and LLM-assisted reasoning.
- **Skeptic Agent**: Challenges proposed roots by presenting counterexamples, alternate derivations, and uncertainty evaluations.
- **Glossator Agent**: Synthesizes and records an official root definition if the Adjudicator approves the proposal.

### Methodology
- **Bottom-Up Semantic Modeling**: Inferring semantic structures from recurring substrings (n-grams) and contextual proximity without presupposing a fixed linguistic framework.
- **Pattern Recognition and Subtractive Morphology**: Identifying potential stems and derivatives by analyzing morphological patterns across related word forms.
- **Semantic Clustering**: Grouping lexical items based on model-inferred proximity to propose and validate candidate roots.

## Current Status
- **Dictionary Construction**: Recently completed, but not yet implemented into the experiments.
- **Embedding Infrastructure**: Initial FastText models functional and generating promising proximity data.
- **Agent Design**: Logical structure of the multi-agent system outlined; implementation will proceed following dictionary stabilization.

## Future Work and Phased Development

### Near-Term Development Goals
- Finalizing and validating the first-pass dictionary.
- Continuing agentic evaluations to expand and refine the speculative root lexicon.
- Examining current approaches to root-word definition and expanding the system to support multiple alternative meanings for individual n-grams, reflecting contextual variation.
- Preparing sample retranslation exercises using the stabilized dictionary for validation testing.

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
