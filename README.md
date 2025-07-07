# Enochian Language Modeling: Experimental Computational Philology

## Overview
This project explores computational approaches to deciphering *Liber Loagaeth*, a 16th-century manuscript associated with John Dee and Edward Kelley. Rather than pursuing esoteric interpretations of the undeciphered corpus, this research treats it as a **synthetic, low-resource linguistic corpus**—an experimental setting for semantic structure discovery, root-word extraction, and agent-driven linguistic modeling.

The precise linguistic status of *Liber Loagaeth* remains uncertain. Its language differs markedly from the better-known Enochian Keys, and scholarly consensus generally regards the text as nonsensical or glossolalic—possibly a deliberate linguistic artifice. This project does not seek to contest this interpretation. Instead, *Liber Loagaeth* provides a challenging, bounded dataset for exploring how AI systems can infer meaningful semantic relationships and construct speculative lexicons from irregular, anomalous text data. Such experiments offer methodological insights into computational semantic analysis, low-resource language modeling, and broader digital humanities applications.

## Project Goals
- **Semantic Structure Discovery:** Identify plausible root morphemes and semantic clusters within the linguistic corpus.
- **Experimental Lexicon Building:** Construct an AI-assisted speculative lexicon through pattern recognition, semantic proximity, and agentic validation.
- **Agent-Based Linguistic Modeling:** Simulate multi-agent debates (and compare with single-agent "Solo Mode") to propose, validate, and refine candidate root structures.
- **Low-Resource Language Methods:** Develop and evaluate computational methods suitable for texts with limited ground-truth data, irregular morphological patterns, and historical ambiguity.

## Core Components

### Embedding and Semantic Modeling (Foundational Stage)
- **FastText Embeddings:** Generated from a limited glossary (~800–1300 entries) based on *Angelical Language Vol. II* by Aaron Leitch, an adjacent corpus with rough translations.
- **Semantic Proximity Analysis:** Calculation of semantic similarities via dynamic clustering methods—including k-nearest neighbors (kNN), agglomerative clustering, fuzzy clustering, and others—selected through automatic parameter tuning.
- **Data Preparation for Agent-Driven Extraction:** Clustering of potential morphemes to serve as input for agentic debates and validations.

### Agent-Based Root Extraction (Root Inference and Validation)
A simulated AI team operates in two distinct modes:

**Debate Mode:**  
- **Linguist Agent:** Proposes candidate roots based on embeddings, heuristics, and semantic proximity.
- **Skeptic Agent:** Critically evaluates proposed roots, presenting counterarguments and alternate hypotheses.
- **Adjudicator Agent:** Finalizes judgments based on linguistic plausibility, semantic consistency, and evidence provided.
- **Glossator Agent:** Synthesizes and records official root definitions approved by the Adjudicator.

**Solo Mode:**  
- A single-agent configuration where a designated linguistic expert reviews and directly adjudicates root candidates without debate, intended as a methodological contrast to the Debate Mode.

Comparative studies between Solo and Debate modes are anticipated as future research avenues, exploring efficiency, quality of definitions, and methodological robustness.

## Methodology
- **Bottom-Up Semantic Modeling:** Inferring semantic structures from recurring substrings (n-grams) and semantic proximity metrics without presupposing fixed linguistic rules.
- **Pattern Recognition and Morphological Analysis:** Identifying stems, derivatives, and root candidates by analyzing morphological and semantic regularities across clusters.
- **Semantic Clustering (Dynamic Tuning):** Dynamically adjusting and selecting clustering methods (kNN, agglomerative, fuzzy) to propose robust candidate root groups.
- **SQLite Database for Data Recording:** Systematic logging of agent deliberations, accepted/rejected definitions, and clustering metadata for reproducibility and further analysis.

## Current Status
- **Dictionary Foundation:** Established initial lexicon from Aaron Leitch's *Angelical Language Vol. II*.
- **Ngram Generation:** Robust n-gram indexing including morphological variants derived from John Dee's irregular spellings.
- **Embedding Infrastructure:** Functional FastText embedding pipelines producing promising initial semantic analyses.
- **Dynamic Clustering:** Implementation and testing of multiple clustering methods with automatic parameter tuning, selecting optimal methods based on internal cluster consistency metrics.
- **Agent Modes:** Both Debate and Solo modes fully implemented; comparative analyses planned as future research.
- **CLI Enhancements:** Improved terminal interface for smoother operation, enhancing readability and significantly reducing unnecessary remote API calls to external LLM services.

## Future Work and Phased Development

### Near-Term Development Goals
- Comprehensive execution until all viable n-grams are processed and defined.
- Enhancement of data collection routines, including further improvements to the SQLite-based definition logging system and supporting metadata capture.

### Phase Two: Lexicon Refinement and Semantic Reconstruction
With a complete speculative root-word lexicon in place, subsequent efforts will focus on refining and standardizing dictionary entries. Human-in-the-loop semantic validation and curation will ensure consistent lexical quality and facilitate practical linguistic reconstruction tasks.

A central methodological validation will involve a "blind retranslation" exercise, reinterpreting the previously translated *Enochian Keys* exclusively using newly derived root meanings, without recourse to historical translations. This test case aims to empirically validate the semantic viability and internal consistency of the speculative lexicon.

#### Anticipated Side-Projects and Additional Experiments
- **Solo vs. Debate Comparative Analysis:** Investigate whether multi-agent debates yield superior semantic quality compared to single-agent evaluations.
- **Root Structure and Syllabic Patterns:** Explore relationships between extracted roots and their phonological and morphological structures, offering potential insights into the linguistic design principles underlying the corpus.

### Phase Three: Full Corpus Application
Following successful methodological validation, the agentic semantic extraction system will be extended to the entire *Liber Loagaeth* corpus. This larger-scale analysis will assess the scalability and robustness of the developed computational methods, potentially uncovering new semantic regularities, lexicon structures, and insights into the underlying linguistic—or deliberately artificial—nature of the corpus.

## Broader Impact
- Demonstrates computational methods for low-resource linguistic corpora, contributing methodological frameworks applicable to other historically ambiguous or synthetic languages.
- Provides interdisciplinary insights connecting computational linguistics, semantic modeling, and digital humanities, serving as an illustrative case study of AI-assisted linguistic reconstruction in low-resource contexts.

## Note on Setup
This project is currently configured for a highly customized local development environment (involving LM Studio, WSL2, Python poetry-based dependencies, and various advanced tooling). Due to this complexity, detailed setup instructions are currently omitted but may be provided in simplified form in future updates.

---

*This project remains experimental and actively evolving. All analyses, definitions, and methodological approaches are speculative, aimed at computational and theoretical exploration rather than definitive linguistic reconstruction.*
