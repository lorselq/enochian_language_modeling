# Concepts to Study for the Enochian Language Modeling Project

The Enochian LM pipeline references a mix of computational linguistics and machine-learning ideas. The topics below provide a study map for understanding the project end-to-end.

- **Morpheme**: The smallest meaningful unit in a language. Understanding morphemes is essential for decomposing Enochian tokens into semantic components.
- **Composite Token**: A surface token assembled from multiple morphemes; treated as the observable unit whose meaning the model reconstructs.
- **Gloss**: A short textual explanation of a token or morpheme. Gloss embeddings supply semantic supervision during factorization.
- **Leave-One-Out Attribution (LOO)**: A perturbation technique that measures how a model’s output changes when a single input feature—in this case, a morpheme vector—is removed.
- **Cosine Similarity**: A similarity metric between two vectors based on the cosine of the angle between them. Used repeatedly for attribution deltas, clustering, and alignment scoring.
- **Pointwise Mutual Information (PMI)**: A statistical measure that quantifies the strength of association between two events relative to their independent occurrence probabilities. Applied to morpheme co-occurrence counts.
- **Log-Likelihood Ratio (LLR)**: Also called the G-test; measures how strongly observed co-occurrence counts differ from what would be expected under independence.
- **Asymmetric Dependency**: A directional association score derived from attribution deltas to reveal which morpheme tends to depend on another.
- **Residual Vector**: The portion of a morpheme’s semantics not captured by direct compositional reconstruction—often modeled as the difference between a morpheme vector and its expected context.
- **Residual Clustering**: Grouping residual vectors to discover thematic clusters (e.g., flame-related morphemes). Implemented with cosine-based k-means.
- **K-Means Clustering**: An iterative algorithm that partitions vectors into K groups by minimizing within-cluster variance; here applied to normalized residual embeddings.
- **Ridge Regression**: A linear regression technique with L2 regularization that stabilizes weight estimation when features are correlated. Used to recover morpheme vectors from gloss embeddings and incidence matrices.
- **Design Matrix (Incidence Matrix)**: A sparse matrix encoding token-to-morpheme relationships (rows = tokens, columns = morphemes) used in factorization.
- **TF-IDF Vectorization**: Text embedding method weighting terms by frequency and inverse document frequency. Utilized for gloss-word and gloss-character embeddings.
- **Hashing Vectorizer**: A fixed-dimensional embedding technique that applies a hashing function to tokens; offers deterministic yet memory-efficient gloss features.
- **Mean Squared Error (MSE)**: A reconstruction-loss metric comparing predicted vs. gold gloss vectors. Alternative cosine-based diagnostics are also supported.
- **Cosine Reconstruction Error**: An evaluation metric defined as one minus the cosine similarity between predicted and target embeddings.
- **Composite Reconstruction Table**: Database table storing gloss vectors, predicted vectors, morph usage, and reconstruction error per token—central for evaluating coverage improvements.
- **Attribution Marginals Table**: Stores aggregated leave-one-out deltas between morpheme pairs; a foundation for collocation statistics.
- **Collocation Statistics Table**: Contains morpheme pair counts with PMI, LLR, and asymmetric dependency scores for co-occurrence analysis.
- **Residual Cluster Membership Table**: Records which morphemes align to each residual cluster and their similarity to cluster centroids.
- **Morpheme Semantic Vectors Table**: Persists the learned semantic vectors (factorization output) and their norms for reuse across the pipeline.
- **Baseline Residuals File**: A JSONL reference capturing earlier coverage/residual metrics used to measure improvements in the pipeline report.
- **Pipeline Report**: The final consolidated summary combining attribution, collocation, clustering, and factorization diagnostics into an auditable artifact.
