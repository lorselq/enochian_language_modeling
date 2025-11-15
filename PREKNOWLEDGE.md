# Core math for representations

* **Singular Value Decomposition (SVD)** (important): Factor matrices, read rank and variance structure.

  * **Linear algebra foundations**: vectors, matrices, orthogonality, norms.
  * **Eigen-stuff**: eigenvalues, eigenvectors, spectral theorem, why low-rank helps.

* **Ridge Regression**: L2-regularized linear regression for stable coefficients.

  * **Least squares**: normal equations, pseudoinverse.
  * **Regularization tradeoffs**: bias–variance, shrinkage vs overfit.

* **Sparse Coding and L1 Regularization** (important): Learn concise representations, reduce leakage.

  * **Convex optimization**: objective functions, constraints, KKT intuition.
  * **L1 vs L2**: geometry of penalties, when sparsity helps interpretability.

* **Group Lasso and Structured Sparsity** (related): Turn feature groups on or off.

  * **Block norms**: L2,1 and group penalties.
  * **Feature grouping**: how to define morph families or templates.

* **Nonnegative Matrix Factorization (NMF)** (related): Parts-based factors for interpretability.

  * **Matrix factorizations**: Frobenius norm, multiplicative updates.
  * **Nonnegativity constraints**: why parts emerge, scaling pitfalls.

* **Tensor Factorization (CP, Tucker)** (related): Capture token×morpheme×context interactions.

  * **Tensors 101**: modes, unfoldings, Kruskal form.
  * **Optimization for tensors**: alternating least squares, identifiability limits.

* **Orthogonal Procrustes Alignment** (related): Align embedding spaces for comparability.

  * **Orthogonal matrices**: rotations, reflections, invariants.
  * **Least squares**: solving constrained argmin with SVD.

* **Identifiability of Embeddings** (important): When vectors are unique up to rotation or scale.

  * **Symmetries/invariances**: what your loss can and can’t pin down.
  * **Alignment metrics**: CKA, Procrustes error as stability checks.

* **K-Means Clustering**: Partition normalized residuals by cosine geometry.

  * **Distance metrics**: cosine vs Euclidean on unit sphere.
  * **Initialization/stability**: k-means++, multiple restarts.

# Probability, statistics, and information theory

* **Information Theory Basics** (important): Entropy, cross-entropy, KL.

  * **Probability rules**: conditional independence, Bayes’ rule.
  * **Logarithms and bases**: bits vs nats, perplexity meaning.

* **Pointwise Mutual Information (PMI)**: Association strength for morpheme pairs.

  * **Joint vs marginal**: estimating counts, smoothing.
  * **Log transforms**: interpretability, zero/rare-event handling.

* **Log-Likelihood Ratio (LLR)**: Deviation from independence for sparse counts.

  * **Likelihoods**: multinomial vs Poisson approximations.
  * **Asymptotics**: chi-square connections, when G-test is robust.

* **Bootstrap, Jackknife, Permutation Tests** (important): Uncertainty and significance.

  * **Sampling with/without replacement**: variance estimation.
  * **Null modeling**: label shuffle to test spurious correlations.

* **Multiple Testing and FDR Control** (important): Honest discovery among many pairs.

  * **p-values vs q-values**: what gets controlled.
  * **BH/BY procedures**: dependence assumptions, pitfalls.

* **Calibration and Uncertainty in Scores** (related): Make confidence honest.

  * **Reliability curves, ECE**: evaluating calibration.
  * **Temperature scaling**: simple post-hoc fix.

* **Random Matrix Theory and Concentration** (related): High-d noise geometry.

  * **Concentration inequalities**: Hoeffding, Bernstein.
  * **Marchenko–Pastur**: separating signal from noise spectrum.

* **Minimum Description Length (MDL)** (related): Prefer compressive hypotheses.

  * **Coding length**: model cost plus data cost.
  * **Overfitting lens**: when extra parameters fail to compress.

# NLP foundations and representation learning

* **Cosine Similarity**: Core similarity for vectors and directions.

  * **Vector norms**: L2, unit normalization effects.
  * **Angles vs distances**: when cosine beats Euclidean.

* **n-gram, Character, and Masked LMs** (important): Baselines and controls.

  * **Markov assumptions**: order, backoff, perplexity.
  * **Transformer basics**: attention, masking, tokenization artifacts.

* **Language Modeling Smoothing** (important): Kneser–Ney, Witten–Bell for low counts.

  * **Count transforms**: discounting, interpolation.
  * **Evaluation**: perplexity vs accuracy on tiny corpora.

* **PMI–SGNS Connection** (related): Why word2vec ≈ factorizing shifted PMI.

  * **Matrix–model duality**: implicit factorization idea.
  * **Negative sampling**: noise distributions shape embeddings.

* **Noise Contrastive Estimation** (related): Learn unnormalized models by ranking real vs noise.

  * **Logistic regression view**: classification as likelihood proxy.
  * **Noise choice**: how distributions affect bias/variance.

* **Representational Similarity Analysis, CKA** (related): Compare representational geometry.

  * **Kernel methods**: linear vs RBF intuitions.
  * **Invariance**: why CKA tolerates rotations.

* **Approximate Nearest Neighbor Search** (related): Fast similarity at scale.

  * **Index structures**: HNSW graphs, IVF lists, product quantization.
  * **Recall–speed tradeoff**: tuning for analysis vs deployment.

* **Tokenization Strategies** (related): BPE, unigram LM, char-level.

  * **Subword math**: frequency merges, likelihood objective.
  * **Impact on morphology**: leakage vs discovery.

# Morphology, phonology, typology

* **Morpheme**: The smallest meaningful unit for decomposition.

  * **Morphology basics**: roots, affixes, clitics, compounds.
  * **Segmentation criteria**: productivity, distributional evidence.

* **Composite Token**: Multi-morpheme surface form you model.

  * **Compositional semantics**: additive vs interaction effects.
  * **Incidence matrices**: encoding token↔morpheme memberships.

* **Gloss**: Short textual semantic cue for a morpheme/token.

  * **Annotation conventions**: Leipzig rules, gloss alignment.
  * **Embedding prep**: tokenization, stopwords, character n-grams.

* **Finite-State Morphology and WFSTs** (important): Encode morphotactics, alternations.

  * **Automata basics**: DFA/NFA, composition.
  * **Weighted transducers**: tropical/log semirings, path costs.

* **Unsupervised Morphological Segmentation (Morfessor)** (important): Split tokens without labels.

  * **MAP estimation**: likelihood plus lexicon penalty.
  * **Model selection**: controlling under/over-segmentation.

* **Allomorphy and Morphophonemics** (related): Context-driven alternations.

  * **Rules vs weights**: rewrite rules, constraint ranking.
  * **Environment features**: boundary, vowel harmony analogs.

* **Phonotactics and Syllable Structure** (related): What sequences are well-formed.

  * **n-gram phonotactics**: character/phoneme models.
  * **Syllabification heuristics**: onsets, codas, sonority.

* **Grapheme–Phoneme Mapping and Orthography** (related): Letters to sounds.

  * **Alignment**: EM or dynamic programming for G2P.
  * **Ambiguity**: many-to-many mapping handling.

* **Typological Universals and Historical Heuristics** (related): Cross-linguistic priors and sanity checks.

  * **WALS-style features**: word order, affix positions.
  * **Change mechanisms**: analogy, borrowing, sound change.

# Probabilistic modeling and Bayesian tools

* **Hidden Markov Models and PCFGs** (related): Sequence and hierarchical generative baselines.

  * **EM algorithm**: forward–backward, inside–outside.
  * **Independence assumptions**: where they help or hurt.

* **Sequence Labeling for Morphology** (related): CRFs or BiLSTM-CRF for boundaries.

  * **Feature design**: char n-grams, affix indicators.
  * **Structured decoding**: Viterbi, CRF potentials.

* **Bayesian Topic Models (LDA, CTM)** (related): Induce thematic fields over glosses/residuals.

  * **Dirichlet-multinomial**: conjugacy, sparsity.
  * **Inference**: collapsed Gibbs vs variational.

* **Bayesian Nonparametrics** (related): Flexible clustering without fixing K.

  * **CRP/DPM**: stick-breaking, exchangeability.
  * **Hyperparameters**: concentration effects on cluster counts.

* **Variational Inference and ELBO** (related): Scale Bayesian models.

  * **KL objectives**: mean-field factorization.
  * **Stochastic VI**: minibatching, reparameterization.

* **MCMC Diagnostics** (related): Trust your samples.

  * **Convergence checks**: R-hat, ESS.
  * **Trace pathologies**: stickiness, multimodality.

# Attribution, causality, and interpretability

* **Leave-One-Out Attribution (LOO)**: Perturb morphemes to see impact on reconstructions.

  * **Counterfactual thinking**: what changes when a feature is removed.
  * **Efficient recomputation**: cached projections, Sherman–Morrison idea.

* **Asymmetric Dependency**: Directional association from deltas.

  * **Conditional probability**: P(A|B) vs P(B|A) differences.
  * **Graph orientation**: heuristics for direction under constraints.

* **Shapley Values and Integrated Gradients** (related): Complement LOO with path or coalition methods.

  * **Linearity/additivity axioms**: why Shapley is fair.
  * **Path integrals**: baseline choice, saturation effects.

* **Influence Functions** (related): Which training examples moved a parameter or prediction.

  * **Implicit differentiation**: Hessian–vector products.
  * **Robustness**: detect mislabeled or outlier examples.

* **Causal Probing and Ablations** (related): Separate correlation from causation via controlled interventions.

  * **Causal graphs**: DAGs, do-operator intuition.
  * **Controls and counterfactuals**: hold confounders fixed, vary one factor.

* **Confounding, Data Shift, and Controls** (important): Make sure effects aren’t artifacts.

  * **Stratified evaluation**: by source, page, or era.
  * **Synthetic tests**: randomized labels, shuffled contexts.

* **Evaluation Design** (important): Intrinsic vs extrinsic, preregistration.

  * **Metric choice**: what your measure implies about success.
  * **Leakage audits**: split by unit, dedupe near-duplicates.

* **Mean Squared Error (MSE)** and **Cosine Reconstruction Error**: Core reconstruction diagnostics.

  * **Loss landscapes**: scale sensitivity, comparability across runs.
  * **Normalization**: why cosine on unit vectors simplifies.

# Graphs, clustering, alignment, and sequence similarity

* **Residual Vector**: Semantic remainder after composition.

  * **Vector projections**: expected vs residual components.
  * **Normalization**: compare residuals fairly.

* **Residual Clustering**: Find thematic pockets in residual space.

  * **Cluster validation**: silhouette, Davies–Bouldin.
  * **Labeling clusters**: nearest neighbors, centroid glosses.

* **Graph Construction and Community Detection** (related): Morpheme graphs from co-usage.

  * **Similarity graphs**: kNN, epsilon graphs, thresholding.
  * **Louvain/Leiden**: modularity maximization basics.

* **Spectral and Density Clustering** (related): Non-spherical clusters.

  * **Graph Laplacian**: eigenmaps, cuts.
  * **HDBSCAN**: density, min cluster size, stability.

* **Dimensionality Reduction for Visualization** (related): PCA for sanity, UMAP/t-SNE for clusters.

  * **Global vs local structure**: don’t overread t-SNE.
  * **Preprocessing**: center, scale, cosine→Euclidean mapping.

* **Edit Distance and String Alignment** (related): Align noisy variants.

  * **Dynamic programming**: Levenshtein recursion.
  * **Costs**: substitutions vs insertions, tuning.

* **Dynamic Time Warping for Strings** (related): Elastic alignment for repeated motifs.

  * **Warping paths**: monotonicity constraints.
  * **Regularization**: windowing to avoid degenerate matches.

# Project infrastructure, data management, and ethics

* **Design Matrix (Incidence Matrix)**: Sparse token↔morpheme encoding.

  * **Sparse data structures**: CSR/COO, memory patterns.
  * **SQL/ETL basics**: efficient construction and indexing.

* **TF-IDF Vectorization**: Weighted gloss features.

  * **Document frequency math**: logs, smoothing, sublinear TF.
  * **Preprocessing**: tokenization, stemming vs lemmatization.

* **Hashing Vectorizer**: Fixed-dim features via hashing.

  * **Hash collisions**: tradeoffs, signed hashing.
  * **Determinism**: reproducibility across runs.

* **Composite Reconstruction Table**: Store predictions, usage, errors.

  * **Schema design**: keys, indexes, foreign keys.
  * **OLAP querying**: aggregations for diagnostics.

* **Attribution Marginals Table**: Aggregated LOO deltas.

  * **Aggregation math**: means, winsorization, robust stats.
  * **Provenance**: record seeds, model hashes.

* **Collocation Statistics Table**: PMI, LLR, directional scores.

  * **Count pipelines**: windows, skip-grams.
  * **Smoothing**: add-k, backoff for rare pairs.

* **Residual Cluster Membership Table**: Morpheme↔cluster ties.

  * **Soft vs hard assignments**: probabilities vs argmax.
  * **Centroid tracking**: version per training run.

* **Morpheme Semantic Vectors Table**: Persist learned embeddings.

  * **Vector storage**: float32 vs float16, quantization.
  * **Indexing**: ANN indexes, cosine-ready norms.

* **Baseline Residuals File**: Reference for improvements.

  * **JSONL hygiene**: schema consistency, checksums.
  * **Change detection**: small multiples, delta thresholds.

* **Pipeline Report**: Single artifact for audits.

  * **Reproducibility**: seeds, configs, environment pinning.
  * **Visualization**: plots that answer specific questions.

* **Data and Experiment Management** (important): DVC, MLflow, or W&B.

  * **Versioning**: datasets, models, metrics.
  * **Lineage**: trace which data produced which results.

* **Annotation Science** (important): High-signal human input.

  * **Agreement metrics**: Cohen’s κ, Krippendorff’s α.
  * **Active learning loops**: uncertainty and diversity sampling.

* **TEI/XML and UD Conventions** (related): Interop with DH standards.

  * **Schema mindset**: elements, attributes, validation.
  * **Conversion**: scripts to round-trip your data.

* **Ethics and Epistemic Risk** (important): Don’t overclaim structure.

  * **Uncertainty reporting**: CIs, ablations, prereg.
  * **Biases**: confirmation bias, p-hacking defenses.

---

## Learning resources (free or cheap)

* **Speech and Language Processing (3e draft)** — Jurafsky & Martin, free online from authors.
* **Introduction to Natural Language Processing** — Jacob Eisenstein, free PDF on author site, MIT Press print is inexpensive used.
* **Deep Learning** — Goodfellow, Bengio, Courville, full text free online.
* **Information Theory, Inference, and Learning Algorithms** — David MacKay, free online text.
* **The Elements of Statistical Learning** — Hastie, Tibshirani, Friedman, free PDF, used copies common.
* **Network Science** — Barabási, free online, good for graph/community detection intuition.
* **An Introduction to Information Retrieval** — Manning, Raghavan, Schütze, free online, crisp TF-IDF and evaluation.
* **Finite-State Morphology** — Beesley & Karttunen, check libraries; pair with free OpenFst docs and tutorials.
* **Morfessor** — open source, code and papers free from the Helsinki group.
* **Stanford CS224n** — free YouTube lectures, transformer and embedding fundamentals.
* **Hugging Face NLP Course** — free, practical labs and tokenization deep dives.
* **The Open Handbook of Linguistic Data Management** — MIT Press Open, free PDF.
