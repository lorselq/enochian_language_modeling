"""Enrich the base dictionary with POS and semantic metadata."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml


RE_VERB = re.compile(r"^to\s+(?P<lemma>[a-z][a-z\-']*)", re.IGNORECASE)
RE_TOKENS = re.compile(r"[a-zA-Z]+")
RE_EMPHASIS = re.compile(r"\*([^*]+)\*")
COPULA_FORMS = {"am", "art", "are", "be", "being", "been", "is", "was", "were"}
PREPOSITIONS = {
    "of",
    "from",
    "in",
    "into",
    "on",
    "onto",
    "upon",
    "with",
    "within",
    "without",
    "between",
    "among",
    "amongst",
    "about",
    "toward",
    "towards",
    "through",
    "beyond",
    "under",
    "above",
    "before",
    "after",
}
COORDINATORS = {"and", "or", "but", "nor"}
STOPWORDS = {"the", "a", "an"}
POSSESSIVE_PRONOUNS = {
    "my",
    "mine",
    "thy",
    "thine",
    "his",
    "her",
    "hers",
    "its",
    "our",
    "ours",
    "your",
    "yours",
    "their",
    "theirs",
    "whose",
}
ADDITIONAL_IGNORE_TOKENS = {"ye", "o"}


WORDNET_POS_BY_LABEL = {
    "NOUN": ("n",),
    "VERB": ("v",),
    "AUX": ("v",),
    "ADJ": ("a", "s"),
    "ADVERB": ("r",),
    "ADV": ("r",),
}


def _normalize_wordnet_candidate(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    cleaned = token.replace("_", " ").replace("-", " ").strip().lower()
    if not cleaned:
        return None
    primary = cleaned.split()[0]
    return singularize(primary)


@dataclass
class DomainConfig:
    """Semantic domain configuration loaded from YAML."""

    domains: Dict[str, str]
    headword_to_domains: Dict[str, List[str]]
    headword_stopwords: Set[str]
    wordnet_lexname_to_domains: Dict[str, List[str]] = field(default_factory=dict)
    angelic_lexnames: Set[str] = field(default_factory=set)
    sacred_indicators: Set[str] = field(default_factory=set)
    use_wordnet_gloss_similarity: bool = False
    wordnet_gloss_similarity_threshold: float = 0.0
    _wordnet_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[List[str], Optional[str]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _wordnet_import_failed: bool = field(default=False, init=False, repr=False)
    _wordnet_missing_corpus: bool = field(default=False, init=False, repr=False)
    _domain_tfidf: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    _domain_norms: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _domain_idf: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _default_idf: float = field(default=1.0, init=False, repr=False)
    _wordnet_download_attempted: bool = field(default=False, init=False, repr=False)
    _sacred_indicator_pattern: Optional[re.Pattern[str]] = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        dictionary_tokens: Optional[Set[str]] = None,
    ) -> "DomainConfig":
        data = yaml.safe_load(path.read_text())
        configured_stopwords = {
            str(token).lower()
            for token in data.get("headword_stopwords", [])
            if token
        }
        vocab = {token.lower() for token in dictionary_tokens or set()}
        headword_map: Dict[str, List[str]] = {}
        for key, value in data.get("headword_to_domains", {}).items():
            hw = str(key or "").lower()
            if not hw or hw in configured_stopwords:
                continue
            if vocab and hw not in vocab:
                continue
            headword_map[hw] = [label.upper() for label in value]
        lexname_mapping: Dict[str, List[str]] = {}
        for key, value in data.get("wordnet_lexname_to_domains", {}).items():
            lexname = str(key or "").strip()
            if not lexname:
                continue
            lexname_mapping[lexname] = [str(label).upper() for label in value or []]
        gloss_similarity_cfg = data.get("wordnet_gloss_similarity", {}) or {}
        use_wordnet_gloss_similarity = bool(gloss_similarity_cfg.get("enabled", False))
        threshold = float(gloss_similarity_cfg.get("threshold", 0.0))
        sacred_indicators = {
            str(token).lower()
            for token in data.get("wordnet_angelic_indicators", [])
            if token
        }
        angelic_lexnames = {
            str(name).lower()
            for name in data.get("wordnet_angelic_lexnames", [])
            if name
        }
        return cls(
            domains=data.get("domains", {}),
            headword_to_domains=headword_map,
            headword_stopwords=configured_stopwords,
            wordnet_lexname_to_domains=lexname_mapping,
            angelic_lexnames=angelic_lexnames,
            sacred_indicators=sacred_indicators,
            use_wordnet_gloss_similarity=use_wordnet_gloss_similarity,
            wordnet_gloss_similarity_threshold=threshold,
        )

    def __post_init__(self) -> None:
        if self.sacred_indicators:
            self.sacred_indicators = {token.lower() for token in self.sacred_indicators}
            escaped = [re.escape(token) for token in self.sacred_indicators]
            self._sacred_indicator_pattern = re.compile(
                rf"\\b({'|'.join(escaped)})\\b", re.IGNORECASE
            )
        if self.angelic_lexnames:
            self.angelic_lexnames = {name.lower() for name in self.angelic_lexnames}
        if self.use_wordnet_gloss_similarity:
            self._initialize_domain_similarity()

    def lookup(
        self, headword: Optional[str], *, heuristic_pos: Optional[Sequence[str]] = None
    ) -> Tuple[List[str], Optional[str]]:
        if not headword:
            return [], None
        hw = headword.lower()
        candidates = [hw]
        singular = singularize(hw)
        if singular != hw:
            candidates.append(singular)
        for candidate in candidates:
            if candidate in self.headword_to_domains:
                return self.headword_to_domains[candidate], None
        for candidate in candidates:
            inferred, notes = self.infer_domains_via_wordnet(
                candidate, heuristic_pos=heuristic_pos
            )
            if inferred:
                return inferred, notes
        return [], None

    def infer_domains_via_wordnet(
        self, headword: Optional[str], *, heuristic_pos: Optional[Sequence[str]] = None
    ) -> Tuple[List[str], Optional[str]]:
        if not headword:
            return [], None
        normalized = headword.lower()
        key = (normalized, tuple(heuristic_pos or []))
        if key in self._wordnet_cache:
            return self._wordnet_cache[key]
        if self._wordnet_import_failed or self._wordnet_missing_corpus:
            self._wordnet_cache[key] = ([], None)
            return [], None
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
        except ImportError:
            self._wordnet_import_failed = True
            self._wordnet_cache[key] = ([], None)
            return [], None

        pos_filters: Set[str] = set()
        for label in heuristic_pos or []:
            pos_filters.update(WORDNET_POS_BY_LABEL.get(str(label).upper(), ()))

        try:
            synsets = self._lookup_wordnet_synsets(wn, normalized, pos_filters)
        except LookupError:
            if not self._ensure_wordnet_corpus():
                self._wordnet_missing_corpus = True
                self._wordnet_cache[key] = ([], None)
                return [], None
            try:
                synsets = self._lookup_wordnet_synsets(wn, normalized, pos_filters)
            except LookupError:
                self._wordnet_missing_corpus = True
                self._wordnet_cache[key] = ([], None)
                return [], None

        inferred: List[str] = []
        notes: Optional[str] = None
        candidate_tokens: Set[str] = set()
        lexname_domains: List[str] = []
        for synset in synsets:
            for lemma in synset.lemma_names():
                normalized_lemma = _normalize_wordnet_candidate(lemma)
                if normalized_lemma:
                    candidate_tokens.add(normalized_lemma)
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemma_names():
                    normalized_hypernym = _normalize_wordnet_candidate(lemma)
                    if normalized_hypernym:
                        candidate_tokens.add(normalized_hypernym)
            lex_domains = [
                str(label).upper()
                for label in self.wordnet_lexname_to_domains.get(synset.lexname(), [])
            ]
            sacred_domains: List[str] = []
            if self._should_tag_as_sacred(synset):
                sacred_domains = ["DIVINE", "CELESTIAL"]

            for label in sacred_domains + lex_domains:
                if label not in lexname_domains:
                    lexname_domains.append(label)

        for token in candidate_tokens:
            if token in self.headword_to_domains:
                for label in self.headword_to_domains[token]:
                    if label not in inferred:
                        inferred.append(label)

        if not inferred and self.use_wordnet_gloss_similarity:
            gloss_texts: List[str] = []
            for synset in synsets:
                gloss_texts.append(synset.definition())
                gloss_texts.extend(synset.examples())
            gloss_inferred, gloss_notes = self._infer_domains_from_glosses(gloss_texts)
            if gloss_inferred:
                inferred = gloss_inferred
                notes = gloss_notes

        if not inferred and lexname_domains:
            inferred = lexname_domains
            notes = "wordnet_lexname"

        self._wordnet_cache[key] = (inferred, notes)
        return inferred, notes

    def _should_tag_as_sacred(self, synset: object) -> bool:
        if not self._sacred_indicator_pattern:
            return False
        try:
            lexname = synset.lexname().lower()
            if lexname not in self.angelic_lexnames:
                return False
            candidate_texts = list(synset.lemma_names())
            candidate_texts.append(synset.definition())
            candidate_texts.extend(synset.examples())
        except Exception:
            return False
        for text in candidate_texts:
            normalized = str(text).replace("_", " ")
            if self._sacred_indicator_pattern.search(normalized):
                return True
        return False

    def _ensure_wordnet_corpus(self) -> bool:
        if self._wordnet_download_attempted:
            return not self._wordnet_missing_corpus
        self._wordnet_download_attempted = True
        try:
            import nltk
        except ImportError:
            return False

        try:
            nltk.download("wordnet", quiet=True, raise_on_error=True)
            nltk.download("omw-1.4", quiet=True, raise_on_error=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _lookup_wordnet_synsets(wn, headword: str, pos_filters: Set[str]):
        synsets = []
        seen: Set[str] = set()
        if pos_filters:
            for pos in pos_filters:
                for synset in wn.synsets(headword, pos=pos):
                    if synset.name() not in seen:
                        seen.add(synset.name())
                        synsets.append(synset)
        else:
            for synset in wn.synsets(headword):
                if synset.name() not in seen:
                    seen.add(synset.name())
                    synsets.append(synset)
        return synsets

    def _initialize_domain_similarity(self) -> None:
        documents: Dict[str, List[str]] = {}
        for label, description in self.domains.items():
            tokens = extract_tokens(description)
            if tokens:
                documents[label.upper()] = tokens
        if not documents:
            self.use_wordnet_gloss_similarity = False
            return

        doc_freq: Counter[str] = Counter()
        for tokens in documents.values():
            doc_freq.update(set(tokens))
        num_docs = len(documents)
        self._default_idf = math.log((1 + num_docs) / 1) + 1.0
        for token, count in doc_freq.items():
            self._domain_idf[token] = math.log((1 + num_docs) / (1 + count)) + 1.0

        for label, tokens in documents.items():
            vec = self._tfidf_vector(tokens)
            norm = math.sqrt(sum(weight * weight for weight in vec.values()))
            if norm:
                self._domain_tfidf[label] = vec
                self._domain_norms[label] = norm

        if not self._domain_tfidf:
            self.use_wordnet_gloss_similarity = False

    def _tfidf_vector(self, tokens: Sequence[str]) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        if not tokens:
            return vec
        token_counts = Counter(tokens)
        total = sum(token_counts.values())
        for token, count in token_counts.items():
            idf = self._domain_idf.get(token, self._default_idf)
            vec[token] = (count / total) * idf
        return vec

    def _infer_domains_from_glosses(self, gloss_texts: Sequence[str]) -> Tuple[List[str], Optional[str]]:
        if not self.use_wordnet_gloss_similarity or not self._domain_tfidf:
            return [], None
        gloss_tokens: List[str] = []
        for text in gloss_texts:
            gloss_tokens.extend(extract_tokens(text))
        if not gloss_tokens:
            return [], None
        gloss_vec = self._tfidf_vector(gloss_tokens)
        gloss_norm = math.sqrt(sum(weight * weight for weight in gloss_vec.values()))
        if not gloss_norm:
            return [], None

        scores: Dict[str, float] = {}
        for label, domain_vec in self._domain_tfidf.items():
            denom = gloss_norm * self._domain_norms.get(label, 0.0)
            if not denom:
                continue
            score = 0.0
            for token, weight in gloss_vec.items():
                if token in domain_vec:
                    score += weight * domain_vec[token]
            if score:
                scores[label] = score / denom

        if not scores:
            return [], None

        max_score = max(scores.values())
        if max_score < self.wordnet_gloss_similarity_threshold:
            return [], None

        best_domains = [label for label, score in scores.items() if score == max_score]
        best_domains.sort()
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_notes = [f"{label}={score:.3f}" for label, score in ranked[:3]]
        note = "wordnet_gloss:" + ",".join(top_notes)
        return best_domains, note


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 2:
        return token[:-2]
    if token.endswith("s") and len(token) > 1:
        return token[:-1]
    return token


def _normalize_vector(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


class TextEmbedder:
    """Abstract text embedder interface."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(TextEmbedder):
    """Encode text using a sentence-transformers model if available."""

    def __init__(self, model_name: str) -> None:
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "sentence-transformers is required for --embedding-fallback when "
                "using the sentence-transformers backend. Install it or choose --embedding-backend glove."
            )
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            list(texts), show_progress_bar=False, normalize_embeddings=True
        )
        return [list(vector) for vector in embeddings]


class GloveAveragingEmbedder(TextEmbedder):
    """Average GloVe token vectors to represent short glosses."""

    def __init__(self, model_name: str = "glove-wiki-gigaword-50") -> None:
        if importlib.util.find_spec("gensim") is None:
            raise RuntimeError(
                "gensim is required for --embedding-backend glove. Install it or choose "
                "--embedding-backend sentence-transformers."
            )
        import gensim.downloader as api

        self._model = api.load(model_name)
        self._dimension = int(self._model.vector_size)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            tokens = [token.lower() for token in extract_tokens(text)]
            if not tokens:
                vectors.append([0.0 for _ in range(self._dimension)])
                continue
            summed: List[float] = [0.0 for _ in range(self._dimension)]
            count = 0
            for token in tokens:
                if token not in self._model:
                    continue
                count += 1
                word_vector = self._model[token]
                for idx in range(self._dimension):
                    summed[idx] += float(word_vector[idx])
            if not count:
                vectors.append([0.0 for _ in range(self._dimension)])
                continue
            averaged = [value / count for value in summed]
            vectors.append(_normalize_vector(averaged))
        return vectors


def build_embedder(backend: str, model_name: str) -> TextEmbedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "glove":
        return GloveAveragingEmbedder(model_name=model_name)
    raise RuntimeError(f"Unknown embedding backend '{backend}'")


def build_domain_embeddings(domain_config: DomainConfig, embedder: TextEmbedder) -> Dict[str, List[float]]:
    seeds: Dict[str, List[str]] = {label.upper(): [] for label in domain_config.domains}
    for headword, labels in domain_config.headword_to_domains.items():
        for label in labels:
            seeds.setdefault(label.upper(), []).append(headword)
    domain_texts: List[str] = []
    ordered_labels: List[str] = []
    for label, description in domain_config.domains.items():
        label_upper = label.upper()
        domain_texts.append(f"{label_upper}: {description} {' '.join(seeds.get(label_upper, []))}")
        ordered_labels.append(label_upper)
    vectors = embedder.embed(domain_texts)
    return {label: vector for label, vector in zip(ordered_labels, vectors)}


def gather_sense_text(sense: Dict) -> str:
    parts: List[str] = []
    definition = sense.get("definition")
    if isinstance(definition, str):
        parts.append(definition)
    examples: Iterable[str] = []
    raw_examples = sense.get("example_sentences") or sense.get("examples")
    if isinstance(raw_examples, str):
        examples = [raw_examples]
    elif isinstance(raw_examples, (list, tuple)):
        examples = [text for text in raw_examples if isinstance(text, str)]
    parts.extend(examples)
    return " ".join(part for part in parts if part).strip()


def select_embedding_domains(
    sense_text: str,
    *,
    embedder: TextEmbedder,
    domain_embeddings: Dict[str, List[float]],
    min_similarity: float,
    similarity_band: float,
) -> List[str]:
    if not sense_text:
        return []
    sense_vector = embedder.embed_one(sense_text)
    scored: List[Tuple[str, float]] = []
    for label, vector in domain_embeddings.items():
        similarity = _cosine_similarity(sense_vector, vector)
        scored.append((label, similarity))
    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored:
        return []
    _, top_score = scored[0]
    if top_score < min_similarity:
        return []
    threshold = max(min_similarity, top_score - similarity_band)
    return [label for label, score in scored if score >= threshold]


def extract_tokens(gloss: str) -> List[str]:
    return [match.group(0).lower() for match in RE_TOKENS.finditer(gloss or "")]


def extract_headword(
    gloss: str,
    *,
    ignore_tokens: Optional[Sequence[str]] = None,
) -> Optional[str]:
    tokens = extract_tokens(gloss)
    if not tokens:
        return None
    # Drop leading infinitive marker or determiners
    idx = 0
    if tokens[0] == "to" and len(tokens) > 1:
        idx = 1
    ignore = set(STOPWORDS)
    ignore |= POSSESSIVE_PRONOUNS
    ignore |= ADDITIONAL_IGNORE_TOKENS
    if ignore_tokens:
        ignore |= {tok.lower() for tok in ignore_tokens}
    while idx < len(tokens) and tokens[idx] in ignore:
        idx += 1
    return tokens[idx] if idx < len(tokens) else None


def detect_compound(gloss: str, tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    if "," in gloss or " / " in gloss or ";" in gloss:
        return True
    compound_markers = {"which", "that", "those", "these", "they", "amongst"}
    if any(marker in tokens for marker in compound_markers):
        return True
    return len(tokens) > 2


def infer_pos(gloss: str, tokens: Sequence[str]) -> Tuple[List[str], bool, bool, Optional[str]]:
    pos: List[str] = []
    notes: Optional[str] = None
    is_copula = False

    gloss_lower = (gloss or "").strip().lower()
    if not gloss_lower:
        pos.append("NOUN")
        notes = "fallback:noun_empty"
        return pos, is_copula, False, notes

    if gloss_lower.startswith("to be ") or gloss_lower in COPULA_FORMS:
        is_copula = True
        if "AUX" not in pos:
            pos.append("AUX")

    if not is_copula and gloss_lower.startswith("to be"):
        is_copula = True
        if "AUX" not in pos:
            pos.append("AUX")

    verb_match = RE_VERB.match(gloss_lower)
    if verb_match and not is_copula:
        if "VERB" not in pos:
            pos.append("VERB")

    if tokens:
        primary = tokens[0]
        if primary in PREPOSITIONS and len(tokens) <= 3:
            if "ADP" not in pos:
                pos.append("ADP")
        if len(tokens) == 1 and primary in COORDINATORS:
            if "CCONJ" not in pos:
                pos.append("CCONJ")
        if len(tokens) == 1 and primary in {"he", "she", "it", "they", "we", "i", "thou", "ye"}:
            if "PRON" not in pos:
                pos.append("PRON")

    if not pos:
        pos.append("NOUN")
        notes = "fallback:noun_default"

    return pos, is_copula, detect_compound(gloss, tokens), notes


class CitationTagger:
    """Abstract base class for POS taggers operating on citation snippets."""

    def tag_phrase(self, phrase: str) -> List[str]:
        raise NotImplementedError

    def tag_phrases(self, phrases: Sequence[str]) -> List[str]:
        tags: List[str] = []
        for phrase in phrases:
            if not phrase:
                continue
            tags.extend(self.tag_phrase(phrase))
        return tags


class SpacyCitationTagger(CitationTagger):
    """spaCy-backed citation tagger that emits coarse POS labels."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "spaCy is required for citation tagging. Install it in your environment "
                "(e.g., `pip install spacy`) or set --citation-tagger none to skip citation POS enrichment."
            ) from exc

        try:
            self._nlp = spacy.load(model, disable=["ner", "parser"])
        except OSError as exc:  # pragma: no cover - model lookup
            raise RuntimeError(
                f"Unable to load spaCy model '{model}'. Install it with `python -m spacy download {model}` "
                "or choose a different model via --spacy-model."
            ) from exc

    def tag_phrase(self, phrase: str) -> List[str]:
        doc = self._nlp(phrase)
        return [token.pos_.upper() for token in doc if token.is_alpha]


def extract_emphasized_phrases(text: str) -> List[str]:
    return [match.group(1).strip() for match in RE_EMPHASIS.finditer(text or "") if match.group(1)]


def infer_citation_votes(sense: Dict, citation_tagger: Optional[CitationTagger]) -> Counter:
    if not citation_tagger:
        return Counter()
    phrases: List[str] = []
    for citation in sense.get("key_citations", []) or []:
        if not isinstance(citation, dict):
            continue
        phrases.extend(extract_emphasized_phrases(str(citation.get("context", ""))))
    if not phrases:
        return Counter()
    tags = citation_tagger.tag_phrases(phrases)
    return Counter(tag for tag in tags if tag)


def combine_pos_labels(heuristic_pos: Sequence[str], citation_votes: Counter) -> List[str]:
    if not heuristic_pos and not citation_votes:
        return []
    order: Dict[str, int] = {}
    votes: Counter = Counter()
    citation_sources: Set[str] = set()

    def register(label: str, weight: float = 1.0, *, source: Optional[str] = None) -> None:
        if not label:
            return
        votes[label] += weight
        if label not in order:
            order[label] = len(order)
        if source == "citation":
            citation_sources.add(label)

    for label in heuristic_pos:
        register(label, 1.0, source="heuristic")
    for label, count in citation_votes.items():
        register(label, float(count), source="citation")

    def sort_key(item: Tuple[str, float]) -> Tuple[float, int, int]:
        label, vote = item
        citation_rank = 0 if label in citation_sources else 1
        return (-vote, citation_rank, order.get(label, len(order)))

    sorted_labels = sorted(votes.items(), key=sort_key)
    return [label for label, _ in sorted_labels]


def enrich_senses(
    entry: Dict,
    domain_config: DomainConfig,
    citation_tagger: Optional[CitationTagger] = None,
    *,
    embedding_fallback: Optional[Dict] = None,
) -> None:
    for sense in entry.get("senses", []):
        gloss = sense.get("definition", "")
        tokens = extract_tokens(gloss)
        pos, is_copula, is_compound, notes = infer_pos(gloss, tokens)
        headword = extract_headword(
            gloss, ignore_tokens=domain_config.headword_stopwords
        )
        domains, notes_semantic = domain_config.lookup(headword, heuristic_pos=pos)
        citation_votes = infer_citation_votes(sense, citation_tagger)
        combined_pos = combine_pos_labels(pos, citation_votes)
        citation_note = None
        if citation_votes:
            details = []
            for label, count in citation_votes.most_common():
                details.append(f"{label}({int(count)})" if count > 1 else label)
            citation_note = "citation:" + ",".join(details)

        notes_components: List[str] = []
        if notes:
            notes_components.append(notes)
        if pos:
            notes_components.append("heuristic:" + ",".join(pos))
        if citation_note:
            notes_components.append(citation_note)
        notes_pos = ";".join(notes_components) if notes_components else None

        if not domains and embedding_fallback:
            sense_text = gather_sense_text(sense)
            fallback_domains = select_embedding_domains(
                sense_text,
                embedder=embedding_fallback["embedder"],
                domain_embeddings=embedding_fallback["domain_embeddings"],
                min_similarity=embedding_fallback["min_similarity"],
                similarity_band=embedding_fallback["similarity_band"],
            )
            if fallback_domains:
                domains = fallback_domains

        sense["parts_of_speech"] = combined_pos or pos
        sense["semantic_domains"] = domains
        sense["is_copula"] = is_copula
        sense["is_compound_standing_for_phrase"] = is_compound
        sense["notes_pos"] = notes_pos
        sense["notes_semantic"] = notes_semantic


def review(enriched_entries: Sequence[Dict]) -> str:
    copula_entries: List[str] = []
    multi_pos: List[str] = []
    missing_domains: List[str] = []

    for entry in enriched_entries:
        word = entry.get("word", "<unknown>")
        for sense in entry.get("senses", []):
            descriptor = f"{word} (sense {sense.get('sense_id')})"
            pos = sense.get("parts_of_speech", [])
            if sense.get("is_copula"):
                copula_entries.append(descriptor)
            if len(pos) > 1:
                multi_pos.append(descriptor + f" -> {pos}")
            if not sense.get("semantic_domains"):
                missing_domains.append(descriptor)

    lines = ["=== Manual Review Report ==="]
    lines.append(f"Copula lemmas: {len(copula_entries)}")
    lines.extend(copula_entries)
    lines.append("")
    lines.append(f"Multiple POS senses: {len(multi_pos)}")
    lines.extend(multi_pos)
    lines.append("")
    lines.append(f"Senses without semantic domains: {len(missing_domains)}")
    lines.extend(missing_domains)
    return "\n".join(lines)


def collect_gloss_vocabulary(entries: Sequence[Dict]) -> Set[str]:
    vocab: Set[str] = set()
    for entry in entries:
        top_gloss = entry.get("definition")
        if isinstance(top_gloss, str):
            for token in extract_tokens(top_gloss):
                vocab.add(token)
                vocab.add(singularize(token))
        for sense in entry.get("senses", []):
            definition = sense.get("definition")
            if not isinstance(definition, str):
                continue
            for token in extract_tokens(definition):
                vocab.add(token)
                vocab.add(singularize(token))
    return {tok for tok in vocab if tok}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "enochian_lm"
        / "root_extraction"
        / "data"
        / "dictionary.json"
    )
    default_output = default_input.with_name("dictionary_enriched.json")
    default_domains = Path(__file__).resolve().parents[1] / "config" / "semantic_domains.yml"

    parser.add_argument("--input", type=Path, default=default_input, help="Path to dictionary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination for enriched dictionary",
    )
    parser.add_argument(
        "--domains",
        type=Path,
        default=default_domains,
        help="Path to semantic_domains.yml",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to save the manual review report",
    )
    parser.add_argument(
        "--citation-tagger",
        choices=["spacy", "none"],
        default="spacy",
        help="Backend used to POS-tag emphasized citation snippets",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model used when --citation-tagger=spacy",
    )
    parser.add_argument(
        "--embedding-fallback",
        action="store_true",
        help="Enable embedding-based semantic domain suggestions for senses that lack domains",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "glove"],
        default="sentence-transformers",
        help="Embedding backend used when --embedding-fallback is enabled",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name used by the embedding backend",
    )
    parser.add_argument(
        "--embedding-min-similarity",
        type=float,
        default=0.35,
        help="Minimum cosine similarity required before adding a semantic domain via the embedding fallback",
    )
    parser.add_argument(
        "--embedding-similarity-band",
        type=float,
        default=0.05,
        help="Similarity band beneath the top score to include additional domains",
    )
    return parser.parse_args()


def build_citation_tagger(name: str, spacy_model: str) -> Optional[CitationTagger]:
    if name == "none":
        return None
    if name == "spacy":
        try:
            return SpacyCitationTagger(model=spacy_model)
        except RuntimeError as exc:
            raise SystemExit(str(exc))
    raise SystemExit(f"Unknown citation tagger '{name}'")


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing file {args.output}. Use --overwrite to replace it."
        )

    entries = json.loads(args.input.read_text())
    vocab = collect_gloss_vocabulary(entries)
    domain_config = DomainConfig.load(
        args.domains, dictionary_tokens=vocab
    )
    citation_tagger = build_citation_tagger(args.citation_tagger, args.spacy_model)
    embedding_fallback: Optional[Dict] = None
    if args.embedding_fallback:
        embedder = build_embedder(args.embedding_backend, args.embedding_model)
        embedding_fallback = {
            "embedder": embedder,
            "domain_embeddings": build_domain_embeddings(domain_config, embedder),
            "min_similarity": args.embedding_min_similarity,
            "similarity_band": args.embedding_similarity_band,
        }
    for entry in entries:
        enrich_senses(
            entry,
            domain_config,
            citation_tagger=citation_tagger,
            embedding_fallback=embedding_fallback,
        )

    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    report_text = review(entries)
    if args.report:
        args.report.write_text(report_text + "\n")
    print(report_text)


if __name__ == "__main__":
    main()
