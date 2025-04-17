import json
from gensim.models import FastText

class MorphemeCandidateFinder:
    def __init__(self, ngram_path, fasttext_model_path, dictionary_entries):
        self.ngram_index = self._load_ngram_index(ngram_path)
        self.fasttext_model = FastText.load(str(fasttext_model_path))
        self.dictionary = {entry["normalized"].lower(): entry for entry in dictionary_entries}
        self.known_words = set(self.dictionary.keys())

    def _load_ngram_index(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_exact_matches(self, ngram):
        return self.ngram_index.get(ngram.lower(), [])

    def get_fuzzy_matches(self, ngram, top_n=10, similarity_threshold=0.4):
        results = []
        try:
            neighbors = self.fasttext_model.wv.similar_by_word(ngram, topn=top_n)
            for word, score in neighbors:
                if word in self.known_words and score >= similarity_threshold:
                    results.append(word)
        except KeyError:
            pass  # ngram not in FastText vocab
        return results

    def get_candidates(self, ngram, top_n=10, similarity_threshold=0.4, include_context=True):
        exact = set(self.get_exact_matches(ngram))
        fuzzy = set(self.get_fuzzy_matches(ngram, top_n, similarity_threshold))
        all_matches = sorted(exact | fuzzy)

        if not include_context:
            return all_matches

        return [
            {
                "word": self.dictionary[word]["word"],
                "normalized": word,
                "definition": self.dictionary[word].get("definition", ""),
                "citations": self.dictionary[word].get("key_citations", [])
            }
            for word in all_matches
        ]
