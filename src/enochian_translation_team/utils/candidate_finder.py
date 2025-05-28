import sqlite3
from gensim.models import FastText

class MorphemeCandidateFinder:
    def __init__(self, ngram_db_path, fasttext_model_path, dictionary_entries):
        self.conn = sqlite3.connect(ngram_db_path)
        self.cursor = self.conn.cursor()
        self.fasttext_model = FastText.load(str(fasttext_model_path))

        self.dictionary = {
            entry["normalized"].lower(): entry
            for entry in dictionary_entries
            if entry.get("normalized")
        }
        self.known_words = set(self.dictionary.keys())

    def get_exact_matches(self, ngram):
        """Query SQLite for variant/canonical pairs containing this n-gram."""
        query = """
            SELECT DISTINCT canonical
            FROM ngrams
            WHERE ngram = ?
        """
        self.cursor.execute(query, (ngram.lower(),))
        return [row[0].lower() for row in self.cursor.fetchall()]

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

        # Build enriched candidate objects
        return [
            {
                "word": self.dictionary[word]["word"],
                "normalized": word,
                "definition": self.dictionary[word].get("definition", ""),
                "citations": self.dictionary[word].get("key_citations", []),
            }
            for word in all_matches if word in self.dictionary
        ]

    def close(self):
        self.conn.close()
