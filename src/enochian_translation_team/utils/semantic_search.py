from itertools import product, combinations
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import util

def normalize_form(word):
    return word.lower()

def generate_variants(word, subst_map, max_subs=2):
    word = word.lower()
    variants = set()
    variants.add(word)
    for positions in combinations(range(len(word)), max_subs):
        for replacements in product(*[subst_map.get(word[i], [word[i]]) for i in positions]):
            temp = list(word)
            for idx, sub in zip(positions, replacements):
                temp[idx] = sub
            variants.add("".join(temp))
    return list(variants)

def definition_similarity(def1, def2, sentence_model):
    if not def1 or not def2:
        return 0.0
    emb1 = sentence_model.encode(def1, convert_to_tensor=True)
    emb2 = sentence_model.encode(def2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def compute_cluster_cohesion(definitions, sentence_model):
    from itertools import combinations
    if len(definitions) < 2:
        return 0.0
    pairs = list(combinations(definitions, 2))
    sims = []
    for a, b in pairs:
        sims.append(definition_similarity(a, b, sentence_model))
    return round(sum(sims) / len(sims), 3)

def find_semantically_similar_words(ft_model, sent_model, entries, target_word, subst_map, topn=10,
                                    fasttext_weight=0.53, definition_weight=0.47, min_similarity=0.0):
    normalized_query = normalize_form(target_word)
    variants = generate_variants(normalized_query, subst_map)

    target_entry = next((e for e in entries if normalize_form(e["normalized"]) == normalized_query), None)
    if not target_entry:
        return []

    results = []
    for entry in entries:
        cand_norm = normalize_form(entry["normalized"])
        if cand_norm == normalized_query:
            continue

        # FastText similarity (max across variants)
        ft_score = 0.0
        if cand_norm in ft_model.wv:
            ft_score = max(
                (ft_model.wv.similarity(v, cand_norm) for v in variants if v in ft_model.wv),
                default=0.0
            )

        # Semantic similarity
        def_score = definition_similarity(
            target_entry.get("definition", ""),
            entry.get("definition", ""),
            sent_model
        )

        final_score = (fasttext_weight * ft_score) + (definition_weight * def_score)

        if cand_norm.startswith(normalized_query) or cand_norm.endswith(normalized_query):
            final_score += 0.11

        if cand_norm.startswith(normalized_query) or cand_norm.endswith(normalized_query):
            priority = 2  # Top tier: starts or ends with the root
        elif normalized_query in cand_norm:
            priority = 1  # Middle tier: contains root
        else:
            priority = 0  # Lowest: no apparent connection
        
        if priority == 2 and final_score > 0.85:
            tier = "🔥 Very Strong"
        elif priority == 2:
            tier = "✅ Strong"
        elif priority == 1:
            tier = "🤷 Possible"
        else:
            tier = "❌ Weak"
        
        if final_score < min_similarity:
            continue

        results.append({
            "word": entry["word"],
            "normalized": entry["normalized"],
            "definition": entry.get("definition", ""),
            "fasttext": round(ft_score, 3),
            "semantic": round(def_score, 3),
            "score": round(final_score, 3),
            "priority": priority,
            "tier": tier,
            "levenshtein": levenshtein_distance(normalized_query, cand_norm),
            "citations": entry.get("key_citations", [])
        })

    results.sort(key=lambda x: (-x["priority"], -x["score"]))
    return results[:topn]
