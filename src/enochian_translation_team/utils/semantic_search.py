import numpy as np
from tqdm import tqdm
from itertools import product, combinations
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering


def normalize_form(word):
    return word.lower()


def generate_variants(word, subst_map, max_subs=2):
    word = word.lower()
    variants = set()
    variants.add(word)
    for positions in combinations(range(len(word)), max_subs):
        for replacements in product(
            *[subst_map.get(word[i], [word[i]]) for i in positions]
        ):
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
    if len(definitions) < 2:
        return 0.0
    embeddings = sentence_model.encode(definitions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    upper_triangle_scores = cosine_scores.triu(diagonal=1).flatten()
    relevant_scores = upper_triangle_scores[upper_triangle_scores != 0]
    return round(float(relevant_scores.mean()), 3) if len(relevant_scores) else 0.0


def cluster_definitions(entries, sentence_model, min_cluster_size=2, distance_threshold=0.35):
    definitions = [e["definition"] for e in entries if e.get("definition")]
    if len(definitions) < min_cluster_size:
        return [entries]  # One cluster only

    # Step 1: Get embeddings
    embeddings = sentence_model.encode(definitions)

    # Step 2: Cluster
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - distance_threshold,  # Cosine similarity to distance
        affinity='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # Step 3: Group entries by label
    clustered = {}
    for idx, label in enumerate(labels):
        clustered.setdefault(label, []).append(entries[idx])

    return list(clustered.values())


def find_semantically_similar_words(
    ft_model,
    sent_model,
    entries,
    target_word,
    subst_map,
    topn=10,
    fasttext_weight=0.40,
    definition_weight=0.60,
    min_similarity=0.0,
):
    normalized_query = normalize_form(target_word)
    variants = generate_variants(normalized_query, subst_map)

    target_entry = next(
        (e for e in entries if normalize_form(e["normalized"]) == normalized_query),
        None,
    )
    if not target_entry:
        return []

    results = []
    for entry in tqdm(entries, desc="Processing dictionary entries..."):
        cand_norm = normalize_form(entry["normalized"])
        if (
            cand_norm == normalized_query
            and entry["word"].lower() != target_word.lower()
        ):
            continue

        # FastText similarity (max across variants)
        ft_score = 0.0
        if cand_norm in ft_model.wv:
            ft_score = max(
                (
                    ft_model.wv.similarity(v, cand_norm)
                    for v in variants
                    if v in ft_model.wv
                ),
                default=0.0,
            )

        # Semantic similarity
        def_score = definition_similarity(
            target_entry.get("definition", ""), entry.get("definition", ""), sent_model
        )

        final_score = (fasttext_weight * ft_score) + (definition_weight * def_score)

        if cand_norm.startswith(normalized_query) or cand_norm.endswith(
            normalized_query
        ):
            final_score += 0.15

        if cand_norm.startswith(normalized_query) or cand_norm.endswith(
            normalized_query
        ):
            priority = 2  # Top tier: starts or ends with the root
        elif normalized_query in cand_norm:
            priority = 1  # Middle tier: contains root
        else:
            priority = 0  # Lowest: no apparent connection

        if priority == 2 and final_score > 0.85:
            tier = "Very strong connection"
        elif priority == 2:
            tier = "Possible connection"
        elif priority == 1:
            tier = "Somewhat possible connection"
        else:
            tier = "Weak to no connection"

        if final_score < min_similarity:
            continue

        # Filter out garbage entries that define letters or numbers, which is frankly just noise at this point
        definition_text = entry.get("definition", "").lower()
        if "enochian letter" in definition_text or "enochian word" in definition_text:
            continue

        results.append(
            {
                "word": entry["word"],
                "normalized": entry["normalized"],
                "definition": entry.get("definition", ""),
                "fasttext": round(ft_score, 3),
                "semantic": round(def_score, 3),
                "score": round(final_score, 3),
                "priority": priority,
                "tier": tier,
                "levenshtein": levenshtein_distance(normalized_query, cand_norm),
                "citations": entry.get("key_citations", []),
            }
        )

    # Semantic cluster coherence
    definitions = [r["definition"] for r in results]
    if len(definitions) > 1:
        embeddings = sent_model.encode(definitions, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)

        for i, entry in tqdm(enumerate(results), "Calculating cluster similarity..."):
            sims = [cosine_scores[i][j].item() for j in range(len(results)) if j != i]
            entry["cluster_similarity"] = sum(sims) / len(sims) if sims else 0.0
    else:
        for entry in results:
            entry["cluster_similarity"] = 0.0

    results.sort(
        key=lambda x: (
            round(x["semantic"], 3),
            round(x["fasttext"], 3),
            round(x["cluster_similarity"], 3),
            round(x["score"], 3),
        ),
        reverse=True,
    )

    print(f"[Debug] Cluster size for '{target_word}': {len(results)}")
    return results[:topn]
