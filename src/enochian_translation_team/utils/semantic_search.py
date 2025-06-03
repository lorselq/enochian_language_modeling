import numpy as np
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from enochian_translation_team.utils.variant_utils import generate_variants


def normalize_form(word):
    return word.lower()


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


def build_enhanced_definition(def_entry):
    base_def = def_entry.get("definition", "").strip()

    usage_examples = [
        c.get("context", "").strip()
        for c in def_entry.get("key_citations", [])
        if c.get("context")
    ]

    if usage_examples:
        formatted_usages = ", ".join(f"`{ex}`" for ex in usage_examples)
        usage_snippet = f" Usage: {formatted_usages}"
    else:
        usage_snippet = ""

    return f"{base_def.lower()}.{usage_snippet.lower()}"


def cluster_definitions(definitions, model, threshold=0.35):
    pairs = [
        (build_enhanced_definition(d), d) for d in definitions if d.get("definition")
    ]
    if len(pairs) < 2:
        return [definitions]

    texts, original_entries = zip(*pairs)

    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    distance_matrix = cosine_distances(embeddings)

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None,
    ).fit(distance_matrix)

    clusters = [[] for _ in range(max(clustering.labels_) + 1)]
    for item, label in zip(original_entries, clustering.labels_):
        clusters[label].append(item)
    return clusters


def find_semantically_similar_words(
    ft_model,
    sentence_model,
    entries,
    target_word,
    subst_map,
    fasttext_weight=0.50,
    definition_weight=0.50,
    min_similarity=0.05,
):
    # 🔒 Hard fix for malformed subst_map
    if isinstance(subst_map, list):
        try:
            subst_map = {
                entry["key"]: entry
                for entry in subst_map
                if isinstance(entry, dict) and "key" in entry and "alternates" in entry
            }
        except Exception as e:
            raise ValueError(
                f"Malformed substitution map passed to semantic search: {subst_map[:2]}"
            ) from e
    
    normalized_query = normalize_form(target_word)
    variants_raw = generate_variants(normalized_query, subst_map, return_subst_meta=True)
    variants = [v[0] for v in variants_raw]

    target_entry = next(
        (e for e in entries if normalize_form(e["normalized"]) == normalized_query),
        None,
    )
    if not target_entry:
        return []

    results = []
    for entry in tqdm(entries, desc="Processing dictionary entries"):
        cand_norm = normalize_form(entry["normalized"])
        if (
            cand_norm == normalized_query
            and entry["word"].lower() != target_word.lower()
        ):
            continue

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

        def_score = definition_similarity(
            build_enhanced_definition(target_entry),
            build_enhanced_definition(entry),
            sentence_model,
        )

        final_score = (fasttext_weight * ft_score) + (definition_weight * def_score)

        if cand_norm.startswith(normalized_query) or cand_norm.endswith(
            normalized_query
        ):
            final_score += 0.30

        if cand_norm.startswith(normalized_query) or cand_norm.endswith(
            normalized_query
        ):
            priority = 2
        elif normalized_query in cand_norm:
            priority = 1
        else:
            priority = 0

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

        definition_text = entry.get(
            "definition", ""
        ).lower()  # this is purposefully not "enhanced_definition"
        if (
            "enochian letter" in definition_text
            or "enochian word" in definition_text
            or "aethyr" in definition_text
        ):
            continue

        results.append(
            {
                "word": entry["word"],
                "normalized": entry["normalized"],
                "definition": entry.get("definition", ""),
                "enhanced_definition": entry.get("enhanced_definition", ""),
                "fasttext": round(ft_score, 3),
                "semantic": round(def_score, 3),
                "score": round(final_score, 3),
                "priority": priority,
                "tier": tier,
                "levenshtein": levenshtein_distance(normalized_query, cand_norm),
                "citations": entry.get("key_citations", []),
            }
        )

    # Calculate cluster similarity for scoring
    if len(results) > 1:
        definitions = [r["enhanced_definition"] for r in results]
        embeddings = sentence_model.encode(definitions, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)

        for i, entry in tqdm(enumerate(results), "Calculating cluster similarity"):
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

    return results

    # clusters = cluster_definitions(results, sentence_model)

    # print(f"[Debug] Created {len(clusters)} semantic clusters for '{target_word}'.")

    # return clusters
