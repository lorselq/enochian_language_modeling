import logging
import math
import sys
import time
from collections import OrderedDict

import numpy as np
import networkx as nx
import torch
from typing import List, Optional
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import util
from skfuzzy.cluster import cmeans
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from enochian_lm.root_extraction.utils.variant_utils import generate_variants


def _entry_get(entry, field: str, default=None):
    if isinstance(entry, dict):
        return entry.get(field, default)
    return getattr(entry, field, default)


def _entry_set(entry, field: str, value) -> None:
    if isinstance(entry, dict):
        entry[field] = value
    else:
        setattr(entry, field, value)


logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self._cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def lookup(self, key: tuple[str, str]) -> torch.Tensor | None:
        if key in self._cache:
            self.hits += 1
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        self.misses += 1
        return None

    def store(self, key: tuple[str, str], embedding: torch.Tensor) -> None:
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = embedding

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_ratio = (self.hits / total) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio,
            "size": len(self._cache),
            "capacity": self.max_size,
        }

    def log_stats(self, *, context: str | None = None) -> None:
        stats = self.stats()
        ctx = f" [{context}]" if context else ""
        logger.info(
            "Embedding cache stats%s: hits=%d misses=%d hit_ratio=%.2f%% size=%d/%d",
            ctx,
            stats["hits"],
            stats["misses"],
            stats["hit_ratio"] * 100,
            stats["size"],
            stats["capacity"],
        )


embedding_cache = EmbeddingCache()


def _batched_cached_embeddings(entries, sentence_model, *, context: str):
    embeddings: List[Optional[torch.Tensor]] = [None] * len(entries)
    to_encode_texts: List[str] = []
    to_encode_keys: List[tuple[str, str]] = []
    to_encode_indices: List[int] = []

    for idx, entry in enumerate(entries):
        key = (
            _entry_get(entry, "canonical", ""),
            _entry_get(entry, "enhanced_definition", ""),
        )
        cached = embedding_cache.lookup(key)
        if cached is not None:
            embeddings[idx] = cached
            continue
        to_encode_indices.append(idx)
        to_encode_keys.append(key)
        to_encode_texts.append(_entry_get(entry, "enhanced_definition", ""))

    if to_encode_texts:
        encoded_batch = sentence_model.encode(
            to_encode_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        if isinstance(encoded_batch, torch.Tensor):
            encoded_batch = encoded_batch.detach().cpu()
        else:
            encoded_batch = torch.as_tensor(encoded_batch)

        for idx, key, embedding in zip(
            to_encode_indices,
            to_encode_keys,
            encoded_batch,
        ):
            embedding_cpu = embedding.clone().detach()
            embedding_cache.store(key, embedding_cpu)
            embeddings[idx] = embedding_cpu

    # Ensure all embeddings were populated before returning.
    result = []
    for embedding in embeddings:
        if embedding is None:
            raise RuntimeError("Embedding cache returned an incomplete result set")
        result.append(embedding)

    embedding_cache.log_stats(context=context)

    return result


def normalize_form(word):
    return word.lower()


def stream_text(text: str, delay: float = 0.02):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            # if you really need to interrupt, break cleanly
            break


def definition_similarity(def1, def2, sentence_model):
    if not def1 or not def2:
        return 0.0
    emb1 = sentence_model.encode(def1, convert_to_tensor=True, show_progress_bar=False)
    emb2 = sentence_model.encode(def2, convert_to_tensor=True, show_progress_bar=False)
    return float(util.cos_sim(emb1, emb2))


def cluster_knn(defs, embeddings, k=5):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    # indices[i] is i plus its k nearest neighbors
    return [[defs[j] for j in idx_list] for idx_list in indices]


def cluster_dbscan(defs, embeddings, eps=0.15, min_samples=2):
    # metric='cosine' directly uses 1 - cosine_similarity
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(
        embeddings
    )
    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(defs[idx])
    return clusters  # note: label = -1 means “noise”


def compute_cluster_cohesion(definitions, sentence_model):
    if len(definitions) < 2:
        return 0.0
    embeddings = sentence_model.encode(definitions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    upper_triangle_scores = cosine_scores.triu(diagonal=1).flatten()
    relevant_scores = upper_triangle_scores[upper_triangle_scores != 0]
    return round(float(relevant_scores.mean()), 3) if len(relevant_scores) else 0.0


def build_enhanced_definition(def_entry):
    # Handle both object and dictionary input
    if isinstance(def_entry, dict):
        senses = def_entry.get("senses", [])
        key_citations = def_entry.get("key_citations", [])
    else:
        senses = getattr(def_entry, "senses", [])
        key_citations = getattr(def_entry, "key_citations", [])

    # quick fix
    if key_citations == None:
        key_citations = []

    base_def = (
        " ".join(
            (
                sense.get("definition", "")
                if isinstance(sense, dict)
                else sense.definition
            ).strip()
            for sense in senses
        )
        if senses
        else ""
    )

    usage_examples = []
    for cite in key_citations:
        if isinstance(cite, dict):
            context = cite.get("context", "")
        else:
            context = getattr(cite, "context", "")
        if context:
            usage_examples.append(context.strip())

    if usage_examples:
        formatted_usages = ", ".join([f"`{ex}`" for ex in usage_examples])
        usage_snippet = f" Usage: {formatted_usages}"
    else:
        usage_snippet = ""

    return f"{base_def.lower()}.{usage_snippet.lower()}"


def score_cluster_array(
    clusters: list[list],
    *,
    min_clusters: int = 3,
    max_clusters: int = 35,
    target_avg_low: float = 6.0,
    target_avg_high: float = 18.0,
    weight_count: float = 1.0,
    weight_avg: float = 1.3,
    weight_std: float = 0.4,
) -> float:
    sizes = np.array([len(c) for c in clusters], dtype=float)
    count = len(sizes)

    if count == 0:
        return float("inf")

    # penalize too few or too many clusters
    if count < min_clusters:
        pen_count = (min_clusters - count) / min_clusters
    elif count > max_clusters:
        pen_count = (count - max_clusters) / max_clusters
    else:
        pen_count = 0.0

    # avg size penalty
    avg_size = sizes.mean()
    if avg_size < target_avg_low:
        pen_avg = (target_avg_low - avg_size) / target_avg_low
    elif avg_size > target_avg_high:
        pen_avg = (avg_size - target_avg_high) / target_avg_high
    else:
        pen_avg = 0.0

    # dispersion penalty
    std_size = sizes.std()
    pen_std = std_size / avg_size if avg_size else std_size

    return weight_count * pen_count + weight_avg * pen_avg + weight_std * pen_std


def cluster_definitions(definitions, model):
    """
    Thin wrapper: preprocess then call tuned clustering.
    """
    # prepare texts & entries
    texts, original_entries = [], []
    for item in definitions:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = (
                item.get("enhanced_definition")
                or item.get("definition")
                or build_enhanced_definition(item)
            ).strip()
        else:
            text = build_enhanced_definition(item).strip()
        if text:
            texts.append(text)
            original_entries.append(item)

    if len(original_entries) < 2:
        return {"clusters": [definitions], "config": "[Error] original entries < 2; no config"}

    # get embeddings + distance matrix
    embeddings = (
        model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        .cpu()
        .numpy()
    )
    dist_matrix = cosine_distances(embeddings, embeddings)

    # auto-tune clusters
    return tuned_cluster_definitions(texts, original_entries, embeddings, dist_matrix)


def tuned_cluster_definitions(texts, original_entries, embeddings, dist_matrix):
    """
    Auto-tune multiple clustering methods by optimizing
    silhouette, Davies-Bouldin, Calinski-Harabasz and penalty,
    implemented via index-based clustering to ensure consistency.
    """
    N = len(texts)

    # 1) Hyperparameter grids
    configs = []
    for t in range(50, 81):
        configs.append(("agglomerative", {"threshold": t / 100}))
    for e in range(50, 81):
        configs.append(("dbscan", {"eps": e / 100, "min_samples": 2}))
    for g in range(50, 81):
        configs.append(("graph", {"threshold": g / 100}))
        configs.append(("ego", {"threshold": g / 100}))
    for k in range(3, min(N, 13)):
        configs.append(("knn", {"k": k}))
    for nc in range(3, 20):
        configs.append(("fuzzy", {"n_clusters": nc, "m": 2.0}))

    # print()
    # stream_text(
    #     f"We are about to evaluate {N} entries using {len(configs)} method/param combinations to identify the way to cluster them."
    # )
    # print()
    # time.sleep(0.6)

    best_score = float("inf")
    best_clusters_idx = []
    best_meta = (
        0,
        0,
        0,
        0,
        0,
    )  # fixes a Pylance error rather than assigning None to the var

    # 2) Evaluate each config
    for method, params in tqdm(configs, desc="Finding best configuration"):
        # print(f"[DEBUG] Trying {method.upper()} with {params}")
        clusters_idx = []

        if method == "agglomerative":
            labels = AgglomerativeClustering(
                metric="precomputed",
                linkage="complete",
                distance_threshold=params["threshold"],
                n_clusters=None,
            ).fit_predict(dist_matrix)
            for c in set(labels):
                clusters_idx.append([i for i, lbl in enumerate(labels) if lbl == c])

        elif method == "dbscan":
            labels = DBSCAN(
                eps=params["eps"],
                min_samples=params["min_samples"],
                metric="precomputed",
            ).fit_predict(dist_matrix)
            cmap = {}
            for i, lbl in enumerate(labels):
                cmap.setdefault(lbl, []).append(i)
            clusters_idx = list(cmap.values())

        elif method == "graph":
            G = nx.Graph()
            G.add_nodes_from(range(N))
            thr = params["threshold"]
            ix, jx = np.where((dist_matrix < thr) & (dist_matrix > 0))
            G.add_edges_from((i, j) for i, j in zip(ix, jx) if i < j)
            clusters_idx = [list(comp) for comp in nx.connected_components(G)]

        elif method == "ego":
            G = nx.Graph()
            G.add_nodes_from(range(N))
            thr = params["threshold"]
            ix, jx = np.where((dist_matrix < thr) & (dist_matrix > 0))
            G.add_edges_from((i, j) for i, j in zip(ix, jx) if i < j)
            clusters_idx = [[i] + list(G.neighbors(i)) for i in G.nodes]

        elif method == "knn":
            nbrs = NearestNeighbors(n_neighbors=params["k"] + 1, metric="cosine").fit(
                embeddings
            )
            _, inds = nbrs.kneighbors(embeddings)
            clusters_idx = [list(row) for row in inds]

        else:  # fuzzy
            data = embeddings.T
            cntr, u, *_ = cmeans(
                data,
                c=params["n_clusters"],
                m=params.get("m", 2.0),
                error=1e-6,
                maxiter=1000,
            )
            labels = np.argmax(u, axis=0)
            for j in range(params["n_clusters"]):
                clusters_idx.append([i for i, lbl in enumerate(labels) if lbl == j])

        # print(f"[DEBUG]   raw clusters idx: {[len(c) for c in clusters_idx]}")

        # 3) Filter trivial clusters
        clusters_idx = [c for c in clusters_idx if 2 <= len(c) <= N]
        if not clusters_idx:
            # print(f"[DEBUG]   no clusters remain after filtering")
            continue
        # print(f"[DEBUG]   filtered → {len(clusters_idx)} clusters idx")

        # 4) Build flat label array
        flat = [-1] * N
        for cid, cl in enumerate(clusters_idx):
            for idx in cl:
                flat[idx] = cid
        unique_labels = set(flat) - {-1}
        if len(unique_labels) < 2:
            # print(f"[DEBUG]   only {len(unique_labels)} labels, skipping")
            continue
        # print(f"[DEBUG]   labels unique: {unique_labels}")

        # 5) Compute metrics
        sil = silhouette_score(dist_matrix, flat, metric="precomputed")
        db = davies_bouldin_score(embeddings, flat)
        ch = calinski_harabasz_score(embeddings, flat)
        penalty = score_cluster_array(
            [[original_entries[i] for i in c] for c in clusters_idx]
        )
        # print(
        #     f"[DEBUG]   scores → sil={sil:.3f}, db={db:.3f}, ch={ch:.1f}, pen={penalty:.3f}"
        # )
        combo = -sil + db - 0.1 * math.log10(ch + 1) + penalty
        if combo < best_score:
            best_score = combo
            best_clusters_idx = clusters_idx
            best_meta = (method, params, sil, db, ch)
            # print(f"[DEBUG]   🎉 New best! combo={combo:.3f}")

    # 6) Map back to entries and return
    best_clusters = [[original_entries[i] for i in cl] for cl in best_clusters_idx]
    best_config = f"{best_meta[0]} {best_meta[1]} (sil={best_meta[2]:.3f}, db={best_meta[3]:.3f}, ch={best_meta[4]:.1f}, score={best_score:.3f})"
    stream_text(
        f"🏆 Final best config: {best_config}\n"
    )
    print()
    # time.sleep(1)
    result = {"clusters": best_clusters, "config": best_config}
    return result


def find_semantically_similar_words(
    ft_model,
    sentence_model,
    entries,
    target_word,
    subst_map,
    fasttext_weight=0.40,
    definition_weight=0.60,
    min_similarity=0.05,
):
    # Hard fix for malformed subst_map
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

    variants_raw = generate_variants(
        normalized_query, subst_map, return_subst_meta=True
    )
    variants = [v[0] for v in variants_raw]

    filtered_entries = []
    for e in entries:
        canon_text = normalize_form(_entry_get(e, "canonical", ""))
        if normalized_query in canon_text or any(v in canon_text for v in variants):
            filtered_entries.append(e)

    entries = filtered_entries

    for e in entries:
        _entry_set(e, "enhanced_definition", build_enhanced_definition(e))

    all_roots = [normalized_query] + variants
    index_entries = [
        e
        for e in entries
        if any(root in normalize_form(_entry_get(e, "canonical", "")) for root in all_roots)
    ]
    if not index_entries:
        return []

    root_embeddings_list = _batched_cached_embeddings(
        index_entries, sentence_model, context="find_semantically_similar_words:index"
    )
    entry_embeddings_list = _batched_cached_embeddings(
        entries, sentence_model, context="find_semantically_similar_words:entries"
    )

    root_embeddings = torch.stack(root_embeddings_list)
    entry_embeddings = torch.stack(entry_embeddings_list)

    cosine_matrix = util.cos_sim(entry_embeddings, root_embeddings)
    enh_def_scores = cosine_matrix.max(dim=1).values.cpu().tolist()

    results = []

    for i, entry in enumerate(
        tqdm(entries, desc="Processing relevant dictionary entries")
    ):
        cand_norm = normalize_form(_entry_get(entry, "canonical", ""))
        if (
            cand_norm == normalized_query
            and _entry_get(entry, "canonical", "").lower() != target_word.lower()
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

        total_score = sum(enh_def_scores)
        count_score = len(enh_def_scores)
        def_score = min((total_score / count_score) * 100, 100)
        final_score = (fasttext_weight * ft_score) + (definition_weight * def_score)

        if cand_norm.startswith(normalized_query) or cand_norm.endswith(
            normalized_query
        ):
            final_score += 0.60

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

        senses = _entry_get(entry, "senses") or []
        if senses:
            first_sense = senses[0]
            if isinstance(first_sense, dict):
                primary_definition = first_sense.get("definition", "")
            else:
                primary_definition = getattr(first_sense, "definition", "")
        else:
            primary_definition = ""
        definition_text = primary_definition.lower()
        if (
            "enochian letter" in definition_text
            or "enochian word" in definition_text
            or "aethyr" in definition_text
        ):
            continue

        results.append(
            {
                "word": _entry_get(entry, "canonical", ""),
                "normalized": _entry_get(entry, "canonical", "").lower(),
                "definition": primary_definition,
                "enhanced_definition": build_enhanced_definition(entry),
                "fasttext": round(ft_score, 3),
                "semantic": round(def_score, 3),
                "score": round(final_score, 3),
                "priority": priority,
                "tier": tier,
                "levenshtein": levenshtein_distance(normalized_query, cand_norm),
                "citations": _entry_get(entry, "key_citations", []),
            }
        )

    # Calculate cluster similarity for scoring
    if len(results) > 1:
        definitions = [r["enhanced_definition"] for r in results]
        embeddings = sentence_model.encode(
            definitions, convert_to_tensor=True, show_progress_bar=False
        )
        cosine_scores = util.cos_sim(embeddings, embeddings)

        for i, entry in tqdm(
            enumerate(results),
            f"Calculating definition similarity across all relevant words (up to {len(entries)})",
        ):
            sims = []
            for j in range(len(results)):
                if j != i:
                    try:
                        sims.append(cosine_scores[i][j].item())
                    except IndexError as e:
                        raise
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
