import re, json
from enochian_common.sqlite_bootstrap import sqlite3
import numpy as np
from typing import List, Optional, Tuple

MAX_WORDS, MAX_CHARS = 10, 90
_STOP_FUNC = {
    "a","an","the","and","or","but","if","then","when","while","where","which","that","this","these","those",
    "of","for","to","in","on","at","as","by","with","from","into","onto","over","under","about","across","through",
    "is","are","was","were","be","being","been","am","do","does","did","have","has","had","not","no","nor","also",
    "even","often","generally","typically","usually","may","might","must","shall","should","will","would","can","could"
}
# Morphological boilerplate / meta you don’t want in labels
_STOP_META = {
    "root","prefix","prefixal","suffix","morpheme","morphology","morphological","base","stem",
    "term","terms","word","words","entry","entries","sense","senses","definition","definitions","meaning","means",
    "compound","compounds","context","contexts","example","examples","guidance","decoding","note","notes","hint","hints"
}
# Common LLM explanation verbs you don’t want
_STOP_VERBS = {
    "denotes","denote","denoting","indicates","indicating","conveys","convey","imparts","impart",
    "provides","provide","anchors","anchor","yields","yield","refines","refine","specifies","specify",
    "prioritize","prioritizes","analyze","analyzes","encountering","encounter"
}
STOPWORDS = _STOP_FUNC | _STOP_META | _STOP_VERBS


def _strip_markdown(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)   # **bold**
    s = re.sub(r"^#+\s*", "", s, flags=re.M)   # headings
    s = re.sub(r"`([^`]+)`", r"\1", s)         # `code`
    return s

def _drop_guidance(s: str) -> str:
    return re.split(r"(?im)^\s*(?:guidance|decoding)\b.*$", s)[0].strip()

def _first_sentence(s: str) -> str:
    m = re.search(r":\s*([^\n]+)", s)
    if m: return m.group(1).strip()
    for line in s.splitlines():
        t = line.strip(" -•*✅☑️:\t")
        if t: return t
    return s.strip()

def _rm_parens(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    return s

def _tokens_keep(s: str) -> List[str]:
    # keep hyphenated tokens (load-bearing), apostrophes, slashes
    # split on punctuation but preserve hyphenated words as one token
    s = re.sub(r"[.;:–—,]", " ", s)
    toks = [t.lower().replace("’","'") for t in re.findall(r"[a-z][a-z\-'/]*", s)]
    out, seen = [], set()
    for t in toks:
        if t in STOPWORDS:         # drop boilerplate
            continue
        if len(t) < 3:             # drop very short bits (e.g., “of”, “to” already covered)
            continue
        if t.endswith("'s"):       # drop possessive
            t = t[:-2]
            if len(t) < 3: 
                continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _cosine_sim(u, v) -> float:
    return float(np.dot(u, v))


def _cluster_by_threshold(embs: np.ndarray, tau: float) -> List[List[int]]:
    n = len(embs)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_sim(embs[i], embs[j]) >= tau:
                adj[i].append(j)
                adj[j].append(i)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            k = stack.pop()
            comp.append(k)
            for nb in adj[k]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        groups.append(sorted(comp))
    return groups


def _cluster_by_jaccard(texts: List[str], tau: float = 0.5) -> List[List[int]]:
    def toks(s):
        return set(re.findall(r"[a-z]+", s.lower()))

    T = [toks(t) for t in texts]
    n = len(texts)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(T[i] & T[j])
            union = len(T[i] | T[j]) or 1
            if inter / union >= tau:
                adj[i].append(j)
                adj[j].append(i)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            k = stack.pop()
            comp.append(k)
            for nb in adj[k]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        groups.append(sorted(comp))
    return groups


def _medoid_idx(embs: np.ndarray, idxs: List[int]) -> int:
    if len(idxs) == 1:
        return 0
    sims = np.dot(embs[idxs], embs[idxs].T)
    dists = 1.0 - sims
    return int(np.argmin(dists.sum(axis=1)))


def _top_keywords(texts: List[str], k: int = 5) -> List[str]:
    from collections import Counter

    toks = []
    for t in texts:
        toks += [w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOPWORDS]
    return [w for w, _ in Counter(toks).most_common(k)]


def reduce_glossator(gloss_text: str, fallback_terms: Optional[List[str]]=None
                     ) -> Tuple[str, str]:
    """
    Deterministic reduction to ≤10 whole words (no stemming).
    Returns (label, notes_json). notes.needs_review=true only on fallbacks.
    """
    notes = {"reducer":"v4","steps":[],"needs_review": False}
    if not gloss_text or not gloss_text.strip():
        notes["steps"].append("empty"); notes["needs_review"] = True
        return "unknown", json.dumps(notes)

    raw = _strip_markdown(gloss_text); notes["steps"].append("strip_md")
    raw = _drop_guidance(raw);        notes["steps"].append("drop_guidance")
    raw = _rm_parens(raw);            notes["steps"].append("rm_parens")
    sent = _first_sentence(raw);      notes["steps"].append("first_sentence")

    toks = _tokens_keep(sent);        notes["steps"].append(f"tokens:{len(toks)}")

    label = None
    if toks:
        label = " ".join(toks[:MAX_WORDS]).strip(" /-")
    elif fallback_terms:
        # fallback: use top keywords (already filtered upstream) if supplied
        ftoks = [t.lower() for t in fallback_terms if t and t.lower() not in STOPWORDS]
        label = " ".join(ftoks[:MAX_WORDS]); notes["steps"].append("fallback_terms"); notes["needs_review"] = True

    if not label:
        # absolute fallback: first 2 alphabetic words from raw
        raw_toks = re.findall(r"[A-Za-z]{3,}", raw)
        label = " ".join(w.lower() for w in raw_toks[:2]) if raw_toks else "unknown"
        notes["steps"].append("fallback_raw2"); notes["needs_review"] = True

    # final tidy/sanitize
    label = re.sub(r"\s*/\s*", "/", label)
    label = re.sub(r"\s+", " ", label).strip()
    if len(label) > MAX_CHARS:
        label = label[:MAX_CHARS].rsplit(" ", 1)[0].strip(); notes["steps"].append("trim_chars")
    if not re.fullmatch(r"[a-z0-9/ '\-]+", label):
        safe = re.findall(r"[a-z0-9\-]+", label)
        label = " ".join(safe[:MAX_WORDS]) if safe else "unknown"
        notes["steps"].append("sanitize"); notes["needs_review"] = True

    return label or "unknown", json.dumps(notes)


def consolidate_ngram_senses(
    conn: sqlite3.Connection,
    ngram: str,
    *,
    sentence_model=None,
    sim_threshold: float = 0.80,
    jaccard_threshold: float = 0.50,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT cluster_id, glossator_def
        FROM clusters
        WHERE ngram = ?
        AND TRIM(COALESCE(glossator_def, '')) <> ''
        ORDER BY cluster_id ASC
    """,
        (ngram.upper(),),
    )
    rows = cur.fetchall()
    if not rows:
        return 0

    cluster_ids = [r[0] for r in rows]
    glosses = [r[1] for r in rows]

    if sentence_model is not None:
        embs = sentence_model.encode(glosses, normalize_embeddings=True).astype(
            np.float32
        )
        groups = _cluster_by_threshold(embs, sim_threshold)
    else:
        groups = _cluster_by_jaccard(glosses, tau=jaccard_threshold)

    written = 0
    for g in groups:
        g_cluster_ids = [cluster_ids[i] for i in g]
        # union all member def_ids from grouped clusters
        qmarks = ",".join("?" * len(g_cluster_ids))
        cur.execute(
            f"SELECT def_id FROM raw_defs WHERE cluster_id IN ({qmarks})", g_cluster_ids
        )
        member_ids = [r[0] for r in cur.fetchall()]

        # choose representative
        if sentence_model is not None:
            rep_local = _medoid_idx(embs, g)
            rep_idx = g[rep_local]
        else:
            rep_idx = g[0]

        rep_gloss = glosses[rep_idx]
        # fallback terms = top keywords across the group (helps if rep is messy)
        fallback_terms = _top_keywords([glosses[i] for i in g], k=MAX_WORDS) or None
        synth_def, notes = reduce_glossator(
            rep_gloss,
            fallback_terms=_top_keywords([glosses[i] for i in g], k=MAX_WORDS),
        )

        method_meta = json.dumps(
            {
                "ngram": ngram.upper(),
                "strategy": (
                    "embed+threshold"
                    if sentence_model is not None
                    else "lexical+jaccard"
                ),
                "threshold": (
                    sim_threshold if sentence_model is not None else jaccard_threshold
                ),
                "group_size": len(g),
                "rep_cluster_id": cluster_ids[rep_idx],
            }
        )

        cur.execute(
            """INSERT INTO synth_defs (ngram, cluster_id, synth_def, members, method_meta, notes)
            VALUES (?,?,?,?,?,?)""",
            (
                ngram.upper(),
                cluster_ids[rep_idx],
                synth_def,
                json.dumps(member_ids),
                method_meta,
                notes,
            ),
        )
        written += 1

    conn.commit()
    return written
