#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build ngram_index.sqlite ("sidecar") with dictionary canonicals as ground truth.

Key design:
- Dictionary canonicals are NEVER renormalized. We only strip to letters and lowercase for storage.
- Substitution/sequence rules are used ONLY to generate plausible surface variants (canonical -> variants)
  to reconcile corpus tokens back to their canonicals.
- DF and ngram_membership are computed from dictionary types; TF comes from the corpus (after redirects).

Creates tables:
- ngrams(ngram TEXT PRIMARY KEY, ngram_length INT, total_occurrences INT, total_containing_words INT)
- morphs(morph_text TEXT PRIMARY KEY, morph_length INT, is_enabled INT, part_of_speech_probability_json TEXT, morphotactic_role TEXT, position_probability_json TEXT, ontology_json TEXT)
- morph_stats(morph TEXT PRIMARY KEY, freq INT, marginal_probability REAL, left_entropy REAL, right_entropy REAL, position_counts_json TEXT)
- morph_adjacency(prev_morph TEXT, next_morph TEXT, adjacency_count INT, p_right_given_left REAL, pmi REAL, PRIMARY KEY(prev_morph,next_morph))
- allomorphs(morph_a TEXT, morph_b TEXT, allomorph_score REAL, allomorph_type TEXT, evidence_json TEXT, PRIMARY KEY(morph_a,morph_b))
- ontology_compat(type_left TEXT, type_right TEXT, compatibility_weight REAL, PRIMARY KEY(type_left,type_right))
- seg_cache(canonical TEXT PRIMARY KEY, best_segmentation_json TEXT, segmentation_class TEXT, scores_json TEXT, kbest_segmentations_json TEXT)
- ngram_membership(ngram TEXT, canonical TEXT, start_positions_json TEXT, PRIMARY KEY(ngram,canonical))
  + VIEW ngram_member_of(ngram, canonical)

Usage:
  python build_ngram_sidecar.py \
    --db ngram_index.sqlite \
    --keys enochian_keys.txt \
    --dict dictionary.json \
    --subst substitution_map.json \
    --compress sequence_compressions.json \
    --min_n 1 --max_n 6 \
    --variant_map variant_redirects.json  # optional
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

# ---------------------- knobs ----------------------
ALPHABET = set("ABCDEFGHIKLMNOPQRSTUVXYZ")  # manuscript alphabet; stored lowercase later
SMOOTH_K = 0.5
DEFAULT_POS_PRIORS = {"V": 0.34, "N": 0.33, "PART": 0.33}
DEFAULT_POSITIONS = {"initial": 0.34, "medial": 0.33, "final": 0.33}
APPROVED_BIGRAMS: Set[str] = set()  # e.g., {"de","bu","he","ca"} to bias greedy segmentation
# ---------------------------------------------------

# ---------------------- IO helpers ----------------------
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def letters_only_upper(s: str) -> str:
    """Keep only A..Z (Enochian alphabet here), uppercase."""
    return "".join(ch for ch in s.upper() if ch in ALPHABET)

def split_paragraphs(raw: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")) if p.strip()]

def tokenize_para(p: str, respect_pauses: bool) -> List[str]:
    HARD_BOUNDARY = {".", "?", "!", ";"}
    PAUSE_PUNCT = {",", ":", "—", "–", "-"}
    raw_bits = re.findall(r"[A-Za-z]+|[.,;:?!\"'()–—-]", p)
    out = []
    for t in raw_bits:
        if t in ('"', "'", "(", ")", "–", "—", "-"):
            continue
        if t in HARD_BOUNDARY:
            continue
        if respect_pauses and t in PAUSE_PUNCT:
            continue
        u = letters_only_upper(t)
        if u:
            out.append(u)
    return out
# ---------------------------------------------------------

# ------------- rules: load & interpret for variants -------------
def load_seq_rules(path: Path) -> List[Dict[str, Any]]:
    """Accept list of dicts, or { "rules": [...] }. Returns list[dict]."""
    if not path or not path.exists():
        return []
    try:
        data = read_json(path)
    except Exception:
        return []
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and isinstance(data.get("rules"), list):
        return [r for r in data["rules"] if isinstance(r, dict)]
    return []

def collect_letter_out_rules(
    subst_cfg: Dict[str, Any],
    allow_levels: Tuple[str, ...] = ("high", "medium", "low"),
) -> List[Tuple[str, str]]:
    """
    From substitution_map.json, collect canonical->alternate **letter** rules for variant generation.
    Only use alternates whose direction allows CANONICAL→ALT: i.e., 'from' or 'both'.
    Only single-letter alternates are used here (multi-letter belongs to sequence rules).
    """
    out: List[Tuple[str, str]] = []
    for _canon_key, spec in subst_cfg.items():
        canon = str(spec.get("canonical", "")).upper()
        if not canon:
            continue
        for alt in spec.get("alternates", []):
            val = str(alt.get("value", "")).upper()
            conf = str(alt.get("confidence", "low")).lower()
            dire = str(alt.get("direction", "to")).lower()
            if not val or conf not in allow_levels:
                continue
            if dire in ("from", "both") and len(canon) == 1 and len(val) == 1:
                out.append((canon, val))
    return out

def collect_seq_out_rules(seq_rules: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    From sequence_compressions.json, collect canonical->surface rules.
    Use entries whose 'direction' allows CANONICAL→ALT: 'to' or 'both'.
    Note: compressions are usually defined as FROM->TO (surface->canonical),
          so we invert for variant generation: canonical(TO) -> surface(FROM).
    """
    out: List[Tuple[str, str]] = []
    for r in seq_rules:
        dire = str(r.get("direction", "from")).lower()
        if dire in ("to", "both"):
            fr = str(r.get("from", "")).upper()
            to = str(r.get("to", "")).upper()
            if fr and to:
                out.append((to, fr))  # invert
    return out

def generate_variants_for_word(
    canon_lower: str,
    letter_out: List[Tuple[str, str]],
    seq_out: List[Tuple[str, str]],
    max_ops: int = 2,
    max_variants: int = 2000,
) -> Set[str]:
    """
    Generate plausible surface variants from a canonical by applying:
      - sequence-level canonical→surface expansions
      - letter-level canonical→alternate substitutions
    Bounded by max_ops and max_variants to prevent blowup.
    Returns lowercase strings.
    """
    base = canon_lower.upper()
    seen: Set[str] = {base}
    frontier: Set[str] = {base}
    ops = 0

    def apply_once(s: str) -> Set[str]:
        out: Set[str] = {s}
        # sequence (multi-char) first
        for (frm, to) in seq_out:
            if frm in s:
                out.add(s.replace(frm, to))
        # letter-level
        for (frm, to) in letter_out:
            if frm in s:
                out.add(s.replace(frm, to))
        return {x for x in out if x}

    while frontier and ops < max_ops and len(seen) < max_variants:
        nxt: Set[str] = set()
        for s in list(frontier):
            for t in apply_once(s):
                if t not in seen:
                    seen.add(t)
                    nxt.add(t)
                    if len(seen) >= max_variants:
                        break
            if len(seen) >= max_variants:
                break
        frontier = nxt
        ops += 1

    return {x.lower() for x in seen}
# ---------------------------------------------------------------

# ------------------------- analysis helpers -------------------------
def iter_ngrams(s: str, nmin: int, nmax: int) -> Iterable[str]:
    L = len(s)
    for n in range(nmin, min(nmax, L) + 1):
        for i in range(0, L - n + 1):
            yield s[i : i + n]

def find_positions(s: str, sub: str) -> List[int]:
    starts, i = [], 0
    while True:
        j = s.find(sub, i)
        if j == -1:
            break
        starts.append(j)
        i = j + 1  # allow overlaps
    return starts

def greedy_segment(token: str) -> List[str]:
    out, i, n = [], 0, len(token)
    while i < n:
        if i + 2 <= n and token[i : i + 2] in APPROVED_BIGRAMS:
            out.append(token[i : i + 2])
            i += 2
        else:
            out.append(token[i])
            i += 1
    return out

def entropy(counter: Dict[str, int]) -> float:
    total = sum(counter.values()) or 1
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)
# -------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS ngrams(
  ngram TEXT PRIMARY KEY,
  ngram_length INT,
  total_occurrences INT,
  total_containing_words INT
);

CREATE TABLE IF NOT EXISTS morphs(
  morph_text TEXT PRIMARY KEY,
  morph_length INT NOT NULL,
  is_enabled INT NOT NULL,
  part_of_speech_probability_json TEXT NOT NULL,
  morphotactic_role TEXT DEFAULT 'unknown',
  position_probability_json TEXT NOT NULL,
  ontology_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS morph_stats(
  morph TEXT PRIMARY KEY,
  freq INT NOT NULL,
  marginal_probability REAL NOT NULL,
  left_entropy REAL,
  right_entropy REAL,
  position_counts_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS morph_adjacency(
  prev_morph TEXT NOT NULL,
  next_morph TEXT NOT NULL,
  adjacency_count INT NOT NULL,
  p_right_given_left REAL,
  pmi REAL,
  PRIMARY KEY(prev_morph,next_morph)
);

CREATE TABLE IF NOT EXISTS allomorphs(
  morph_a TEXT NOT NULL,
  morph_b TEXT NOT NULL,
  allomorph_score REAL NOT NULL,
  allomorph_type TEXT,
  evidence_json TEXT NOT NULL,
  PRIMARY KEY(morph_a, morph_b)
);

CREATE TABLE IF NOT EXISTS ontology_compat(
  type_left TEXT NOT NULL,
  type_right TEXT NOT NULL,
  compatibility_weight REAL NOT NULL,
  PRIMARY KEY(type_left, type_right)
);

CREATE TABLE IF NOT EXISTS seg_cache(
  canonical TEXT PRIMARY KEY,
  best_segmentation_json TEXT NOT NULL,
  segmentation_class TEXT DEFAULT 'UNKNOWN',
  scores_json TEXT NOT NULL,
  kbest_segmentations_json TEXT
);

CREATE TABLE IF NOT EXISTS ngram_membership(
  ngram TEXT NOT NULL,
  canonical TEXT NOT NULL,
  start_positions_json TEXT NOT NULL,
  PRIMARY KEY(ngram, canonical)
);
CREATE INDEX IF NOT EXISTS idx_ngram_membership_canonical ON ngram_membership(canonical);
CREATE VIEW IF NOT EXISTS ngram_member_of AS
  SELECT ngram, canonical FROM ngram_membership;
"""

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()

# --------------------------- main build ---------------------------
def build_sidecar(
    db: Path,
    keys_txt: Path,
    dict_json: Path,
    subst_json: Path,
    compress_json: Path,
    min_n: int = 1,
    max_n: int = 6,
    respect_pauses: bool = True,
    variant_map_path: Path | None = None,
) -> None:

    # 1) Load authoritative dictionary canonicals (letters-only, lower)
    dictionary = read_json(dict_json)
    dict_types: Set[str] = set()
    for entry in dictionary:
        norm = (entry.get("normalized") or "").strip()
        if not norm:
            continue
        canon = letters_only_upper(norm).lower()  # NO rule-based mutation
        if canon:
            dict_types.add(canon)

    # 2) Load corpus tokens (letters-only, lower; NO rule-based normalization)
    raw = keys_txt.read_text(encoding="utf-8")
    tokens_raw: List[str] = []
    for p in split_paragraphs(raw):
        tokens_raw.extend(tokenize_para(p, respect_pauses=respect_pauses))
    corpus_tokens: List[str] = []
    for t in tokens_raw:
        u = letters_only_upper(t).lower()
        if u:
            corpus_tokens.append(u)
    corpus_types: Set[str] = set(corpus_tokens)

    # 3) Build variant index from dictionary canonicals using rules (canonical -> variants)
    subst_cfg: Dict[str, Any] = read_json(subst_json) if subst_json.exists() else {}
    seq_rules: List[Dict[str, Any]] = load_seq_rules(compress_json)

    letter_out = collect_letter_out_rules(subst_cfg, allow_levels=("high", "medium"))
    seq_out = collect_seq_out_rules(seq_rules)

    variant_index: Dict[str, Set[str]] = defaultdict(set)
    for canon in dict_types:
        variants = generate_variants_for_word(canon, letter_out, seq_out, max_ops=2)
        for v in variants:
            variant_index[v].add(canon)

    # 3b) Optional manual overrides (variant map): { "surface": "canonical" }
    manual_map: Dict[str, str] = {}
    if variant_map_path and variant_map_path.exists():
        try:
            manual_map = {k.lower(): v.lower() for k, v in read_json(variant_map_path).items()}
        except Exception:
            manual_map = {}

    # 4) Reconcile corpus types to dictionary canonicals
    redirects: Dict[str, str] = {}
    ambiguous: Dict[str, List[str]] = {}
    unresolved: List[str] = []

    for t in sorted(corpus_types):
        if t in dict_types:
            continue
        # manual override first
        if t in manual_map and manual_map[t] in dict_types:
            redirects[t] = manual_map[t]
            continue
        # auto variant mapping
        cands = sorted(variant_index.get(t, []))
        if len(cands) == 1:
            redirects[t] = cands[0]
        elif len(cands) > 1:
            ambiguous[t] = cands
        else:
            unresolved.append(t)

    # Apply redirects to corpus tokens and rebuild corpus types
    if redirects:
        corpus_tokens = [redirects.get(tok, tok) for tok in corpus_tokens]
        corpus_types = set(corpus_tokens)

    # Fail only on truly unresolved / ambiguous so you can fix source or add to variant map
    if unresolved or ambiguous:
        msgs = []
        if unresolved:
            msgs.append(
                "[VARIANT RECONCILIATION] Unresolved corpus types (no variant match in dictionary): "
                + ", ".join(unresolved[:50])
                + (" ..." if len(unresolved) > 50 else "")
            )
        if ambiguous:
            preview = [f"{k} -> {', '.join(v[:5])}" for k, v in list(ambiguous.items())[:10]]
            msgs.append("[VARIANT RECONCILIATION] Ambiguous mappings (choose one canonical):\n  " + "\n  ".join(preview))
        raise SystemExit("\n".join(msgs))

    # 5) n-gram counts: TF from corpus instances; DF from dictionary types
    tf_counts: Dict[str, int] = Counter()
    for tok in corpus_tokens:
        for ng in iter_ngrams(tok, min_n, max_n):
            tf_counts[ng] += 1

    df_sets: Dict[str, Set[str]] = defaultdict(set)
    for t in dict_types:
        for ng in iter_ngrams(t, min_n, max_n):
            df_sets[ng].add(t)
    df_counts: Dict[str, int] = {ng: len(s) for ng, s in df_sets.items()}

    all_ngrams = sorted(set(tf_counts.keys()) | set(df_counts.keys()))

    # 6) Segmentation and morph stats/adjacency
    # Segment ALL dictionary types for cache (classify as CORPUS or DICT_ONLY).
    segs_corpus: Dict[str, List[str]] = {}
    segs_dictonly: Dict[str, List[str]] = {}
    for tok in sorted(corpus_types):
        segs_corpus[tok] = greedy_segment(tok)
    for tok in sorted(dict_types - corpus_types):
        segs_dictonly[tok] = greedy_segment(tok)

    morph_freq = Counter()
    left_neighbors = defaultdict(Counter)
    right_neighbors = defaultdict(Counter)
    pos_counts = defaultdict(lambda: {"initial": 0, "medial": 0, "final": 0})
    adjacency = Counter()

    # counts only from CORPUS types
    for tok, seg in segs_corpus.items():
        for i, m in enumerate(seg):
            morph_freq[m] += 1
            if i == 0:
                pos_counts[m]["initial"] += 1
            if i == len(seg) - 1:
                pos_counts[m]["final"] += 1
            if 0 < i < len(seg) - 1:
                pos_counts[m]["medial"] += 1
            if i > 0:
                prev = seg[i - 1]
                right_neighbors[prev][m] += 1
                left_neighbors[m][prev] += 1
                adjacency[(prev, m)] += 1

    morph_inventory = sorted(
        {m for seg in list(segs_corpus.values()) + list(segs_dictonly.values()) for m in seg}
    )
    total_morphs = sum(morph_freq.values()) or 1
    p_marg = {m: (morph_freq[m] / total_morphs if total_morphs else 0.0) for m in morph_inventory}
    V = max(1, len(morph_inventory))

    # 7) Write DB
    conn = sqlite3.connect(str(db))
    init_db(conn)
    cur = conn.cursor()

    # ngrams (store lowercase)
    cur.executemany(
        "INSERT OR REPLACE INTO ngrams(ngram, ngram_length, total_occurrences, total_containing_words) VALUES (?,?,?,?)",
        [(ng, len(ng), int(tf_counts.get(ng, 0)), int(df_counts.get(ng, 0))) for ng in all_ngrams],
    )
    conn.commit()

    # ngram_membership: dictionary types only (authoritative)
    membership_rows = []
    for canonical in sorted(dict_types):
        for ng in iter_ngrams(canonical, min_n, max_n):
            starts = find_positions(canonical, ng)
            if starts:
                membership_rows.append((ng, canonical, json.dumps(starts)))
    if membership_rows:
        cur.executemany(
            "INSERT OR REPLACE INTO ngram_membership(ngram, canonical, start_positions_json) VALUES (?,?,?)",
            membership_rows,
        )
        conn.commit()

    # morphs
    cur.executemany(
        """
        INSERT OR REPLACE INTO morphs
            (morph_text, morph_length, is_enabled,
             part_of_speech_probability_json, morphotactic_role,
             position_probability_json, ontology_json)
        VALUES (?,?,?,?,?,?,?)
        """,
        [
            (
                m,
                len(m),
                1,
                json.dumps(DEFAULT_POS_PRIORS),
                "unknown",
                json.dumps(DEFAULT_POSITIONS),
                json.dumps([]),
            )
            for m in morph_inventory
        ],
    )
    conn.commit()

    # morph_stats
    def pos_json(m: str) -> Dict[str, int]:
        return pos_counts[m] if m in pos_counts else {"initial": 0, "medial": 0, "final": 0}

    cur.executemany(
        """
        INSERT OR REPLACE INTO morph_stats
            (morph, freq, marginal_probability, left_entropy, right_entropy, position_counts_json)
        VALUES (?,?,?,?,?,?)
        """,
        [
            (
                m,
                int(morph_freq.get(m, 0)),
                float(p_marg.get(m, 0.0)),
                float(entropy(left_neighbors[m])) if m in left_neighbors else 0.0,
                float(entropy(right_neighbors[m])) if m in right_neighbors else 0.0,
                json.dumps(pos_json(m)),
            )
            for m in morph_inventory
        ],
    )
    conn.commit()

    # morph_adjacency (+ PMI)
    adj_rows = []
    for (prev, nxt), c in adjacency.items():
        denom = sum(right_neighbors[prev].values()) + SMOOTH_K * V
        p_right = (c + SMOOTH_K) / denom if denom > 0 else 0.0
        p_xy = c / (total_morphs or 1)
        denom_pmi = (p_marg.get(prev, 1e-12) * p_marg.get(nxt, 1e-12))
        pmi_bits = math.log2(p_xy / denom_pmi) if (p_xy > 0 and denom_pmi > 0) else 0.0
        adj_rows.append((prev, nxt, int(c), float(p_right), float(pmi_bits)))
    if adj_rows:
        cur.executemany(
            "INSERT OR REPLACE INTO morph_adjacency(prev_morph, next_morph, adjacency_count, p_right_given_left, pmi) VALUES (?,?,?,?,?)",
            adj_rows,
        )
        conn.commit()

    # allomorphs (conservative; corpus evidence only)
    def js_div(p: Dict[str, int], q: Dict[str, int]) -> float:
        keys = set(p.keys()) | set(q.keys())
        tp = sum(p.values()) or 1
        tq = sum(q.values()) or 1
        P = {k: p.get(k, 0) / tp for k in keys}
        Q = {k: q.get(k, 0) / tq for k in keys}
        M = {k: 0.5 * (P[k] + Q[k]) for k in keys}
        def KL(A: Dict[str, float], B: Dict[str, float]) -> float:
            eps = 1e-12
            return sum(A[k] * math.log2((A[k] + eps) / (B[k] + eps)) for k in keys if A[k] > 0)
        return 0.5 * KL(P, M) + 0.5 * KL(Q, M)

    allo_rows = []
    bigrams = [m for m in morph_inventory if len(m) == 2]
    buckets: Dict[str, List[str]] = defaultdict(list)
    for bg in bigrams:
        buckets[bg[0]].append(bg)
    for initial, bunch in buckets.items():
        n = len(bunch)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = bunch[i], bunch[j]
                d = js_div(right_neighbors[a], right_neighbors[b])
                if d <= 0.15 and (morph_freq.get(a, 0) >= 2 and morph_freq.get(b, 0) >= 2):
                    score = round(1.0 - d, 3)
                    ev = {"metric": "JSdiv_right_neighbors", "value": round(d, 3)}
                    allo_rows.append((a, b, score, "CONTEXTUAL", json.dumps(ev)))
    if allo_rows:
        cur.executemany(
            "INSERT OR REPLACE INTO allomorphs(morph_a, morph_b, allomorph_score, allomorph_type, evidence_json) VALUES (?,?,?,?,?)",
            allo_rows,
        )
        conn.commit()

    # ontology_compat (placeholder seed; replace later with your real matrix)
    cur.executemany(
        "INSERT OR IGNORE INTO ontology_compat(type_left, type_right, compatibility_weight) VALUES (?,?,?)",
        [("RELATOR", "SPEECH", 0.6), ("SPEECH", "LIGHT", 0.8), ("LIGHT", "ASPECT:INGRESS", 0.9)],
    )
    conn.commit()

    # seg_cache
    for tok, seg in {**segs_corpus, **segs_dictonly}.items():
        cls = "CORPUS" if tok in segs_corpus else "DICT_ONLY"
        cur.execute(
            "INSERT OR REPLACE INTO seg_cache(canonical, best_segmentation_json, segmentation_class, scores_json, kbest_segmentations_json) VALUES (?,?,?,?,?)",
            (tok, json.dumps(seg), cls, json.dumps({"note": "greedy bootstrap"}), json.dumps([])),
        )
    conn.commit()
    conn.close()

    print(
        f"Built sidecar at {db} | ngrams: {len(all_ngrams)} | dict types: {len(dict_types)} | "
        f"corpus types: {len(segs_corpus)} | membership rows: {len(membership_rows)}"
    )

# ----------------------------- CLI -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("ngram_index.sqlite"))
    ap.add_argument("--keys", type=Path, default=Path("enochian_keys.txt"))
    ap.add_argument("--dict", type=Path, default=Path("dictionary.json"))
    ap.add_argument("--subst", type=Path, default=Path("substitution_map.json"))
    ap.add_argument("--compress", type=Path, default=Path("sequence_compressions.json"))
    ap.add_argument("--min_n", type=int, default=1)
    ap.add_argument("--max_n", type=int, default=6)
    ap.add_argument("--respect_pauses", type=int, default=1)
    ap.add_argument("--variant_map", type=Path, default=None)  # optional manual overrides
    args = ap.parse_args()

    build_sidecar(
        db=args.db,
        keys_txt=args.keys,
        dict_json=args.dict,
        subst_json=args.subst,
        compress_json=args.compress,
        min_n=args.min_n,
        max_n=args.max_n,
        respect_pauses=bool(args.respect_pauses),
        variant_map_path=args.variant_map,
    )

if __name__ == "__main__":
    main()
