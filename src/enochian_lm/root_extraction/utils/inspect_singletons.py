# utils/singletons.py
# ------------------------------------------------------------
# Singleton inference helpers (anchor-index, Δ-embeddings, persistence)
# ------------------------------------------------------------
# NOTE: not in use yet. This should be reviewed LATER once the solo analysis is done.
# ------------------------------------------------------------
from __future__ import annotations

import json
from enochian_lm.common.sqlite_bootstrap import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# -----------------------------
# Helper dataclasses
# -----------------------------
@dataclass
class Anchor:
  morph: str
  uses: int
  avg_cohesion: float
  any_gloss_json: Optional[str]  # JSON string or None

@dataclass
class SingletonCandidate:
  cluster_id: int
  source_word: str
  residual_fragment: str
  residual_ratio: float
  coverage_ratio: float
  candidate_root: str  # c.ngram for context


# -----------------------------
# Embedding helpers (thin)
# -----------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
  n = np.linalg.norm(v)
  return v / n if n > 0 else v

def _embed_texts(texts: List[str], sbert) -> Optional[np.ndarray]:
  texts = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
  if not texts:
    return None
  vecs = sbert.encode(texts, normalize_embeddings=True)
  vecs = np.array(vecs, dtype=np.float32)
  return _normalize(vecs.mean(axis=0))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
  if a is None or b is None:
    return 0.0
  return float(np.dot(_normalize(a), _normalize(b)))


# -----------------------------
# Phase B: Build anchor index
# -----------------------------
def build_anchor_index(conn: sqlite3.Connection,
                       min_uses: int = 3,
                       min_cohesion: float = 0.40) -> Dict[str, Anchor]:
  """
  Returns {morph -> Anchor} for accepted, stable morphemes.
  Expects `anchors` VIEW to exist (see ANCHORS_VIEW_SQL).
  """
  conn.row_factory = sqlite3.Row
  cur = conn.execute("""
    SELECT morph, uses, avg_cohesion, any_gloss_json
    FROM anchors
    WHERE uses >= ? AND avg_cohesion >= ?
    ORDER BY uses DESC, avg_cohesion DESC
  """, (min_uses, min_cohesion))
  out: Dict[str, Anchor] = {}
  for r in cur:
    out[r["morph"].upper()] = Anchor(
      morph=r["morph"].upper(),
      uses=int(r["uses"] or 0),
      avg_cohesion=float(r["avg_cohesion"] or 0.0),
      any_gloss_json=r["any_gloss_json"]
    )
  return out


# -----------------------------
# Fetch defs for embeddings
# -----------------------------
def _defs_for_word(conn: sqlite3.Connection, word_upper: str) -> List[str]:
  cur = conn.execute("""
    SELECT definition
    FROM raw_defs
    WHERE source_word = ?
      AND TRIM(COALESCE(definition,'')) <> ''
  """, (word_upper,))
  return [r[0] for r in cur.fetchall()]

def _defs_for_morph(conn: sqlite3.Connection, morph_upper: str) -> List[str]:
  # accepted defs of that n-gram via any cluster
  cur = conn.execute("""
    SELECT rd.definition
    FROM clusters c
    JOIN raw_defs rd ON rd.cluster_id = c.cluster_id
    WHERE c.ngram = ?
      AND TRIM(COALESCE(c.glossator_def,'')) <> ''
      AND TRIM(COALESCE(rd.definition,'')) <> ''
  """, (morph_upper,))
  rows = [r[0] for r in cur.fetchall()]
  if rows:
    return rows
  # fallback: try the gloss JSON text (if any) as a definition carrier
  cur2 = conn.execute("""
    SELECT glossator_def
    FROM clusters
    WHERE ngram = ?
      AND TRIM(COALESCE(glossator_def,'')) <> ''
    LIMIT 5
  """, (morph_upper,))
  gl = []
  for (g,) in cur2.fetchall():
    try:
      obj = json.loads(g)
      # common key you store: a single-line definition
      if isinstance(obj, dict):
        # try a few keys defensively
        for k in ("DEFINITION","Definition","def","gloss","GLOSS","glossator_def"):
          if k in obj and isinstance(obj[k], str):
            gl.append(obj[k])
    except Exception:
      pass
  return gl


# -----------------------------
# Phase C: Mine singleton candidates
# -----------------------------
def fetch_singleton_candidates(conn: sqlite3.Connection,
                               min_residual: float = 0.35,
                               max_uncovered: int = 1,
                               limit: int = 500) -> List[SingletonCandidate]:
  """
  Pull high-residue words with exactly `max_uncovered` uncovered fragments.
  """
  conn.row_factory = sqlite3.Row
  cur = conn.execute(f"""
    WITH rd AS (
      SELECT residual_id, cluster_id, normalized, definition,
             residual_ratio, coverage_ratio, uncovered_json
      FROM residual_details
      WHERE residual_ratio >= ?
    )
    SELECT
      c.cluster_id,
      rd.normalized AS source_word,
      json_extract(rd.uncovered_json, '$[0].text') AS frag,
      rd.residual_ratio,
      rd.coverage_ratio,
      c.ngram AS candidate_root
    FROM rd
    JOIN clusters c ON c.cluster_id = rd.cluster_id
    WHERE json_array_length(rd.uncovered_json) = ?
    ORDER BY rd.residual_ratio DESC
    LIMIT ?
  """, (min_residual, max_uncovered, limit))
  out: List[SingletonCandidate] = []
  for r in cur:
    frag = (r["frag"] or "").strip()
    if not frag:
      continue
    out.append(SingletonCandidate(
      cluster_id=int(r["cluster_id"]),
      source_word=str(r["source_word"]).upper(),
      residual_fragment=str(frag).upper(),
      residual_ratio=float(r["residual_ratio"] or 0.0),
      coverage_ratio=float(r["coverage_ratio"] or 0.0),
      candidate_root=str(r["candidate_root"]).upper()
    ))
  return out


def _choose_anchor_in_word(source_word: str,
                           anchor_index: Dict[str, Anchor]) -> Optional[str]:
  """
  Very simple heuristic: pick the longest anchor substring present in the word.
  You can replace this with segmentation-aware selection later.
  """
  word = source_word.upper()
  hits = [a for a in anchor_index.keys() if a in word]
  if not hits:
    return None
  # longest wins; tie-break by uses, then cohesion
  hits.sort(key=lambda a: (len(a), anchor_index[a].uses, anchor_index[a].avg_cohesion), reverse=True)
  return hits[0]


# -----------------------------
# Phase C: Singleton inference (Δ trick)
# -----------------------------
@dataclass
class SingletonHypothesis:
  morph: str
  source_word: str
  anchor: Optional[str]
  seed_glosses: List[str]
  proposed_gloss: str
  rationale: str
  delta_cosine: float
  residual_before: float
  residual_after: float
  accepted: int  # 0/1


def infer_singleton(conn: sqlite3.Connection,
                    sbert,                         # your sentence-transformer
                    anchor_index: Dict[str, Anchor],
                    candidate: SingletonCandidate,
                    sense_matrix: np.ndarray,      # precomputed (N x d) embeddings
                    sense_labels: List[str],       # N labels for seeds
                    topk: int = 5) -> Optional[SingletonHypothesis]:
  """
  Build a Δ-embedding: meaning(source_word) - meaning(anchor).
  NN over dictionary senses to get seed glosses. The *proposed_gloss* and
  *rationale* are expected to be filled by your LLM layer upstream—here we
  compute Δ and select seeds, returning a hypothesis shell you can enrich.
  """
  # 1) find anchor (skip if none)
  anchor = _choose_anchor_in_word(candidate.source_word, anchor_index)
  if not anchor:
    return None

  # 2) compute embeddings
  defs_w = _defs_for_word(conn, candidate.source_word)
  defs_a = _defs_for_morph(conn, anchor)
  v_w = _embed_texts(defs_w, sbert)
  v_a = _embed_texts(defs_a, sbert)
  if v_w is None or v_a is None:
    return None

  delta = _normalize(v_w - v_a)

  # 3) nearest neighbors over sense_matrix
  if sense_matrix is None or len(sense_matrix) == 0:
    seeds = []
    best = 0.0
  else:
    sims = (sense_matrix @ delta).astype(np.float32)  # cosine if rows are normalized
    idx = np.argsort(-sims)[:topk]
    seeds = [sense_labels[i] for i in idx]
    best = float(sims[idx[0]])

  # 4) return a partial hypothesis (LLM will fill gloss + rationale)
  return SingletonHypothesis(
    morph=candidate.residual_fragment,
    source_word=candidate.source_word,
    anchor=anchor,
    seed_glosses=seeds,
    proposed_gloss="",   # fill via LLM
    rationale="",        # fill via LLM
    delta_cosine=best,
    residual_before=candidate.residual_ratio,
    residual_after=candidate.residual_ratio,  # will be updated after eval
    accepted=0
  )


# -----------------------------
# Phase D: Evaluate + persist
# -----------------------------
def evaluate_and_persist(conn: sqlite3.Connection,
                         hypothesis: SingletonHypothesis,
                         recompute_residual_fn,
                         accept_delta: float = 0.10) -> int:
  """
  - recompute new residual_after via your segmentation/breakdown function
    (inject the proposed morph/gloss temporarily).
  - accept if residual drops by >= accept_delta.
  - persist to morph_hypotheses.
  Returns: 1 if accepted, else 0.
  """
  # recompute new residual ratio for the source word with the new morph
  new_residual = recompute_residual_fn(
    word=hypothesis.source_word,
    new_morph=hypothesis.morph,
    new_gloss=hypothesis.proposed_gloss
  )
  hypothesis.residual_after = float(new_residual or hypothesis.residual_before)

  delta = hypothesis.residual_before - hypothesis.residual_after
  hypothesis.accepted = 1 if delta >= accept_delta else 0

  conn.execute("""
    INSERT OR REPLACE INTO morph_hypotheses
      (morph, source_word, anchor, seed_glosses, proposed_gloss, rationale,
       delta_cosine, residual_before, residual_after, accepted)
    VALUES (?,?,?,?,?,?,?,?,?,?)
  """, (
    hypothesis.morph.upper(),
    hypothesis.source_word.upper(),
    (hypothesis.anchor or "").upper(),
    json.dumps(hypothesis.seed_glosses, ensure_ascii=False),
    hypothesis.proposed_gloss,
    hypothesis.rationale,
    float(hypothesis.delta_cosine or 0.0),
    float(hypothesis.residual_before or 0.0),
    float(hypothesis.residual_after or 0.0),
    int(hypothesis.accepted)
  ))
  conn.commit()
  return hypothesis.accepted

"""
how to wire this (later)

run once at startup:

conn.executescript(ANCHORS_VIEW_SQL)

conn.executescript(MORPH_HYPOTHESES_TABLE_SQL)

build anchors: anchors = build_anchor_index(conn)

fetch candidates: cands = fetch_singleton_candidates(conn)

precompute a sense matrix:

embed all dictionary senses or your clusters.raw_defs.definition as rows → (N, d) normalized

keep sense_labels (short human-readable tokens)

for each candidate:

hyp = infer_singleton(conn, sbert, anchors, cand, sense_matrix, sense_labels)

call your Solo agent with residual_guidance_json + seeds to fill proposed_gloss/rationale

evaluate_and_persist(conn, hyp, recompute_residual_fn)

quick notes

the “choose anchor” heuristic is intentionally dumb but safe; swap in your breakdown segments when ready.

the Δ vector trick assumes the source word’s meaning ≈ anchor + residual; for multi-residual words, handle greedily or jointly (you can extend infer_singleton to accept multiple residuals later).

recompute_residual_fn is your bridge into MorphemeCandidateFinder—inject the provisional morph and re-run the breakdown to get a fresh residual ratio.
"""