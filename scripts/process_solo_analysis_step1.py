#!/usr/bin/env python3
"""
Step 2: Strict, minimal repair of glossator_def for two shapes only:

1) EXAMPLE must be an array:
   "EXAMPLE": "a.", "b.", "c."
   → "EXAMPLE": ["a.", "b.", "c."]

2) SIGNATURE.contribution must be an object when it was encoded as:
   "contribution": ["k": 0.8, "m": 0.2]
   → "contribution": {"k": 0.8, "m": 0.2}

Everything else is left as-is. After repair, we parse and lightly normalize
(UPPERCASE ROOT; slot "various"→"mixed"; EXAMPLE coerced to list).

Run:
  python scripts/process_solo_analysis_step2.py
  python scripts/process_solo_analysis_step2.py --dry-run
  python scripts/process_solo_analysis_step2.py --only-accepted
"""
from __future__ import annotations
from enochian_translation_team.utils import sqlite_bootstrap  # noqa: F401
import argparse, json, re, sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------- tiny JSON-lexing helpers -----------------------

def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i

def _read_json_string(s: str, i: int) -> Tuple[str, int] | None:
    # expects s[i] == '"'; returns ('"…"', next_index)
    if i >= len(s) or s[i] != '"': return None
    j = i + 1; esc = False
    while j < len(s):
        c = s[j]
        if esc: esc = False
        elif c == '\\': esc = True
        elif c == '"': return (s[i:j+1], j+1)
        j += 1
    return None

# Accept JSON literals true/false/null
def _read_json_literal(s: str, i: int) -> tuple[str, int] | None:
    for lit in ("true", "false", "null"):
        if s.startswith(lit, i):
            return (lit, i + len(lit))
    return None

_NUM_RE = re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')
def _read_json_number(s: str, i: int) -> Tuple[str, int] | None:
    m = _NUM_RE.match(s, i)
    if not m: return None
    return (m.group(0), m.end())

def _is_next_key_at(s: str, i: int) -> bool:
    """Return True if s[i:] starts with ,"KEY": (comma optional if caller already positioned)."""
    j = _skip_ws(s, i)
    if j < len(s) and s[j] == ',':
        j += 1
        j = _skip_ws(s, j)
    if j >= len(s) or s[j] != '"': return False
    key = _read_json_string(s, j)
    if not key: return False
    _, k_end = key
    k_end = _skip_ws(s, k_end)
    return k_end < len(s) and s[k_end] == ':'

def find_first_json_object_block(text: str) -> Optional[str]:
    if not text: return None
    try:
        start = text.index("{")
    except ValueError:
        return None
    depth = 0; i = start; in_str = False; esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0: return text[start:i+1]
        i += 1
    return None

# ----------------------- pre-repair A: EXAMPLE → array -----------------------

def repair_example_array(block: str) -> str:
    """
    If we see: "EXAMPLE": "a." (TAG) | "b." (TAG) "c." (TAG)
    rewrite value as ["a. (TAG)", "b. (TAG)", "c. (TAG)"].
    Handles single or multiple items, optional pipes, stray quotes, and preserves
    the boundary comma separating EXAMPLE from the next key.
    """
    out: list[str] = []
    i = 0
    while True:
        k = block.find('"EXAMPLE"', i)
        if k == -1:
            out.append(block[i:]); break

        out.append(block[i:k]); out.append('"EXAMPLE"')
        j = k + len('"EXAMPLE"')
        j = _skip_ws(block, j)
        if j < len(block) and block[j] == ':':
            out.append(':'); j += 1
        j = _skip_ws(block, j)

        # Already an array? copy balanced and continue
        if j < len(block) and block[j] == '[':
            out.append('['); j += 1
            depth = 1; in_str = False; esc = False; p = j
            while p < len(block) and depth > 0:
                ch = block[p]
                if in_str:
                    if esc: esc = False
                    elif ch == '\\': esc = True
                    elif ch == '"': in_str = False
                else:
                    if ch == '"': in_str = True
                    elif ch == '[': depth += 1
                    elif ch == ']': depth -= 1
                p += 1
            out.append(block[j:p]); i = p
            continue

        # Broken form → gather one or more quoted strings
        if j < len(block) and block[j] == '"':
            elems: list[str] = []
            p = j
            while True:
                lit = _read_json_string(block, p)
                if not lit: break
                token, p = lit
                text = json.loads(token)  # unescape

                # Optional: consume an out-of-string annotation like (TAG)
                q = _skip_ws(block, p)
                if q < len(block) and block[q] == '(':
                    # read balanced parens (no nested parens needed here)
                    r = q + 1
                    while r < len(block) and block[r] != ')':
                        r += 1
                    if r < len(block) and block[r] == ')':
                        tag = block[q+1:r].strip()
                        if tag:
                            text = f"{text} ({tag})"
                        p = r + 1
                    else:
                        # unmatched '(', leave it
                        p = q

                # Add to array
                elems.append(json.dumps(text))

                # Skip optional separators: pipes, stray quotes, whitespace
                while True:
                    q = _skip_ws(block, p)
                    advanced = False
                    if q < len(block) and block[q] in {'|'}:
                        p = q + 1; advanced = True
                    elif q < len(block) and block[q] == '"':
                        # stray quote without content → skip
                        p = q + 1; advanced = True
                    else:
                        p = q
                    if not advanced:
                        break

                # If comma + next *string* follows, keep consuming; otherwise stop before comma
                if p < len(block) and block[p] == ',':
                    q2 = _skip_ws(block, p + 1)
                    if q2 < len(block) and block[q2] == '"':
                        # if this comma actually starts the next key (,"KEY":), stop
                        if _is_next_key_at(block, p):
                            # leave comma for outer context
                            pass
                        else:
                            # consume next string in loop
                            p = q2
                            continue
                break

            out.append('[' + ', '.join(elems) + ']')
            i = p
            continue

        # Not string/array: copy one char and move on
        out.append(block[j:j+1]); i = j + 1

    return ''.join(out)


# ---------------- pre-repair B: contribution array-of-pairs → object ------------

def repair_contribution_object(block: str) -> str:
    """
    Convert "contribution": ["k": VALUE, ...] → "contribution": {"k": VALUE, ...}
    VALUE may be string, number, true, false, or null. Works with a single pair too.
    Leaves already-correct objects/arrays alone.
    """
    out: list[str] = []; i = 0
    while True:
        k = block.find('"contribution"', i)
        if k == -1:
            out.append(block[i:]); break

        out.append(block[i:k]); out.append('"contribution"')
        j = k + len('"contribution"'); j = _skip_ws(block, j)
        if j < len(block) and block[j] == ':':
            out.append(':'); j += 1
        j = _skip_ws(block, j)

        # Already an object: copy balanced
        if j < len(block) and block[j] == '{':
            out.append('{'); j += 1
            depth = 1; in_str = False; esc = False; p = j
            while p < len(block) and depth > 0:
                ch = block[p]
                if in_str:
                    if esc: esc = False
                    elif ch == '\\': esc = True
                    elif ch == '"': in_str = False
                else:
                    if ch == '"': in_str = True
                    elif ch == '{': depth += 1
                    elif ch == '}': depth -= 1
                p += 1
            out.append(block[j:p]); i = p
            continue

        # Array? Try to parse as ["label": VALUE, ...]
        if j < len(block) and block[j] == '[':
            p = j + 1
            pairs: list[tuple[str, str]] = []; ok = True
            while True:
                p = _skip_ws(block, p)
                if p >= len(block): ok = False; break
                if block[p] == ']': p += 1; break

                lbl = _read_json_string(block, p)
                if not lbl: ok = False; break
                label_token, p = lbl

                p = _skip_ws(block, p)
                if p >= len(block) or block[p] != ':': ok = False; break
                p += 1; p = _skip_ws(block, p)

                val_token = None
                sread = _read_json_string(block, p)
                if sread: val_token, p = sread
                else:
                    nread = _read_json_number(block, p)
                    if nread: val_token, p = nread
                    else:
                        lread = _read_json_literal(block, p)  # true/false/null
                        if lread: val_token, p = lread

                if val_token is None: ok = False; break

                pairs.append((label_token, val_token))
                p = _skip_ws(block, p)
                if p < len(block) and block[p] == ',':
                    p += 1; continue
                elif p < len(block) and block[p] == ']':
                    p += 1; break

            if ok and pairs:
                kvs = []
                for lbl_tok, val_tok in pairs:
                    key = json.loads(lbl_tok)
                    kvs.append(json.dumps(key) + ": " + val_tok)
                out.append('{' + ', '.join(kvs) + '}')
                i = p
                continue
            else:
                # Not our shape → copy array balanced
                out.append('[')
                p = j + 1
                depth = 1; in_str = False; esc = False
                while p < len(block) and depth > 0:
                    ch = block[p]
                    if in_str:
                        if esc: esc = False
                        elif ch == '\\': esc = True
                        elif ch == '"': in_str = False
                    else:
                        if ch == '"': in_str = True
                        elif ch == '[': depth += 1
                        elif ch == ']': depth -= 1
                    p += 1
                out.append(block[j+1:p]); i = p
                continue

        # Not object/array: copy one char and move on
        out.append(block[j:j+1]); i = j + 1

    return ''.join(out)


def pre_repair(block: str) -> str:
    # Order matters: EXAMPLE first (so commas around it are clean), then contribution
    s = repair_example_array(block)
    s = repair_contribution_object(s)
    return s

# ---------------------- light normalization (text preserved) -------------------

POS_ALLOWED = {"prefix","infix","suffix","root","particle","variable"}
BND_ALLOWED = {"bound","clitic","free","unknown"}
SLOT_ALLOWED = {"initial","medial","final","mixed"}
EVAL_ALLOWED = {"accepted","rejected"}
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z*"])')

def _as_nonempty_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None

def _coerce_examples(x: Any) -> List[str]:
    if isinstance(x, list):
        return [s.strip() for s in x if isinstance(s, str) and s.strip()]
    if isinstance(x, str):
        parts = [p.strip().strip(",") for p in SENT_SPLIT_RE.split(x.strip()) if p.strip().strip(",")]
        return parts if parts else [x.strip()]
    return []

def _strip_brackets_in_strings(x: Any) -> Any:
    """Recursively remove '[' and ']' from any string values."""
    if isinstance(x, dict):
        return {k: _strip_brackets_in_strings(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_brackets_in_strings(v) for v in x]
    if isinstance(x, str):
        return x.replace("[", "").replace("]", "")
    return x

def normalize_record(d: Dict[str, Any]) -> Dict[str, Any]:
    # ---- EVALUATION + ROOT ----
    evaluation_raw = d.get("EVALUATION")
    evaluation = evaluation_raw.lower() if isinstance(evaluation_raw, str) else None
    evaluation = evaluation if evaluation in EVAL_ALLOWED else "rejected"
    is_rejected = (evaluation == "rejected")

    root = (_as_nonempty_str(d.get("ROOT")) or "?").upper()

    # ---- SIGNATURE (type-safe dict for Pylance and runtime) ----
    sig_raw = d.get("SIGNATURE")
    sig: Dict[str, Any] = dict(sig_raw) if isinstance(sig_raw, dict) else {}

    # helpers: return "" for rejected rows if missing/invalid; otherwise default to a safe enum
    def _norm_sig_enum(val: Any, allowed: set[str], default_if_accepted: str) -> str:
        if isinstance(val, str):
            s = val.strip()
            if s in allowed:
                return s
            # allow empty string ("") to survive *only* in rejected rows
            if is_rejected and s == "":
                return ""
        # missing or invalid
        return "" if is_rejected else default_if_accepted

    # collapse pipe-separated options to canonical enums
    pos_raw = sig.get("position")
    if isinstance(pos_raw, str) and '|' in pos_raw:
        sig["position"] = "variable"

    slot_raw = sig.get("slot")
    if isinstance(slot_raw, str) and '|' in slot_raw:
        sig["slot"] = "mixed"

    # position / boundness / slot
    sig["position"]  = _norm_sig_enum(sig.get("position"),  POS_ALLOWED,  "root")
    sig["boundness"] = _norm_sig_enum(sig.get("boundness"), BND_ALLOWED,  "unknown")
    # treat odd labels like "various" as invalid → "" (rejected) or "mixed" (accepted)
    sig["slot"]      = _norm_sig_enum(sig.get("slot"),      SLOT_ALLOWED, "mixed")

    # contribution: prefer {} when empty/missing, otherwise preserve (object or list)
    contrib = sig.get("contribution")
    if contrib is None or (isinstance(contrib, list) and len(contrib) == 0) or (isinstance(contrib, str) and contrib.strip() == ""):
        sig["contribution"] = {}
    else:
        sig["contribution"] = _strip_brackets_in_strings(contrib)

    # ontology: keep as list if present; else []
    if not isinstance(sig.get("ontology"), list):
        sig["ontology"] = []

    # ---- Top-level fields ----
    reason      = _as_nonempty_str(d.get("REASON")) or ""
    definition  = _as_nonempty_str(d.get("DEFINITION")) or ""
    decoding    = _as_nonempty_str(d.get("DECODING_GUIDE")) or ""

    # EXAMPLE: listify; rejected rows can naturally end up with []
    examples = _coerce_examples(d.get("EXAMPLE"))

    # SEMANTIC_CORE: keep list if list; if it's a string, split like examples; else []
    if isinstance(d.get("SEMANTIC_CORE"), list):
        sem_core = d["SEMANTIC_CORE"]
    elif isinstance(d.get("SEMANTIC_CORE"), str):
        sem_core = _coerce_examples(d["SEMANTIC_CORE"])
    else:
        sem_core = []

    # RULES / NEGATIVE_CONTRAST: preserve if list, else []
    rules = d.get("RULES") if isinstance(d.get("RULES"), list) else []
    neg   = d.get("NEGATIVE_CONTRAST") if isinstance(d.get("NEGATIVE_CONTRAST"), list) else []

    return {
        "ROOT": root,
        "EVALUATION": evaluation,
        "REASON": reason,
        "DEFINITION": definition,
        "EXAMPLE": examples,
        "DECODING_GUIDE": decoding,
        "SEMANTIC_CORE": sem_core,
        "SIGNATURE": {
            "position":    sig.get("position",  "" if is_rejected else "root"),
            "boundness":   sig.get("boundness", "" if is_rejected else "unknown"),
            "slot":        sig.get("slot",      "" if is_rejected else "mixed"),
            "contribution": sig.get("contribution", {}),
            "ontology":     sig.get("ontology", []),
        },
        "RULES": rules,
        "NEGATIVE_CONTRAST": neg,
    }

# ----------------------------- main ------------------------------------------

def main():
    error_blobs = []
    parser = argparse.ArgumentParser(description="Repair EXAMPLE and contribution containers and normalize lightly.")
    default_db = (
        Path(__file__).resolve().parents[1] /
        "src" / "enochian_translation_team" / "data" /
        "raw_solo_analysis_derived_definitions.sqlite3"
    )
    parser.add_argument("--db", type=Path, default=default_db)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-accepted", action="store_true")
    args = parser.parse_args()

    db = args.db.expanduser().resolve()
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    with sqlite3.connect(db.as_posix()) as con:
        con.execute("PRAGMA foreign_keys=ON;")
        con.execute("PRAGMA journal_mode=WAL;")

        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='clusters'").fetchone():
            raise SystemExit("Table 'clusters' not found.")

        rows = con.execute("""
            SELECT cluster_id, glossator_def
            FROM clusters
            WHERE glossator_def IS NOT NULL AND TRIM(glossator_def) <> ''
        """).fetchall()

        total = len(rows)
        updated = 0
        failures: List[Tuple[int,str]] = []

        if not args.dry_run:
            con.execute("BEGIN;")

        for cluster_id, blob in rows:
            raw = blob or ""
            block = find_first_json_object_block(raw)

            # check for error message
            raw = blob or ""
            if "[ERROR]" in raw and "No content returned" in raw:
                error_blobs.append(cluster_id)
                continue  # skip normalization for this row
            
            if not block:
                failures.append((cluster_id, "no JSON object found"))
                continue

            repaired = pre_repair(block)

            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as e:
                failures.append((cluster_id, f"JSON decode error after repair: {e}"))
                continue

            norm = normalize_record(data)
            if args.only_accepted and norm.get("EVALUATION") != "accepted":
                continue

            out_text = json.dumps(norm, ensure_ascii=False, indent=2, separators=(", ", ": "))
            if raw.strip() != out_text.strip():
                updated += 1
                if not args.dry_run:
                    con.execute("UPDATE clusters SET glossator_def = ? WHERE cluster_id = ?", (out_text, cluster_id))

        if not args.dry_run:
            con.commit()

    print(f"Scanned {total} rows. Changed {updated} rows.")
    print(f"{len(error_blobs)} rows had no content from the model.")
    if failures:
        print(f"{len(failures)} failed to normalize (showing up to 20):")
        for cid, msg in failures[:20]:
            print(f"  - cluster_id={cid}: {msg}")
    if error_blobs:
        list = (
            Path(__file__).resolve().parents[1] /
            "logs" / "processing_reports" /
            "solo_glossator_no_content.txt"
        )
        with open(list, "w", encoding="utf-8") as f:
            f.write("cluster_id\n")
            for cid in error_blobs:
                f.write(f"{cid}\n")
        print(f"Saved list to {list}")

if __name__ == "__main__":
    main()
