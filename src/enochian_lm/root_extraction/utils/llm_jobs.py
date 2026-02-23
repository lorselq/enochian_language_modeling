from __future__ import annotations

import hashlib, json, datetime, sqlite3

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_prompt_hash(*, system_prompt: str, user_prompt: str, role: str, model: str, temperature: float, base_url: str) -> str:
    # Normalize to keep hashes stable
    blob = json.dumps({
        "system": system_prompt,
        "user": user_prompt,
        "role": role,
        "model": model,
        "temperature": round(float(temperature), 3),
        "base_url": base_url,
    }, sort_keys=True, ensure_ascii=False)
    return _sha256(blob)

def llm_job_try_cache(conn: sqlite3.Connection, prompt_hash: str) -> dict | None:
    row = conn.execute(
        """
        SELECT response_text, model, status
          FROM llm_job
         WHERE prompt_hash = ?
         ORDER BY
           CASE WHEN finished_at IS NULL THEN 1 ELSE 0 END,
           finished_at DESC,
           job_id DESC
         LIMIT 1
        """,
        (prompt_hash,)
    ).fetchone()
    if not row:
        return None
    response_text = row[0] or ""
    status = (row[2] or "").lower()

    # Only reuse successful, non-empty, non-error responses
    if status in ("ok", "cached") and response_text.strip() and not response_text.startswith("[ERROR]"):
        return {"response_text": response_text, "gloss_model": row[1]}
    return None

def llm_job_start(conn: sqlite3.Connection, *, run_id: str, prompt_hash: str, role: str,
                  model: str, base_url: str, temperature: float,
                  system_prompt: str, user_prompt: str, request_json: dict) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO llm_job (run_id, prompt_hash, role, model, base_url, temperature,
                             system_prompt, user_prompt, request_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued')
        ON CONFLICT(prompt_hash) DO NOTHING
        """,
        (run_id, prompt_hash, role, model, base_url, float(temperature),
         system_prompt, user_prompt, json.dumps(request_json, ensure_ascii=False))
    )
    # Retrieve job_id (row may pre-exist)
    row = cur.execute("SELECT job_id FROM llm_job WHERE prompt_hash = ?", (prompt_hash,)).fetchone()
    return int(row[0])

def llm_job_finish(conn: sqlite3.Connection, job_id: int, *, response_text: str,
                   tokens_in: int | None = None, tokens_out: int | None = None,
                   cost_usd: float | None = None, status: str = "ok", error: str | None = None) -> None:
    conn.execute(
        """
        UPDATE llm_job
           SET response_text = ?,
               tokens_in = ?, tokens_out = ?, cost_usd = ?,
               status = ?, error = ?, finished_at = ?
         WHERE job_id = ?
        """,
        (response_text, tokens_in, tokens_out, cost_usd, status, error,
         datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", job_id)
    )
    conn.commit()
