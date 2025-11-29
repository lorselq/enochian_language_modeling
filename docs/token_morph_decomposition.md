# Token-level morph decomposition

The `token_morph_decomp` table stores root-independent segmentations for tokens. Each row captures one ordered morph with its span and score, keyed by `(run_id, token, seg_index)`.

```sql
CREATE TABLE IF NOT EXISTS token_morph_decomp (
  run_id      TEXT NOT NULL,
  token       TEXT NOT NULL,
  seg_index   INTEGER NOT NULL,
  morph       TEXT NOT NULL,
  span_start  INTEGER NOT NULL,
  span_end    INTEGER NOT NULL,
  score       REAL,
  source      TEXT,
  PRIMARY KEY (run_id, token, seg_index)
);
```

## Computing decompositions

Use the analysis CLI to compute decompositions once per token (independent of any root hypothesis):

```bash
poetry run enochian-analysis morph decompose --run-id <RUN> \
  --source seg_v1 --min-morph-len 2 --min-segments 1
```

The command reports how many tokens land in each bucket and records the ordered morph sequences for reuse by composite reconstruction and downstream collocation utilities. Example outcomes:

| token      | morphs          | spans   | notes              |
| ---------- | --------------- | ------- | ------------------ |
| `NAZPSAD`  | [`naz`, `ps`, `ad`] | [0-3], [3-5], [5-7] | multi-morph with three spans |
| `MADZILODARP` | [`mad`, `zi`, `lod`, `arp`] | [0-3], [3-5], [5-8], [8-11] | four-way split |
| `L`        | [`l`]           | [0-1]   | single-morph fallback |

To reuse these decompositions for composite reconstruction, pass `--use-token-decomp` to `enochian-analysis composite backfill`.
