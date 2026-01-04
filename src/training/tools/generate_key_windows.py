"""
Generate sliding-window Enochian ↔ English gloss training pairs.

The script reads the Keys corpus from ``enochian_keys.txt``, tokenizes each
Key, pairs every token with its dictionary gloss and in-Key citations, and
emits fixed-width sliding windows suitable for fine-tuning data pipelines.

Example
-------
python tools/generate_key_windows.py \
    --keys src/enochian_lm/root_extraction/data/enochian_keys.txt \
    --dictionary src/enochian_lm/root_extraction/data/dictionary_enriched.json \
    --window-size 5 --stride 2 --format jsonl --output windows.jsonl \
    --max-windows 100

Each output record now contains the Enochian token window plus parallel
definition and citation token sequences so that models can learn both the
dictionary gloss and the natural English contexts. By default, tokens are
lowercased and punctuation is stripped. Use ``--no-lowercase`` or
``--keep-punctuation`` to disable those normalizations. ``--max-windows``
is available for sampling small subsets (e.g., ``--max-windows 100``).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z']+")


@dataclass
class WindowRecord:
    key_id: int
    start: int
    end: int
    enochian_tokens: list[str]
    definition_tokens: list[str]
    citation_tokens: list[str]

    def to_mapping(self) -> dict:
        definition_text = " ".join(self.definition_tokens)
        citation_text = " ".join(self.citation_tokens)
        return {
            "key_id": self.key_id,
            "start": self.start,
            "end": self.end,
            "input_tokens": self.enochian_tokens,
            "target_tokens": self.definition_tokens,
            "input_text": " ".join(self.enochian_tokens),
            "target_text": definition_text,
            "definition_tokens": self.definition_tokens,
            "definition_text": definition_text,
            "citation_tokens": self.citation_tokens,
            "citation_text": citation_text,
        }


@dataclass
class Config:
    keys_path: Path
    dictionary_path: Path
    output_path: Path
    window_size: int
    stride: int
    lowercase: bool
    strip_punctuation: bool
    output_format: str
    max_windows: int | None


@dataclass
class GlossSources:
    definition_tokens: list[str]
    citation_tokens: list[str]


def tokenize(text: str, lowercase: bool = True, strip_punctuation: bool = True) -> list[str]:
    """Tokenize a string with optional lowercasing and punctuation stripping."""
    tokens = TOKEN_RE.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    if not strip_punctuation:
        # When preserving punctuation we fall back to whitespace splitting but keep
        # the regex-derived tokens available for compatibility.
        extra = []
        for chunk in re.split(r"\s+", text.strip()):
            if chunk:
                extra.append(chunk.lower() if lowercase else chunk)
        if extra:
            return extra
    return tokens


def load_dictionary_glosses(dictionary_path: Path, lowercase: bool, strip_punctuation: bool) -> dict[str, GlossSources]:
    entries = json.loads(dictionary_path.read_text())
    mapping: dict[str, GlossSources] = {}
    for entry in entries:
        word = entry.get("word") or entry.get("normalized")
        if not isinstance(word, str):
            continue
        key = word.lower() if lowercase else word

        definition_text = str(entry.get("definition") or "").strip()
        if not definition_text:
            senses = entry.get("senses") or []
            if senses:
                definition_text = str(senses[0].get("definition") or "").strip()
        definition_tokens = tokenize(
            definition_text, lowercase=lowercase, strip_punctuation=strip_punctuation
        )

        citation_tokens: list[str] = []
        for citation in entry.get("key_citations", []) or []:
            context = citation.get("context")
            if not isinstance(context, str):
                continue
            citation_tokens.extend(
                tokenize(context, lowercase=lowercase, strip_punctuation=strip_punctuation)
            )

        mapping[key] = GlossSources(
            definition_tokens=definition_tokens,
            citation_tokens=citation_tokens,
        )
    return mapping


def sliding_windows(tokens: Sequence[str], window_size: int, stride: int) -> Iterable[tuple[int, int, list[str]]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    for start in range(0, max(len(tokens) - window_size + 1, 0), stride):
        end = start + window_size
        yield start, end, list(tokens[start:end])


def build_records(keys_text: str, config: Config, gloss_map: dict[str, GlossSources]) -> list[WindowRecord]:
    keys = [part for part in keys_text.split("\n\n") if part.strip()]
    records: list[WindowRecord] = []
    for idx, key_text in enumerate(keys, start=1):
        key_tokens = tokenize(key_text, lowercase=config.lowercase, strip_punctuation=config.strip_punctuation)
        for start, end, window in sliding_windows(key_tokens, config.window_size, config.stride):
            definition_window_tokens: list[str] = []
            citation_window_tokens: list[str] = []
            for tok in window:
                key = tok if not config.lowercase else tok.lower()
                sources = gloss_map.get(key)
                if not sources:
                    continue
                definition_window_tokens.extend(sources.definition_tokens)
                citation_window_tokens.extend(sources.citation_tokens)
            records.append(
                WindowRecord(
                    key_id=idx,
                    start=start,
                    end=end,
                    enochian_tokens=window,
                    definition_tokens=definition_window_tokens,
                    citation_tokens=citation_window_tokens,
                )
            )
            if config.max_windows and len(records) >= config.max_windows:
                return records
    return records


def write_jsonl(records: Sequence[WindowRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_mapping(), ensure_ascii=False))
            f.write("\n")


def write_csv(records: Sequence[WindowRecord], output_path: Path) -> None:
    fieldnames = [
        "key_id",
        "start",
        "end",
        "input_text",
        "target_text",
        "definition_text",
        "citation_text",
        "input_tokens",
        "target_tokens",
        "definition_tokens",
        "citation_tokens",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            mapping = rec.to_mapping()
            writer.writerow(mapping)


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Generate sliding window pairs from the Enochian Keys.")
    ap.add_argument("--keys", type=Path, default=Path("src/enochian_lm/root_extraction/data/enochian_keys.txt"))
    ap.add_argument("--dictionary", type=Path, default=Path("src/enochian_lm/root_extraction/data/dictionary_enriched.json"))
    ap.add_argument("--output", type=Path, default=Path("keys_windows.jsonl"))
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", dest="output_format")
    ap.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True, help="Lowercase tokens before alignment")
    ap.add_argument(
        "--strip-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip punctuation when tokenizing Enochian and gloss text",
    )
    ap.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit the number of sliding windows emitted (useful for sampling)",
    )
    args = ap.parse_args()
    return Config(
        keys_path=args.keys,
        dictionary_path=args.dictionary,
        output_path=args.output,
        window_size=args.window_size,
        stride=args.stride,
        lowercase=bool(args.lowercase),
        strip_punctuation=bool(args.strip_punctuation),
        output_format=args.output_format,
        max_windows=args.max_windows,
    )


def main() -> None:
    config = parse_args()
    keys_text = config.keys_path.read_text(encoding="utf-8")
    gloss_map = load_dictionary_glosses(
        dictionary_path=config.dictionary_path,
        lowercase=config.lowercase,
        strip_punctuation=config.strip_punctuation,
    )
    records = build_records(keys_text, config, gloss_map)
    if config.output_format == "jsonl":
        write_jsonl(records, config.output_path)
    else:
        write_csv(records, config.output_path)
    print(f"Wrote {len(records)} windows to {config.output_path}")


if __name__ == "__main__":
    main()
