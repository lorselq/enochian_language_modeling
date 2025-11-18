"""Enrich the base dictionary with POS and semantic metadata."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml


RE_VERB = re.compile(r"^to\s+(?P<lemma>[a-z][a-z\-']*)", re.IGNORECASE)
RE_TOKENS = re.compile(r"[a-zA-Z]+")
COPULA_FORMS = {"am", "art", "are", "be", "being", "been", "is", "was", "were"}
PREPOSITIONS = {
    "of",
    "from",
    "in",
    "into",
    "on",
    "onto",
    "upon",
    "with",
    "within",
    "without",
    "between",
    "among",
    "amongst",
    "about",
    "toward",
    "towards",
    "through",
    "beyond",
    "under",
    "above",
    "before",
    "after",
}
COORDINATORS = {"and", "or", "but", "nor"}
STOPWORDS = {"the", "a", "an"}


@dataclass
class DomainConfig:
    """Semantic domain configuration loaded from YAML."""

    domains: Dict[str, str]
    headword_to_domains: Dict[str, List[str]]

    @classmethod
    def load(cls, path: Path) -> "DomainConfig":
        data = yaml.safe_load(path.read_text())
        headword_map = {
            key.lower(): [label.upper() for label in value]
            for key, value in data.get("headword_to_domains", {}).items()
        }
        return cls(domains=data.get("domains", {}), headword_to_domains=headword_map)

    def lookup(self, headword: Optional[str]) -> List[str]:
        if not headword:
            return []
        hw = headword.lower()
        if hw in self.headword_to_domains:
            return self.headword_to_domains[hw]
        singular = singularize(hw)
        return self.headword_to_domains.get(singular, [])


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 2:
        return token[:-2]
    if token.endswith("s") and len(token) > 1:
        return token[:-1]
    return token


def extract_tokens(gloss: str) -> List[str]:
    return [match.group(0).lower() for match in RE_TOKENS.finditer(gloss or "")]


def extract_headword(gloss: str) -> Optional[str]:
    tokens = extract_tokens(gloss)
    if not tokens:
        return None
    # Drop leading infinitive marker or determiners
    idx = 0
    if tokens[0] == "to" and len(tokens) > 1:
        idx = 1
    while idx < len(tokens) and tokens[idx] in STOPWORDS:
        idx += 1
    return tokens[idx] if idx < len(tokens) else None


def detect_compound(gloss: str, tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    if "," in gloss or " / " in gloss or ";" in gloss:
        return True
    compound_markers = {"which", "that", "those", "these", "they", "amongst"}
    if any(marker in tokens for marker in compound_markers):
        return True
    return len(tokens) > 2


def infer_pos(gloss: str, tokens: Sequence[str]) -> Tuple[List[str], bool, bool, Optional[str]]:
    pos: List[str] = []
    notes: Optional[str] = None
    is_copula = False

    gloss_lower = (gloss or "").strip().lower()
    if not gloss_lower:
        pos.append("NOUN")
        notes = "fallback:noun_empty"
        return pos, is_copula, False, notes

    if gloss_lower.startswith("to be ") or gloss_lower in COPULA_FORMS:
        is_copula = True
        if "AUX" not in pos:
            pos.append("AUX")

    if not is_copula and gloss_lower.startswith("to be"):
        is_copula = True
        if "AUX" not in pos:
            pos.append("AUX")

    verb_match = RE_VERB.match(gloss_lower)
    if verb_match and not is_copula:
        if "VERB" not in pos:
            pos.append("VERB")

    if tokens:
        primary = tokens[0]
        if primary in PREPOSITIONS and len(tokens) <= 3:
            if "ADP" not in pos:
                pos.append("ADP")
        if len(tokens) == 1 and primary in COORDINATORS:
            if "CCONJ" not in pos:
                pos.append("CCONJ")
        if len(tokens) == 1 and primary in {"he", "she", "it", "they", "we", "i", "thou", "ye"}:
            if "PRON" not in pos:
                pos.append("PRON")

    if not pos:
        pos.append("NOUN")
        notes = "fallback:noun_default"

    return pos, is_copula, detect_compound(gloss, tokens), notes


def enrich_senses(entry: Dict, domain_config: DomainConfig) -> None:
    for sense in entry.get("senses", []):
        gloss = sense.get("definition", "")
        tokens = extract_tokens(gloss)
        pos, is_copula, is_compound, notes = infer_pos(gloss, tokens)
        headword = extract_headword(gloss)
        domains = domain_config.lookup(headword)
        sense["parts_of_speech"] = pos
        sense["semantic_domains"] = domains
        sense["is_copula"] = is_copula
        sense["is_compound_standing_for_phrase"] = is_compound
        sense["notes_pos"] = notes


def review(enriched_entries: Sequence[Dict]) -> str:
    copula_entries: List[str] = []
    multi_pos: List[str] = []
    missing_domains: List[str] = []

    for entry in enriched_entries:
        word = entry.get("word", "<unknown>")
        for sense in entry.get("senses", []):
            descriptor = f"{word} (sense {sense.get('sense_id')})"
            pos = sense.get("parts_of_speech", [])
            if sense.get("is_copula"):
                copula_entries.append(descriptor)
            if len(pos) > 1:
                multi_pos.append(descriptor + f" -> {pos}")
            if not sense.get("semantic_domains"):
                missing_domains.append(descriptor)

    lines = ["=== Manual Review Report ==="]
    lines.append(f"Copula lemmas: {len(copula_entries)}")
    lines.extend(copula_entries)
    lines.append("")
    lines.append(f"Multiple POS senses: {len(multi_pos)}")
    lines.extend(multi_pos)
    lines.append("")
    lines.append(f"Senses without semantic domains: {len(missing_domains)}")
    lines.extend(missing_domains)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "enochian_lm"
        / "root_extraction"
        / "data"
        / "dictionary.json"
    )
    default_output = default_input.with_name("dictionary_enriched.json")
    default_domains = Path(__file__).resolve().parents[1] / "config" / "semantic_domains.yml"

    parser.add_argument("--input", type=Path, default=default_input, help="Path to dictionary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination for enriched dictionary",
    )
    parser.add_argument(
        "--domains",
        type=Path,
        default=default_domains,
        help="Path to semantic_domains.yml",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to save the manual review report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing file {args.output}. Use --overwrite to replace it."
        )

    entries = json.loads(args.input.read_text())
    domain_config = DomainConfig.load(args.domains)
    for entry in entries:
        enrich_senses(entry, domain_config)

    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    report_text = review(entries)
    if args.report:
        args.report.write_text(report_text + "\n")
    print(report_text)


if __name__ == "__main__":
    main()
