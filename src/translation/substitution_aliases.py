from __future__ import annotations

"""Substitution-map helpers for translation evidence lookup.

The root-extraction sidecar records anomalous spellings in
``substitution_map.json``. Translation needs a lightweight view of those rules
so an observed surface token can borrow evidence from configured lookup forms
without weakening the evidence-backed decomposition pipeline.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from enochian_lm.common.config import get_config_paths


@dataclass(frozen=True)
class LookupSubstitutionRule:
    """Represent one surface-to-lookup substitution.

    Rules can be single-letter or multi-letter (for example ``Q <-> QU``).
    Translation uses them as alias transforms: when the surface contains
    ``source``, lookup evidence for ``target`` may be considered for that span.
    """

    source: str
    target: str


def load_lookup_substitution_rules(
    path: Path | None = None,
) -> tuple[LookupSubstitutionRule, ...]:
    """Load lookup aliases from the configured substitution map.

    This function exists so phrase and word translation can share root-
    extraction's anomalous-letter policy without depending on pipeline scripts.
    """

    map_path = (
        Path(path)
        if path is not None
        else Path(get_config_paths()["substitution_map"])
    )
    with map_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        return ()
    return collect_lookup_substitution_rules(payload)


def collect_lookup_substitution_rules(
    payload: Mapping[str, object],
) -> tuple[LookupSubstitutionRule, ...]:
    """Collect deterministic lookup aliases from a substitution payload.

    Keeping this parser separate from file I/O makes the direction semantics
    easy to test and keeps malformed non-dictionary entries from leaking into
    translation lookup.
    """

    rules: list[LookupSubstitutionRule] = []
    seen: set[tuple[str, str]] = set()

    def _add_rule(source: str, target: str) -> None:
        normalized_source = source.strip().upper()
        normalized_target = target.strip().upper()
        if not normalized_source or not normalized_target:
            return
        if normalized_source == normalized_target:
            return
        key_pair = (normalized_source, normalized_target)
        if key_pair in seen:
            return
        rules.append(
            LookupSubstitutionRule(
                source=normalized_source,
                target=normalized_target,
            )
        )
        seen.add(key_pair)

    for key, raw_spec in payload.items():
        if not isinstance(raw_spec, Mapping):
            continue
        canonical = str(raw_spec.get("canonical") or key).strip().upper()
        if not canonical:
            continue
        alternates = raw_spec.get("alternates")
        if not isinstance(alternates, list):
            continue
        for raw_alternate in alternates:
            if not isinstance(raw_alternate, Mapping):
                continue
            direction = str(raw_alternate.get("direction") or "").strip().lower()
            target = str(raw_alternate.get("value") or "").strip().upper()
            if direction in {"from", "both"}:
                _add_rule(canonical, target)
            if direction in {"to", "both"}:
                _add_rule(target, canonical)
    return tuple(rules)


def build_lookup_variants(
    surface: str,
    rules: Iterable[LookupSubstitutionRule],
    *,
    max_variants: int = 128,
    max_operations: int = 2,
) -> tuple[str, ...]:
    """Return lookup spellings produced by applying all configured aliases.

    Decomposition lookup needs every applicable anomalous letter to be tried,
    but the variant set must stay bounded for long words. The original surface
    is returned first, followed by deterministic substitution variants sorted
    lexically for reproducible diagnostics and tests.
    """

    normalized = (surface or "").strip().upper()
    if not normalized:
        return ()

    rule_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for rule in rules:
        source = rule.source.upper()
        target = rule.target.upper()
        if not source or not target or source == target:
            continue
        pair = (source, target)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rule_pairs.append(pair)
    if not rule_pairs:
        return (normalized,)

    variants: set[str] = {normalized}
    frontier: set[str] = {normalized}
    max_steps = max(1, int(max_operations))
    for _step in range(max_steps):
        next_frontier: set[str] = set()
        for candidate in frontier:
            for source, replacement in rule_pairs:
                start = 0
                while True:
                    index = candidate.find(source, start)
                    if index < 0:
                        break
                    if _is_redundant_expansion(
                        candidate,
                        index=index,
                        source=source,
                        replacement=replacement,
                    ):
                        start = index + 1
                        continue
                    next_candidate = (
                        candidate[:index]
                        + replacement
                        + candidate[index + len(source) :]
                    )
                    if next_candidate not in variants:
                        variants.add(next_candidate)
                        next_frontier.add(next_candidate)
                        if len(variants) >= max_variants:
                            return _ordered_variants(normalized, variants)
                    start = index + 1
        if not next_frontier:
            break
        frontier = next_frontier

    return _ordered_variants(normalized, variants)


def build_lookup_aliases(
    surfaces: Iterable[str],
    rules: Iterable[LookupSubstitutionRule],
    *,
    max_variants_per_surface: int = 128,
    max_operations: int = 2,
) -> dict[str, set[str]]:
    """Map lookup spellings back to the surface spans that should use them.

    The service fetches evidence once for lookup forms, then clones matching
    records back onto the original surface substring. This preserves positional
    decomposition on the user's token while allowing configured anomalous
    letters such as ``Y -> I`` to contribute support.
    """

    aliases: dict[str, set[str]] = {}
    rule_tuple = tuple(rules)
    for raw_surface in surfaces:
        surface = (raw_surface or "").strip().upper()
        if not surface:
            continue
        for lookup in build_lookup_variants(
            surface,
            rule_tuple,
            max_variants=max_variants_per_surface,
            max_operations=max_operations,
        ):
            if lookup == surface:
                continue
            aliases.setdefault(lookup, set()).add(surface)
    return aliases


def _ordered_variants(original: str, variants: set[str]) -> tuple[str, ...]:
    """Return variants with the original spelling first and stable ordering after."""

    ordered = [original]
    ordered.extend(sorted(variant for variant in variants if variant != original))
    return tuple(ordered)


def _is_redundant_expansion(
    candidate: str,
    *,
    index: int,
    source: str,
    replacement: str,
) -> bool:
    """Return whether replacing at ``index`` would duplicate an existing expansion.

    Rules like ``Q -> QU`` can otherwise re-expand a sequence that already
    starts with ``QU`` (for example ``QU`` -> ``QUU``), which adds noisy lookup
    forms without adding useful evidence paths.
    """

    if len(replacement) <= len(source):
        return False
    end = index + len(replacement)
    return candidate[index:end] == replacement
