from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from enochian_lm.root_extraction.utils.residual_analysis import compute_word_break_subtractions

DICTIONARY_PATH = (
    PROJECT_ROOT / "src" / "enochian_lm" / "root_extraction" / "data" / "dictionary.json"
)


def _load_normalized_dictionary_words() -> list[str]:
    """Load dictionary words as normalized lowercase tokens.

    What: parse dictionary JSON and extract normalized spellings.
    Why: Addendum E requires dictionary-backed host/root pairs rather than synthetic text.
    Big picture: keeps subtraction behavior anchored to corpus-like inventory.
    """

    rows = json.loads(DICTIONARY_PATH.read_text())
    return sorted(
        {
            str(row.get("normalized", "")).strip().lower()
            for row in rows
            if str(row.get("normalized", "")).strip()
        }
    )


def _select_deterministic_substring_pairs(words: list[str], limit: int = 20) -> list[tuple[str, str]]:
    """Build a stable sample where root is a strict substring of host.

    Policy: iterate hosts lexicographically and pick the longest lexical-first root
    that is strictly contained in the host. This gives deterministic coverage across runs.
    """

    pairs: list[tuple[str, str]] = []
    for host in words:
        candidates = [root for root in words if root != host and root in host]
        if not candidates:
            continue
        root = sorted(candidates, key=lambda item: (-len(item), item))[0]
        pairs.append((host, root))
        if len(pairs) >= limit:
            break
    return pairs


def _assert_candidate_internal_consistency(host: str, root: str, candidate: dict[str, object]) -> None:
    """Validate spans/artifacts are internally coherent for one subtraction row."""

    start = int(candidate.get("start", -1))
    end = int(candidate.get("end", -1))
    assert 0 <= start < end <= len(host)
    assert host[start:end].lower() == root.lower()

    artifacts = [frag for frag in (candidate.get("residuals") or []) if isinstance(frag, dict)]
    for frag in artifacts:
        frag_start = int(frag.get("start", -1))
        frag_end = int(frag.get("end", -1))
        assert 0 <= frag_start <= frag_end <= len(host)
        assert host[frag_start:frag_end] == str(frag.get("artifact", ""))

    residual_from_artifacts = "".join(str(f.get("artifact", "")) for f in artifacts)
    assert residual_from_artifacts == str(candidate.get("residual", ""))


def test_dictionary_sample_pairs_produce_valid_subtraction_candidates() -> None:
    words = _load_normalized_dictionary_words()
    sampled_pairs = _select_deterministic_substring_pairs(words, limit=25)

    assert sampled_pairs, "expected at least one dictionary-backed substring pair"

    for host, root in sampled_pairs:
        candidates = compute_word_break_subtractions(host, root)
        assert candidates, f"expected subtraction options for {host} - {root}"
        for cand in candidates:
            _assert_candidate_internal_consistency(host, root, cand)


def test_dictionary_pinned_cases_remain_stable() -> None:
    """Pin key corpus-backed pairs so behavior stays stable if sampling changes."""

    words = set(_load_normalized_dictionary_words())
    pinned_pairs = [
        ("nazpsad", "naz"),
        ("aaiom", "aai"),
        ("acurtoh", "toh"),
        ("adepoad", "ad"),
    ]

    for host, root in pinned_pairs:
        assert host in words
        assert root in words
        assert host != root and root in host

        candidates = compute_word_break_subtractions(host, root)
        assert candidates, f"expected subtraction candidates for pinned pair {host} - {root}"
        for cand in candidates:
            _assert_candidate_internal_consistency(host, root, cand)


def test_repeated_root_hosts_expose_ambiguity_options() -> None:
    """Repeated-root host rows must expose single-occurrence and remove-all options."""

    host = "adepoad"
    root = "ad"
    candidates = compute_word_break_subtractions(host, root)

    # Repeated roots should create at least two single-occurrence options.
    assert len(candidates) >= 2
    assert any(int(c.get("total_occurrences", 0)) >= 2 for c in candidates)

    single_residuals = {str(c.get("residual", "")) for c in candidates}
    remove_all_payload = candidates[0].get("remove_all_occurrences")
    assert isinstance(remove_all_payload, dict)

    removed_spans = remove_all_payload.get("removed_spans") or []
    assert len(removed_spans) >= 2

    all_residual = str(remove_all_payload.get("residual", ""))
    # Big-picture ambiguity check: remove-all option should be represented explicitly
    # and differ from at least one single-removal residual in repeated-root hosts.
    assert any(all_residual != single for single in single_residuals)
