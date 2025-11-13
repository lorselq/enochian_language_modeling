"""Text and randomness utilities for Enochian analysis tooling."""
from __future__ import annotations

import logging
import random
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def normalize(s: str) -> str:
    """Normalize an Enochian string.

    The current implementation is an identity function. Future updates should
    replace v/w with u, j/y with i, convert k to c, strip diacritics, and
    normalize case.
    """

    # TODO: implement v/w->u, j/y->i, k->c, strip diacritics, normalize case.
    return s


def split_morphemes(token: str, known: dict[str, list[str]] | None = None) -> list[str]:
    """Split *token* into morphemes.

    If a *known* dictionary is provided, return its mapping when available;
    otherwise return the token as a single-morpheme list.
    """

    if known is not None and token in known:
        return list(known[token])
    return [token]


def set_global_seeds(seed: int = 42) -> None:
    """Seed deterministic modules for repeatable experiments."""

    random.seed(seed)
    logger.info("Global random seed set", extra={"seed": seed})


def utcnow_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format with a trailing 'Z'."""

    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")
