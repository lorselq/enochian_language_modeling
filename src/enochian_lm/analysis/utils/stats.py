"""Statistical helpers for Enochian analysis."""
from __future__ import annotations

import math


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Return *a / b* guarding against division by zero."""

    return default if b == 0 else a / b


def pmi(count_ab: int, count_a: int, count_b: int, total: int) -> float:
    """Compute base-2 pointwise mutual information."""

    return math.log2(safe_div(count_ab * total, count_a * count_b, default=1e-9))


def llr(k11: int, k12: int, k21: int, k22: int) -> float:
    """Compute the Dunning (1993) log-likelihood ratio for a 2×2 table."""

    def _xlogx(x: float) -> float:
        return 0.0 if x <= 0 else x * math.log(x)

    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22
    total = row1 + row2

    return 2 * (
        _xlogx(k11)
        + _xlogx(k12)
        + _xlogx(k21)
        + _xlogx(k22)
        - _xlogx(row1)
        - _xlogx(row2)
        - _xlogx(col1)
        - _xlogx(col2)
        + _xlogx(total)
    )


__all__ = ["safe_div", "pmi", "llr"]
