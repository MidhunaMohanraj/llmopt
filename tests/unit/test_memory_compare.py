"""Tests for comparing a memory estimate with a measured peak."""

from __future__ import annotations

import pytest

from llmopt.models.config import compare_with_measured

GIB = 1024**3


def test_gap_is_measured_minus_estimate() -> None:
    result = compare_with_measured(1 * GIB, 2 * GIB)
    assert result.gap_bytes == GIB
    assert result.gap_pct == pytest.approx(100.0)
    assert result.estimate_is_lower_bound


def test_estimate_above_measured_is_flagged() -> None:
    result = compare_with_measured(3 * GIB, 2 * GIB)
    assert result.gap_bytes == -GIB
    assert not result.estimate_is_lower_bound


def test_non_positive_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        compare_with_measured(0, GIB)
    with pytest.raises(ValueError, match="greater than 0"):
        compare_with_measured(GIB, -1)
