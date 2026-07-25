import json
import math

from eval.frontier import (
    GSM8K_RELAXED_REGRESSION_FLOOR_PCT,
    is_regression,
    merge_frontier_scores,
    pct_delta,
    regression_floor_pct,
    triton_pct_delta,
)
from eval.score import score


def test_merge_frontier_updates_gsm8k_when_improved():
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    candidate = {"triton": 0.42, "gsm8k": 0.65}
    merged, updates = merge_frontier_scores(frontier, candidate)
    assert merged["gsm8k"] == 0.65
    assert merged["triton"] == 0.428
    assert updates == ["gsm8k"]


def test_gsm8k_floor_relaxes_when_triton_improves_at_least_2pct():
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    candidate = {"triton": 0.48, "gsm8k": 0.6}
    assert triton_pct_delta(candidate, frontier) >= 2.0
    assert (
        regression_floor_pct("gsm8k", triton_pct=triton_pct_delta(candidate, frontier))
        == GSM8K_RELAXED_REGRESSION_FLOOR_PCT
    )


def test_score_allows_gsm8k_regression_up_to_2pct_when_triton_improves():
    candidate = {"triton": 0.48, "gsm8k": 0.591}
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    report = score(candidate, frontier)
    assert report["label"] == "eval:L"
    assert report["regressions"] == []
    assert report["gsm8k_regression_floor_pct"] == 2.0


def test_score_rejects_gsm8k_regression_beyond_relaxed_floor():
    candidate = {"triton": 0.48, "gsm8k": 0.58}
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    report = score(candidate, frontier)
    assert report["label"] == "eval:REJECT"
    assert "regression-gsm8k" in report["regressions"]


def test_score_keeps_1pct_gsm8k_floor_when_triton_gain_is_small():
    candidate = {"triton": 0.433, "gsm8k": 0.591}
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    report = score(candidate, frontier)
    assert report["label"] == "eval:REJECT"
    assert "regression-gsm8k" in report["regressions"]
    assert report["gsm8k_regression_floor_pct"] == 1.0


def test_score_reports_frontier_updates_on_verified_run():
    candidate = {"triton": 0.48, "gsm8k": 0.65}
    frontier = {"triton": 0.428, "gsm8k": 0.6}
    report = score(candidate, frontier)
    assert set(report["frontier_updates"]) == {"triton", "gsm8k"}
    assert report["frontier_scores"]["gsm8k"] == 0.65
    assert report["frontier_scores"]["triton"] == 0.48


def test_pct_delta_is_always_finite_over_a_zero_frontier():
    """`inf` cleared every tier band and serialized as invalid JSON."""
    assert math.isfinite(pct_delta(0.001, 0.0))
    assert math.isfinite(pct_delta(0.9, 0.0))
    assert pct_delta(0.0, 0.0) == 0.0


def test_pct_delta_over_zero_frontier_scales_with_the_absolute_gain():
    """With no ratio available, percentage points keep the ordering meaningful."""
    assert pct_delta(0.001, 0.0) == 0.1
    assert pct_delta(0.9, 0.0) == 90.0
    assert pct_delta(0.02, 0.0) == 2.0


def test_pct_delta_unchanged_for_a_nonzero_frontier():
    assert pct_delta(0.48, 0.428) == (0.48 - 0.428) / 0.428 * 100.0
    assert pct_delta(0.428, 0.428) == 0.0


def test_rounding_error_gain_over_a_zero_frontier_is_not_the_top_tier():
    """A bucket seeded at triton 0.0 must not pay eval:XL for a 0.001 candidate."""
    report = score({"triton": 0.001, "gsm8k": 0.6}, {"triton": 0.0, "gsm8k": 0.6})
    assert report["label"] == "eval:none"
    assert report["best_pct_delta"] == 0.1


def test_real_gain_over_a_zero_frontier_still_tiers_xl():
    report = score({"triton": 0.9, "gsm8k": 0.6}, {"triton": 0.0, "gsm8k": 0.6})
    assert report["label"] == "eval:XL"


def test_zero_frontier_report_is_valid_json():
    """best_pct_delta lands in runs/ledger.jsonl and runs/<id>/result.json."""
    report = score({"triton": 0.001, "gsm8k": 0.6}, {"triton": 0.0, "gsm8k": 0.6})
    encoded = json.dumps(report)
    assert "Infinity" not in encoded
    # A strict RFC 8259 parser must accept it; Python's default would not have complained.
    json.loads(encoded, parse_constant=_reject_constant)


def _reject_constant(name: str):
    raise AssertionError(f"non-JSON constant in report: {name}")


def test_zero_frontier_is_never_a_regression():
    """Candidates are fractions >= 0, so the regression path over 0 is unchanged."""
    assert is_regression("gsm8k", 0.0, 0.0, triton_pct=None) is False
    assert is_regression("gsm8k", 0.5, 0.0, triton_pct=None) is False
