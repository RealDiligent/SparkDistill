import json
import math

from eval.score import _pct_delta, score


def test_score_improvement_gets_expected_tier():
    candidate = {"gsm8k": 88.0, "humaneval": 80.0}
    frontier = {"gsm8k": 80.0, "humaneval": 80.0}
    report = score(candidate, frontier)
    assert report["label"] == "eval:L"  # (88-80)/80 = 10.0% -> L band
    assert report["best_benchmark"] == "gsm8k"
    assert report["regressions"] == []


def test_score_rejects_on_regression_beyond_floor():
    candidate = {"gsm8k": 88.0, "humaneval": 70.0}
    frontier = {"gsm8k": 80.0, "humaneval": 80.0}
    report = score(candidate, frontier)
    assert report["label"] == "eval:REJECT"
    assert "regression-humaneval" in report["regressions"]


def test_score_none_below_minimum_tier():
    candidate = {"gsm8k": 80.5}
    frontier = {"gsm8k": 80.0}
    report = score(candidate, frontier)
    assert report["label"] == "eval:none"


def test_zero_frontier_does_not_mint_top_tier():
    # AIME24/GPQA start at 0.0 for a small student; a candidate landing any non-zero
    # score there must NOT jump to eval:XL (max emission) off an unestablished frontier.
    candidate = {"aime24": 3.3, "gsm8k": 80.0}
    frontier = {"aime24": 0.0, "gsm8k": 80.0}
    report = score(candidate, frontier)
    assert report["label"] != "eval:XL"
    # gsm8k is flat (0% delta) and aime24 is unscoreable -> no rewardable improvement.
    assert report["label"] in ("eval:none", "eval:REJECT")
    assert report["best_benchmark"] != "aime24"
    # aime24 is still reported for humans, just excluded from tiering.
    assert report["per_benchmark"]["aime24"]["pct_delta"] is None


def test_zero_frontier_still_scores_other_benchmarks():
    # A real, established-benchmark improvement is unaffected by a co-submitted
    # zero-frontier benchmark.
    candidate = {"aime24": 3.3, "gsm8k": 96.0}
    frontier = {"aime24": 0.0, "gsm8k": 80.0}
    report = score(candidate, frontier)
    assert report["best_benchmark"] == "gsm8k"
    assert report["label"] == "eval:XL"  # (96-80)/80 = 20% -> XL band


def test_pct_delta_is_never_non_finite():
    # The value that flows into report.json / the ledger must always be JSON-valid.
    assert _pct_delta(3.3, 0.0) is None
    assert _pct_delta(0.0, 0.0) is None
    delta = _pct_delta(96.0, 80.0)
    assert delta is not None and math.isfinite(delta)


def test_report_serializes_as_strict_json_with_zero_frontier():
    report = score({"aime24": 3.3}, {"aime24": 0.0})
    # allow_nan=False mirrors eval.score.main: a non-finite delta would raise here.
    dumped = json.dumps(report, allow_nan=False)
    assert "Infinity" not in dumped
