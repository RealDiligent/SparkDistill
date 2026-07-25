"""Frontier merge + per-benchmark regression floors for mining-track scoring."""

from __future__ import annotations

from eval.benchmarks import BENCHMARKS

TIER_BENCHMARK = "triton"
GSM8K_BENCHMARK = "gsm8k"

# When triton improves at least this much vs frontier, GSM8K may regress up to 2%.
GSM8K_RELAX_TRITON_IMPROVEMENT_PCT = 2.0
GSM8K_RELAXED_REGRESSION_FLOOR_PCT = 2.0


def pct_delta(candidate: float, frontier: float) -> float:
    """Percent change vs. the frontier score. Always finite.

    Scores are fractions in `[0, 1]` (`eval.benchmarks.assert_fraction_scores`), so a
    frontier score of exactly 0 has no relative baseline to divide by. Returning
    `float("inf")` there broke two invariants the pipeline states elsewhere:

    - **Tiering.** `eval.score._TIER_BANDS` reads this value, and `inf` clears every
      band, so *any* nonzero candidate scored the top tier: over a 0.0 frontier,
      `0.001` earned the same `eval:XL` as `0.9`. A bucket seeded at 0 by its own
      `eval:BASELINE` run (`merge_frontier_scores` keeps a 0 high until it is beaten,
      and `runs/frontiers.json` already carries 0.0 entries) therefore handed out the
      maximum multiplier for a rounding-error gain.
    - **Serialization.** `inf` reaches `report["best_pct_delta"]`, which is written with
      `json.dumps` into `runs/ledger.jsonl`, `runs/<run-id>/result.json` and the gate
      report — as a bare `Infinity` literal, which is not valid JSON (RFC 8259) and is
      rejected by any strict parser. `assert_fraction_scores` already refuses NaN/Inf
      at ingestion for exactly this reason; nothing should mint one downstream.

    With no ratio available, fall back to the absolute gain in **percentage points** —
    the unit the tier bands and `regression_floor_pct` already speak. A genuine 0 -> 0.9
    jump still tiers `XL` (90.0), while 0 -> 0.001 tiers `none` (0.1). Candidates cannot
    be negative, so the regression path over a 0 frontier is unchanged (delta >= 0).
    """
    if frontier == 0:
        return (candidate - frontier) * 100.0
    return (candidate - frontier) / frontier * 100.0


def triton_pct_delta(candidate: dict[str, float], frontier: dict[str, float]) -> float | None:
    if TIER_BENCHMARK not in candidate or TIER_BENCHMARK not in frontier:
        return None
    return pct_delta(float(candidate[TIER_BENCHMARK]), float(frontier[TIER_BENCHMARK]))


def regression_floor_pct(benchmark_key: str, *, triton_pct: float | None) -> float:
    """Per-benchmark regression floor; GSM8K relaxes when triton improves enough."""
    benchmark = BENCHMARKS[benchmark_key]
    if benchmark_key == GSM8K_BENCHMARK and triton_pct is not None:
        if triton_pct >= GSM8K_RELAX_TRITON_IMPROVEMENT_PCT:
            return GSM8K_RELAXED_REGRESSION_FLOOR_PCT
    return benchmark.regression_floor_pct


def is_regression(
    benchmark_key: str,
    candidate_score: float,
    frontier_score: float,
    *,
    triton_pct: float | None,
) -> bool:
    delta = pct_delta(candidate_score, frontier_score)
    if delta >= 0:
        return False
    return abs(delta) > regression_floor_pct(benchmark_key, triton_pct=triton_pct)


def merge_frontier_scores(
    current: dict[str, float],
    candidate: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Raise per-benchmark frontier highs from a verified candidate.

    Any benchmark that beats the current frontier is updated — including GSM8K
    when a miner improves math reasoning even if Triton is flat.
    """
    merged = dict(current)
    updates: list[str] = []
    for key in BENCHMARKS:
        if key not in candidate:
            continue
        value = float(candidate[key])
        if key not in merged or value > float(merged[key]):
            merged[key] = value
            updates.append(key)
    return merged, updates
