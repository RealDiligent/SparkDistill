"""Score a candidate checkpoint's benchmark results against the frontier.

Mirrors `sparkinfer`'s speedup tiering (XL/L/M/S/XS bands over the frontier)
but applied to quality-benchmark deltas instead of decode speed.

    python -m eval.score --candidate eval/results/candidate.json --frontier eval/results/frontier.json --out eval/results/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.benchmarks import BENCHMARKS

# (lower_bound_pct, label) — first match wins, checked highest-to-lowest.
_TIER_BANDS = [
    (18.0, "XL"),
    (10.0, "L"),
    (6.0, "M"),
    (3.5, "S"),
    (2.0, "XS"),
]


def _pct_delta(candidate: float, frontier: float) -> float | None:
    """Percent improvement of `candidate` over `frontier`, or None when unscoreable.

    A non-positive frontier has no meaningful percentage baseline: returning a
    finite ratio (or +inf) here would mint a spurious tier from a single benchmark
    whose frontier has not been established yet — e.g. AIME24/GPQA sit at 0.0 for a
    small student, so the first candidate to land any non-zero score would jump to
    +inf and be tiered `XL` (max emission) for essentially no improvement, while also
    writing the non-finite token `Infinity` (invalid JSON) into the run ledger. Such
    a benchmark is still reported in `per_benchmark`, but excluded from tiering.
    """
    if frontier <= 0:
        return None
    return (candidate - frontier) / frontier * 100.0


def _tier_for(pct: float) -> str:
    for lower_bound, label in _TIER_BANDS:
        if pct >= lower_bound:
            return label
    return "none"


def score(candidate: dict[str, float], frontier: dict[str, float]) -> dict:
    per_benchmark = {}
    regressions = []
    best_pct = float("-inf")
    best_key = None

    for key, benchmark in BENCHMARKS.items():
        if key not in candidate or key not in frontier:
            continue
        pct = _pct_delta(candidate[key], frontier[key])
        per_benchmark[key] = {"candidate": candidate[key], "frontier": frontier[key], "pct_delta": pct}
        if pct is None:
            continue
        if pct > best_pct:
            best_pct, best_key = pct, key
        if pct < 0 and abs(pct) > benchmark.regression_floor_pct:
            regressions.append(key)

    if regressions:
        label = "REJECT"
    elif best_key is None:
        label = "REJECT"
    else:
        label = _tier_for(best_pct)

    return {
        "label": f"eval:{label}",
        "best_benchmark": best_key,
        "best_pct_delta": None if best_key is None else best_pct,
        "regressions": [f"regression-{key}" for key in regressions],
        "per_benchmark": per_benchmark,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True, help="scores json from eval.harness for the candidate")
    parser.add_argument("--frontier", type=Path, required=True, help="scores json from eval.harness for the frontier")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    candidate = json.loads(args.candidate.read_text())["scores"]
    frontier = json.loads(args.frontier.read_text())["scores"]
    report = score(candidate, frontier)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a non-finite delta must never silently reach report.json (and
    # from there the immutable run ledger) as the non-standard `Infinity`/`NaN` token.
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"{report['label']} (best: {report['best_benchmark']} {report['best_pct_delta']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
