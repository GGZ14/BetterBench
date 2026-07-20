"""Statistics for BetterBench.

Two families of numbers:

1. Distribution summaries for a single result set (per category / combined):
   TTFT, inter-token latency (ITL), and per-run tokens/sec — reported as
   1%-low / median / average / 99%-high, plus the gaming-style "1% low"
   (mean of the worst 1% of instantaneous token rates).

2. Paired A/B comparison (the repeatability path, see plan §7): given paired
   per-trial samples of a metric measured by interleaving config A and B on
   the same warmed box, report the difference with a bootstrap confidence
   interval and a significance verdict, and the run count needed to resolve a
   target minimum-detectable-effect (MDE).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Distribution summary
# --------------------------------------------------------------------------- #
@dataclass
class Dist:
    """Summary of one metric's distribution. Latencies in ms, rates in tok/s."""
    n: int
    mean: float
    median: float
    p1: float
    p99: float
    stdev: float
    cv: float                    # coefficient of variation (stdev/mean)
    low_1pct: float | None = None    # mean of worst 1% (rate-style metrics only)
    low_01pct: float | None = None   # mean of worst 0.1%
    iqr: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(samples: Sequence[float], rate_like: bool = False) -> Dist:
    """Summarize a metric. If rate_like (tokens/sec), also compute the
    gaming-style low-1%/0.1% (mean of the slowest tail)."""
    a = np.asarray([s for s in samples if s is not None and math.isfinite(s)], dtype=float)
    if a.size == 0:
        return Dist(0, 0, 0, 0, 0, 0, 0)
    mean = float(a.mean())
    stdev = float(a.std(ddof=1)) if a.size > 1 else 0.0
    q1, med, q3 = (float(x) for x in np.percentile(a, [25, 50, 75]))
    d = Dist(
        n=int(a.size),
        mean=mean,
        median=med,
        p1=float(np.percentile(a, 1)),
        p99=float(np.percentile(a, 99)),
        stdev=stdev,
        cv=(stdev / mean) if mean else 0.0,
        iqr=q3 - q1,
    )
    if rate_like:
        d.low_1pct = _worst_tail_mean(a, 0.01)
        d.low_01pct = _worst_tail_mean(a, 0.001)
    return d


def _worst_tail_mean(a: np.ndarray, frac: float) -> float:
    """Mean of the worst (lowest) `frac` fraction of values — the gaming '1% low'."""
    k = max(1, int(round(a.size * frac)))
    return float(np.sort(a)[:k].mean())


def itl_to_rate_samples(itl_ms: Sequence[float]) -> np.ndarray:
    """Convert per-token inter-token latencies (ms) to instantaneous tok/s."""
    a = np.asarray([x for x in itl_ms if x and x > 0], dtype=float)
    return 1000.0 / a if a.size else a


# --------------------------------------------------------------------------- #
# Paired A/B comparison
# --------------------------------------------------------------------------- #
@dataclass
class PairedResult:
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float            # B - A (absolute)
    pct_diff: float             # (B - A) / A * 100
    ci_low_pct: float           # CI on the percentage difference
    ci_high_pct: float
    conf: float                 # e.g. 0.95
    significant: bool           # does the CI exclude zero?
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def _t_crit(df: int, conf: float) -> float:
    """Two-sided t critical value, dependency-free (Cornish-Fisher expansion of
    the normal quantile). Accurate to <1% for df>=2; exact-ish table not needed."""
    z = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}.get(conf, 1.9599640)
    if df <= 0:
        return float("inf")
    g1 = (z**3 + z) / 4.0
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96.0
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df**2 + g3 / df**3


def paired_compare(
    a: Sequence[float],
    b: Sequence[float],
    metric: str = "metric",
    conf: float = 0.95,
    higher_is_better: bool = True,
    **_ignored,
) -> PairedResult:
    """Paired-t comparison of B vs A on the *per-trial difference*.

    a[i], b[i] are the same trial measured under config A and B (interleaved).
    Pairing cancels common-mode drift, so the CI is on the difference — the only
    honest way to resolve sub-few-percent effects. A paired-t interval is used
    (well-calibrated at small n) rather than a percentile bootstrap (which
    under-covers and can manufacture phantom significance)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if n < 2:
        return PairedResult(metric, n, float(a.mean() if n else 0),
                            float(b.mean() if n else 0), 0, 0, 0, 0, conf, False,
                            "insufficient pairs")
    diff = b - a
    mean_a = float(a.mean())
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1)) / math.sqrt(n)
    t = _t_crit(n - 1, conf)
    lo_abs, hi_abs = mean_diff - t * se, mean_diff + t * se

    pct = 100.0 * mean_diff / mean_a if mean_a else 0.0
    lo_pct = 100.0 * lo_abs / mean_a if mean_a else 0.0
    hi_pct = 100.0 * hi_abs / mean_a if mean_a else 0.0

    significant = (lo_abs > 0) or (hi_abs < 0)   # CI excludes zero
    if not significant:
        verdict = f"within noise (not distinguishable, {int(conf*100)}% CI straddles 0)"
    else:
        direction = "faster" if (mean_diff > 0) == higher_is_better else "slower"
        verdict = f"B is {abs(pct):.2f}% {direction} — SIGNIFICANT"
    return PairedResult(metric, n, mean_a, float(b.mean()), mean_diff, pct,
                        float(lo_pct), float(hi_pct), conf, significant, verdict)


# --------------------------------------------------------------------------- #
# Power analysis / minimum detectable effect
# --------------------------------------------------------------------------- #
_Z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def required_pairs_for_mde(paired_cv: float, mde_pct: float, conf: float = 0.95,
                           power: float = 0.80) -> int:
    """Approx number of interleaved pairs needed to detect an mde_pct effect,
    given the observed coefficient of variation of the *paired difference*.
    n ~= ((z_alpha + z_beta) * CV / effect)^2  (effect as a fraction)."""
    z_a = _Z.get(conf, 1.96)
    z_b = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}.get(power, 0.842)
    eff = mde_pct / 100.0
    if eff <= 0 or paired_cv <= 0:
        return 0
    return int(math.ceil(((z_a + z_b) * paired_cv / eff) ** 2))


def paired_ci_halfwidth_pct(a: Sequence[float], b: Sequence[float],
                            conf: float = 0.95) -> float:
    """Current half-width of the CI on the % difference — used by the
    sequential 'run until tight enough' stopping rule."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(a.size, b.size)
    if n < 2:
        return float("inf")
    diff = (b[:n] - a[:n])
    mean_a = float(a[:n].mean())
    if mean_a == 0:
        return float("inf")
    se_pct = 100.0 * (diff.std(ddof=1) / math.sqrt(n)) / abs(mean_a)
    return _t_crit(n - 1, conf) * se_pct


def enough_samples_for_percentile(n: int, pct: float) -> bool:
    """Is n big enough for a p1/p99 to mean anything? Rule of thumb: need at
    least ~10 samples beyond the tail, i.e. n * min(pct, 1-pct) >= 5."""
    tail = min(pct, 100 - pct) / 100.0
    return n * tail >= 5
