"""Small-sample statistics for the paired experiments (no scipy: the numbered
scripts that use these are executed at import by the tests)."""

import math

# t_{0.975, df} for the paired intervals; the experiments pair 3..12 units.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042}


def describe(values: list[float], ns: list[int] | None = None) -> dict[str, float]:
    """Mean, sample sd, range; and, given the test sizes, the between-draw sd
    left after subtracting the binomial sampling variance p(1-p)/n."""
    k = len(values)
    mean = sum(values) / k
    var = sum((v - mean) ** 2 for v in values) / (k - 1) if k > 1 else 0.0
    out = {"n_runs": k, "mean": mean, "sd": math.sqrt(var), "min": min(values), "max": max(values),
           "range": max(values) - min(values)}
    if ns:
        sampling_var = sum(v * (1 - v) / n for v, n in zip(values, ns)) / k
        out["sampling_sd"] = math.sqrt(sampling_var)
        out["excess_sd"] = math.sqrt(max(var - sampling_var, 0.0))
    return out


def paired_diffs(diffs: dict) -> dict:
    """Summary of one set of within-unit differences: mean, sd, standard error,
    a 95% t interval, the paired t, and how many units have a positive sign."""
    vals = list(diffs.values())
    d = describe(vals)
    k = len(vals)
    se = d["sd"] / math.sqrt(k) if k > 1 else None
    t = T_975.get(k - 1)
    d.update({"se": se, "ci95": [d["mean"] - t * se, d["mean"] + t * se] if se is not None and t else None,
              "t": d["mean"] / se if se else None, "n_pos": sum(v > 0 for v in vals), "diffs": diffs})
    return d


def paired_stats(cells: dict[tuple, dict[str, float]], metrics: list[str],
                 contrasts: list[tuple[str, str]]) -> dict[str, dict[str, dict]]:
    """Within-unit differences between arms. `cells` maps (unit, arm) -> metrics;
    a unit is a split, or a (split, seed) pair. For each (a, b) contrast and
    metric, paired_diffs() over the units that have both arms."""
    units = sorted({u for u, _ in cells})
    out: dict[str, dict[str, dict]] = {}
    for a, b in contrasts:
        for m in metrics:
            diffs = {u: cells[(u, a)][m] - cells[(u, b)][m]
                     for u in units if (u, a) in cells and (u, b) in cells}
            if diffs:
                out.setdefault(f"{a}-{b}", {})[m] = paired_diffs(diffs)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)
