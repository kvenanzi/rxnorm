"""Retrieval metrics shared by the baselines and the trained model.

Every method produces a `Retrieval` (top-k candidate ids + scores per query);
everything downstream is identical, so numbers across W&B runs are comparable.
"""

from dataclasses import dataclass

import numpy as np

from .data import PRIMARY_SOURCE, Data

SPLITS = ["train", "val", "test"]
TOP_K = 5
COVERAGES = [i / 10 for i in range(1, 11)]  # 10%, 20%, ... 100%
N_FAILURES = 200
METRIC_COLS = ["n", "acc@1", "recall@5", "ingredient_acc", "strength_acc", "dose_form_acc",
               "precision@50%", "precision@80%", "coverage@99%precision"]


@dataclass
class Retrieval:
    top: np.ndarray         # (n_queries, TOP_K) candidate ids; -1 = no candidate
    scores: np.ndarray      # (n_queries, TOP_K) similarity, 0 where no candidate


def precision_coverage(correct: np.ndarray, confidence: np.ndarray) -> dict:
    """Accept the most confident X% of predictions; how many of those are right?"""
    order = np.argsort(-confidence, kind="stable")
    cum_precision = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    n = len(order)
    curve = {c: float(cum_precision[max(1, int(round(c * n))) - 1]) for c in COVERAGES}
    ok = np.flatnonzero(cum_precision >= 0.99)
    coverage_at_99 = float((ok[-1] + 1) / n) if len(ok) else 0.0
    return {"curve": curve, "coverage@99%precision": coverage_at_99}


def evaluate(data: Data, ret: Retrieval, rows: np.ndarray, confidence: np.ndarray) -> dict:
    """Metrics for the queries at `rows`, using `confidence` (same length) for the curve."""
    top = ret.top[rows]
    valid = [data.valid[i] for i in rows]
    correct = np.array([top[q, 0] in v for q, v in enumerate(valid)])
    recall5 = np.array([any(t in v for t in top[q] if t >= 0) for q, v in enumerate(valid)])

    # Error taxonomy: is the top-1 right about each component, even when it's the
    # wrong drug overall? Compared against every valid target's labels.
    comp_correct = {k: np.zeros(len(rows), dtype=bool) for k in ("ingredient", "strength", "dose_form")}
    for q, v in enumerate(valid):
        if top[q, 0] < 0:
            continue
        pred = data.cand_key[top[q, 0]]
        truths = [data.cand_key[t] for t in v]
        for j, k in enumerate(comp_correct):
            comp_correct[k][q] = any(pred[j] == t[j] for t in truths)

    pc = precision_coverage(correct, confidence)
    metrics = {
        "n": int(len(rows)),
        "acc@1": float(correct.mean()),
        "recall@5": float(recall5.mean()),
        **{f"{k}_acc": float(v.mean()) for k, v in comp_correct.items()},
        "precision@50%": pc["curve"][0.5],
        "precision@80%": pc["curve"][0.8],
        "coverage@99%precision": pc["coverage@99%precision"],
    }
    return {"metrics": metrics, "curve": pc["curve"], "correct": correct, "comp_correct": comp_correct}


def evaluate_splits(data: Data, ret: Retrieval, confidence: np.ndarray,
                    margin: np.ndarray | None = None, extra_sources: bool = True) -> dict[str, dict]:
    """evaluate() on every split; optionally also the curve for a margin confidence.

    The keys train/val/test are the VA strings only, as in every published run.
    A folder that also carries another source's strings (e.g. MTHSPL) gets
    extra keys val_mthspl / test_mthspl. Empty splits (the all-train folder of
    the k-fold experiment has no val or test) are skipped rather than scored."""
    def one(rows: np.ndarray) -> dict:
        ev = evaluate(data, ret, rows, confidence[rows])
        if margin is not None:
            ev_m = evaluate(data, ret, rows, margin[rows])
            ev["curve_margin"] = ev_m["curve"]
            ev["metrics"]["margin/coverage@99%precision"] = ev_m["metrics"]["coverage@99%precision"]
        return ev

    results = {}
    for split in SPLITS:
        rows = data.rows(split)
        if len(rows):
            results[split] = one(rows)
    if extra_sources:
        for src in data.sources:
            if src == PRIMARY_SOURCE:
                continue
            for split in ("val", "test"):
                rows = data.rows(split, src)
                if len(rows):
                    results[f"{split}_{src.lower()}"] = one(rows)
    return results


def failure_rows(data: Data, ret: Retrieval, rows: np.ndarray, ev: dict, limit: int) -> list[list]:
    out = []
    for q_local, q in enumerate(rows):
        if ev["correct"][q_local]:
            continue
        wrong = [k for k, arr in ev["comp_correct"].items() if not arr[q_local]]
        truth = ", ".join(data.cand["name"][t] for t in sorted(data.valid[q]))
        top5 = [
            f"{ret.scores[q, j]:.3f}  {data.cand['name'][t]}"
            for j, t in enumerate(ret.top[q]) if t >= 0
        ]
        out.append([data.queries["vandf_string"][q], truth, "\n".join(top5) or "(no candidate)",
                    ", ".join(wrong) or "(none: wrong concept, same components)"])
        if len(out) >= limit:
            break
    return out


def print_summary(name: str, results: dict[str, dict]) -> None:
    print(f"\n{name}")
    print(f"{'split':<12}" + "".join(f"{c:>23}" for c in METRIC_COLS))
    for split, ev in results.items():
        m = ev["metrics"]
        print(f"{split:<12}" + "".join(
            f"{m[c]:>23}" if c == "n" else f"{m[c]:>23.3f}" for c in METRIC_COLS))


def log_results(run, name: str, data: Data, ret: Retrieval, results: dict[str, dict]) -> None:
    """Summary metrics, precision-vs-coverage plots, and a test-failure Table on a W&B run."""
    import wandb

    for split, ev in results.items():
        for k, v in ev["metrics"].items():
            run.summary[f"{split}/{k}"] = v
        ys, keys = [list(ev["curve"].values())], ["score"]
        if "curve_margin" in ev:
            ys.append(list(ev["curve_margin"].values()))
            keys.append("margin")
        run.log({f"{split}/precision_vs_coverage": wandb.plot.line_series(
            xs=COVERAGES, ys=ys, keys=keys,
            title=f"{name}: precision at automation rate ({split})", xname="coverage")})

    if "test" in results:
        table = wandb.Table(columns=["vandf_string", "truth", "top5", "wrong_components"],
                            data=failure_rows(data, ret, data.rows("test"), results["test"], N_FAILURES))
        run.log({"test/failures": table})
