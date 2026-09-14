"""Figures for the write-up, from W&B run data and local models.

Writes docs/post/figures/*.png and figures/data.json (the numbers behind each).
Run from the repo root:  uv run scripts/11_figures.py
                         uv run scripts/11_figures.py --only split_seeds   # one figure; data.json merged
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rxnorm_vandf.wb import STORY_RUNS, SWEEP_ID, api, latest_calibrate_run, line_series, run

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "post" / "figures"

# Palette: categorical slots in fixed order (blue, orange, aqua, yellow), light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "text.color": INK,
    "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})


def style(ax, xgrid=True):
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_progression(summaries: dict) -> dict:
    names = list(STORY_RUNS)
    labels = ["Exact match", "TF-IDF", "MiniLM", "MiniLM\n+ normalizer", "SapBERT + negs\n+ normalizer"]
    acc = [summaries[n]["test/acc@1"] for n in names]
    r5 = [summaries[n]["test/recall@5"] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(names))
    h = 0.34
    ax.barh(y + h / 2 + 0.02, acc, height=h, color=SERIES[0], label="acc@1")
    ax.barh(y - h / 2 - 0.02, r5, height=h, color=SERIES[1], label="recall@5")
    for yi, v in zip(y, acc):
        ax.text(v + 0.01, yi + h / 2 + 0.02, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Test split, ingredients never seen in training")
    ax.set_title("From string matching to a fine-tuned bi-encoder", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right")   # the exact-match row is empty there
    style(ax)
    fig.savefig(OUT / "progression.png")
    plt.close(fig)
    return {"labels": labels, "acc@1": acc, "recall@5": r5}


def fig_sweep_effects() -> dict:
    sweep = api().sweep(f"kettle-labs/rxnorm-vandf/{SWEEP_ID}")
    rows = [(r.config["base_model"].split("/")[-1], r.config["negatives"],
             r.config["normalize_strength"], r.summary["val/acc@1"]) for r in sweep.runs]
    factors = {
        "Encoder": {"SapBERT": 0, "all-MiniLM-L6-v2": 0, "bge-small-en-v1.5": 0},
        "Hard negatives": {"ingredient": 1, "tfidf": 1, "none": 1},
        "Strength normalizer": {"on": 2, "off": 2},
    }
    means = {}
    for m, n, s, v in rows:
        means.setdefault(("Encoder", "SapBERT" if "SapBERT" in m else m), []).append(v)
        means.setdefault(("Hard negatives", n), []).append(v)
        means.setdefault(("Strength normalizer", "on" if s else "off"), []).append(v)
    means = {k: float(np.mean(v)) for k, v in means.items()}

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ytick, ylab, headers, yi = [], [], [], 0
    for factor, levels in factors.items():
        ytick.append(yi); ylab.append(factor); headers.append(len(ylab) - 1)   # header row
        yi += 1
        for level in levels:
            v = means[(factor, level)]
            ax.plot([0.70, v], [yi, yi], color=GRID, linewidth=1.2, zorder=1)
            ax.scatter([v], [yi], s=70, color=SERIES[0], zorder=2, edgecolor=SURFACE, linewidth=1.5)
            ax.text(v + 0.006, yi, f"{v:.3f}", va="center", color=INK2, fontsize=9)
            ytick.append(yi); ylab.append(f"    {level}")
            yi += 1
        yi += 0.5
    ax.set_yticks(ytick, ylab)
    for i, lab in enumerate(ax.get_yticklabels()):
        if i in headers:
            lab.set_fontweight("bold"); lab.set_color(INK)
    ax.invert_yaxis()
    ax.set_xlim(0.70, 0.92)
    ax.set_xlabel("Mean validation acc@1 across the other two factors (18-run grid)")
    ax.set_title("What moved the needle in the sweep", loc="left", color=INK, fontsize=11)
    style(ax)
    fig.savefig(OUT / "sweep_effects.png")
    plt.close(fig)
    return {f"{k[0]} / {k[1]}": v for k, v in means.items()}


def fig_precision_coverage() -> dict:
    cal = latest_calibrate_run()
    signals = ["cosine", "margin", "softmax", "platt"]
    data = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, (pop, title) in zip(axes, [("matched", "Strings that have an RxNorm answer"),
                                       ("+hard", "+ real drugs with no answer")]):
        curves = line_series(cal, f"test/{pop}/precision_vs_coverage")
        data[pop] = {s: curves[s] for s in signals}
        ax.axhline(0.99, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)))
        for s, c in zip(signals, SERIES):
            xs, ys = zip(*curves[s])
            ax.plot(xs, ys, color=c, linewidth=2, label=s, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("Fraction auto-accepted (coverage)")
        ax.set_xlim(0.08, 1.02)
        ax.set_ylim(0.72, 1.008)
        style(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes[1].text(0.62, 0.9935, "99% precision", color=MUTED, fontsize=8.5, va="bottom")
    axes[0].set_ylabel("Precision of accepted answers")
    axes[1].legend(frameon=False, loc="lower left", title="confidence signal", title_fontsize=9)
    fig.suptitle("Final model, test: which confidence signal you threshold on matters", x=0.01, y=1.04,
                 ha="left", color=INK, fontsize=11)
    fig.savefig(OUT / "precision_coverage.png")
    plt.close(fig)
    return data


def fig_error_taxonomy() -> dict:
    """Share of test failures by which component was wrong, recomputed locally."""
    from rxnorm_vandf.data import load_data
    from rxnorm_vandf.eval import evaluate
    from rxnorm_vandf.tfidf import tfidf_retrieve

    data = load_data(ROOT / "data" / "processed")
    rows = data.rows("test")
    queries = data.queries["vandf_string"].tolist()

    def taxonomy(ret) -> Counter:
        ev = evaluate(data, ret, rows, ret.scores[rows, 0])
        c = Counter()
        for i in range(len(rows)):
            if ev["correct"][i]:
                continue
            wrong = [k for k, a in ev["comp_correct"].items() if not a[i]]
            if not wrong:
                key = "concept only (SCD/SBD twin)"
            elif "ingredient" in wrong:
                key = "ingredient"
            elif wrong == ["strength"]:
                key = "strength"
            elif wrong == ["dose_form"]:
                key = "dose form"
            else:
                key = "strength + dose form"
            c[key] += 1
        return c

    results = {"TF-IDF": taxonomy(tfidf_retrieve(data.cand["name"].tolist(), queries))}
    from sentence_transformers import SentenceTransformer
    from rxnorm_vandf.train import TrainConfig, retrieve
    for label, d in [("MiniLM", "all-minilm-l6-v2-ingredient"), ("SapBERT + normalizer", "sapbert-ingredient-strength-final")]:
        mdir = ROOT / "models" / d / "best"
        cfg = TrainConfig(**json.loads((mdir / "train_config.json").read_text()))
        m = SentenceTransformer(str(mdir)); m.max_seq_length = cfg.max_seq_length
        results[label] = taxonomy(retrieve(m, data, cfg, rows))

    cats = ["strength", "strength + dose form", "dose form", "concept only (SCD/SBD twin)", "ingredient"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    labels = list(results)
    left = np.zeros(len(labels))
    colors = SERIES + ["#898781"]
    for cat, color in zip(cats, colors):
        vals = np.array([results[l].get(cat, 0) for l in labels])
        ax.barh(labels, vals, left=left, color=color, label=cat, edgecolor=SURFACE, linewidth=1.5)
        left += vals
    for i, l in enumerate(labels):
        ax.text(left[i] + 8, i, f"{int(left[i])} wrong of {len(rows):,}", va="center", color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.25)
    ax.set_xlabel("Test strings mapped to the wrong clinical drug, by what was wrong")
    ax.set_title("What's still wrong, and how the mix changed", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8.5)
    style(ax)
    fig.savefig(OUT / "error_taxonomy.png")
    plt.close(fig)
    return {l: dict(c) for l, c in results.items()}


def fig_split_seeds() -> dict:
    """Val and test acc@1 of the final configuration on six ingredient splits, the
    seed repeats on the published split, and TF-IDF on the same splits. Reads
    outputs/split_seeds/summary.json (scripts/12_split_seeds.py summarize)."""
    summary = json.loads((ROOT / "outputs" / "split_seeds" / "summary.json").read_text())
    rows = {r["job"]: r for r in summary["rows"]}
    splits = [f"v{i}" for i in range(1, 7)]
    seed_rows = [rows[j] for j in ("seed-1", "seed-2") if j in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True,
                             gridspec_kw={"width_ratios": [3, 2]})
    y = np.arange(len(splits))
    for ax, (label, val_key, test_key, xlim) in zip(axes, [
            ("SapBERT + negatives + normalizer", "val/acc@1", "test/acc@1", None),
            ("TF-IDF (no training)", "tfidf/val/acc@1", "tfidf/test/acc@1", None)]):
        for yi, key in zip(y, splits):
            r = rows.get(f"split-{key}")
            if r is None or r.get(val_key) is None:
                continue
            v, t = r[val_key], r[test_key]
            ax.plot([v, t], [yi, yi], color=GRID, linewidth=2, zorder=1)
            ax.scatter([t], [yi], s=60, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=1.2,
                       label="test" if yi == 0 else None)
            ax.scatter([v], [yi], s=60, color=SERIES[1], zorder=3, edgecolor=SURFACE, linewidth=1.2,
                       label="validation" if yi == 0 else None)
        if val_key == "val/acc@1":
            for k, r in enumerate(seed_rows):
                off = 0.22 * (k + 1)
                ax.scatter([r["test/acc@1"]], [off], s=34, facecolor=SURFACE, edgecolor=SERIES[0], linewidth=1.4,
                           zorder=3, label="test, other training seed" if k == 0 else None)
                ax.scatter([r["val/acc@1"]], [off], s=34, facecolor=SURFACE, edgecolor=SERIES[1], linewidth=1.4,
                           zorder=3, label="validation, other training seed" if k == 0 else None)
            st = summary["stats"].get("splits", {}).get("test/acc@1")
            if st:
                ax.axvspan(st["mean"] - st["sd"], st["mean"] + st["sd"], color=SERIES[0], alpha=0.08, zorder=0)
                ax.axvline(st["mean"], color=SERIES[0], alpha=0.35, linewidth=1, zorder=0)
        ax.set_title(label, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("acc@1 against all 27,287 candidates")
        style(ax)
    axes[0].set_yticks(y, [f"split {k}" + ("  (published)" if k == "v1" else "") for k in splits])
    axes[0].set_ylim(len(splits) - 0.5, -0.8)       # inverted; headroom for the seed markers on the v1 row
    for ax in axes:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The same recipe on six ingredient draws: the split moves the number more than the seed",
                 x=0.01, y=1.04, ha="left", color=INK, fontsize=11)
    fig.savefig(OUT / "split_seeds.png")
    plt.close(fig)
    return {"rows": summary["rows"], "stats": summary["stats"], "published": summary["published"]}


def fig_negatives_by_split() -> dict:
    """The hard-negative strategy on every split (sweeps/negatives_by_split.yaml):
    test and val acc@1 per split for ingredient / tfidf / none, and the paired
    ingredient − none differences. Reads outputs/split_seeds/negatives.json."""
    neg = json.loads((ROOT / "outputs" / "split_seeds" / "negatives.json").read_text())
    levels = ["ingredient", "tfidf", "none"]
    splits = [f"v{i}" for i in range(1, 7)]
    cells = neg["cells"]
    y = np.arange(len(splits))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True, gridspec_kw={"width_ratios": [3, 3, 2]})
    for ax, (metric, title) in zip(axes[:2], [("test/acc@1", "Test acc@1"), ("val/acc@1", "Validation acc@1")]):
        for yi, k in zip(y, splits):
            vals = [cells.get(f"{k}/{n}", {}).get(metric) for n in levels]
            present = [v for v in vals if v is not None]
            if len(present) > 1:
                ax.plot([min(present), max(present)], [yi, yi], color=GRID, linewidth=2, zorder=1)
            for v, c, n in zip(vals, SERIES, levels):
                if v is not None:
                    ax.scatter([v], [yi], s=60, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.2,
                               label=n if yi == 0 else None)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("acc@1, all 27,287 candidates")
        style(ax)
    ax = axes[2]
    d = neg["stats"].get("ingredient-none", {}).get("test/acc@1", {})
    diffs = d.get("diffs", {})
    ax.axvline(0, color=BASELINE, linewidth=1)
    for yi, k in zip(y, splits):
        if k in diffs:
            ax.plot([0, diffs[k]], [yi, yi], color=GRID, linewidth=2, zorder=1)
            ax.scatter([diffs[k]], [yi], s=60, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=1.2)
    if d.get("ci95"):
        ax.axvspan(d["ci95"][0], d["ci95"][1], color=SERIES[0], alpha=0.08, zorder=0)
        ax.axvline(d["mean"], color=SERIES[0], alpha=0.35, linewidth=1, zorder=0)
    ax.set_title("ingredient − none, test", loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel("paired difference (band: mean, 95% CI)")
    style(ax)
    axes[0].set_yticks(y, [f"split {k}" for k in splits])
    axes[0].set_ylim(len(splits) - 0.5, -0.6)
    for a in axes:
        lo, hi = a.get_xlim()
        a.set_xlim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), title="hard negatives", title_fontsize=8.5)
    fig.suptitle("Does ingredient-matched hard-negative mining survive a re-drawn split?",
                 x=0.01, y=1.04, ha="left", color=INK, fontsize=11)
    fig.savefig(OUT / "negatives_by_split.png")
    plt.close(fig)
    return {"cells": cells, "stats": neg["stats"], "replication": neg["replication"], "missing": neg["missing"]}


FIGURES = {
    "progression": lambda: fig_progression({name: dict(run(rid).summary) for name, rid in STORY_RUNS.items()}),
    "sweep_effects": fig_sweep_effects,
    "precision_coverage": fig_precision_coverage,
    "error_taxonomy": fig_error_taxonomy,
    "split_seeds": fig_split_seeds,
    "negatives_by_split": fig_negatives_by_split,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FIGURES), action="append",
                    help="build only these figures; their keys are merged into the existing data.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = args.only or list(FIGURES)
    data_path = OUT / "data.json"
    payload = json.loads(data_path.read_text()) if args.only and data_path.exists() else {}
    for name in names:
        payload[name] = FIGURES[name]()
        print("wrote", (OUT / f"{name}.png").relative_to(ROOT))
    data_path.write_text(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
