"""Build the W&B Report with live panels.

1. Logs a small `report-data` run whose metrics are the precision-vs-coverage
   curves as plain scalar series (coverage on x), so standard line panels can
   render them. (The `line_series` custom charts logged by the other scripts
   don't embed cleanly in programmatic reports.)
2. Creates the Report (private) with panels bound to the real runs: run
   comparer, val/acc@1 per epoch, the sweep's parallel coordinates, and the
   curves. Prints the URL. Edit the prose in the W&B editor; panels stay live.

Run from the repo root:  uv run scripts/10_report.py [--skip-data]
The follow-up's report:  uv run scripts/10_report.py --part 2   (tables from outputs/levers and outputs/kfold)
"""

import argparse
import json
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr

from rxnorm_vandf.wb import (ENTITY, NEGATIVES_SWEEP_ID, PROJECT, SPLIT_SEED_RUNS, STORY_RUNS, SWEEP_ID,
                             latest_calibrate_run, line_series, run)

TITLE = "VANDF → RxNorm: how far a small model gets at the clinical-drug level"
DESCRIPTION = ("Mapping VA drug strings to full RxNorm clinical drugs (ingredient + strength + dose form) "
               "with a fine-tuned bi-encoder, calibrated confidence, and abstention.")
GITHUB = "https://github.com/kvenanzi/rxnorm"
HF_MODEL = "https://huggingface.co/kvenanzi/vandf-rxnorm-biencoder"
TRAIN_RUNS = ["minilm", "minilm+normalizer", "final"]  # keys into STORY_RUNS
SIGNALS = ["cosine", "margin", "softmax", "platt"]


def log_report_data() -> None:
    """One run holding every curve the report needs, keyed by coverage."""
    curves = {}
    for name in ["tfidf", "minilm", "minilm+normalizer", "final"]:
        curves[f"progression/{name}"] = line_series(run(STORY_RUNS[name]), "test/precision_vs_coverage")["score"]
    cal = latest_calibrate_run()
    for pop, label in [("matched", "matched"), ("+hard", "hard")]:
        series = line_series(cal, f"test/{pop}/precision_vs_coverage")
        for s in SIGNALS:
            curves[f"final_{label}/{s}"] = series[s]

    r = wandb.init(project=PROJECT, entity=ENTITY, job_type="report", name="report-data",
                   config={"report_data": True, "calibrate_run": cal.id})
    wandb.define_metric("coverage")
    wandb.define_metric("*", step_metric="coverage")
    xs = sorted({x for pts in curves.values() for x, _ in pts})
    for x in xs:
        row = {"coverage": x}
        for key, pts in curves.items():
            for px, py in pts:
                if abs(px - x) < 1e-9:
                    row[key] = py
        wandb.log(row)
    r.finish()
    print(f"logged report-data run {r.id} with {len(curves)} curves")


def ids_filter(ids) -> str:
    # By run ID, not name: two sweep runs share names with the story runs, and a
    # name filter pulls them into these panels too.
    return "ID in [" + ", ".join(f"'{i}'" for i in ids) + "]"


def build_report() -> wr.Report:
    story = wr.Runset(entity=ENTITY, project=PROJECT, name="Baselines and trained models",
                      filters=ids_filter(STORY_RUNS.values()))
    trained = wr.Runset(entity=ENTITY, project=PROJECT, name="Trained models",
                        filters=ids_filter(STORY_RUNS[k] for k in TRAIN_RUNS))
    # By sweep id: the split-seed runs also have log_model=False and smoke=False,
    # and a config filter would pull them into the 18-run panels. (Booleans in
    # these filters must be Python-style: a lowercase `false` is saved as the
    # *string* "false", which matches no run and empties every panel.)
    sweep = wr.Runset(entity=ENTITY, project=PROJECT, name="Sweep (18 runs)",
                      filters=f"Metric('Sweep') == '{SWEEP_ID}'")
    curves = wr.Runset(entity=ENTITY, project=PROJECT, name="Curves", filters="Config('report_data') == True")
    seeds = wr.Runset(entity=ENTITY, project=PROJECT, name="Split and seed variance (8 runs)",
                      filters=ids_filter(SPLIT_SEED_RUNS.values()))
    negatives = wr.Runset(entity=ENTITY, project=PROJECT, name="Hard negatives × six splits (18 runs)",
                          filters=f"Metric('Sweep') == '{NEGATIVES_SWEEP_ID}'")

    W = 24  # panel grid width units
    blocks = [
        wr.TableOfContents(),
        wr.H1("The problem"),
        wr.P("Tools that normalize drug names mostly stop at the ingredient: they'll tell you a string is "
             "metoprolol. Interaction checking, reconciliation, and research cohorts need the clinical drug: "
             "metoprolol tartrate 25 MG oral tablet. Getting ingredient, strength, and dose form right at once "
             "is a compositional problem, and the VA's National Drug File (VANDF) comes pre-linked to RxNorm by "
             "NLM, so the labels are free. This report is the live record of the runs behind the write-up."),
        wr.P([wr.Link(text="Code and data prep on GitHub", url=GITHUB), " · ",
              wr.Link(text="Model on Hugging Face", url=HF_MODEL)]),

        wr.H1("Results at a glance"),
        wr.MarkdownBlock(
            "Test split: 1,848 VA strings whose ingredients never appear in training; the candidate pool is "
            "all 27,287 active RxNorm SCD/SBD.\n\n"
            "| Method | acc@1 | recall@5 | strength acc | dose form acc |\n|---|---|---|---|---|\n"
            "| Exact match | 0.000 | 0.000 | – | – |\n"
            "| TF-IDF char n-grams | 0.509 | 0.820 | 0.617 | 0.698 |\n"
            "| MiniLM fine-tuned | 0.836 | 0.967 | 0.884 | 0.935 |\n"
            "| + strength normalizer | 0.886 | 0.975 | 0.941 | 0.943 |\n"
            "| **SapBERT + negatives + normalizer** | **0.931** | **0.984** | 0.961 | 0.972 |\n\n"
            "The final row is the published split; across six ingredient draws the same recipe scores "
            "0.907 ± 0.019, and ingredient-matched negatives beat in-batch-only on every draw "
            "(see *Split and seed variance* below)."),
        wr.PanelGrid(runsets=[story], panels=[
            wr.BarPlot(title="Test acc@1", metrics=["test/acc@1"], layout=wr.Layout(x=0, y=0, w=W // 2, h=8)),
            wr.BarPlot(title="Test recall@5", metrics=["test/recall@5"], layout=wr.Layout(x=W // 2, y=0, w=W // 2, h=8)),
            wr.BarPlot(title="Ingredient / strength / dose-form accuracy of the top-1",
                       metrics=["test/ingredient_acc", "test/strength_acc", "test/dose_form_acc"],
                       layout=wr.Layout(x=0, y=8, w=W, h=8)),
            wr.RunComparer(diff_only=True, layout=wr.Layout(x=0, y=16, w=W, h=12)),
        ]),

        wr.H1("Training"),
        wr.P("Each model is a sentence-transformers bi-encoder trained with MultipleNegativesRankingLoss on "
             "(VA string, RxNorm name, hard negative) triplets, where the hard negative is a train-split product "
             "with the same ingredients and a different strength or dose form. Validation is scored after every "
             "epoch against the full candidate pool; the best epoch is kept and test is scored once."),
        wr.PanelGrid(runsets=[trained], panels=[
            wr.LinePlot(title="Validation acc@1 by epoch", x="epoch", y=["val/acc@1"],
                        layout=wr.Layout(x=0, y=0, w=W // 2, h=9)),
            wr.LinePlot(title="Training loss", x="train/global_step", y=["train/loss"], smoothing_factor=0.6,
                        layout=wr.Layout(x=W // 2, y=0, w=W // 2, h=9)),
        ]),

        wr.H1("The sweep: encoder × hard negatives × strength normalizer"),
        wr.P("An 18-run grid. Three effects, roughly additive: a biomedical encoder pre-trained on UMLS synonyms "
             "(SapBERT) over general-English encoders, +8.6 points val acc@1; a 40-line rule that appends the "
             "RxNorm-style concentration to the input (0.25% → 2.5 mg/ml), +4.8; ingredient-matched hard negatives "
             "over TF-IDF-mined or in-batch only, +3."),
        wr.PanelGrid(runsets=[sweep], panels=[
            wr.ParallelCoordinatesPlot(title="Sweep", columns=[
                # Typed metrics: a plain string is always read as a summary key, so
                # "c::base_model" became summary:c::base_model, which no run has.
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("base_model"), display_name="encoder"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("negatives"), display_name="hard negatives"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("normalize_strength"),
                                                 display_name="strength normalizer"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("val/acc@1")),
            ], layout=wr.Layout(x=0, y=0, w=W, h=10)),
            wr.ParameterImportancePlot(with_respect_to="val/acc@1", layout=wr.Layout(x=0, y=10, w=W // 2, h=8)),
            wr.BarPlot(title="val/acc@1 by encoder", metrics=["val/acc@1"], groupby="base_model",
                       groupby_aggfunc="mean", layout=wr.Layout(x=W // 2, y=10, w=W // 2, h=8)),
        ]),

        wr.H1("Split and seed variance"),
        wr.P("Every number above is one training run on one ingredient split. The final recipe was retrained on "
             "six draws of the split (v1 is the published one; v2–v6 are new hash salts) at seed 42, and on the "
             "published split at two more training seeds. Runs are named sapbert-ingredient-strength-split-v* and "
             "-seed-*; the v1 × seed 42 run reproduced the published run to the third decimal."),
        wr.MarkdownBlock(
            "| Split | val acc@1 | test acc@1 | val − test | test recall@5 | TF-IDF test acc@1 |\n"
            "|---|---|---|---|---|---|\n"
            "| v1 (published) | 0.881 | **0.931** | −0.049 | 0.984 | 0.509 |\n"
            "| v2 | 0.913 | 0.905 | +0.007 | 0.974 | 0.475 |\n"
            "| v3 | 0.919 | 0.905 | +0.014 | 0.979 | 0.555 |\n"
            "| v4 | 0.908 | 0.874 | +0.034 | 0.981 | 0.477 |\n"
            "| v5 | 0.895 | 0.921 | −0.026 | 0.985 | 0.490 |\n"
            "| v6 | 0.884 | 0.910 | −0.026 | 0.988 | 0.480 |\n"
            "| **mean ± sd** | 0.900 ± 0.015 | **0.907 ± 0.019** | −0.007 ± 0.031 | 0.982 ± 0.005 | 0.498 ± 0.031 |\n"
            "| v1, seeds 1 and 2 | 0.881, 0.880 | 0.930, 0.932 | | 0.984, 0.984 | |\n\n"
            "The split moves test acc@1 by about two points (sampling alone predicts 0.007); the training seed "
            "moves it by 0.001. The published draw is the most favorable of the six, and the validation-harder-than-"
            "test gap seen on it flips sign on half of the re-draws. Recall@5 and the component accuracies barely move."),
        wr.PanelGrid(runsets=[seeds], panels=[
            wr.BarPlot(title="Test acc@1 by run", metrics=["test/acc@1"], layout=wr.Layout(x=0, y=0, w=W // 2, h=9)),
            wr.BarPlot(title="Validation acc@1 by run", metrics=["val/acc@1"], layout=wr.Layout(x=W // 2, y=0, w=W // 2, h=9)),
            wr.BarPlot(title="Test recall@5 by run", metrics=["test/recall@5"], layout=wr.Layout(x=0, y=9, w=W // 2, h=8)),
            wr.LinePlot(title="Validation acc@1 by epoch", x="epoch", y=["val/acc@1"],
                        layout=wr.Layout(x=W // 2, y=9, w=W // 2, h=8)),
        ]),
        wr.H2("Does the hard-negatives effect survive a re-drawn split?"),
        wr.P("The sweep measured its three effects on one split, and the +3 for ingredient-matched negatives is "
             "about the size of the split noise. A second sweep re-ran that one factor on all six splits with "
             "SapBERT and the normalizer on (18 runs, one A100), so each comparison is paired within a split."),
        wr.MarkdownBlock(
            "| contrast | test acc@1, mean diff | 95% CI | splits with + sign | val acc@1, mean diff |\n"
            "|---|---|---|---|---|\n"
            "| ingredient − none | **+0.016** | +0.008 to +0.024 | 6 / 6 | +0.021 |\n"
            "| ingredient − tfidf | +0.013 | +0.002 to +0.023 | 6 / 6 | +0.013 |\n"
            "| tfidf − none | +0.003 | −0.013 to +0.019 | 4 / 6 | +0.008 |\n\n"
            "Ingredient-matched negatives win on every split, by 1.6 points on test on average (0.2 to 2.3). "
            "TF-IDF-mined negatives are indistinguishable from in-batch only. Recall@5 does not move with the "
            "strategy. The v1 cells reproduced the original sweep's to four decimals."),
        wr.PanelGrid(runsets=[negatives], panels=[
            wr.BarPlot(title="Test acc@1 by hard-negative strategy (mean over six splits)", metrics=["test/acc@1"],
                       groupby="negatives", groupby_aggfunc="mean", layout=wr.Layout(x=0, y=0, w=W // 2, h=8)),
            wr.BarPlot(title="Test acc@1 by split (mean over strategies)", metrics=["test/acc@1"],
                       groupby="dataset_subdir", groupby_aggfunc="mean", layout=wr.Layout(x=W // 2, y=0, w=W // 2, h=8)),
            wr.ParallelCoordinatesPlot(title="Split × strategy", columns=[
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("dataset_subdir"), display_name="split"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("negatives"), display_name="hard negatives"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("test/acc@1")),
            ], layout=wr.Layout(x=0, y=8, w=W, h=9)),
        ]),

        wr.H1("Calibration and abstention"),
        wr.P("A cosine score isn't a probability. Four confidence signals were compared: raw cosine, the top-1 − "
             "top-2 margin, a temperature-scaled softmax over the top-20, and a Platt (logistic) layer over all "
             "three. Thresholds are chosen on validation and applied to test. On strings that have an RxNorm "
             "answer, 79% can be auto-accepted at 98.8% precision. Once real drugs with no SCD/SBD (packs, "
             "ingredient-only concepts) are added, margin collapses and only the Platt layer works: 46% at 98.2%."),
        wr.PanelGrid(runsets=[curves], panels=[
            wr.LinePlot(title="Precision at automation rate: baseline → final (top-1 score as confidence)",
                        x="coverage", y=[f"progression/{n}" for n in ["tfidf", "minilm", "minilm+normalizer", "final"]],
                        range_y=(0.5, 1.0), layout=wr.Layout(x=0, y=0, w=W, h=9)),
            wr.LinePlot(title="Final model: strings with an answer", x="coverage",
                        y=[f"final_matched/{s}" for s in SIGNALS], range_y=(0.5, 1.0),
                        layout=wr.Layout(x=0, y=9, w=W // 2, h=9)),
            wr.LinePlot(title="Final model: + real drugs with no answer", x="coverage",
                        y=[f"final_hard/{s}" for s in SIGNALS], range_y=(0.5, 1.0),
                        layout=wr.Layout(x=W // 2, y=9, w=W // 2, h=9)),
        ]),

        wr.H1("What's still wrong"),
        wr.P("The final model gets 128 of 1,848 test strings wrong. TF-IDF's failures were dominated by strength "
             "(669 of 907 involved it); the normalizer and SapBERT cut that to 46 of 128, and what remains is an "
             "even spread: strength 46 (often an underdetermined string like MANNITOL 250MG/ML INJ, where RxNorm's "
             "answer carries a 50 ML volume the input never states), dose form 43, ingredient 31, and 25 SCD-vs-SBD "
             "twins with identical components. The test/failures table on the final run lists every one with its "
             "top-5 candidates."),
        wr.P([wr.Link(text="Final run: sapbert-ingredient-strength-final (see Tables → test/failures)",
                      url=f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{STORY_RUNS['final']}")]),

        wr.H1("Limitations"),
        wr.UnorderedList(items=[
            "Trained and evaluated on VA strings only; other systems' naming conventions are unmeasured.",
            "The headline 0.931 is the best of six ingredient draws; the same recipe averages 0.907 ± 0.019 across "
            "them, so read it as a band of about two points. The hard-negatives effect was re-checked on all six "
            "splits; the encoder and normalizer effects, and the calibration, rest on one split.",
            "The 99% precision target chosen on validation lands at 98.2–98.8% on test.",
            "RxNorm 2026-09-08; the candidate pool changes monthly.",
        ]),
    ]
    return blocks


# ------------------------------------------------------------- Part II (docs/post-2)

ROOT = Path(__file__).resolve().parent.parent
PART1_POST = "https://withinnoise.dev/blog/posts/vandf-rxnorm-biencoder/"
PART2_POST = "https://withinnoise.dev/blog/posts/vandf-rxnorm-interventions/"
PART1_REPORT = ("https://wandb.ai/kettle-labs/rxnorm-vandf/reports/"
                "VANDF-RxNorm-how-far-a-small-model-gets-at-the-clinical-drug-level--VmlldzoxNzkxNzIzNA")
LEVERS_TITLE = "VANDF → RxNorm, Part II: a second source vocabulary, a strength head, and cross-validation by ingredient"
LEVERS_DESCRIPTION = ("Two training interventions as a paired 2 × 2 grid over six ingredient draws and two seeds, "
                      "two controls, and seven-fold cross-validation with calibration on out-of-fold predictions.")
PILOT_RUNS = ["lqf7dett", "tiotc7iy", "9zw6jvov", "0usjnp44", "spja8f8i"]   # Part I's split-v1 run, then weights 0.01..0.5
CONTROL_RUNS = ["us4peb8v", "vxkly49t", "u4pby8j3", "b9lti8jv"]            # steps v1, v4; size v1, v4
ARM_LABELS = {"base": "VA only", "mthspl": "+ SPL", "aux": "+ head", "both": "+ SPL + head"}


def ci(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{lo:+.3f} to {hi:+.3f}"


def levers_tables() -> dict:
    """Markdown tables for the Part II report, built from the analysis outputs, and the run ids
    each panel is bound to (the grid's by the cell each run fills, so failed and superseded runs
    never appear)."""
    lev = json.loads((ROOT / "outputs" / "levers" / "levers.json").read_text())
    oof = json.loads((ROOT / "outputs" / "kfold" / "oof.json").read_text())
    cal = json.loads((ROOT / "outputs" / "kfold" / "calibration_oof.json").read_text())
    cells, main = lev["cells"], lev["stats"]["main"]
    effects = ["| effect | metric | mean | 95% CI | units positive |", "|---|---|---|---|---|"]
    for key, label in (("data", "SPL strings"), ("head", "strength head"), ("interaction", "interaction")):
        for m, mlabel in (("test/acc@1", "VA test acc@1"), ("val/acc@1", "VA val acc@1"),
                          ("test/strength_acc", "VA test strength accuracy"), ("test_mthspl/acc@1", "SPL test acc@1")):
            d = main[m][key]
            effects.append(f"| {label} | {mlabel} | {d['mean']:+.4f} | {ci(d)} | {d['n_pos']} / {d['n_runs']} |")
    arms = ["| arm (12 runs each) | VA test acc@1 | VA val acc@1 | SPL test acc@1 |", "|---|---|---|---|"]
    for a, label in ARM_LABELS.items():
        per = lev["stats"]["arm_means"][a]
        arms.append(f"| {label} | " + " | ".join(f"{per[m]['mean']:.3f} ± {per[m]['sd']:.3f}"
                                                 for m in ("test/acc@1", "val/acc@1", "test_mthspl/acc@1")) + " |")
    metrics = ("test/acc@1", "val/acc@1", "test_mthspl/acc@1")
    controls = ["| run (seed 42) | v1 VA test | v1 VA val | v1 SPL test | v4 VA test | v4 VA val | v4 SPL test |",
                "|---|---|---|---|---|---|---|"]
    for label, per in (("VA only, 4 epochs", {k: cells[f"{k}/s42/base"] for k in ("v1", "v4")}),
                       ("VA + all SPL, 4 epochs", {k: cells[f"{k}/s42/mthspl"] for k in ("v1", "v4")}),
                       ("VA only, 16 epochs (steps control)", {k: lev["controls"]["steps"][k]["control"] for k in ("v1", "v4")}),
                       ("VA + 9,300 SPL, 4 epochs (size control)", {k: lev["controls"]["size"][k]["control"] for k in ("v1", "v4")})):
        controls.append(f"| {label} | " + " | ".join(f"{per[k][m]:.3f}" for k in ("v1", "v4") for m in metrics) + " |")
    folds = ["| fold | test n | best epoch | val acc@1 | test acc@1 | test recall@5 |", "|---|---|---|---|---|---|"]
    for f, r in oof["folds"].items():
        folds.append(f"| {f}{' (published test set)' if f == 'fold0' else ''} | {r['test/n']:,} | {r['best_epoch']} | "
                     f"{r['val/acc@1']:.3f} | {r['test/acc@1']:.3f} | {r['test/recall@5']:.3f} |")
    o = oof["oof"]
    folds.append(f"| **pooled out-of-fold** | {o['n']:,} | | | **{o['acc@1']:.3f}** "
                 f"(Wilson 95% {o['wilson95'][0]:.3f}–{o['wilson95'][1]:.3f}) | {o['recall@5']:.3f} |")
    calib = ["| target | precision, threshold chosen on the other six folds: mean ± sd (range) | coverage | "
             "precision, threshold from one fold applied to another: mean (minimum) |", "|---|---|---|---|"]
    for key in ("p95", "p99"):
        lofo, tr = cal["leave_one_fold_out"][key], cal["transfer"][key]
        precs = [v["precision"] for f, v in lofo.items() if f.startswith("fold")]
        calib.append(f"| {key[1:]}% | {lofo['precision']['mean']:.3f} ± {lofo['precision']['sd']:.3f} "
                     f"({min(precs):.3f}–{max(precs):.3f}) | {lofo['coverage']['mean']:.3f} | "
                     f"{tr['off_diagonal_mean']:.3f} ({tr['off_diagonal_min']:.3f}) |")
    return {"effects": "\n".join(effects), "arms": "\n".join(arms), "controls": "\n".join(controls),
            "folds": "\n".join(folds), "calibration": "\n".join(calib),
            "grid_ids": [v["run_id"] for v in cells.values()],
            "control_ids": CONTROL_RUNS + [cells[f"{k}/s42/{a}"]["run_id"] for k in ("v1", "v4") for a in ("base", "mthspl")],
            "fold_ids": [r["run_id"] for r in oof["folds"].values()],
            "n_cells": len(cells)}


def build_levers_report() -> list:
    t = levers_tables()
    assert t["n_cells"] == 48, f"the grid has {t['n_cells']} of 48 cells; run 13_levers.py summarize first"
    grid = wr.Runset(entity=ENTITY, project=PROJECT, name="Grid (48 runs)", filters=ids_filter(t["grid_ids"]))
    controls = wr.Runset(entity=ENTITY, project=PROJECT, name="Controls and the grid cells they are compared with",
                         filters=ids_filter(t["control_ids"]))
    folds = wr.Runset(entity=ENTITY, project=PROJECT, name="Fold models (7 runs)", filters=ids_filter(t["fold_ids"]))
    pilot = wr.Runset(entity=ENTITY, project=PROJECT, name="Head-weight pilot and its reference (5 runs)",
                      filters=ids_filter(PILOT_RUNS))
    W = 24
    return [
        wr.TableOfContents(),
        wr.P(["This report holds the runs behind ", wr.Link(text="Part II of the write-up", url=PART2_POST),
              ", which follows ", wr.Link(text="Part I", url=PART1_POST), " and ",
              wr.Link(text="its report", url=PART1_REPORT), ". Code: ", wr.Link(text="GitHub", url=GITHUB), "."]),

        wr.H1("Design"),
        wr.P("Part I's recipe (SapBERT fine-tuned with an in-batch softmax loss, ingredient-matched hard negatives, and "
             "a strength normalizer) is crossed with two training interventions. The first adds the FDA Structured "
             "Product Label names (MTHSPL), the only large RxNorm source at UMLS restriction level 0 besides the two "
             "already used, as training strings. The second adds an auxiliary strength-classification head on the "
             "query embedding, at loss weight 0.01. Each of the four arms is trained on the six ingredient draws of "
             "Part I at seeds 42 and 1, so every contrast is paired within a (draw, seed) unit and reported over "
             "twelve units. The VA validation and test sets are those of Part I; the SPL strings of each draw are "
             "scored under separate keys (val_mthspl, test_mthspl)."),

        wr.H1("The grid"),
        wr.MarkdownBlock("Main effects, each the mean of its two simple contrasts within a unit, with t intervals "
                         "over the twelve units:\n\n" + t["effects"] + "\n\nArm means over the twelve units:\n\n"
                         + t["arms"] + "\n\nNeither intervention changes accuracy on VA strings. The head lowers "
                         "validation accuracy on all twelve units and leaves strength accuracy unchanged. The SPL "
                         "strings raise accuracy on held-out SPL strings on all twelve units."),
        wr.PanelGrid(runsets=[grid], panels=[
            wr.ParallelCoordinatesPlot(title="Draw × seed × training sources × head", columns=[
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("dataset_subdir"), display_name="draw"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("seed"), display_name="seed"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("train_sources"), display_name="training sources"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.Config("aux"), display_name="head"),
                wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("test/acc@1")),
            ], layout=wr.Layout(x=0, y=0, w=W, h=10)),
            wr.BarPlot(title="VA test acc@1 by training sources (mean of 24 runs)", metrics=["test/acc@1"],
                       groupby="train_sources", groupby_aggfunc="mean", layout=wr.Layout(x=0, y=10, w=W // 2, h=8)),
            wr.BarPlot(title="VA test acc@1 by head (mean of 24 runs)", metrics=["test/acc@1"],
                       groupby="aux", groupby_aggfunc="mean", layout=wr.Layout(x=W // 2, y=10, w=W // 2, h=8)),
            wr.BarPlot(title="SPL test acc@1 by training sources (mean of 24 runs)", metrics=["test_mthspl/acc@1"],
                       groupby="train_sources", groupby_aggfunc="mean", layout=wr.Layout(x=0, y=18, w=W // 2, h=8)),
            wr.BarPlot(title="VA test strength accuracy by head (mean of 24 runs)", metrics=["test/strength_acc"],
                       groupby="aux", groupby_aggfunc="mean", layout=wr.Layout(x=W // 2, y=18, w=W // 2, h=8)),
        ]),
        wr.H2("Seed variation"),
        wr.P("Two seeds per draw and arm give 24 same-draw pairs. The seed contributes a standard deviation of 0.0045 "
             "to a single run's VA test acc@1, against a between-draw standard deviation of 0.018 for the VA-only "
             "arm. Part I's estimate of 0.001 came from three seeds on one draw on a GTX 1070. Re-runs at the same "
             "seed on the A100 agree to within 0.0005."),

        wr.H1("Controls"),
        wr.MarkdownBlock("One run per control at seed 42 on draws v1 and v4, with the grid cells they are compared "
                         "with. The steps control trains on VA strings for sixteen epochs, about the SPL arm's number "
                         "of optimizer steps; the size control adds 9,300 SPL rows, as many as there are VA rows.\n\n"
                         + t["controls"] + "\n\nOn SPL strings the size control retains the whole transfer increase, "
                         "and the steps control lowers transfer below the VA-only arm. On VA strings the differences "
                         "are single runs of the order of the seed variation."),
        wr.PanelGrid(runsets=[controls], panels=[
            wr.BarPlot(title="VA test acc@1", metrics=["test/acc@1"], layout=wr.Layout(x=0, y=0, w=W // 2, h=8)),
            wr.BarPlot(title="SPL test acc@1", metrics=["test_mthspl/acc@1"], layout=wr.Layout(x=W // 2, y=0, w=W // 2, h=8)),
            wr.RunComparer(diff_only=True, layout=wr.Layout(x=0, y=8, w=W, h=10)),
        ]),

        wr.H1("Cross-validation by ingredient"),
        wr.MarkdownBlock("Seven folds of Part I's ingredient hash; fold 0's test set is the published test set. Each "
                         "fold model trains the VA-only recipe at seed 42 and selects its epoch on the next fold. The "
                         "all-data model trains on every ingredient family for the median best epoch, which is 1.\n\n"
                         + t["folds"]),
        wr.PanelGrid(runsets=[folds], panels=[
            wr.BarPlot(title="Test acc@1 by fold", metrics=["test/acc@1"], layout=wr.Layout(x=0, y=0, w=W // 2, h=8)),
            wr.LinePlot(title="Validation acc@1 by epoch", x="epoch", y=["val/acc@1"],
                        layout=wr.Layout(x=W // 2, y=0, w=W // 2, h=8)),
        ]),
        wr.H2("Calibration on out-of-fold predictions"),
        wr.MarkdownBlock("Temperature and Platt layer fit on the pooled out-of-fold predictions of the +hard "
                         "population (12,227 matched and 819 hard unmatched strings). Each held-out value chooses the "
                         "threshold on some folds and measures precision on another.\n\n" + t["calibration"]
                         + "\n\nIn Part I the 99% target, with the threshold chosen on one validation split, achieved "
                         "0.982 on the +hard test population."),

        wr.H1("The head-weight pilot"),
        wr.P("Four head weights on the published draw at seed 42, VA strings only, on the GTX 1070, against the Part I "
             "run on the same draw. Validation accuracy falls as the weight rises, and strength accuracy does not "
             "change at any weight; the grid runs the head at 0.01."),
        wr.PanelGrid(runsets=[pilot], panels=[
            wr.LinePlot(title="Validation acc@1 by epoch", x="epoch", y=["val/acc@1"],
                        layout=wr.Layout(x=0, y=0, w=W // 2, h=8)),
            wr.BarPlot(title="Best validation acc@1", metrics=["val/acc@1"], layout=wr.Layout(x=W // 2, y=0, w=W // 2, h=8)),
        ]),

        wr.H1("Limitations"),
        wr.UnorderedList(items=[
            "Two seeds per draw; the seed's standard deviation of 0.0045 enters every contrast between arms that do "
            "not share batches, which includes the SPL contrast.",
            "The controls are single runs on two draws.",
            "One head weight, 0.01.",
            "The folds are one partition at one seed; 15% of VA strings, combination drugs whose ingredients fall in "
            "different folds, are never tested.",
            "RxNorm 2026-09-08; two naming conventions, the VA's and the FDA label names.",
        ]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2], default=1,
                    help="1: the first write-up's report; 2: the follow-up's (docs/post-2)")
    ap.add_argument("--skip-data", action="store_true", help="don't re-log the report-data run (part 1)")
    ap.add_argument("--url", help="update this existing report in place; replaces every block, "
                                  "including prose edited in the W&B editor")
    args = ap.parse_args()
    title, description, build = ((TITLE, DESCRIPTION, build_report) if args.part == 1
                                 else (LEVERS_TITLE, LEVERS_DESCRIPTION, build_levers_report))
    if args.part == 1 and not args.skip_data:
        log_report_data()
    if args.url:
        report = wr.Report.from_url(args.url)
        report.blocks = build()
    else:
        report = wr.Report(entity=ENTITY, project=PROJECT, title=title, description=description, blocks=build())
    report.save()
    print(f"\nreport: {report.url}")


if __name__ == "__main__":
    main()
