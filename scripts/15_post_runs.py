# /// script
# requires-python = ">=3.10"
# dependencies = ["plotly>=5.24"]
# ///
"""Run records and interactive charts for the two write-ups, from the local W&B export.

    uv run scripts/15_post_runs.py [--export wandb-export/rxnorm-vandf]

Writes:
  docs/post/figures/runs.json             Part I runs (before 2026-09-14 19:00 UTC): id, name, job
  docs/post-2/figures/runs.json           type, group, sweep, tags, created time, config, summary
  docs/post/figures/sweep_parallel.html   the 18-run sweep: parallel coordinates, settings to scores
  docs/post/figures/sweep_epochs.html     validation metrics by epoch for each sweep run
  docs/post/figures/progression_coverage.html  precision against the share accepted, for each model
                                          from TF-IDF to the final one (the runs' test tables)
  docs/post/figures/training_curves.html  validation acc@1 by epoch and training loss, three models
  docs/post-2/figures/levers_cells.html   every cell of the Part II grid, with a metric selector
                                          (from outputs/levers/levers.json, 13_levers.py summarize)
  docs/post-2/figures/kfold_epochs.html   the seven fold models by epoch, and their training loss

The export is written by wandb_export.py and is not committed; the files above are.
"""

import argparse
import json
import math
import re
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
OUT1, OUT2 = ROOT / "docs/post/figures", ROOT / "docs/post-2/figures"
PART2_START = "2026-09-14T19:00"
SWEEP_ID = "idhaaw5i"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK)
SECRET = re.compile(r"token|secret|password|api_key|apikey", re.I)
DROP_SUMMARY = ("_wandb", "_runtime", "_timestamp", "_step")
# The runs Part I is built from (rxnorm_vandf/wb.py STORY_RUNS), with the post's names.
STORY = {"xr7g32b0": "TF-IDF", "fked1jx1": "MiniLM", "x9p04t9i": "MiniLM + normalizer",
         "0d9ntjls": "SapBERT + normalizer (final)"}
KFOLD_GROUP = "kfold-k7"
ENCODER = {"SapBERT-from-PubMedBERT-fulltext": "SapBERT", "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
           "bge-small-en-v1.5": "bge-small-en-v1.5"}


def clean(v):
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 6)
    if isinstance(v, dict):
        if str(v.get("_type", "")).endswith("-file"):
            return {"file": v.get("path")}
        return {k: clean(x) for k, x in v.items() if not SECRET.search(k)}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def load_runs(export: Path) -> list[dict]:
    runs = []
    for row in json.loads((export / "runs.json").read_text()):
        r = json.loads((export / "runs" / row["id"] / "run.json").read_text())
        config = {k: (v["value"] if isinstance(v, dict) and "value" in v else v)
                  for k, v in r["config"].items()}
        config = {k: v for k, v in config.items()
                  if not k.startswith("_") and not SECRET.search(k) and v not in (None, [], {})}
        summary = {k: v for k, v in r["summary"].items() if not k.startswith(DROP_SUMMARY)}
        runs.append({k: r.get(k) for k in ("id", "name", "group", "job_type", "state", "sweep",
                                           "tags", "notes", "created_at")}
                    | {"config": clean(config), "summary": clean(summary)})
    return sorted(runs, key=lambda r: r["created_at"] or "")


def write_runs(runs, export, out, label):
    f = export / "export.json"   # written last; absent while an export is still running
    meta = json.loads(f.read_text()) if f.exists() else {"project": "within-noise/rxnorm-vandf",
                                                         "exported_at": "(export incomplete)"}
    doc = {"source": f"W&B project {meta['project']}, exported {meta['exported_at']}",
           "selection": label, "n_runs": len(runs), "runs": runs}
    (out / "runs.json").write_text(json.dumps(doc, indent=1))
    print("wrote", out / "runs.json", f"({len(runs)} runs)")


def layout(fig, height, **kw):
    fig.update_layout(template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=FONT, height=height, **kw)


# On a phone-width page the legend moves below the plot, away from the metric menu.
NARROW = {"legend.y": -0.28, "legend.yanchor": "top", "legend.x": 0, "legend.xanchor": "left",
          "margin.l": 55, "margin.r": 10, "margin.b": 150, "xaxis.tickangle": -90}


def save(fig, path, narrow=None):
    """Write a standalone page. `narrow` adds layout changes for widths under 600 px."""
    small, wide = NARROW | (narrow or {}), {}
    for k in small:   # the wide values to restore, read from the figure as built
        obj = fig.layout
        for part in k.split("."):
            obj = getattr(obj, part, None) if obj is not None else None
        wide[k] = obj
    script = (f"var gd=document.getElementById('{{plot_id}}');var S={json.dumps(small)},"
              f"W={json.dumps(wide, default=lambda v: v.to_plotly_json())};"
              "function fit(){Plotly.relayout(gd,window.innerWidth<600?S:W);}"
              "fit();window.addEventListener('resize',fit);")
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, post_script=script,
                   config={"displaylogo": False, "responsive": True,
                           "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
    # Shown in an iframe on the blog: no body margin, and the figures' surface colour.
    html = path.read_text().replace(
        "<head>", f"<head><style>html,body{{margin:0;background:{SURFACE}}}</style>", 1)
    path.write_text(html)
    print("wrote", path)


def sweep_rows(export, runs):
    rows = []
    for r in runs:
        if r["sweep"] != SWEEP_ID:
            continue
        c = r["config"]
        hist = json.loads((export / "runs" / r["id"] / "history.json").read_text())
        rows.append({"run": r, "encoder": ENCODER[c["base_model"].split("/")[-1]],
                     "negatives": c["negatives"],
                     "normalizer": "on" if c["normalize_strength"] else "off",
                     "epochs": [h for h in hist if h.get("val/acc@1") is not None]})
    assert len(rows) == 18, len(rows)
    return rows


def sweep_parallel(rows):
    """Each sweep run as one line across its three settings and its two scores. Plotly's
    parcoords trace lets a reader drag along an axis to select a range of runs."""
    # Axes run bottom to top, so each setting's best level (Figure 2) is listed last.
    levels = {"encoder": ["all-MiniLM-L6-v2", "bge-small-en-v1.5", "SapBERT"],
              "negatives": ["none", "tfidf", "ingredient"], "normalizer": ["off", "on"]}
    dims = [dict(label=k, values=[levels[k].index(x[k]) for x in rows],
                 tickvals=list(range(len(v))), ticktext=v, range=[-0.3, len(v) - 0.7])
            for k, v in levels.items()]
    val = [x["run"]["summary"]["val/acc@1"] for x in rows]
    test = [x["run"]["summary"]["test/acc@1"] for x in rows]
    lo = min(val + test) - 0.01
    dims += [dict(label="validation acc@1", values=val, range=[lo, 0.9], tickformat=".2f"),
             dict(label="test acc@1", values=test, range=[lo, 0.94], tickformat=".2f")]
    fig = go.Figure(go.Parcoords(
        dimensions=dims, labelfont=dict(size=13, color=INK), tickfont=dict(size=12, color=INK2),
        rangefont=dict(size=1, color=SURFACE),
        line=dict(color=val, colorscale=[[0, "#a9c8ee"], [1, "#123f80"]], showscale=True,
                  colorbar=dict(title="validation<br>acc@1", thickness=12, tickformat=".2f"))))
    layout(fig, 460, margin=dict(l=150, r=40, t=50, b=30))
    save(fig, OUT1 / "sweep_parallel.html", {"margin.l": 120, "margin.b": 30, "margin.r": 20})


def sweep_epochs(rows):
    metrics = [("val/acc@1", "validation acc@1"), ("val/recall@5", "validation recall@5"),
               ("val/strength_acc", "validation strength accuracy"),
               ("val/ingredient_acc", "validation ingredient accuracy"),
               ("val/dose_form_acc", "validation dose-form accuracy"),
               ("val/coverage@99%precision", "validation coverage at 99% precision")]
    colour = dict(zip(ENCODER.values(), SERIES))
    dash = {"ingredient": "solid", "tfidf": "dash", "none": "dot"}
    fig, n = go.Figure(), len(metrics)
    seen = set()
    for x in rows:
        label = f"{x['encoder']}, {x['negatives']} negatives, normalizer {x['normalizer']}"
        for j, (key, name) in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=[h["epoch"] for h in x["epochs"]], y=[h[key] for h in x["epochs"]],
                mode="lines+markers", visible=(j == 0), name=x["encoder"], legendgroup=x["encoder"],
                showlegend=(j == 0 and x["encoder"] not in seen),
                line=dict(color=colour[x["encoder"]], dash=dash[x["negatives"]],
                          width=2.2 if x["normalizer"] == "on" else 1.2),
                marker=dict(size=5), opacity=0.9,
                hovertemplate=f"{label}<br>run {x['run']['id']}<br>epoch %{{x:.0f}}<br>"
                              f"{name}: %{{y:.4f}}<extra></extra>"))
        seen.add(x["encoder"])
    buttons = [dict(label=name, method="update",
                    args=[{"visible": [i % n == j for i in range(len(rows) * n)]},
                          {"yaxis.title.text": name}]) for j, (_, name) in enumerate(metrics)]
    axis = dict(gridcolor=GRID, showgrid=True, linecolor=INK2, tickcolor=INK2)
    layout(fig, 520, margin=dict(l=70, r=20, t=70, b=80), hovermode="closest",
           legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
           updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.12, yanchor="bottom",
                             bgcolor=SURFACE, bordercolor=GRID)],
           xaxis=axis | dict(title="epoch", dtick=1), yaxis=axis | dict(title=metrics[0][1]),
           annotations=[dict(text="line style: solid = ingredient negatives, dashed = TF-IDF, "
                                  "dotted = none; thick = normalizer on",
                             xref="paper", yref="paper", x=0, y=-0.2, showarrow=False,
                             font=dict(size=12, color=INK2), xanchor="left")])
    save(fig, OUT1 / "sweep_epochs.html", {"annotations[0].y": -0.62, "annotations[0].text": "solid = ingredient negatives, "
         "dashed = TF-IDF,<br>dotted = none; thick = normalizer on", "legend.y": -0.25,
         "margin.b": 210, "xaxis.tickangle": 0})


def run_table(export, run_id, key):
    """A wandb.plot.line_series chart as {line: [(x, y), ...]}, from the run's table artifact."""
    squashed = key.replace("/", "")
    path = next((export / "artifacts/run_table").glob(f"run-{run_id}-{squashed}_table-*/**/*.table.json"))
    data = json.loads(path.read_text())
    cols = data["columns"]
    xi, ki, yi = cols.index("step"), cols.index("lineKey"), cols.index("lineVal")
    out = {}
    for row in data["data"]:
        out.setdefault(row[ki], []).append((float(row[xi]), float(row[yi])))
    return {k: sorted(v) for k, v in out.items()}


def menu_chart(traces, menus, path, height=480, narrow=None, legend=True):
    """traces: one list of go.Scatter per menu entry; menus: (label, layout changes)."""
    fig = go.Figure()
    sizes = [len(ts) for ts in traces]
    for j, ts in enumerate(traces):
        for tr in ts:
            tr.visible = j == 0
            fig.add_trace(tr)
    starts = [sum(sizes[:j]) for j in range(len(sizes))]
    buttons = [dict(label=label, method="update",
                    args=[{"visible": [starts[j] <= i < starts[j] + sizes[j] for i in range(sum(sizes))]},
                          changes]) for j, (label, changes) in enumerate(menus)]
    axis = dict(gridcolor=GRID, showgrid=True, linecolor=INK2, tickcolor=INK2)
    layout(fig, height, margin=dict(l=70, r=20, t=70, b=60), hovermode="closest", showlegend=legend,
           legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
           updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.12, yanchor="bottom",
                             bgcolor=SURFACE, bordercolor=GRID)],
           xaxis=axis, yaxis=axis)
    fig.update_layout({k.replace(".", "_"): v for k, v in menus[0][1].items()})
    save(fig, path, (narrow or {}) | {"xaxis.tickangle": 0})


def progression_coverage(export):
    signals = [("score", "top-1 score"), ("margin", "margin between the top two candidates")]
    curves = {rid: run_table(export, rid, "test/precision_vs_coverage") for rid in STORY}
    traces = []
    for key, sname in signals:
        traces.append([go.Scatter(
            x=[x for x, _ in curves[rid][key]], y=[y for _, y in curves[rid][key]],
            mode="lines+markers", name=label, line=dict(color=SERIES[i], width=2),
            marker=dict(size=6),
            hovertemplate=f"{label} (run {rid})<br>signal: {sname}<br>"
                          "accept the %{x:.0%} most confident<br>precision %{y:.3f}<extra></extra>")
            for i, (rid, label) in enumerate(STORY.items())])
    axes = {"xaxis.title.text": "share accepted (most confident first)",
            "xaxis.tickformat": ".0%", "yaxis.title.text": "precision of the accepted strings"}
    menu_chart(traces, [(f"confidence: {s}", axes) for _, s in signals],
               OUT1 / "progression_coverage.html", narrow={"legend.y": -0.3, "margin.b": 170})


def history(export, rid):
    return json.loads((export / "runs" / rid / "history.json").read_text())


def epoch_rows(h, key):
    return [(r["epoch"], r[key]) for r in h if r.get(key) is not None and r.get("epoch") is not None]


def loss_rows(h):
    return [(r["train/global_step"], r["train/loss"]) for r in h if r.get("train/loss") is not None]


def training_curves(export):
    rids = ["fked1jx1", "x9p04t9i", "0d9ntjls"]
    hs = {rid: history(export, rid) for rid in rids}
    def lines(rows_of, unit):
        return [go.Scatter(
            x=[x for x, _ in rows_of(hs[rid])], y=[y for _, y in rows_of(hs[rid])],
            mode="lines+markers" if unit == "epoch" else "lines", name=STORY[rid],
            line=dict(color=SERIES[i + 1], width=2), marker=dict(size=7),
            hovertemplate=f"{STORY[rid]} (run {rid})<br>{unit} %{{x:.0f}}<br>%{{y:.4f}}<extra></extra>")
            for i, rid in enumerate(rids)]
    menus = [("validation acc@1 by epoch", {"xaxis.title.text": "epoch", "xaxis.dtick": 1,
                                            "yaxis.title.text": "validation acc@1"}),
             ("training loss", {"xaxis.title.text": "training step", "xaxis.dtick": None,
                                "yaxis.title.text": "training loss (mean of each 10 steps)"})]
    menu_chart([lines(lambda h: epoch_rows(h, "val/acc@1"), "epoch"), lines(loss_rows, "step")],
               menus, OUT1 / "training_curves.html", narrow={"legend.y": -0.3, "margin.b": 170})


def kfold_epochs(export, runs):
    folds = sorted((r for r in runs if r["group"] == KFOLD_GROUP and "-fold" in r["name"]),
                   key=lambda r: r["name"])
    assert len(folds) == 7, len(folds)
    hs = {r["id"]: history(export, r["id"]) for r in folds}
    def lines(rows_of, unit):
        return [go.Scatter(
            x=[x for x, _ in rows_of(hs[r["id"]])], y=[y for _, y in rows_of(hs[r["id"]])],
            mode="lines+markers" if unit == "epoch" else "lines", name=f"fold {r['name'][-1]}",
            line=dict(color=SERIES[0], width=1.6), marker=dict(size=6), opacity=0.8,
            hovertemplate=f"fold {r['name'][-1]} (run {r['id']})<br>{unit} %{{x:.0f}}<br>"
                          "%{y:.4f}<extra></extra>") for r in folds]
    epoch = {"xaxis.title.text": "epoch", "xaxis.dtick": 1}
    menus = [("validation acc@1", epoch | {"yaxis.title.text": "validation acc@1"}),
             ("validation strength accuracy", epoch | {"yaxis.title.text": "validation strength accuracy"}),
             ("training loss", {"xaxis.title.text": "training step", "xaxis.dtick": None,
                                "yaxis.title.text": "training loss (mean of each 10 steps)"})]
    menu_chart([lines(lambda h: epoch_rows(h, "val/acc@1"), "epoch"),
                lines(lambda h: epoch_rows(h, "val/strength_acc"), "epoch"),
                lines(loss_rows, "step")], menus, OUT2 / "kfold_epochs.html", height=440,
               legend=False)


def levers_cells(levers: Path):
    d = json.loads(levers.read_text())
    arms = {"base": "VANDF only", "mthspl": "+ MTHSPL", "aux": "+ strength head", "both": "both"}
    metrics = [("test/acc@1", "test acc@1"), ("test/strength_acc", "test strength accuracy"),
               ("test_mthspl/acc@1", "held-out SPL strings, acc@1"),
               ("val/acc@1", "validation acc@1"), ("test/recall@5", "test recall@5")]
    units = sorted({k.rsplit("/", 1)[0] for k in d["cells"]},
                   key=lambda u: (u.split("/")[0], int(u.split("/")[1][1:])))
    fig, n = go.Figure(), len(metrics)
    for a, (arm, label) in enumerate(arms.items()):
        cells = [(u, d["cells"][f"{u}/{arm}"]) for u in units]
        xs = [units.index(u) + (a - 1.5) * 0.16 for u, _ in cells]
        for j, (key, name) in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=xs, y=[c.get(key) for _, c in cells], mode="markers", visible=(j == 0),
                name=label, marker=dict(color=SERIES[a], size=10, line=dict(color=SURFACE, width=1)),
                customdata=[[u.split("/")[0], u.split("/")[1][1:], c["run_id"], c["best_epoch"]]
                            for u, c in cells],
                hovertemplate=f"{label}<br>draw %{{customdata[0]}}, seed %{{customdata[1]}}<br>"
                              "run %{customdata[2]}, best epoch %{customdata[3]}<br>"
                              f"{name}: %{{y:.4f}}<extra></extra>"))
    buttons = [dict(label=name, method="update",
                    args=[{"visible": [i % n == j for i in range(len(arms) * n)]},
                          {"yaxis.title.text": name}]) for j, (_, name) in enumerate(metrics)]
    axis = dict(gridcolor=GRID, linecolor=INK2, tickcolor=INK2)
    layout(fig, 500, margin=dict(l=70, r=20, t=70, b=60), hovermode="closest",
           legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
           updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.12, yanchor="bottom",
                             bgcolor=SURFACE, bordercolor=GRID)],
           xaxis=axis | dict(title="ingredient draw and seed", showgrid=False,
                             tickvals=list(range(len(units))),
                             ticktext=[u.replace("/s", "<br>s") for u in units]),
           yaxis=axis | dict(title=metrics[0][1], showgrid=True))
    save(fig, OUT2 / "levers_cells.html",
         {"xaxis.ticktext": [u.replace("/", " ") for u in units], "margin.b": 190,
          "legend.y": -0.45})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, default=ROOT / "wandb-export/rxnorm-vandf")
    ap.add_argument("--levers", type=Path, default=ROOT / "outputs/levers/levers.json")
    args = ap.parse_args()
    runs = load_runs(args.export)
    write_runs([r for r in runs if r["created_at"] < PART2_START], args.export, OUT1,
               f"runs created before {PART2_START} UTC (Part I)")
    write_runs([r for r in runs if r["created_at"] >= PART2_START], args.export, OUT2,
               f"runs created from {PART2_START} UTC (Part II)")
    rows = sweep_rows(args.export, runs)
    sweep_parallel(rows)
    sweep_epochs(rows)
    levers_cells(args.levers)
    progression_coverage(args.export)
    training_curves(args.export)
    kfold_epochs(args.export, runs)


if __name__ == "__main__":
    main()
