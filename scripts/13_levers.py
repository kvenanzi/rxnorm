"""The two training levers, paired within split and seed (docs/post-2).

Does more training data (the FDA label names, MTHSPL) or an auxiliary strength
head move the final recipe? Both are run as one W&B grid (sweeps/levers.yaml):
2 data arms x 2 head arms x the six ingredient splits of §5.8 x two seeds =
48 runs, so every contrast is a within-(split, seed) difference. Two small
control sweeps separate "more steps" and "more rows" from "different strings".
Nothing here touches data/processed, data/splits, or the published artifacts:

  data/multi/<key>/       the six splits rebuilt with --sources VANDF,MTHSPL
  outputs/levers/         levers.json, levers.md
  W&B artifact vandf-rxnorm-multi (Colab reads the folders from it)

Run from the repo root:
  uv run scripts/13_levers.py build                     # ~20 s; six folders
  uv run scripts/13_levers.py upload                    # log them as vandf-rxnorm-multi
  uv run scripts/08_sweep.py --create --config sweeps/levers.yaml            # then notebooks/05_levers.ipynb
  uv run scripts/08_sweep.py --create --config sweeps/levers_control_steps.yaml
  uv run scripts/08_sweep.py --create --config sweeps/levers_control_size.yaml
  uv run scripts/13_levers.py summarize [--sweep ID ...] [--steps ID] [--size ID]
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))      # a bare Colab clone has no `pip install -e`

from rxnorm_vandf.stats import describe, paired_diffs, paired_stats   # noqa: E402


def load_script(name: str):
    """Import a numbered sibling script (its name is not an identifier)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SPLITS = load_script("12_split_seeds").SPLITS          # v1..v6 -> salt
SOURCES = ("VANDF", "MTHSPL")
MULTI_DIR = ROOT / "data" / "multi"
OUT = ROOT / "outputs" / "levers"
ARTIFACT = "vandf-rxnorm-multi"
SEEDS = (42, 1)

# Arm labels: (train_sources, aux) -> short name.
ARMS = {("VANDF", "none"): "base", ("VANDF,MTHSPL", "none"): "mthspl",
        ("VANDF", "strength"): "aux", ("VANDF,MTHSPL", "strength"): "both"}
METRICS = ["test/acc@1", "val/acc@1", "test/recall@5", "test/strength_acc", "test/dose_form_acc",
           "test/ingredient_acc", "test_mthspl/acc@1", "val_mthspl/acc@1", "test_mthspl/strength_acc"]
NS = ["test/n", "val/n", "test_mthspl/n"]
# Simple contrasts: the data effect at each head level, the head effect at each data level.
CONTRASTS = [("mthspl", "base"), ("both", "aux"), ("aux", "base"), ("both", "mthspl")]


def multi_dir(key: str) -> Path:
    return MULTI_DIR / key


def arm_of(config: dict) -> str | None:
    srcs = config.get("train_sources", "VANDF")
    if isinstance(srcs, (list, tuple)):
        srcs = ",".join(srcs)
    return ARMS.get((srcs.upper(), config.get("aux", "none")))


# ------------------------------------------------------------------------ build

def cmd_build(args: argparse.Namespace) -> None:
    """The six splits again, with the MTHSPL strings alongside the VA ones. Same
    salts, so the VA rows and their splits are identical to data/splits/<key>."""
    MULTI_DIR.mkdir(parents=True, exist_ok=True)
    for key, salt in SPLITS.items():
        d = multi_dir(key)
        if (d / "pairs.parquet").exists() and not args.rebuild:
            print(f"{key}: exists, skipping")
            continue
        subprocess.run([sys.executable, str(ROOT / "scripts" / "03_build_dataset.py"), "--salt", salt,
                        "--sources", ",".join(SOURCES), "--out-dir", str(d)], check=True, stdout=subprocess.DEVNULL)
        print(f"{key}: built with salt {salt}")
    print()
    print(f"{'split':<6}{'source':<8}{'train':>7}{'val':>7}{'test':>7}{'mixed':>7}")
    for key in SPLITS:
        for src, c in counts(key).items():
            print(f"{key:<6}{src:<8}{c.get('train', 0):>7}{c.get('val', 0):>7}{c.get('test', 0):>7}{c.get('mixed', 0):>7}")


def counts(key: str) -> dict[str, dict[str, int]]:
    import duckdb
    rows = duckdb.sql(f"SELECT source, split, count(*) FROM '{multi_dir(key) / 'pairs.parquet'}' GROUP BY 1, 2").fetchall()
    out: dict[str, dict[str, int]] = {}
    for src, split, n in rows:
        out.setdefault(src, {})[split] = int(n)
    return out


def cmd_upload(args: argparse.Namespace) -> None:
    import wandb
    from rxnorm_vandf import WANDB_PROJECT

    run = wandb.init(project=WANDB_PROJECT, job_type="dataset", name="upload-multi", group="levers")
    art = wandb.Artifact(ARTIFACT, type="dataset",
                         description="The six ingredient-hash splits (v1 published, v2..v6 re-draws) with the "
                                     "FDA label names (MTHSPL, license level 0) paired alongside the VA names; "
                                     "pairs.parquet has a `source` column. Built by scripts/13_levers.py.",
                         metadata={"splits": SPLITS, "sources": SOURCES, "rows": {k: counts(k) for k in SPLITS}})
    art.add_dir(str(MULTI_DIR))
    run.log_artifact(art)
    run.finish()
    print(f"logged {ARTIFACT}")


# --------------------------------------------------------------------- summarize

def fetch_cells(sweep_id: str) -> tuple[dict, dict]:
    """Finished, non-smoke runs of a sweep keyed by ((split, seed), arm); a re-run
    replaces an older finished cell. Returns (cells, run ids)."""
    from rxnorm_vandf.wb import ENTITY, PROJECT, api

    cells, ids = {}, {}
    for r in api().sweep(f"{ENTITY}/{PROJECT}/{sweep_id}").runs:
        if r.state != "finished" or r.config.get("smoke"):
            continue
        arm = arm_of(r.config)
        key = ((r.config.get("dataset_subdir"), int(r.config.get("seed", 42))), arm)
        if key in cells and r.created_at < ids[key][1]:
            continue
        s = dict(r.summary)
        cells[key] = {m: s.get(m) for m in METRICS + NS + ["best_epoch", "aux/num_labels", "aux/label_coverage"]}
        cells[key]["config"] = {k: r.config.get(k) for k in ("epochs", "max_extra_pairs", "train_sources", "aux")}
        ids[key] = (r.id, r.created_at)
    return cells, ids


def unit_key(unit: tuple) -> str:
    """(split, seed) -> "v1/s42": a JSON-safe unit name used in every diffs dict."""
    return f"{unit[0]}/s{unit[1]}"


def analyse(cells: dict) -> dict:
    """Paired contrasts over (split, seed) units, the two main effects (the mean
    of the two simple contrasts per unit), and the interaction."""
    have = {(unit_key(u), a): v for (u, a), v in cells.items()
            if all(v.get(m) is not None for m in ("test/acc@1", "val/acc@1"))}
    stats = {"simple": paired_stats(have, [m for m in METRICS if m not in ("val_mthspl/acc@1",)], CONTRASTS)}
    units = sorted({u for u, _ in have})
    full = [u for u in units if all((u, a) in have for a in ARMS.values())]
    for m in METRICS:
        if any(have[(u, "base")].get(m) is None for u in full):
            continue
        g = lambda u, a: have[(u, a)][m]
        data = {u: ((g(u, "mthspl") - g(u, "base")) + (g(u, "both") - g(u, "aux"))) / 2 for u in full}
        head = {u: ((g(u, "aux") - g(u, "base")) + (g(u, "both") - g(u, "mthspl"))) / 2 for u in full}
        inter = {u: g(u, "both") - g(u, "mthspl") - g(u, "aux") + g(u, "base") for u in full}
        if full:
            stats.setdefault("main", {})[m] = {"data": paired_diffs(data), "head": paired_diffs(head),
                                               "interaction": paired_diffs(inter)}
    stats["arm_means"] = {a: {m: describe([have[(u, a)][m] for u in units if (u, a) in have and have[(u, a)].get(m) is not None])
                              for m in METRICS if any((u, a) in have and have[(u, a)].get(m) is not None for u in units)}
                          for a in ARMS.values()}
    stats["units"] = full
    return stats


def controls(cells: dict, steps: dict, size: dict) -> dict:
    """Each control run against the base and mthspl cells of the same split at seed 42."""
    out = {}
    for name, ctl in (("steps", steps), ("size", size)):
        for (unit, _), c in ctl.items():
            base, mth = cells.get((unit, "base")), cells.get((unit, "mthspl"))
            if not base or not mth:
                continue
            out.setdefault(name, {})[unit[0]] = {
                "control": {m: c.get(m) for m in ("test/acc@1", "val/acc@1", "test_mthspl/acc@1")},
                "config": c["config"],
                "control-base": {m: c[m] - base[m] for m in ("test/acc@1", "val/acc@1") if c.get(m) is not None},
                "mthspl-control": {m: mth[m] - c[m] for m in ("test/acc@1", "val/acc@1") if c.get(m) is not None}}
    return out


def cmd_summarize(args: argparse.Namespace) -> None:
    from rxnorm_vandf.wb import LEVERS_CONTROL_SWEEP_IDS, LEVERS_SWEEP_IDS

    sweep_ids = args.sweep or LEVERS_SWEEP_IDS
    if not sweep_ids:
        raise SystemExit("no sweep id: pass --sweep or fill in LEVERS_SWEEP_IDS in rxnorm_vandf/wb.py")
    # Several sweeps can hold cells of the one grid (a make-up sweep for cells
    # that failed); the newest finished run of a cell wins across all of them.
    cells, ids = {}, {}
    for sid in sweep_ids:
        c, i = fetch_cells(sid)
        for key in c:
            if key not in cells or i[key][1] > ids[key][1]:
                cells[key], ids[key] = c[key], i[key]
    sweep_id = "+".join(sweep_ids)
    expected = [((k, s), a) for k in SPLITS for s in SEEDS for a in ARMS.values()]
    missing = [f"{k}/s{s}/{a}" for (k, s), a in expected if ((k, s), a) not in cells]
    if missing:
        print(f"missing {len(missing)} cells:", ", ".join(missing))
    stats = analyse(cells)
    ctl = {}
    ctl_ids = {"steps": args.steps or LEVERS_CONTROL_SWEEP_IDS.get("steps"),
               "size": args.size or LEVERS_CONTROL_SWEEP_IDS.get("size")}
    ctl_cells = {name: (fetch_cells(sid)[0] if sid else {}) for name, sid in ctl_ids.items()}
    ctl = controls(cells, ctl_cells["steps"], ctl_cells["size"])

    payload = {"sweep_id": sweep_id, "control_sweep_ids": ctl_ids, "arms": {f"{s}|{a}": n for (s, a), n in ARMS.items()},
               "cells": {f"{unit_key(u)}/{a}": {**v, "run_id": ids[(u, a)][0]} for (u, a), v in cells.items()},
               "missing": missing, "stats": stats, "controls": ctl}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "levers.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    md = render_markdown(cells, stats, ctl)
    (OUT / "levers.md").write_text(md)
    print(md)
    print(f"wrote {(OUT / 'levers.json').relative_to(ROOT)} and levers.md")


def render_markdown(cells: dict, stats: dict, ctl: dict) -> str:
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    arms = list(ARMS.values())
    lines = ["| split | seed | " + " | ".join(f"{a}: test (val)" for a in arms) + " | mthspl − base | aux − base |",
             "|---|---|" + "---|" * (len(arms) + 2)]
    for k in SPLITS:
        for s in SEEDS:
            row = [k, str(s)]
            for a in arms:
                c = cells.get(((k, s), a))
                row.append("–" if c is None else f"{f3(c['test/acc@1'])} ({f3(c['val/acc@1'])})")
            for con in ("mthspl-base", "aux-base"):
                dd = stats["simple"].get(con, {}).get("test/acc@1", {}).get("diffs", {})
                u = unit_key((k, s))
                row.append(f"{dd[u]:+.3f}" if u in dd else "–")
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "| effect (paired over split × seed) | metric | units | mean | sd | 95% CI | units with + sign | paired t |",
              "|---|---|---|---|---|---|---|---|"]
    for m, per in stats.get("main", {}).items():
        for name, d in per.items():
            ci = "–" if not d.get("ci95") else f"{d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}"
            lines.append(f"| {name} | {m} | {d['n_runs']} | {d['mean']:+.4f} | {d['sd']:.4f} | {ci} | "
                         f"{d['n_pos']} / {d['n_runs']} | {f3(d.get('t'))} |")
    lines += ["", "| simple contrast | metric | units | mean | sd | 95% CI | units with + sign |", "|---|---|---|---|---|---|---|"]
    for con, per in stats["simple"].items():
        for m, d in per.items():
            ci = "–" if not d.get("ci95") else f"{d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}"
            lines.append(f"| {con} | {m} | {d['n_runs']} | {d['mean']:+.4f} | {d['sd']:.4f} | {ci} | {d['n_pos']} / {d['n_runs']} |")
    lines += ["", "| arm | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for a, per in stats["arm_means"].items():
        lines.append(f"| {a} | " + " | ".join(f3(per[m]["mean"]) + f" ± {per[m]['sd']:.3f}" if m in per else "–" for m in METRICS) + " |")
    if ctl:
        lines += ["", "| control | split | test acc@1 | val acc@1 | control − base (test) | mthspl − control (test) |",
                  "|---|---|---|---|---|---|"]
        for name, per in ctl.items():
            for k, d in per.items():
                lines.append(f"| {name} ({d['config']}) | {k} | {f3(d['control']['test/acc@1'])} | {f3(d['control']['val/acc@1'])} | "
                             f"{d['control-base'].get('test/acc@1', float('nan')):+.3f} | {d['mthspl-control'].get('test/acc@1', float('nan')):+.3f} |")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--rebuild", action="store_true"); b.set_defaults(fn=cmd_build)
    sub.add_parser("upload").set_defaults(fn=cmd_upload)
    s = sub.add_parser("summarize")
    s.add_argument("--sweep", action="append", help="levers sweep id, repeatable (default: wb.LEVERS_SWEEP_IDS)")
    s.add_argument("--steps", help="steps-control sweep id"); s.add_argument("--size", help="size-control sweep id")
    s.set_defaults(fn=cmd_summarize)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
