"""Split and seed variance: retrain the final configuration on re-drawn splits.

Every number in the write-up comes from one training seed on one ingredient
hash. This script draws five more splits (salts v2..v6), retrains the final
configuration (SapBERT + ingredient negatives + strength normalizer) on each,
and repeats the original split with two more training seeds, so split
variance and run-to-run noise can be told apart. Nothing it writes goes near
the published split, models, or artifacts:

  data/splits/<key>/          one folder per split (v1 is a copy of data/processed)
  models/split-seeds/<run>/   checkpoints (no model artifact is logged)
  outputs/split_seeds/        tfidf.json, runs.json (ledger), summary.json, summary.md
  W&B artifact vandf-rxnorm-splits (Colab only; vandf-rxnorm-pairs is untouched)

Run from the repo root:
  uv run scripts/12_split_seeds.py build                # ~10s; draws the splits
  uv run scripts/12_split_seeds.py tfidf                # ~1 min; TF-IDF baseline per split
  uv run scripts/12_split_seeds.py train --only split-v2 --smoke --offline   # pipeline check
  uv run scripts/12_split_seeds.py train                # 8 runs, ~20 min each on a GTX 1070
  uv run scripts/12_split_seeds.py upload               # Colab: log the splits as an artifact
  uv run scripts/12_split_seeds.py summarize            # table + summary.json from W&B
  uv run scripts/12_split_seeds.py negatives            # paired analysis of sweeps/negatives_by_split.yaml
In Colab (notebooks/03_split_seeds.ipynb):  python scripts/12_split_seeds.py train --from-artifact
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))      # a bare Colab clone has no `pip install -e`

SPLITS = {f"v{i}": f"rxnorm-2026-09-08-v{i}" for i in range(1, 7)}
PUBLISHED_SPLIT = "v1"
SPLITS_DIR = ROOT / "data" / "splits"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "split-seeds"
OUT = ROOT / "outputs" / "split_seeds"
LEDGER = OUT / "runs.json"
GROUP = "split-seeds"
ARTIFACT = "vandf-rxnorm-splits"
RUN_PREFIX = "sapbert-ingredient-strength"

# job -> (split key, training seed). Six splits at the published seed, then the
# published split at two more seeds.
JOBS = {**{f"split-{k}": (k, 42) for k in SPLITS}, "seed-1": ("v1", 1), "seed-2": ("v1", 2)}
SPLIT_JOBS = [j for j, (_, s) in JOBS.items() if s == 42]
SEED_JOBS = [j for j, (k, _) in JOBS.items() if k == PUBLISHED_SPLIT]

# The final configuration, models/sapbert-ingredient-strength-final/best/train_config.json.
FINAL = dict(base_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", epochs=4, batch_size=64,
             lr=2e-5, warmup_ratio=0.1, max_seq_length=96, negatives="ingredient",
             normalize_text=True, normalize_strength=True)

METRICS = ["acc@1", "recall@5", "ingredient_acc", "strength_acc", "dose_form_acc"]


def run_name(job: str) -> str:
    return f"{RUN_PREFIX}-{job}"


def split_dir(key: str) -> Path:
    return SPLITS_DIR / key


# ------------------------------------------------------------------------ build

def cmd_build(args: argparse.Namespace) -> None:
    """Copy the published split to v1 and draw v2..v6 with 03_build_dataset.py."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for key, salt in SPLITS.items():
        d = split_dir(key)
        if (d / "pairs.parquet").exists() and not args.rebuild:
            print(f"{key}: exists, skipping")
            continue
        if key == PUBLISHED_SPLIT:
            # The exact files behind every published number, never a rebuild.
            d.mkdir(exist_ok=True)
            for f in ("pairs.parquet", "candidates.parquet", "unmatched.parquet"):
                shutil.copy2(PROCESSED / f, d / f)
            (d / "split.json").write_text(json.dumps(
                {"salt": salt, "test_pct": 15, "val_pct": 15, "rxnorm_release": "2026-09-08",
                 "copied_from": "data/processed"}, indent=2) + "\n")
            print(f"{key}: copied data/processed")
        else:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "03_build_dataset.py"),
                            "--salt", salt, "--out-dir", str(d)], check=True, stdout=subprocess.DEVNULL)
            print(f"{key}: built with salt {salt}")
    print()
    print(f"{'split':<6}{'salt':<24}{'train':>7}{'val':>7}{'test':>7}{'mixed':>7}{'cands':>7}")
    for key, salt in SPLITS.items():
        c = split_counts(key)
        print(f"{key:<6}{salt:<24}{c['train']:>7}{c['val']:>7}{c['test']:>7}{c['mixed']:>7}{c['candidates']:>7}")


def split_counts(key: str) -> dict[str, int]:
    import duckdb
    d = split_dir(key)
    rows = duckdb.sql(f"SELECT split, count(*) FROM '{d / 'pairs.parquet'}' GROUP BY 1").fetchall()
    counts = {s: int(n) for s, n in rows}
    counts["candidates"] = int(duckdb.sql(f"SELECT count(*) FROM '{d / 'candidates.parquet'}'").fetchone()[0])
    return counts


# ------------------------------------------------------------------------ tfidf

def cmd_tfidf(args: argparse.Namespace) -> None:
    """The character n-gram baseline on every split: a model-free measure of how hard each draw is."""
    from rxnorm_vandf.data import load_data
    from rxnorm_vandf.eval import evaluate_splits, print_summary
    from rxnorm_vandf.tfidf import tfidf_retrieve

    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for key in SPLITS:
        data = load_data(split_dir(key))
        ret = tfidf_retrieve(data.cand["name"].tolist(), data.queries["vandf_string"].tolist())
        results = evaluate_splits(data, ret, ret.scores[:, 0])
        print_summary(f"tfidf on split {key}", results)
        out[key] = {split: {m: ev["metrics"][m] for m in METRICS + ["n"]} for split, ev in results.items()}
    (OUT / "tfidf.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {(OUT / 'tfidf.json').relative_to(ROOT)}")


# ------------------------------------------------------------------------ upload

def cmd_upload(args: argparse.Namespace) -> None:
    """Log data/splits as one artifact so Colab can train on them. A new name:
    vandf-rxnorm-pairs (what the published notebooks resolve) is never re-logged."""
    import wandb
    from rxnorm_vandf import WANDB_PROJECT

    run = wandb.init(project=WANDB_PROJECT, job_type="dataset", name="upload-splits", group=GROUP)
    art = wandb.Artifact(ARTIFACT, type="dataset",
                         description="Six ingredient-hash splits of the VANDF -> RxNorm pairs (v1 is the "
                                     "published split; v2..v6 are re-draws) for the split-variance experiment.",
                         metadata={"splits": SPLITS, "rows": {k: split_counts(k) for k in SPLITS}})
    art.add_dir(str(SPLITS_DIR))
    run.log_artifact(art)
    run.finish()
    print(f"logged {ARTIFACT}")


# ------------------------------------------------------------------------ train

def cmd_train(args: argparse.Namespace) -> None:
    """Run the matrix, one subprocess per job: a clean CUDA context each time,
    Ctrl-C stops between runs, and a crash in one run does not take the rest."""
    jobs = [args.only] if args.only else list(JOBS)
    done = {e["job"] for e in read_ledger() if e.get("state") == "finished"}
    for job in jobs:
        if job in done and not args.only:
            print(f"{job}: finished earlier (see {LEDGER.relative_to(ROOT)}), skipping")
            continue
        cmd = [sys.executable, __file__, "_one", job]
        for flag in ("smoke", "offline"):
            if getattr(args, flag):
                cmd.append(f"--{flag}")
        if args.from_artifact:
            cmd += ["--from-artifact", args.from_artifact]
        print(f"\n=== {job}: {' '.join(cmd[2:])}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            raise SystemExit(f"{job} failed with exit code {proc.returncode}; rerun with --only {job}")


def cmd_one(args: argparse.Namespace) -> None:
    """Train one job in this process (called by cmd_train)."""
    import wandb
    from rxnorm_vandf.train import TrainConfig, train

    key, seed = JOBS[args.job]
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    name = run_name(args.job) + ("-smoke" if args.smoke else "")
    cfg = TrainConfig(**FINAL, seed=seed, smoke=args.smoke, log_model=False,
                      output_dir=str(MODELS_DIR / "smoke" if args.smoke else MODELS_DIR),
                      run_name=name, group=GROUP,
                      tags=[GROUP, f"split-{key}", f"seed-{seed}"])
    if args.from_artifact:
        cfg.data_dir, cfg.dataset_artifact, cfg.dataset_subdir = None, args.from_artifact, key
    else:
        cfg.data_dir = str(split_dir(key))
    train(cfg)
    if not args.smoke:
        append_ledger({"job": args.job, "run_name": name, "split": key, "salt": SPLITS[key], "seed": seed,
                       "run_id": latest_run_id(), "state": "finished",
                       "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def latest_run_id() -> str | None:
    """The id of the run that just finished: W&B leaves `wandb/latest-run` pointing at run-<ts>-<id>."""
    link = ROOT / "wandb" / "latest-run"
    try:
        return link.resolve().name.rsplit("-", 1)[-1]
    except OSError:
        return None


def read_ledger() -> list[dict]:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def append_ledger(entry: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = [e for e in read_ledger() if e["job"] != entry["job"]] + [entry]
    LEDGER.write_text(json.dumps(entries, indent=2) + "\n")


# --------------------------------------------------------------------- summarize

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


def fetch_runs() -> dict[str, dict]:
    """Finished runs of the group from W&B, keyed by job; the newest run wins a name clash."""
    from rxnorm_vandf.wb import ENTITY, PROJECT, api
    by_name = {}
    for r in api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP, "jobType": "train"}):
        if r.state != "finished" or r.config.get("smoke"):
            continue
        if r.name not in by_name or r.created_at > by_name[r.name].created_at:
            by_name[r.name] = r
    out = {}
    for job in JOBS:
        r = by_name.get(run_name(job))
        if r is None:
            continue
        s = dict(r.summary)
        out[job] = {"run_id": r.id, "created_at": str(r.created_at), "split_salt": r.config.get("split_salt"),
                    **{f"{split}/{m}": s.get(f"{split}/{m}") for split in ("val", "test") for m in METRICS + ["n"]}}
    return out


def cmd_summarize(args: argparse.Namespace) -> None:
    from rxnorm_vandf.wb import STORY_RUNS, run as wb_run

    runs = fetch_runs()
    missing = [j for j in JOBS if j not in runs]
    if missing:
        print(f"not finished yet: {', '.join(missing)}")
    tfidf = json.loads((OUT / "tfidf.json").read_text()) if (OUT / "tfidf.json").exists() else {}
    counts = {k: split_counts(k) for k in SPLITS if (split_dir(k) / "pairs.parquet").exists()}
    published = dict(wb_run(STORY_RUNS["final"]).summary)

    rows = []
    for job, r in runs.items():
        key, seed = JOBS[job]
        rows.append({"job": job, "split": key, "seed": seed, **r, "counts": counts.get(key),
                     "tfidf/test/acc@1": tfidf.get(key, {}).get("test", {}).get("acc@1"),
                     "tfidf/val/acc@1": tfidf.get(key, {}).get("val", {}).get("acc@1"),
                     "gap": r["val/acc@1"] - r["test/acc@1"]})
    stats = {}
    split_rows = [x for x in rows if x["job"] in SPLIT_JOBS]
    seed_rows = [x for x in rows if x["job"] in SEED_JOBS]
    if len(split_rows) > 1:
        stats["splits"] = {m: describe([x[m] for x in split_rows], [x[n] for x in split_rows])
                           for m, n in [("test/acc@1", "test/n"), ("val/acc@1", "val/n")]}
        stats["splits"]["gap"] = describe([x["gap"] for x in split_rows])
        stats["splits"]["test/recall@5"] = describe([x["test/recall@5"] for x in split_rows])
        if all(x["tfidf/test/acc@1"] is not None for x in split_rows):
            stats["splits"]["tfidf/test/acc@1"] = describe([x["tfidf/test/acc@1"] for x in split_rows])
    if len(seed_rows) > 1:
        stats["seeds"] = {m: describe([x[m] for x in seed_rows]) for m in ("test/acc@1", "val/acc@1")}
    payload = {"jobs": JOBS, "rows": rows, "stats": stats,
               "published": {"run_id": STORY_RUNS["final"],
                             **{f"{s}/{m}": published.get(f"{s}/{m}") for s in ("val", "test") for m in METRICS + ["n"]}}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    md = render_markdown(rows, stats, payload["published"])
    (OUT / "summary.md").write_text(md)
    print(md)
    print(f"wrote {(OUT / 'summary.json').relative_to(ROOT)} and summary.md")


def render_markdown(rows: list[dict], stats: dict, published: dict) -> str:
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    lines = ["| run | split | seed | train / val / test / mixed pairs | val acc@1 | test acc@1 | val−test | "
             "test recall@5 | test ingredient | test strength | test dose form | TF-IDF test acc@1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        c = x["counts"] or {}
        n = " / ".join(f"{c.get(s, 0):,}" for s in ("train", "val", "test", "mixed"))
        lines.append(f"| {x['job']} | {x['split']} | {x['seed']} | {n} | {f3(x['val/acc@1'])} | {f3(x['test/acc@1'])} | "
                     f"{x['gap']:+.3f} | {f3(x['test/recall@5'])} | {f3(x['test/ingredient_acc'])} | "
                     f"{f3(x['test/strength_acc'])} | {f3(x['test/dose_form_acc'])} | {f3(x['tfidf/test/acc@1'])} |")
    lines.append(f"| published run `{published['run_id']}` | v1 | 42 | – | {f3(published['val/acc@1'])} | "
                 f"{f3(published['test/acc@1'])} | {published['val/acc@1'] - published['test/acc@1']:+.3f} | "
                 f"{f3(published['test/recall@5'])} | {f3(published['test/ingredient_acc'])} | "
                 f"{f3(published['test/strength_acc'])} | {f3(published['test/dose_form_acc'])} | – |")
    lines.append("")
    if stats:
        lines += ["| statistic | metric | runs | mean | sd | min | max | sampling sd | excess sd |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for group, label in (("splits", "across splits (seed 42)"), ("seeds", "across seeds (split v1)")):
            for m, d in stats.get(group, {}).items():
                lines.append(f"| {label} | {m} | {d['n_runs']} | {d['mean']:.3f} | {d['sd']:.3f} | {d['min']:.3f} | "
                             f"{d['max']:.3f} | {f3(d.get('sampling_sd'))} | {f3(d.get('excess_sd'))} |")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------- negatives across splits

NEGATIVE_LEVELS = ["ingredient", "tfidf", "none"]
CONTRASTS = [("ingredient", "none"), ("tfidf", "none"), ("ingredient", "tfidf")]
PAIRED_METRICS = ["test/acc@1", "val/acc@1", "test/recall@5"]
# t_{0.975, df} for the paired interval; n is small and fixed, so no scipy import
# (tests execute this module at import).
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def paired_stats(cells: dict[tuple[str, str], dict[str, float]],
                 metrics: list[str] = PAIRED_METRICS) -> dict[str, dict[str, dict]]:
    """Within-split differences between negative strategies. `cells` maps
    (split, negatives) -> metrics. For each contrast and metric: the per-split
    differences, mean, sd, standard error, a 95% t interval, and how many
    splits have the expected (positive) sign."""
    splits = sorted({s for s, _ in cells})
    out: dict[str, dict[str, dict]] = {}
    for a, b in CONTRASTS:
        for m in metrics:
            diffs = {s: cells[(s, a)][m] - cells[(s, b)][m]
                     for s in splits if (s, a) in cells and (s, b) in cells}
            if not diffs:
                continue
            vals = list(diffs.values())
            d = describe(vals)
            k = len(vals)
            se = d["sd"] / math.sqrt(k) if k > 1 else None
            t = T_975.get(k - 1)
            d.update({"se": se, "ci95": [d["mean"] - t * se, d["mean"] + t * se] if se is not None and t else None,
                      "t": d["mean"] / se if se else None, "n_pos": sum(v > 0 for v in vals), "diffs": diffs})
            out.setdefault(f"{a}-{b}", {})[m] = d
    return out


def cmd_negatives(args: argparse.Namespace) -> None:
    """Table and paired contrasts for the negatives-by-split sweep, plus the
    replication rows: the v1 cells of the original sweep and the split-v1 run."""
    from rxnorm_vandf.wb import ENTITY, PROJECT, NEGATIVES_SWEEP_ID, SPLIT_SEED_RUNS, SWEEP_ID, api, run as wb_run

    sweep_id = args.sweep or NEGATIVES_SWEEP_ID
    if not sweep_id:
        raise SystemExit("no sweep id: pass --sweep or fill in NEGATIVES_SWEEP_ID in rxnorm_vandf/wb.py")

    def metrics_of(r) -> dict[str, float]:
        s = dict(r.summary)
        return {m: s.get(m) for m in PAIRED_METRICS + ["test/n", "val/n", "test/strength_acc", "test/dose_form_acc",
                                                        "test/ingredient_acc"]}

    cells, run_ids = {}, {}
    for r in api().sweep(f"{ENTITY}/{PROJECT}/{sweep_id}").runs:
        if r.state != "finished" or r.config.get("smoke"):
            continue
        key = (r.config.get("dataset_subdir"), r.config.get("negatives"))
        if key in cells and r.created_at < run_ids[key][1]:
            continue                                   # a re-run replaces an older finished cell
        cells[key], run_ids[key] = metrics_of(r), (r.id, r.created_at)
    expected = [(k, n) for k in SPLITS for n in NEGATIVE_LEVELS]
    missing = [f"{k}/{n}" for k, n in expected if (k, n) not in cells]
    if missing:
        print("missing cells:", ", ".join(missing))

    replication = {}
    for r in api().sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}").runs:
        if "SapBERT" in r.config.get("base_model", "") and r.config.get("normalize_strength") and r.state == "finished":
            replication[f"v1/{r.config['negatives']} (T4 sweep {SWEEP_ID}, {r.id})"] = metrics_of(r)
    if "split-v1" in SPLIT_SEED_RUNS:
        replication[f"v1/ingredient (split-seeds, {SPLIT_SEED_RUNS['split-v1']})"] = metrics_of(wb_run(SPLIT_SEED_RUNS["split-v1"]))

    stats = paired_stats(cells)
    payload = {"sweep_id": sweep_id,
               "cells": {f"{k}/{n}": {**v, "run_id": run_ids[(k, n)][0]} for (k, n), v in cells.items()},
               "missing": missing, "stats": stats, "replication": replication}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "negatives.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    md = render_negatives_markdown(cells, stats, replication)
    (OUT / "negatives.md").write_text(md)
    print(md)
    print(f"wrote {(OUT / 'negatives.json').relative_to(ROOT)} and negatives.md")


def render_negatives_markdown(cells: dict, stats: dict, replication: dict) -> str:
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    lines = ["| split | " + " | ".join(f"{n}: test (val)" for n in NEGATIVE_LEVELS)
             + " | ingredient − none, test | ingredient − none, val |",
             "|---|" + "---|" * (len(NEGATIVE_LEVELS) + 2)]
    for k in SPLITS:
        row = [k]
        for n in NEGATIVE_LEVELS:
            c = cells.get((k, n))
            row.append("–" if c is None else f"{f3(c['test/acc@1'])} ({f3(c['val/acc@1'])})")
        for m in ("test/acc@1", "val/acc@1"):
            dd = stats.get("ingredient-none", {}).get(m, {}).get("diffs", {})
            row.append(f"{dd[k]:+.3f}" if k in dd else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "| contrast | metric | splits | mean diff | sd | 95% CI | splits with + sign | paired t |",
              "|---|---|---|---|---|---|---|---|"]
    for contrast, per in stats.items():
        for m, d in per.items():
            ci = "–" if not d.get("ci95") else f"{d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}"
            lines.append(f"| {contrast} | {m} | {d['n_runs']} | {d['mean']:+.3f} | {d['sd']:.3f} | {ci} | "
                         f"{d['n_pos']} / {d['n_runs']} | {f3(d.get('t'))} |")
    if replication:
        lines += ["", "| replication row | test acc@1 | val acc@1 | test recall@5 |", "|---|---|---|---|"]
        for name, c in replication.items():
            lines.append(f"| {name} | {f3(c['test/acc@1'])} | {f3(c['val/acc@1'])} | {f3(c['test/recall@5'])} |")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--rebuild", action="store_true"); b.set_defaults(fn=cmd_build)
    sub.add_parser("tfidf").set_defaults(fn=cmd_tfidf)
    sub.add_parser("upload").set_defaults(fn=cmd_upload)
    sub.add_parser("summarize").set_defaults(fn=cmd_summarize)
    n = sub.add_parser("negatives", help="paired analysis of the negatives-by-split sweep")
    n.add_argument("--sweep", help="sweep id (default: wb.NEGATIVES_SWEEP_ID)"); n.set_defaults(fn=cmd_negatives)
    for name, fn in (("train", cmd_train), ("_one", cmd_one)):
        p = sub.add_parser(name)
        if name == "train":
            p.add_argument("--only", choices=list(JOBS), help="run one job (ignores the ledger)")
        else:
            p.add_argument("job", choices=list(JOBS))
        p.add_argument("--smoke", action="store_true", help="1 epoch on 256 pairs; nothing recorded")
        p.add_argument("--offline", action="store_true", help="WANDB_MODE=offline")
        p.add_argument("--from-artifact", nargs="?", const=f"{ARTIFACT}:v0", default=None, metavar="NAME:VERSION",
                       help=f"Colab: download the splits from W&B (default {ARTIFACT}:v0)")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
