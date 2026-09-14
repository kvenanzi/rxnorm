"""K-fold by ingredient, the out-of-fold estimate, and the all-data model (docs/post-2).

Seven folds of the published salt: an ingredient's md5 bucket (0..99) maps to
fold bucket*7//100, so fold 0's test set is exactly the published test set and
the other six are new draws. Fold i trains on five folds, selects its epoch on
fold i+1, and is tested once on fold i; pooling the seven test sets gives one
out-of-fold accuracy over every VA string that is ever held out. The final
model then trains on every ingredient (no val, no test) for the median best
epoch of the folds, and is calibrated on the pooled out-of-fold predictions.
Nothing here touches data/processed, data/splits, or the published artifacts:

  data/kfold/fold<i>/, data/kfold/all/    the folders (03_build_dataset.py --kfold / --all-train)
  models/kfold/<run>/                     checkpoints; only the final run logs a model artifact
  outputs/kfold/                          runs.json (ledger), oof.{json,md,parquet}, calibration.{json,md}
  W&B: group kfold-k7; artifacts vandf-rxnorm-kfold (data), vandf-rxnorm-predictions (per fold),
       vandf-rxnorm-biencoder-final-all (the model; vandf-rxnorm-biencoder is never touched)

Run from the repo root (the recipe flags default to the published recipe; pass
the winning arm of scripts/13_levers.py, e.g. --train-sources VANDF,MTHSPL --aux strength):
  uv run scripts/14_kfold.py build [--sources VANDF,MTHSPL]
  uv run scripts/14_kfold.py upload
  uv run scripts/14_kfold.py train [--only fold3] [--smoke --offline] [--from-artifact] [recipe flags]
  uv run scripts/14_kfold.py final [--epochs N] [--from-artifact] [recipe flags]
  uv run scripts/14_kfold.py oof                       # pooled out-of-fold table from W&B
  uv run scripts/14_kfold.py calibrate-oof             # calibration.json + threshold transfer across folds
In Colab (notebooks/06_kfold.ipynb):  python scripts/14_kfold.py train --from-artifact; then final --from-artifact
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))      # a bare Colab clone has no `pip install -e`

from rxnorm_vandf.stats import describe, wilson   # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_seeds = load_script("12_split_seeds")
FINAL = _seeds.FINAL                     # the published recipe; the levers flags layer on top
SALT = _seeds.SPLITS[_seeds.PUBLISHED_SPLIT]
K = 7
FOLDS = [f"fold{i}" for i in range(K)]
ALL = "all"
KFOLD_DIR = ROOT / "data" / "kfold"
MODELS_DIR = ROOT / "models" / "kfold"
OUT = ROOT / "outputs" / "kfold"
LEDGER = OUT / "runs.json"
GROUP = "kfold-k7"
DATA_ARTIFACT = "vandf-rxnorm-kfold"
MODEL_ARTIFACT = "vandf-rxnorm-biencoder-final-all"
RUN_PREFIX = "sapbert-ingredient-strength-kfold"
FINAL_RUN = "sapbert-ingredient-strength-final-all"
TARGETS = {"p95": 0.95, "p99": 0.99}


def rel(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def run_name(job: str) -> str:
    return f"{RUN_PREFIX}-{job}" if job != ALL else FINAL_RUN


def folder(job: str) -> Path:
    return KFOLD_DIR / job


def recipe_of(args: argparse.Namespace) -> dict:
    """TrainConfig fields for the recipe under test, from the CLI flags."""
    return {**FINAL, "train_sources": args.train_sources, "aux": args.aux, "aux_weight": args.aux_weight}


def unmatched_fold(string: str) -> int:
    """Unmatched strings are never trained on, so any fold's model scores them
    out-of-fold; each string is assigned to one fold so it is counted once."""
    return int(hashlib.md5(string.encode()).hexdigest(), 16) % K


# ------------------------------------------------------------------------ build

def cmd_build(args: argparse.Namespace) -> None:
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(f"fold{i}", ["--kfold", str(K), "--fold", str(i)]) for i in range(K)] + [(ALL, ["--all-train"])]
    for job, flags in jobs:
        d = folder(job)
        if (d / "pairs.parquet").exists() and not args.rebuild:
            print(f"{job}: exists, skipping")
            continue
        subprocess.run([sys.executable, str(ROOT / "scripts" / "03_build_dataset.py"), "--salt", SALT,
                        "--sources", args.sources, "--out-dir", str(d), *flags], check=True, stdout=subprocess.DEVNULL)
        print(f"{job}: built ({' '.join(flags)}, sources {args.sources})")
    print()
    print(f"{'folder':<7}{'train':>8}{'val':>8}{'test':>8}{'mixed':>8}   (VA strings)")
    for job, _ in jobs:
        c = counts(job)
        print(f"{job:<7}{c.get('train', 0):>8}{c.get('val', 0):>8}{c.get('test', 0):>8}{c.get('mixed', 0):>8}")


def counts(job: str) -> dict[str, int]:
    import duckdb
    src = "WHERE source = 'VANDF'" if has_source(job) else ""
    rows = duckdb.sql(f"SELECT split, count(*) FROM '{folder(job) / 'pairs.parquet'}' {src} GROUP BY 1").fetchall()
    return {s: int(n) for s, n in rows}


def has_source(job: str) -> bool:
    import duckdb
    return "source" in duckdb.sql(f"SELECT * FROM '{folder(job) / 'pairs.parquet'}' LIMIT 0").columns


def cmd_upload(args: argparse.Namespace) -> None:
    import wandb
    from rxnorm_vandf import WANDB_PROJECT

    run = wandb.init(project=WANDB_PROJECT, job_type="dataset", name="upload-kfold", group=GROUP)
    art = wandb.Artifact(DATA_ARTIFACT, type="dataset",
                         description=f"{K} folds by ingredient of the published salt (fold0's test set is the "
                                     "published test set) plus the all-train folder. Built by scripts/14_kfold.py.",
                         metadata={"k": K, "salt": SALT, "rows": {j: counts(j) for j in FOLDS + [ALL]}})
    art.add_dir(str(KFOLD_DIR))
    run.log_artifact(art)
    run.finish()
    print(f"logged {DATA_ARTIFACT}")


# ------------------------------------------------------------------------ train

def add_recipe_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-sources", default="VANDF", help="e.g. VANDF,MTHSPL (the folder must carry the source)")
    p.add_argument("--aux", default="none", choices=["none", "strength"])
    p.add_argument("--aux-weight", type=float, default=0.2)
    p.add_argument("--smoke", action="store_true", help="1 epoch on 256 pairs; nothing recorded")
    p.add_argument("--offline", action="store_true", help="WANDB_MODE=offline")
    p.add_argument("--from-artifact", nargs="?", const=f"{DATA_ARTIFACT}:v0", default=None, metavar="NAME:VERSION",
                   help=f"Colab: download the folders from W&B (default {DATA_ARTIFACT}:v0)")


def forward_flags(args: argparse.Namespace) -> list[str]:
    cmd = ["--train-sources", args.train_sources, "--aux", args.aux, "--aux-weight", str(args.aux_weight)]
    for flag in ("smoke", "offline"):
        if getattr(args, flag):
            cmd.append(f"--{flag}")
    if args.from_artifact:
        cmd += ["--from-artifact", args.from_artifact]
    return cmd


def cmd_train(args: argparse.Namespace) -> None:
    """One subprocess per fold: a clean CUDA context each time; a crash in one
    fold does not take the rest; finished folds are skipped via the ledger."""
    jobs = [args.only] if args.only else FOLDS
    done = {e["job"] for e in read_ledger() if e.get("state") == "finished"}
    for job in jobs:
        if job in done and not args.only:
            print(f"{job}: finished earlier (see {rel(LEDGER)}), skipping")
            continue
        cmd = [sys.executable, __file__, "_one", job, *forward_flags(args)]
        print(f"\n=== {job}: {' '.join(cmd[2:])}", flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise SystemExit(f"{job} failed; rerun with --only {job}")


def cmd_final(args: argparse.Namespace) -> None:
    """The all-data model: every ingredient in train, the median best epoch of
    the folds (or --epochs), weights logged as {MODEL_ARTIFACT}."""
    epochs = args.epochs or median_best_epoch()
    cmd = [sys.executable, __file__, "_one", ALL, "--epochs", str(epochs), *forward_flags(args)]
    print(f"=== {ALL}: {' '.join(cmd[2:])}", flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit("final run failed")


def median_best_epoch() -> int:
    runs = fetch_fold_runs()
    epochs = [int(r["best_epoch"]) for r in runs.values() if r.get("best_epoch")]
    if len(epochs) < K:
        raise SystemExit(f"only {len(epochs)} of {K} fold runs have a best_epoch on W&B; pass --epochs")
    return int(round(statistics.median(epochs)))


def cmd_one(args: argparse.Namespace) -> None:
    import wandb
    from rxnorm_vandf.train import TrainConfig, train

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    job = args.job
    name = run_name(job) + ("-smoke" if args.smoke else "")
    final = job == ALL
    cfg = TrainConfig(**recipe_of(args), seed=42, smoke=args.smoke,
                      epochs=args.epochs or FINAL["epochs"],
                      log_model=final and not args.smoke, model_artifact=MODEL_ARTIFACT,
                      log_predictions=True,
                      output_dir=str(MODELS_DIR / "smoke" if args.smoke else MODELS_DIR),
                      run_name=name, group=GROUP,
                      tags=[GROUP, "final-all" if final else job])
    if args.from_artifact:
        cfg.data_dir, cfg.dataset_artifact, cfg.dataset_subdir = None, args.from_artifact, job
    else:
        cfg.data_dir = str(folder(job))
    train(cfg)
    if not args.smoke:
        append_ledger({"job": job, "run_name": name, "recipe": recipe_of(args), "epochs": cfg.epochs,
                       "run_id": _seeds.latest_run_id(), "state": "finished",
                       "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def read_ledger() -> list[dict]:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def append_ledger(entry: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = [e for e in read_ledger() if e["job"] != entry["job"]] + [entry]
    LEDGER.write_text(json.dumps(entries, indent=2) + "\n")


# ------------------------------------------------------------------------- oof

def fetch_fold_runs() -> dict[str, dict]:
    """Finished fold runs of the group from W&B, keyed by fold; newest wins."""
    from rxnorm_vandf.wb import ENTITY, PROJECT, api

    by_name = {}
    for r in api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP, "jobType": "train"}):
        if r.state != "finished" or r.config.get("smoke"):
            continue
        if r.name not in by_name or r.created_at > by_name[r.name].created_at:
            by_name[r.name] = r
    out = {}
    for job in FOLDS + [ALL]:
        r = by_name.get(run_name(job))
        if r is not None:
            s = dict(r.summary)
            out[job] = {"run": r, "run_id": r.id, "best_epoch": s.get("best_epoch"),
                        **{k: s.get(k) for k in ("val/acc@1", "test/acc@1", "test/recall@5", "test/n", "val/n",
                                                 "test/strength_acc", "test/dose_form_acc", "test/ingredient_acc")}}
    return out


def predictions_of(run) -> "pd.DataFrame":
    """The predictions.parquet a fold run logged (rxnorm_vandf.train.PREDICTIONS_ARTIFACT)."""
    import pandas as pd

    for art in run.logged_artifacts():
        if art.type == "predictions":
            return pd.read_parquet(Path(art.download()) / "predictions.parquet")
    raise KeyError(f"run {run.name} logged no predictions artifact")


def oof_frame(runs: dict[str, dict]) -> "pd.DataFrame":
    """One row per out-of-fold prediction: each fold's VA test strings, plus the
    unmatched val/test strings assigned to that fold by hash."""
    import pandas as pd

    parts = []
    for job in FOLDS:
        if job not in runs:
            continue
        df = predictions_of(runs[job]["run"])
        df = df[df["source"] == "VANDF"]
        test = df[(df["kind"] == "matched") & (df["split"] == "test")].copy()
        um = df[df["kind"] == "unmatched"].copy()
        um = um[[unmatched_fold(s) == FOLDS.index(job) for s in um["string"]]]
        for part in (test, um):
            part["fold"] = FOLDS.index(job)
        parts += [test, um]
    out = pd.concat(parts, ignore_index=True)
    out["correct"] = [len(t) > 0 and top[0] in set(t) for top, t in zip(out["top_rxcuis"], out["truth_rxcuis"])]
    out["hard"] = out["rxnorm_tty"].notna() & (out["kind"] == "unmatched")
    return out


def cmd_oof(args: argparse.Namespace) -> None:
    import pandas as pd

    runs = fetch_fold_runs()
    missing = [j for j in FOLDS if j not in runs]
    if missing:
        print("not finished yet:", ", ".join(missing))
    df = oof_frame(runs)
    matched = df[df["kind"] == "matched"]
    k, n = int(matched["correct"].sum()), len(matched)
    lo, hi = wilson(k, n)
    per_fold = {j: runs[j] for j in FOLDS if j in runs}
    accs = [r["test/acc@1"] for r in per_fold.values()]
    ns = [r["test/n"] for r in per_fold.values()]
    total_va = counts(ALL).get("train") if (folder(ALL) / "pairs.parquet").exists() else None
    payload = {"k": K, "salt": SALT,
               "folds": {j: {kk: v for kk, v in r.items() if kk != "run"} for j, r in per_fold.items()},
               "oof": {"n": n, "correct": k, "acc@1": k / n if n else None, "wilson95": [lo, hi],
                       "recall@5": float(np.mean([len(t) > 0 and any(x in set(t) for x in top[:5])
                                                  for top, t in zip(matched["top_rxcuis"], matched["truth_rxcuis"])])) if n else None,
                       "coverage": (n / total_va) if total_va else None, "n_va_strings": total_va},
               "across_folds": {"test/acc@1": describe(accs, ns), "val/acc@1": describe([r["val/acc@1"] for r in per_fold.values()]),
                                "test/recall@5": describe([r["test/recall@5"] for r in per_fold.values()])} if len(accs) > 1 else {},
               "final": {kk: v for kk, v in runs[ALL].items() if kk != "run"} if ALL in runs else None}
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "oof.parquet", index=False)
    (OUT / "oof.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    md = render_oof(payload)
    (OUT / "oof.md").write_text(md)
    print(md)
    print(f"wrote {rel(OUT / 'oof.json')}, oof.md, oof.parquet")


def render_oof(p: dict) -> str:
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    lines = ["| fold | test n | best epoch | val acc@1 | test acc@1 | test recall@5 | strength | dose form | ingredient |",
             "|---|---|---|---|---|---|---|---|---|"]
    for j, r in p["folds"].items():
        lines.append(f"| {j}{' (published test set)' if j == 'fold0' else ''} | {r['test/n']} | {r['best_epoch']} | "
                     f"{f3(r['val/acc@1'])} | {f3(r['test/acc@1'])} | {f3(r['test/recall@5'])} | "
                     f"{f3(r['test/strength_acc'])} | {f3(r['test/dose_form_acc'])} | {f3(r['test/ingredient_acc'])} |")
    o = p["oof"]
    lines += ["", f"Pooled out-of-fold: acc@1 **{f3(o['acc@1'])}** (Wilson 95% {f3(o['wilson95'][0])}–{f3(o['wilson95'][1])}, "
                  f"n = {o['n']:,}), recall@5 {f3(o['recall@5'])}; covers {f3(o['coverage'])} of the "
                  f"{o['n_va_strings']:,} VA strings (the rest are combination drugs whose ingredients span folds)."]
    for m, d in p.get("across_folds", {}).items():
        lines.append(f"- across folds, {m}: {d['mean']:.3f} ± {d['sd']:.3f} (range {d['min']:.3f}–{d['max']:.3f}"
                     + (f"; sampling sd {d['sampling_sd']:.3f}, excess {d['excess_sd']:.3f}" if "excess_sd" in d else "") + ")")
    if p.get("final"):
        lines.append(f"- final all-data model: run {p['final']['run_id']}, {p['final']['best_epoch']} epochs, no held-out score by construction.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- calibrate-oof

def features(df: "pd.DataFrame", cal) -> dict[str, np.ndarray]:
    from rxnorm_vandf.eval import Retrieval
    scores = np.array(df["scores"].tolist(), dtype=np.float32)
    return cal.features(Retrieval(np.zeros_like(scores, dtype=np.int64), scores))


def correct_pos(df: "pd.DataFrame") -> np.ndarray:
    pos = np.full(len(df), -1, dtype=np.int64)
    for i, (top, truth) in enumerate(zip(df["top_rxcuis"], df["truth_rxcuis"])):
        t = set(truth)
        for j, x in enumerate(top):
            if x in t:
                pos[i] = j
                break
    return pos


def cmd_calibrate_oof(args: argparse.Namespace) -> None:
    """Temperature and Platt fit on the pooled out-of-fold predictions (matched
    test strings + hard unmatched strings, each seen by exactly one fold model),
    written as calibration.json for the all-data model. Then the honest check:
    thresholds chosen on every fold but j, applied to fold j, for the 95% and
    99% targets, and the full fold-by-fold transfer matrix."""
    import pandas as pd
    from rxnorm_vandf.calibrate import Calibrator, at_threshold, auroc, ece, fit_temperature, save_json, select_threshold

    path = OUT / "oof.parquet"
    if not path.exists():
        raise SystemExit("run `14_kfold.py oof` first")
    df = pd.read_parquet(path)
    pop = df[(df["kind"] == "matched") | df["hard"]].reset_index(drop=True)   # the +hard population
    matched = pop[pop["kind"] == "matched"]
    cal = Calibrator(temperature=fit_temperature(np.array(matched["scores"].tolist(), dtype=np.float32), correct_pos(matched)))
    cal.fit_platt(features(pop, cal), pop["correct"].to_numpy())
    feats = features(pop, cal)
    y = pop["correct"].to_numpy()
    folds = pop["fold"].to_numpy()

    result = {"temperature": cal.temperature, "n": len(pop), "n_matched": len(matched), "n_hard": int(pop["hard"].sum()),
              "auroc": {s: auroc(feats[s], y) for s in feats}, "ece": {s: ece(feats[s], y) for s in ("softmax", "platt")},
              "transfer": {}, "leave_one_fold_out": {}}
    for key, target in TARGETS.items():
        thr_all = select_threshold(feats["platt"], y, target)
        cal.thresholds.setdefault("+hard", {})[key] = thr_all
        cov, prec = at_threshold(feats["platt"], y, thr_all)
        result[f"pooled@{key}"] = {"threshold": thr_all, "coverage": cov, "precision": prec}
        matrix = np.full((K, K), np.nan)
        lofo = {}
        for i in range(K):
            fit_i = folds == i
            thr = select_threshold(feats["platt"][fit_i], y[fit_i], target)
            for j in range(K):
                on_j = folds == j
                matrix[i, j] = at_threshold(feats["platt"][on_j], y[on_j], thr)[1]
            rest = folds != i
            thr_rest = select_threshold(feats["platt"][rest], y[rest], target)
            cov_i, prec_i = at_threshold(feats["platt"][fit_i], y[fit_i], thr_rest)
            lofo[f"fold{i}"] = {"threshold": thr_rest, "coverage": cov_i, "precision": prec_i}
        off = matrix[~np.eye(K, dtype=bool)]
        result["transfer"][key] = {"precision_matrix": matrix.tolist(), "off_diagonal_mean": float(np.nanmean(off)),
                                   "off_diagonal_min": float(np.nanmin(off))}
        result["leave_one_fold_out"][key] = {**lofo,
                                             "precision": describe([v["precision"] for v in lofo.values()]),
                                             "coverage": describe([v["coverage"] for v in lofo.values()])}

    OUT.mkdir(parents=True, exist_ok=True)
    save_json(cal, OUT / "calibration.json")
    (OUT / "calibration_oof.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    md = render_calibration(result)
    (OUT / "calibration.md").write_text(md)
    print(md)
    final_dir = MODELS_DIR / FINAL_RUN / "best"
    if final_dir.exists():
        save_json(cal, final_dir / "calibration.json")
        print(f"also wrote {final_dir / 'calibration.json'}")
    print(f"wrote {rel(OUT / 'calibration.json')}, calibration_oof.json, calibration.md")


def render_calibration(r: dict) -> str:
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    lines = [f"Pooled out-of-fold calibration (+hard population, n = {r['n']:,}: {r['n_matched']:,} matched + {r['n_hard']:,} hard "
             f"unmatched): temperature {r['temperature']:.4f}; AUROC platt {f3(r['auroc']['platt'])}, softmax {f3(r['auroc']['softmax'])}, "
             f"cosine {f3(r['auroc']['cosine'])}; ECE platt {f3(r['ece']['platt'])}.", "",
             "| target | pooled threshold | pooled coverage | pooled precision | leave-one-fold-out precision (mean ± sd, min) | "
             "leave-one-fold-out coverage (mean ± sd) | fold-to-fold precision, off-diagonal (mean, min) |",
             "|---|---|---|---|---|---|---|"]
    for key in TARGETS:
        p, l, t = r[f"pooled@{key}"], r["leave_one_fold_out"][key], r["transfer"][key]
        lines.append(f"| {key} | {p['threshold']:.3f} | {p['coverage']:.3f} | {p['precision']:.3f} | "
                     f"{l['precision']['mean']:.3f} ± {l['precision']['sd']:.3f}, min {l['precision']['min']:.3f} | "
                     f"{l['coverage']['mean']:.3f} ± {l['coverage']['sd']:.3f} | {t['off_diagonal_mean']:.3f}, {t['off_diagonal_min']:.3f} |")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", default="VANDF,MTHSPL", help="sources carried by the folders; the recipe picks train_sources at train time")
    b.add_argument("--rebuild", action="store_true"); b.set_defaults(fn=cmd_build)
    sub.add_parser("upload").set_defaults(fn=cmd_upload)
    t = sub.add_parser("train"); t.add_argument("--only", choices=FOLDS); add_recipe_flags(t); t.set_defaults(fn=cmd_train)
    f = sub.add_parser("final"); f.add_argument("--epochs", type=int, default=None); add_recipe_flags(f); f.set_defaults(fn=cmd_final)
    o = sub.add_parser("_one"); o.add_argument("job", choices=FOLDS + [ALL]); o.add_argument("--epochs", type=int, default=None)
    add_recipe_flags(o); o.set_defaults(fn=cmd_one)
    sub.add_parser("oof").set_defaults(fn=cmd_oof)
    sub.add_parser("calibrate-oof").set_defaults(fn=cmd_calibrate_oof)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
