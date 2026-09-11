"""Calibrate a trained model's confidence and evaluate abstention.

Fits a temperature and a Platt (logistic) calibrator on the validation split,
chooses abstention thresholds on validation for 95% and 99% precision, and
reports coverage and precision at those thresholds on test, for three
populations of increasing realism:

  matched   test VA strings that have an SCD/SBD answer
  +hard     ... plus test VA names that are real drugs with no SCD/SBD
            (packs, ingredient-only concepts): every answer is wrong
  +all      ... plus the supplies/devices/nutrition names (easy rejections)

Run from the repo root:
  uv run scripts/07_calibrate.py                                   # local best model
  uv run scripts/07_calibrate.py --model vandf-rxnorm-biencoder:v1  # a W&B artifact
  uv run scripts/07_calibrate.py --offline
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from rxnorm_vandf import MODEL_ARTIFACT, WANDB_PROJECT
from rxnorm_vandf.calibrate import (SIGNALS, TARGETS, Calibrator, at_threshold, auroc,
                                    correct_position, ece, fit_temperature,
                                    reliability_table, save_json, select_threshold)
from rxnorm_vandf.data import load_data
from rxnorm_vandf.eval import COVERAGES, precision_coverage
from rxnorm_vandf.train import TrainConfig, encode, retrieve, retrieve_texts

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEFAULT_MODEL = ROOT / "models" / "all-minilm-l6-v2-ingredient" / "best"
TOP_K = 20
POPULATIONS = ["matched", "+hard", "+all"]
CALIBRATION_ARTIFACT = "vandf-rxnorm-calibration"


def load_model(ref: str, run, offline: bool):
    from sentence_transformers import SentenceTransformer

    path = Path(ref)
    if not path.exists():
        if offline:
            raise SystemExit(f"{ref} is not a local directory and artifacts need W&B online")
        path = Path(run.use_artifact(ref).download())
    elif not offline and ref == str(DEFAULT_MODEL):
        run.use_artifact(f"{MODEL_ARTIFACT}:v0")   # lineage: the local model is v0
    model = SentenceTransformer(str(path))
    return model, path


def main() -> None:
    import wandb

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="local model dir or W&B artifact ref")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(project=WANDB_PROJECT, job_type="calibrate",
                     config={"model": args.model, "top_k": TOP_K, "platt_fit_on": "val matched + val hard unmatched"})
    model, model_path = load_model(args.model, run, args.offline)
    # Queries must be preprocessed exactly as during training (e.g. strength normalization).
    cfg = TrainConfig(**json.loads((model_path / "train_config.json").read_text()))
    model.max_seq_length = cfg.max_seq_length
    run.config.update({"train_config": asdict(cfg)})

    data = load_data(PROCESSED)
    unmatched = pd.read_parquet(PROCESSED / "unmatched.parquet")
    unmatched["hard"] = unmatched["rxnorm_tty"].notna()

    # Retrieval once, top-20 so the softmax sees the runners-up.
    cand_emb = encode(model, data.cand["name"].tolist(), cfg, query=False)
    ret = retrieve(model, data, cfg, top_k=TOP_K, cand_emb=cand_emb)
    ret_unm = {split: retrieve_texts(model, data, unmatched.loc[unmatched.split == split, "vandf_string"].tolist(),
                                     cfg, TOP_K, cand_emb) for split in ("val", "test")}

    # Fit on validation only.
    val_rows = data.rows("val")
    cal = Calibrator(temperature=fit_temperature(ret.scores[val_rows], correct_position(ret, data.valid, val_rows)))
    print(f"temperature = {cal.temperature:.4f}")

    def population(split: str, pop: str) -> tuple[dict, np.ndarray, list[str]]:
        """Features, correctness, and a label per row for one split and population."""
        rows = data.rows(split)
        feats = cal.features(ret, rows)
        correct = np.array([ret.top[q, 0] in data.valid[q] for q in rows])
        labels = [f"matched: {data.cand['name'][ret.top[q, 0]]}" for q in rows]
        if pop != "matched":
            um = unmatched[unmatched.split == split].reset_index(drop=True)
            keep = um["hard"].values if pop == "+hard" else np.ones(len(um), dtype=bool)
            f_um = cal.features(ret_unm[split], np.flatnonzero(keep))
            feats = {k: np.concatenate([feats[k], f_um[k]]) for k in feats}
            correct = np.concatenate([correct, np.zeros(keep.sum(), dtype=bool)])
            labels += [f"unmatched ({t or 'nothing'})" for t in um.loc[keep, "rxnorm_tty"]]
        return feats, correct, labels

    # Platt is fit on val matched + hard unmatched: the realistic hard population.
    f, c, _ = population("val", "+hard")
    cal.fit_platt(f, c)

    # Thresholds on val, results on test, for every population x signal.
    rows_out = []
    for pop in POPULATIONS:
        f_val, c_val, _ = population("val", pop)
        f_test, c_test, _ = population("test", pop)
        cal.thresholds[pop] = {}
        curves = {}
        for sig in SIGNALS:
            p_val, p_test = f_val[sig], f_test[sig]
            m = {"ece": ece(p_test, c_test) if sig in ("softmax", "platt") else float("nan"),
                 "auroc": auroc(p_test, c_test)}
            for t in TARGETS:
                thr = select_threshold(p_val, c_val, t)
                cov, prec = at_threshold(p_test, c_test, thr)
                key = f"p{int(t * 100)}"
                m[f"coverage@{key}"], m[f"precision@{key}"] = cov, prec
                if sig == "platt":
                    cal.thresholds[pop][key] = thr
            curves[sig] = precision_coverage(c_test, p_test)["curve"]
            for k, v in m.items():
                run.summary[f"test/{pop}/{sig}/{k}"] = v
            rows_out.append([pop, sig, len(c_test), *m.values()])
        run.log({f"test/{pop}/precision_vs_coverage": wandb.plot.line_series(
            xs=COVERAGES, ys=[list(curves[s].values()) for s in SIGNALS], keys=SIGNALS,
            title=f"precision at automation rate ({pop})", xname="coverage")})

    cols = ["population", "signal", "n", "ece", "auroc", "coverage@p95", "precision@p95",
            "coverage@p99", "precision@p99"]
    table = pd.DataFrame(rows_out, columns=cols)
    print("\nThresholds chosen on val, measured on test")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    run.log({"test/summary": wandb.Table(dataframe=table)})

    # Reliability of the Platt confidence on the realistic population.
    f_test, c_test, labels = population("test", "+hard")
    rel = reliability_table(f_test["platt"], c_test)
    run.log({"test/+hard/reliability": wandb.Table(columns=["bin", "confidence", "accuracy", "n"], data=rel)})
    print("\nReliability (test, +hard, platt): confidence vs accuracy per bin")
    for b, conf, acc, n in rel:
        print(f"  {b}  conf {conf:.3f}  acc {acc:.3f}  n {n}")

    # The errors that would slip through: highest Platt confidence, wrong.
    wrong = np.flatnonzero(~c_test)
    worst = wrong[np.argsort(-f_test["platt"][wrong])][:100]
    test_rows = data.rows("test")
    strings = [data.queries["vandf_string"][q] for q in test_rows] + \
              unmatched.loc[(unmatched.split == "test") & unmatched["hard"], "vandf_string"].tolist()
    truths = [", ".join(data.cand["name"][t] for t in sorted(data.valid[q])) for q in test_rows] + \
             ["(no SCD/SBD)"] * int(((unmatched.split == "test") & unmatched["hard"]).sum())
    run.log({"test/+hard/confident_errors": wandb.Table(
        columns=["vandf_string", "truth", "top1", "cosine", "platt"],
        data=[[strings[i], truths[i], labels[i], float(f_test["cosine"][i]), float(f_test["platt"][i])] for i in worst])})

    out = model_path / "calibration.json"
    save_json(cal, out)
    art = wandb.Artifact(CALIBRATION_ARTIFACT, type="calibration", metadata=cal.to_json())
    art.add_file(str(out))
    run.log_artifact(art)
    run.finish()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
