"""Score two baselines and log them to Weights & Biases.

  exact   normalized string equality against the 27,287 candidate names
  tfidf   TF-IDF over character 3-5-grams, cosine similarity, top-5 retrieval

Both search the full candidate pool, the way a real mapper would. The trained
trained model has to beat tfidf on held-out ingredients.

Run from the repo root after 04_upload_dataset.py:
  uv run scripts/05_baselines.py                  # both baselines, logged online
  uv run scripts/05_baselines.py --baseline tfidf
  uv run scripts/05_baselines.py --offline        # smoke test without a W&B login
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from rxnorm_vandf import DATASET_ARTIFACT, WANDB_PROJECT
from rxnorm_vandf.data import Data, load_data, normalize
from rxnorm_vandf.eval import TOP_K, Retrieval, evaluate_splits, log_results, print_summary
from rxnorm_vandf.tfidf import tfidf_retrieve

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def run_exact(data: Data, queries: list[str]) -> Retrieval:
    by_name = defaultdict(list)
    for i, name in enumerate(data.cand["name"]):
        by_name[normalize(name)].append(i)
    top = np.full((len(queries), TOP_K), -1, dtype=np.int64)
    scores = np.zeros((len(queries), TOP_K), dtype=np.float32)
    for q, s in enumerate(queries):
        hits = by_name.get(normalize(s), [])[:TOP_K]
        top[q, : len(hits)] = hits
        scores[q, : len(hits)] = 1.0
    return Retrieval(top, scores)


def run_tfidf(data: Data, queries: list[str]) -> Retrieval:
    return tfidf_retrieve(data.cand["name"].tolist(), queries)


def run_baseline(name: str, data: Data, offline: bool) -> None:
    import wandb

    queries = data.queries["vandf_string"].tolist()
    config = {"baseline": name, "candidate_pool": len(data.cand), "top_k": TOP_K,
              "normalize": "lower, punct->space, keep %./"}
    if name == "tfidf":
        config.update(analyzer="char_wb", ngram_range=[3, 5], sublinear_tf=True)

    run = wandb.init(project=WANDB_PROJECT, job_type="baseline", name=name, config=config)
    if not offline:
        # Records that this run consumed the dataset artifact (lineage in the UI).
        run.use_artifact(f"{DATASET_ARTIFACT}:latest")

    ret = run_exact(data, queries) if name == "exact" else run_tfidf(data, queries)

    # Confidence for the precision/coverage curve. For tfidf also try the margin
    # between the top two scores: a small ablation of "what makes a good confidence".
    margin = ret.scores[:, 0] - ret.scores[:, 1] if name == "tfidf" else None
    results = evaluate_splits(data, ret, ret.scores[:, 0], margin)

    log_results(run, name, data, ret, results)
    print_summary(name, results)
    run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["exact", "tfidf", "all"], default="all")
    ap.add_argument("--offline", action="store_true", help="run without a W&B login")
    args = ap.parse_args()
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    data = load_data(PROCESSED)
    print(f"{len(data.queries):,} distinct VA strings, {len(data.cand):,} candidates")

    # Sanity check on the retrieval plumbing before anything is logged.
    probe = "METOPROLOL TARTRATE 12.5MG TAB"
    hit = data.cand["name"][run_tfidf(data, [probe]).top[0, 0]]
    assert "metoprolol" in hit, f"tfidf probe failed: {probe!r} -> {hit!r}"

    names = ["exact", "tfidf"] if args.baseline == "all" else [args.baseline]
    for name in names:
        run_baseline(name, data, args.offline)


if __name__ == "__main__":
    main()
