"""Publish the processed dataset as a Weights & Biases Artifact.

An Artifact is a versioned bundle of files that W&B stores and tracks. Logging
the dataset once gives every later run (baselines here, training in Colab) a
fixed, named input: `vandf-rxnorm-pairs:v0`. Rerunning after the data changes
creates v1, v2, ...; W&B deduplicates files that didn't change.

Run from the repo root after 03_build_dataset.py:  uv run scripts/04_upload_dataset.py
"""

from pathlib import Path

import duckdb
import wandb

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FILES = ["pairs.parquet", "candidates.parquet", "unmatched.parquet"]

WANDB_PROJECT = "rxnorm-vandf"
ARTIFACT_NAME = "vandf-rxnorm-pairs"


def row_counts() -> dict[str, dict[str, int]]:
    con = duckdb.connect()
    counts = {}
    for name in FILES:
        rows = con.execute(
            f"SELECT split, count(*) FROM '{PROCESSED / name}' GROUP BY split ORDER BY split"
        ).fetchall()
        counts[name.removesuffix(".parquet")] = {split: n for split, n in rows}
    return counts


def main() -> None:
    # A run is one unit of work W&B records. job_type is a free-form label that
    # lets the UI group runs: "dataset" here, "baseline" and "train" later.
    run = wandb.init(project=WANDB_PROJECT, job_type="dataset")

    artifact = wandb.Artifact(
        name=ARTIFACT_NAME,
        type="dataset",
        description=(
            "VANDF drug strings paired with RxNorm SCD/SBD targets, the full SCD/SBD "
            "candidate pool, and unmatched VA names. Derived only from the VANDF and "
            "RXNORM sources (license restriction level 0) of the RxNorm 2026-09-08 "
            "full release. Built by scripts/03_build_dataset.py."
        ),
        metadata={
            "rxnorm_release": "2026-09-08",
            "split_salt": "rxnorm-2026-09-08-v1",
            "split_pct": {"train": 70, "val": 15, "test": 15},
            "rows": row_counts(),
        },
    )
    for name in FILES:
        artifact.add_file(str(PROCESSED / name))

    run.log_artifact(artifact)
    run.finish()
    print(f"\nLogged artifact {ARTIFACT_NAME} to project {WANDB_PROJECT}")


if __name__ == "__main__":
    main()
