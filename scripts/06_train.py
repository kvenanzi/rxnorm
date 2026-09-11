"""Train the bi-encoder on the local GPU.

Run from the repo root:
  uv run scripts/06_train.py --smoke --offline   # ~2 min pipeline check, nothing uploaded
  uv run scripts/06_train.py                     # the real run, logged to W&B
  uv run scripts/06_train.py --negatives tfidf --epochs 6 --run-name minilm-tfidf-negs
"""

import argparse
import os
from dataclasses import fields
from pathlib import Path

from rxnorm_vandf.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        if f.name in ("data_dir", "output_dir", "tags"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            ap.add_argument(flag, action="store_true", default=getattr(defaults, f.name))
        else:
            ap.add_argument(flag, type=type(getattr(defaults, f.name)) if getattr(defaults, f.name) is not None else str,
                            default=getattr(defaults, f.name))
    ap.add_argument("--offline", action="store_true", help="run without a W&B login")
    args = vars(ap.parse_args())
    if args.pop("offline"):
        os.environ["WANDB_MODE"] = "offline"

    cfg = TrainConfig(**args, data_dir=str(ROOT / "data" / "processed"),
                      output_dir=str(ROOT / "models"))
    best_dir = train(cfg)
    print(f"\nBest model saved to {best_dir}")


if __name__ == "__main__":
    main()
