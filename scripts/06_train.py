"""Train the bi-encoder on the local GPU.

Run from the repo root:
  uv run scripts/06_train.py --smoke --offline   # ~2 min pipeline check, nothing uploaded
  uv run scripts/06_train.py                     # the real run, logged to W&B
  uv run scripts/06_train.py --negatives tfidf --epochs 6 --run-name minilm-tfidf-negs
  uv run scripts/06_train.py --data-dir data/multi/v1 --train-sources VANDF,MTHSPL --aux strength --no-log-model
"""

import argparse
import os
from dataclasses import fields
from pathlib import Path

from rxnorm_vandf.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
INT_OR_NONE = {"max_extra_pairs"}     # None by default, an integer when given


def main() -> None:
    ap = argparse.ArgumentParser()
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        if f.name in ("data_dir", "output_dir", "tags"):
            continue
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if f.type is bool:
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif f.name in INT_OR_NONE:
            ap.add_argument(flag, type=int, default=default)
        elif default is None or isinstance(default, (tuple, list)):
            ap.add_argument(flag, type=str, default=default)   # tuples come as "A,B"; TrainConfig splits them
        else:
            ap.add_argument(flag, type=type(default), default=default)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed",
                    help="a dataset folder (default: data/processed)")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "models",
                    help="where <run name>/best goes (default: models/)")
    ap.add_argument("--offline", action="store_true", help="run without a W&B login")
    args = vars(ap.parse_args())
    if args.pop("offline"):
        os.environ["WANDB_MODE"] = "offline"
    args["data_dir"], args["output_dir"] = str(args["data_dir"]), str(args["output_dir"])

    cfg = TrainConfig(**args)
    best_dir = train(cfg)
    print(f"\nBest model saved to {best_dir}")


if __name__ == "__main__":
    main()
