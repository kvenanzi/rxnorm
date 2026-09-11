"""Register a W&B Sweep and, optionally, run an agent for it locally.

A sweep is a search over training settings that W&B coordinates: you register
the grid once, then any number of *agents* (a Colab notebook, this machine)
ask W&B for the next untried combination, train it, and report back.

Run from the repo root:
  uv run scripts/08_sweep.py --create                 # prints the sweep id and URL
  uv run scripts/08_sweep.py --create --smoke         # 1-trial throwaway sweep to test the plumbing
  uv run scripts/08_sweep.py --agent <sweep id>       # run trials on the local GPU
  uv run scripts/08_sweep.py --agent <sweep id> --count 3
"""

import argparse
import gc
from pathlib import Path

import torch
import wandb
import yaml

from rxnorm_vandf import WANDB_PROJECT
from rxnorm_vandf.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
SWEEP_YAML = ROOT / "sweeps" / "grid.yaml"


def run_one() -> None:
    """One trial: the agent has started the run; train() reads its parameters."""
    train(TrainConfig(data_dir=str(ROOT / "data" / "processed"), output_dir=str(ROOT / "models"),
                      log_model=False, tags=["sweep", "local"]), sweep=True)
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="with --create: a 1-trial sweep on 256 pairs")
    ap.add_argument("--agent", metavar="SWEEP_ID")
    ap.add_argument("--count", type=int, default=None, help="trials for this agent (default: until the grid is done)")
    args = ap.parse_args()

    if args.create:
        config = yaml.safe_load(SWEEP_YAML.read_text())
        if args.smoke:
            config["parameters"].update({
                "base_model": {"value": "sentence-transformers/all-MiniLM-L6-v2"},
                "negatives": {"value": "ingredient"},
                "normalize_strength": {"value": False},
                "smoke": {"value": True},
            })
            config["name"] = "smoke"
        sweep_id = wandb.sweep(config, project=WANDB_PROJECT)
        print(f"\nsweep id: {sweep_id}")
        print("Paste it into notebooks/02_sweep.ipynb, or run:  "
              f"uv run scripts/08_sweep.py --agent {sweep_id}")
    if args.agent:
        wandb.agent(args.agent, function=run_one, count=args.count, project=WANDB_PROJECT)


if __name__ == "__main__":
    main()
