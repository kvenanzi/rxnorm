"""Register a W&B Sweep and, optionally, run an agent for it locally.

A sweep is a search over training settings that W&B coordinates: you register
the grid once, then any number of *agents* (a Colab notebook, this machine)
ask W&B for the next untried combination, train it, and report back.

Run from the repo root:
  uv run scripts/08_sweep.py --create                 # prints the sweep id and URL
  uv run scripts/08_sweep.py --create --smoke         # 1-trial throwaway sweep to test the plumbing
  uv run scripts/08_sweep.py --create --config sweeps/negatives_by_split.yaml
  uv run scripts/08_sweep.py --agent <sweep id>       # run trials on the local GPU
  uv run scripts/08_sweep.py --agent <sweep id> --count 3
  uv run scripts/08_sweep.py --agent <sweep id> --count 1 --from-artifact   # data from the sweep's artifact
"""

import argparse
import copy
import gc
from pathlib import Path

import torch
import wandb
import yaml

from rxnorm_vandf import WANDB_PROJECT
from rxnorm_vandf.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
SWEEP_YAML = ROOT / "sweeps" / "grid.yaml"


def smoke_config(config: dict) -> dict:
    """A one-trial version of a sweep config: every `values` list collapses to
    its first entry, `smoke` is on, and the name says so. Proves the plumbing
    of any grid in a minute or two without spending a real cell."""
    out = copy.deepcopy(config)
    for name, spec in out["parameters"].items():
        if "values" in spec:
            out["parameters"][name] = {"value": spec["values"][0]}
    out["parameters"]["smoke"] = {"value": True}
    out["name"] = f"smoke-{config.get('name', 'sweep')}"
    return out


def make_run_one(from_artifact: bool):
    """One trial: the agent has started the run; train() reads its parameters.
    With from_artifact the data comes from the config's dataset_artifact (and
    dataset_subdir), exactly as it does in Colab."""
    def run_one() -> None:
        train(TrainConfig(data_dir=None if from_artifact else str(ROOT / "data" / "processed"),
                          output_dir=str(ROOT / "models"), log_model=False, tags=["sweep", "local"]),
              sweep=True)
        gc.collect()
        torch.cuda.empty_cache()
    return run_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--config", type=Path, default=SWEEP_YAML, help="sweep YAML (default: sweeps/grid.yaml)")
    ap.add_argument("--smoke", action="store_true", help="with --create: a 1-trial sweep on 256 pairs")
    ap.add_argument("--agent", metavar="SWEEP_ID")
    ap.add_argument("--count", type=int, default=None, help="trials for this agent (default: until the grid is done)")
    ap.add_argument("--from-artifact", action="store_true",
                    help="with --agent: read data from the sweep's dataset_artifact instead of data/processed")
    args = ap.parse_args()

    if args.create:
        config = yaml.safe_load(args.config.read_text())
        if args.smoke:
            config = smoke_config(config)
        sweep_id = wandb.sweep(config, project=WANDB_PROJECT)
        print(f"\nsweep id: {sweep_id}")
        print("Paste it into the sweep notebook, or run:  "
              f"uv run scripts/08_sweep.py --agent {sweep_id}")
    if args.agent:
        wandb.agent(args.agent, function=make_run_one(args.from_artifact), count=args.count, project=WANDB_PROJECT)


if __name__ == "__main__":
    main()
