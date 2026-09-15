"""Read things back out of Weights & Biases for reports and figures."""

import json
import re
from pathlib import Path

import wandb

ENTITY = "kettle-labs"
PROJECT = "rxnorm-vandf"

# The runs the write-up is built from, in story order.
STORY_RUNS = {
    "exact": "pd69r8t4",
    "tfidf": "xr7g32b0",
    "minilm": "fked1jx1",
    "minilm+normalizer": "x9p04t9i",
    "final": "0d9ntjls",
}
SWEEP_ID = "idhaaw5i"
NEGATIVES_SWEEP_ID = "uukeyzw7"   # sweeps/negatives_by_split.yaml: hard negatives x six splits

# The follow-up experiment (docs/post-2): filled in once each sweep is registered.
# levers.yaml + levers_rerun.yaml (aux-off v3..v6) + the three head make-ups (levers_rerun_head.yaml v1/s42,
# levers_rerun_head_v1s1.yaml, levers_rerun_head_v2s42.yaml); j1zdu28j and 5cr5ze02 were registered but never ran.
LEVERS_SWEEP_IDS = ["nypttmi8", "ad9nakej", "2v3am5xr", "94onbtb5", "zcsece1l"]
LEVERS_SWEEP_ID = LEVERS_SWEEP_IDS[0]
LEVERS_CONTROL_SWEEP_IDS = {"steps": "2exoct4h", "size": "ocmpf5na"}   # sweeps/levers_control_*.yaml
KFOLD_GROUP = "kfold-k7"          # scripts/14_kfold.py: seven folds by ingredient + the all-data model

# The split-and-seed-variance runs (scripts/12_split_seeds.py), job -> run id.
# Selected by id in the report, like STORY_RUNS, so a crashed or duplicate run
# in the same group can never land in a panel.
SPLIT_SEED_RUNS: dict[str, str] = {
    "split-v1": "lqf7dett",
    "split-v2": "0c2jqaf1",
    "split-v3": "i5pj0dbc",
    "split-v4": "aa7hcyap",
    "split-v5": "r3asp4in",
    "split-v6": "kflhpyjx",
    "seed-1": "0x22n7cb",
    "seed-2": "zuj9ec29",
}


def api() -> wandb.Api:
    return wandb.Api()


def run(run_id: str):
    return api().run(f"{ENTITY}/{PROJECT}/{run_id}")


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def line_series(run, key: str) -> dict[str, list[tuple[float, float]]]:
    """Recover a `wandb.plot.line_series` chart logged under `key` as
    {line name: [(x, y), ...]}. W&B stores the chart's data as a run table
    artifact named after the key."""
    want = _squash(f"{key}_table")
    for art in run.logged_artifacts():
        if art.type == "run_table" and want in _squash(art.name):
            root = Path(art.download())
            table_file = next(root.rglob("*.table.json"))
            data = json.loads(table_file.read_text())
            cols = data["columns"]
            xi, ki, yi = cols.index("step"), cols.index("lineKey"), cols.index("lineVal")
            out: dict[str, list[tuple[float, float]]] = {}
            for row in data["data"]:
                out.setdefault(row[ki], []).append((float(row[xi]), float(row[yi])))
            return {k: sorted(v) for k, v in out.items()}
    raise KeyError(f"no line_series table for {key!r} on run {run.name}")


def latest_calibrate_run():
    runs = [r for r in api().runs(f"{ENTITY}/{PROJECT}", filters={"jobType": "calibrate"})]
    return max(runs, key=lambda r: r.created_at)
