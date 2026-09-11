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
