# Weights & Biases Primer

W&B is not a modeling framework. The model is built in PyTorch; W&B is the
lab notebook kept *around* it: what ran, with what settings, what came out,
and which data went in. This primer covers the pieces this project uses.

## The vocabulary

| Term | What it is | Where it shows up here |
|---|---|---|
| **Entity** | The account or team that owns things. Yours is your username. | The first part of every URL: `wandb.ai/<entity>/...` |
| **Project** | A folder of related runs and artifacts. | `rxnorm-vandf`. Everything for this project lives in it. |
| **Run** | One unit of work: a script or a training job. Each has a unique id and a page in the UI. | `04_upload_dataset.py` is one run; each baseline is one run; each training session (local or Colab) is one run. |
| **Job type** | A free-form label on a run so you can group them. | `dataset`, `baseline`, `train`, `sweep`, `calibrate`. |
| **Config** | The settings a run was started with. Set once at `wandb.init(config=...)`. | Baseline name and n-gram range; for training, learning rate, batch size, encoder, negatives strategy. |
| **`wandb.log`** | Records a value at a *step*. Call it repeatedly and you get a curve. | Training loss every batch; the precision/coverage chart. |
| **Summary** | One final value per key for the run. Set with `run.summary[key] = value` (or W&B fills it with the last logged value). | `test/acc@1`, `test/recall@5`, and the rest of the metrics. This is what the runs table compares. |
| **Artifact** | A versioned bundle of files. Log it once, then any run can `use_artifact` it. | `vandf-rxnorm-pairs`: the three parquet files. Colab pulls this instead of needing the raw RxNorm release. |
| **Lineage** | The graph of which runs produced and consumed which artifacts. | The artifact page shows `04_upload_dataset` produced it and the baseline runs used it. |
| **Table** | A spreadsheet logged to a run; sortable and filterable in the UI. | `test/failures`: 200 test strings the baseline got wrong, with its top 5 guesses. |
| **Sweep** | W&B picks settings from a grid or distribution and launches a run per setting. | `sweeps/grid.yaml`: encoder × hard-negative strategy × strength normalizer, 18 runs. |
| **Report** | A document with live charts pulled from runs, publishable at a URL. | The live companion to the [write-up](post/README.md): training curves, the sweep, calibration. |

## The lifecycle of a run

```python
import wandb

run = wandb.init(project="rxnorm-vandf", job_type="baseline", config={"lr": 1e-4})
run.use_artifact("vandf-rxnorm-pairs:latest")   # optional: record the input data
for step in range(100):
    run.log({"loss": ...})                       # curves
run.summary["test/acc@1"] = 0.51                 # final numbers
run.log({"failures": wandb.Table(...)})          # tables, plots
run.finish()
```

`init` starts the run and prints its URL. `finish` uploads whatever is pending
and closes it. A script that crashes without `finish` shows as "crashed" in the UI.

## Logging in

W&B needs an API key, which identifies your account. Get it at
https://wandb.ai/authorize. It is a secret: never paste it into a script or
commit it.

- **Locally:** `uv run wandb login`, paste the key once. It's saved to `~/.netrc`
  and every later script picks it up.
- **Colab:** either call `wandb.login()` in a cell and paste the key when prompted,
  or store the key in Colab's Secrets panel (the key icon in the left sidebar) as
  `WANDB_API_KEY` and load it with
  `os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")` before `wandb.init`.
  Secrets are the better option: the key never appears in the notebook.
- **Offline mode:** `WANDB_MODE=offline` writes runs to the local `wandb/` folder
  instead of uploading. `05_baselines.py --offline` uses this for smoke tests.
  `wandb sync wandb/offline-run-*` uploads them later if you want them.

## Finding your way around the UI

- **Project page** (`wandb.ai/<you>/rxnorm-vandf`): the runs table. Each row is a
  run; the columns are config and summary values. Pin `test/acc@1` and
  `test/recall@5` to compare runs at a glance.
- **Run page:** tabs for *Overview* (config, command, git commit, duration),
  *Charts* (everything logged with `wandb.log`), *Tables*, *Files*, *Logs*
  (the script's stdout).
- **Artifacts** (left sidebar): each artifact and its versions (`v0`, `v1`, …).
  A version page has *Files*, *Metadata*, and *Lineage* tabs.
- **Sweeps** and **Reports** are also in the left sidebar: the 18-run grid and the report built by `scripts/10_report.py`.

## Why bother with the dataset artifact

Three reasons:

1. **Colab needs the data.** The raw RxNorm release can't leave this machine, and
   the processed parquet files shouldn't be committed to git. `run.use_artifact(...).download()`
   in Colab fetches exactly the version you logged.
2. **Every result is tied to a data version.** If `03_build_dataset.py` changes the
   split, that's `v1`, and the runs table shows which runs used `v0` vs `v1`.
3. **It is the reproducible input.** The artifact (mirrored on Hugging Face as
   `kvenanzi/vandf-rxnorm-pairs`) is the exact dataset behind every reported number.
