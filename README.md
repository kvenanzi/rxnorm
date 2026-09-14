# VANDF → RxNorm Clinical Drug Normalization

Map VA National Drug File (VANDF) drug strings to RxNorm clinical drugs
(SCD/SBD): resolve ingredient, strength, **and** dose form together, and
abstain when the string doesn't determine the answer.

Published tools such as RxMap stop at the ingredient level (IN/MIN). This
project asks how far a small trained model gets at the full clinical-drug
level, using the fact that RxNorm already links every VANDF name to its
concept, so the labels are free.

**Result:** on 1,848 test strings whose ingredients never appear in training,
the final model (SapBERT fine-tuned with ingredient-matched hard negatives and
a strength-normalizing preprocessor) maps **93.1%** (95% CI 91.8–94.1) to the
exact clinical drug, with the right answer in the top five 98.4% of the time,
up from 50.9% for TF-IDF and 0% for exact match. With calibrated confidence,
79% of answerable strings can be auto-accepted at 98.8% precision; including
real drugs that have no clinical-drug concept, 46% at 98.2%.

The full write-up, with hypotheses, the 18-run sweep, calibration, error
analysis, and the experiment log, is [docs/post/README.md](docs/post/README.md).

## Use the model

```bash
pip install "rxnorm-vandf @ git+https://github.com/kvenanzi/rxnorm"
```

```python
from rxnorm_vandf.infer import Mapper

mapper = Mapper.from_pretrained("kvenanzi/vandf-rxnorm-biencoder")   # Hugging Face, ~450 MB
for p in mapper.map(["METOPROLOL TARTRATE 12.5MG TAB", "CATHETER,FOLEY SILICONE 22FR 5CC"]):
    print(p.rxcui, p.name, f"{p.confidence:.2f}", "accept" if p.accept else "review")
```

Model: [kvenanzi/vandf-rxnorm-biencoder](https://huggingface.co/kvenanzi/vandf-rxnorm-biencoder) ·
Dataset: [kvenanzi/vandf-rxnorm-pairs](https://huggingface.co/datasets/kvenanzi/vandf-rxnorm-pairs) ·
Runs: [wandb.ai/kettle-labs/rxnorm-vandf](https://wandb.ai/kettle-labs/rxnorm-vandf) ·
Report: [live panels](https://wandb.ai/kettle-labs/rxnorm-vandf/reports/VANDF-RxNorm-how-far-a-small-model-gets-at-the-clinical-drug-level--VmlldzoxNzkxNzIzNA)

## Documentation

- [docs/post/README.md](docs/post/README.md): the write-up
- [docs/rxnorm-primer.md](docs/rxnorm-primer.md): the RxNorm data model this all rests on
- [docs/training-primer.md](docs/training-primer.md): what `rxnorm_vandf/train.py` does and why
- [docs/calibration-and-sweeps.md](docs/calibration-and-sweeps.md): confidence, abstention, and the sweep
- [docs/wandb-primer.md](docs/wandb-primer.md): the W&B concepts used here (runs, artifacts, tables, sweeps, reports)

## Layout

```
data/          RxNorm release, rxnorm.duckdb, processed parquet (gitignored; never committed)
docs/          write-up and figures; RxNorm / training / calibration / W&B primers
rxnorm_vandf/  the package: data, evaluation, TF-IDF, training, calibration, strength rules, inference
scripts/       numbered pipeline: load, checkpoint, build dataset, upload, baselines, train, calibrate, sweep, publish, report, figures
notebooks/     Colab notebooks: train one model; run the sweep
sweeps/        W&B sweep definitions
tests/         pytest unit tests
models/        local training outputs (gitignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a UTS account to download the
[RxNorm Full Monthly Release](https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html).
Unzip the release into `data/rxnorm_2026_09_08/`.

```bash
uv sync                              # create .venv with pinned dependencies
uv run scripts/01_load_duckdb.py     # ~15s; builds data/rxnorm.duckdb
uv run scripts/02_checkpoint.py      # VANDF -> SCD/SBD pair counts and samples
uv run scripts/03_build_dataset.py   # ~2s; writes data/processed/*.parquet

uv run wandb login                   # once; paste your API key from wandb.ai/authorize
uv run scripts/04_upload_dataset.py  # logs data/processed as a W&B Artifact
uv run scripts/05_baselines.py       # ~12s; exact + TF-IDF baselines, logged to W&B

uv run scripts/06_train.py --smoke --offline   # ~30s pipeline check on the local GPU
uv run scripts/06_train.py                     # train the bi-encoder (~5 min on a GTX 1070)
uv run scripts/06_train.py --normalize-strength   # ablation: append RxNorm-style strengths to inputs

uv run scripts/07_calibrate.py                 # calibrate confidence, choose abstention thresholds
uv run scripts/08_sweep.py --create            # register the 18-run sweep; run it in Colab (below)

uv run scripts/09_publish_hf.py --dry-run      # stage the HF model + dataset repos (drop --dry-run to upload)
uv run scripts/10_report.py                    # build the W&B Report from the runs
uv run scripts/11_figures.py                   # figures for docs/post/
uv run pytest                                  # unit tests (strength normalizer, inference wrapper)
```

### Training in Colab

[`notebooks/01_train_biencoder.ipynb`](notebooks/01_train_biencoder.ipynb) runs the
same `rxnorm_vandf.train.train()` on a Colab GPU. Open it from GitHub in Colab, set
the runtime to T4, and add `WANDB_API_KEY` as a Colab Secret (key icon in the
sidebar, notebook access on). The notebook pulls the dataset from the W&B
Artifact, so it never needs the RxNorm release.

[`notebooks/02_sweep.ipynb`](notebooks/02_sweep.ipynb) runs a W&B agent for the
sweep in `sweeps/grid.yaml` (encoder × negatives × strength normalizer, 18 runs,
~30 min on a T4). Register the sweep locally first with `08_sweep.py --create`.

## Where things run

| Work | Where | Why |
|---|---|---|
| Data prep, pair building, baselines | Local (uv venv) | CPU-only work; the 1.8 GB release and its restricted sources stay on one machine |
| Bi-encoder training, sweeps | Local GTX 1070 or Google Colab (T4) | Same `train()` either way; Colab opens notebooks straight from this repo |
| Handoff between the two | W&B Artifact | Versioned dataset: Colab pulls exactly the pairs file the local scripts produced |

## Data licensing

The RxNorm full release bundles sources with extra license restrictions:
SNOMED CT US (restriction level 9) and GS, MMX, and NDDF (level 3). This
project uses only `VANDF` and `RXNORM`, both restriction level 0, and anything
published (datasets, models, reports) is derived from those two sources alone.
