# Calibration, Abstention, and Sweeps

What `scripts/07_calibrate.py` and the W&B Sweep do, and how to read their output.
The numbers here are from the *first* model (MiniLM); the final model's calibration
and the full sweep table are in the [write-up](post/README.md).

## Why a cosine score isn't a confidence

The model's cosine similarity says how close two strings sit in embedding
space. It does not say how likely the answer is to be right. On the first
model, correct answers averaged 0.84 and wrong ones 0.74, with wide overlap:
a threshold that keeps 80% of the correct answers also lets half the wrong
ones through. And nothing about "0.84" means "84% likely": the scale is arbitrary.

**Calibration** learns the map from raw scores to an actual probability of
being right, using the validation split. Once a score of 0.9 means "right about
90% of the time", you can set a threshold that means something.

## The four signals compared

| Signal | What it is | Why it might work |
|---|---|---|
| **cosine** | top-1 similarity, as is | The obvious choice; the baseline for the others |
| **margin** | top-1 minus top-2 | A clear winner is more trustworthy than a photo finish |
| **softmax** | P(top-1) from a temperature-scaled softmax over the top-20 scores | Turns the whole ranking into a probability; the temperature (one number, fit on val) sets how sharp it is |
| **platt** | logistic regression on [cosine, margin, log softmax] → P(correct) | Combines the signals and is trained directly on "was this right?" |

**Temperature scaling** divides all scores by a constant T before the softmax.
Small T makes the top answer's probability near 1 whenever it leads at all;
large T flattens everything. T is chosen to make the probabilities match
reality on val (maximum likelihood). The first model's T = 0.034.

**Platt scaling** is a logistic regression: it learns weights for each signal
and a bias so that its output is P(correct). Mine is fit on val strings that
have an answer plus the *hard* unmatched val strings (real drugs with no
SCD/SBD), so it also learns what "there is no right answer" looks like.

## Measuring calibration

- **Reliability diagram / table:** bucket predictions by confidence (0.0–0.1, …,
  0.9–1.0). In each bucket, compare the mean confidence to the actual accuracy.
  A calibrated model has them equal: the 0.8–0.9 bucket is right about 85% of the time.
- **ECE (expected calibration error):** the average gap between confidence and
  accuracy across buckets, weighted by bucket size. 0 is perfect; under 0.05 is good.
- **AUROC:** how well the signal *ranks* right above wrong, ignoring the scale.
  0.5 is random, 1.0 is perfect separation. Useful for comparing cosine and margin,
  which aren't probabilities.

## Abstention: choosing the threshold honestly

`coverage@99%precision` in the baseline script is computed *on test*: find the
best threshold for test, report it on test. That flatters the number. The
honest procedure:

1. On **val**, find the lowest confidence at which everything above it is
   ≥ 99% precise (and separately ≥ 95%).
2. Apply that fixed threshold to **test**. Report the coverage (fraction
   auto-accepted) and the precision actually achieved.

If test precision lands at 97.8% when the target was 99%, that gap is real
information about how stable the threshold is.

### Three populations

Which strings count changes the answer, so the script reports all three:

| Population | Contents | Realism |
|---|---|---|
| **matched** | test VA strings that have an SCD/SBD | The usual benchmark |
| **+hard** | plus test VA names that are real drugs with *no* SCD/SBD (packs, ingredient-only concepts). Any answer is wrong. | What a mapper faces on the drug portion of a VA file |
| **+all** | plus supplies, devices, nutrition products | The whole VA file. 80% of it has no answer, so *coverage* is naturally small; watch precision and how many unmatched slip through |

## Results for the first model (MiniLM, ingredient negatives; thresholds from val, measured on test)

| Population | Signal | AUROC | ECE | coverage @ 95% target | precision | coverage @ 99% target | precision |
|---|---|---|---|---|---|---|---|
| matched | cosine | 0.736 | – | 47.8% | 0.920 | 13.0% | 0.963 |
| matched | margin | 0.871 | – | 61.6% | 0.966 | 32.7% | 0.995 |
| matched | softmax | 0.885 | 0.028 | 65.8% | 0.963 | 40.1% | 0.995 |
| matched | platt | 0.860 | 0.075 | 67.4% | 0.951 | 36.4% | 0.985 |
| +hard | cosine | 0.851 | – | 34.7% | 0.925 | 10.8% | 0.955 |
| +hard | margin | 0.841 | – | 21.3% | 0.940 | 1.4% | 0.906 |
| +hard | softmax | 0.876 | 0.061 | 27.8% | 0.947 | 2.4% | 0.926 |
| +hard | **platt** | **0.899** | **0.035** | **41.3%** | 0.958 | **21.7%** | 0.978 |

What it says:

- **Softmax and margin beat cosine badly on matched strings**: 40% of test strings
  can be auto-accepted at 99.5% precision, versus 13% for raw cosine (and 0.3%
  when the threshold wasn't chosen honestly at all).
- **Hard unmatched strings change the picture.** Margin collapses (1.4% coverage):
  a pack like "Medrol Dose Pack" retrieves one clear winner that is nevertheless
  wrong. Cosine, which margin ignores, carries the "this doesn't look like any
  candidate" information. Platt, which sees both, is the best signal on the
  realistic population: 21.7% at 97.8% precision, 41% at 95.8%.
- **Platt is well calibrated on +hard** (ECE 0.035): the 0.9–1.0 bucket is right
  97% of the time, the 0.5–0.6 bucket 53%.
- **The 99% target wasn't quite met on test** (97.8%). Val and test differ; the
  threshold transfers imperfectly. Report it as such.

## W&B Sweeps

A **sweep** is a search over settings that W&B coordinates. You describe the
space once (`sweeps/grid.yaml`); W&B hands out one combination at a time to
any **agent** that asks. An agent is just a loop: get a config, call `train()`,
report the metric, repeat. Agents can run anywhere (Colab, this machine) and
share the same grid.

- **method: grid** tries every combination (18 here). `random` samples; `bayes`
  uses earlier results to pick promising settings. Grid is right when the space
  is small and every cell is wanted for the analysis.
- **metric** is what the sweep page sorts and colors by: `val/acc@1`. Never test.
- **In `train()`,** `sweep=True` means "the agent already started the run; take
  this trial's parameters from `wandb.config`". Everything else is identical to
  a normal run, so sweep runs and hand-launched runs sit in the same runs table.
- Sweep runs don't log a model artifact (`log_model=False`). The winner is
  retrained once, with the artifact, and calibrated.

### Reading the sweep page

- **Parallel coordinates:** each run is a line through the parameter axes, colored
  by the metric. The axis where high-metric lines bunch together is the one that matters.
- **Parameter importance:** W&B's estimate of which parameter correlates most
  with the metric. Treat it as a hint, not a result: 18 runs is small.
- **Runs table, grouped:** group by `base_model` to see the average per encoder,
  by `negatives` for the negatives strategy, by `normalize_strength` for the ablation.
