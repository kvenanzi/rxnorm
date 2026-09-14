# Three Levers on the VANDF → RxNorm Mapper: More Data, a Strength Head, and Folds

| created | modified | status | confidence | importance |
|---|---|---|---|---|
| 2026-09-14 | 2026-09-14 | in progress | – | 3 |

> **Abstract.** *To be written from the results.* The [first write-up](../post/README.md) left a small bi-encoder at 0.907 ± 0.019 test acc@1 on VA drug strings with held-out ingredients, and named three things that might move it rather than measure it better: more training strings from another RxNorm source vocabulary, an auxiliary head that has to name the strength, and training on every ingredient family instead of holding fifteen percent out. This follow-up runs all three against that band, paired within the same six ingredient draws and two training seeds, and adds the k-fold estimate and the calibration transfer that §7 asked for.

## 1. What is being tested, and against what

The first write-up established the measuring stick. Its final recipe (SapBERT, ingredient-matched hard negatives, the strength normalizer) scores 0.931 on the published split and 0.907 ± 0.019 across six ingredient draws; training-seed noise is 0.001; the one factor whose effect was inside the split noise, hard negatives, held on all six draws when paired within split. Any lever tested here has to be measured the same way: paired within (split, seed), with the sign count and a 95% interval over the units, not as a single number on the favourable draw.

The three levers are the ones §7 of that write-up named.

1. **More data from another source vocabulary.** The same trick that built the dataset works for any RxNorm source: an atom that shares an RXCUI with an SCD/SBD is a labelled pair. Of the thirteen sources in the release, the only one that is both large and restriction level 0 is MTHSPL, the FDA Structured Product Label names: 48,985 (string, SCD/SBD) pairs over 16,000 targets, none of which is a VA string, in a very different house style (`PAROXETINE HYDROCHLORIDE 40 mg ORAL TABLET, FILM COATED [Paroxetine]`). The split is by ingredient on the candidate side, so the MTHSPL strings inherit the six existing splits: a split's VA test set is byte-identical with or without them, and only train-split MTHSPL strings are ever trained on. On the published split that is 29,505 extra training rows on top of 9,290.
2. **An auxiliary strength head.** Strength was the largest error category of every model in the first write-up. A linear classifier on the anchor embedding, trained jointly with the contrastive loss, that has to name the target's strength string (`250 MG`, `0.025 MG/ML`; 232 classes with five or more training examples cover 79% of the VA training rows, the rest get no head loss). The head is a training signal only: it lives in the loss, never in the saved model, so inference is unchanged.
3. **K-fold by ingredient, and the model that sees every ingredient family.** Seven folds of the published ingredient hash, chosen so that fold 0's test set is exactly the published test set (buckets 0–14 of 100); each fold trains on five, selects its epoch on one, and is tested once on the seventh. Pooling the seven test sets gives one out-of-fold accuracy over every VA string that is ever held out, with a sampling interval instead of a six-draw band. The all-data model then trains on every ingredient, for the median best epoch of the folds, and its calibration is fit on the pooled out-of-fold predictions, which also answers §7's first question: do the abstention thresholds transfer across draws?

Levers 1 and 2 are one 2 × 2 grid on the six splits and two seeds, 48 runs (`sweeps/levers.yaml`), plus two two-run controls that separate "more optimizer steps" (VANDF only, 16 epochs) and "more rows" (MTHSPL capped at the VA row count) from "different strings". Lever 3 runs afterwards with whichever arm of the grid wins on validation.

## 2. Hypotheses

Written before any of the 48 runs had been started; the pilot in Appendix A chose the head's loss weight and nothing else.

- **H1.** MTHSPL training strings raise VA test acc@1 by a small amount (a point or so, inside the split band but consistent in sign across the twelve units), and raise accuracy on held-out MTHSPL strings by a large amount, because the VA-only model has never seen the FDA house style.
- **H2.** The strength head lowers the VA strength-error rate and leaves ingredient and dose-form accuracy unchanged; the effect on acc@1 is small and may not clear the split noise.
- **H3.** The two effects are additive: the interaction term is indistinguishable from zero.
- **H4.** The compute-matched control (VANDF only, 16 epochs) gains less than the MTHSPL arm, and the size-matched control (MTHSPL capped at 9,300 rows) captures most of the MTHSPL gain; that is, the value is in the different strings, not the step count.
- **H5.** The pooled out-of-fold acc@1 of the recipe lands inside 0.907 ± 0.019, and a 99%-precision threshold chosen on six folds achieves at least 98% on the seventh, as the single-split thresholds did on their test set.

## 3. Data and method (what changed)

- `03_build_dataset.py --sources VANDF,MTHSPL` unions the MTHSPL `DP` atoms (the NLM-generated `MTH_RXN_DP` duplicates are excluded, as `MTH_RXN_CD` was for the VA) into the input table with a `source` column; the pairs, candidates, and split logic are otherwise unchanged, and the six folders under `data/multi/` reproduce the VA rows of `data/splits/` exactly. Unmatched strings are kept per source.
- `train_sources` selects which sources' train rows feed the triplets; every run evaluates the VA val/test of its split under the usual keys and the MTHSPL val/test rows under `val_mthspl` / `test_mthspl`, so the VA-only arm reports MTHSPL transfer for free.
- `aux=strength` adds an integer label column and swaps the loss for `MNRLWithStrengthHead` (`rxnorm_vandf/losses.py`): the contrastive loss on the same batches, plus `aux_weight` × cross-entropy of a linear head on the anchor embedding. The head's initialization is seeded separately so the two arms of a seed see identical batches. `aux_weight` = 0.01, chosen on validation in the pilot of Appendix A from {0.01, 0.05, 0.2, 0.5}.
- `03_build_dataset.py --kfold 7 --fold i` and `--all-train` draw the fold folders; `14_kfold.py` runs them, pools each fold's logged predictions (`predictions.parquet`, top-20 per string), fits the calibration on the pool, and reports leave-one-fold-out threshold transfer.
- Hyper-parameters are those of the first write-up throughout (4 epochs, batch 64, lr 2e-5, seed 42 and 1). Epochs are not rescaled for the larger MTHSPL arm; the control sweep covers that.

## 4. Results

*Pending the sweep.* Sections to come: 4.1 the 2 × 2 (`outputs/levers/levers.md`, Figure 1), 4.2 the controls, 4.3 MTHSPL transfer, 4.4 k-fold and the all-data model (`outputs/kfold/oof.md`, Figure 2), 4.5 calibration on out-of-fold predictions (`outputs/kfold/calibration.md`).

## 5. Discussion

*Pending.*

## 6. Reproduction

Everything runs from the commands in the repository README under "The follow-up". Sweeps `nypttmi8` (main) and `5cr5ze02` (twelve aux-off cells on v3, v4, v5 re-run after the loader fix noted in Appendix A; `j1zdu28j` was registered with the pilot's 0.2 weight and never run), `2exoct4h` (steps control), `ocmpf5na` (size control); W&B group `kfold-k7`; artifacts `vandf-rxnorm-multi:v0` and `vandf-rxnorm-kfold:v0` for the data, `vandf-rxnorm-predictions` per fold, `vandf-rxnorm-biencoder-final-all` for the all-data model. The published `vandf-rxnorm-pairs`, `vandf-rxnorm-splits`, and `vandf-rxnorm-biencoder` artifacts are untouched.

## Appendix A: Experiment log

- **Tooling** (2026-09-14, branch `levers`). Multi-source pairs, the strength-head loss, k-fold folders, predictions artifacts, the `--log-model` / `--model-artifact` switches; 19 new unit tests. The default dataset build emits byte-identical SQL. Local smoke runs of every new path on the GTX 1070.
- **Pilot** (2026-09-14, W&B group `levers-pilot`, GTX 1070, split v1, seed 42, VA strings only, everything else the published recipe; the reference is the split-v1 run of the first write-up's §5.8, `lqf7dett`, same machine). The head's cross-entropy over 232 classes starts near 4 against a contrastive loss near 0.05, so the weight decides whether the head is a regularizer or the objective. Four weights, selected on validation acc@1:

  | head weight | run | val acc@1 | test acc@1 | strength | ingredient | dose form | best epoch |
  |---|---|---|---|---|---|---|---|
  | none (reference) | `lqf7dett` | 0.882 | 0.931 | 0.961 | 0.983 | 0.972 | – |
  | 0.01 | `tiotc7iy` | 0.880 | 0.928 | 0.963 | 0.985 | 0.970 | 3 |
  | 0.05 | `9zw6jvov` | 0.874 | 0.923 | 0.962 | 0.983 | 0.969 | 1 |
  | 0.2 | `0usjnp44` | 0.864 | 0.904 | 0.963 | 0.975 | 0.964 | 2 |
  | 0.5 | `spja8f8i` | 0.850 | 0.890 | 0.957 | 0.973 | 0.956 | 1 |

  A monotone dose-response: the heavier the head, the lower the validation accuracy, the earlier the best epoch, and the more ingredient and dose-form accuracy give way, while strength accuracy never moves. The grid uses 0.01, the largest weight that is not already worse than no head on one split; H2 is tested at that weight. The pilot was run on the published split and is not part of the paired analysis.

- **First sweep session** (2026-09-14, A100). The v1 VA-only cell reproduced the first write-up's 0.927 test acc@1 exactly. Twelve cells on v3, v4, v5 then failed in seconds: six, four, and one MTHSPL label names in those draws map to two products whose ingredients fall in different splits, and the new loader raised on them rather than dropping them. No VA string does this on any draw. The loader now drops such strings (they can be neither trained on nor scored without leaking), and the twelve cells run again as sweep `5cr5ze02`; a grid sweep never re-issues a failed cell.

## Appendix B: Decisions

| Decision | Reason |
|---|---|
| MTHSPL and no other extra source | The only large source at restriction level 0; MMSL is level 1, GS/MMX/NDDF level 3, SNOMED CT level 9, and CVX/MTHCMSFRF add 190 strings between them |
| MTHSPL strings inherit the existing splits | The split is a function of the target's ingredients; the VA test sets stay identical and every comparison is paired |
| `val/*` and `test/*` stay VA-only | Every published panel, table, and script reads those keys; new sources get new keys |
| The head lives in the loss module, not the model | The trainer optimizes the loss's parameters; `model.save()` never writes the head; inference is unchanged |
| Rare strengths get no head loss rather than an OTHER class | An OTHER class holding a fifth of the rows would teach the head a junk cluster |
| Epochs fixed at 4 in the grid; a 16-epoch control instead | "Same recipe, more data" is the deployment question; the control separates steps from strings |
| Seven folds, contiguous buckets | 71/14/14 like the published split, and fold 0 reproduces the published test set |
| Unmatched strings assigned to one fold by hash | They are never trained on, so any fold's model scores them out-of-fold; the hash counts each once |
| Only the all-data model persists its weights | 48 sweep checkpoints would be about 21 GB for nothing; the fold models are summarized by their predictions |
| A separate artifact name for the all-data model | `vandf-rxnorm-biencoder:latest` must keep pointing at the published model |
| Strings whose targets span splits are dropped at load time | Eleven MTHSPL names across three draws; leaving them in would leak a held-out ingredient into training or score an unanswerable query |
