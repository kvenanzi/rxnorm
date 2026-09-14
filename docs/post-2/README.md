# Three Levers on a Drug-Name Mapper: More Sources, a Strength Head, and Folds

| created | modified | status | confidence | importance |
|---|---|---|---|---|
| 2026-09-14 | 2026-09-14 | in progress | –[^conf] | 3 |

> **Abstract.** The [first write-up](../post/README.md) trained a small bi-encoder to map VA drug names to RxNorm clinical drugs and, after re-drawing the ingredient split five times, put its accuracy on unseen ingredients at 0.907 ± 0.019 rather than the 0.931 of the published draw. It ended by naming the levers that might move that number instead of measuring it better. This follow-up pulls three of them under the same discipline, paired within the same six ingredient draws and two training seeds, so that an effect has to beat a two-point split noise, not a favourable draw. The levers: (1) a second source vocabulary from the same release, the FDA's Structured Product Label names (MTHSPL), which is the only large source at UMLS restriction level 0 besides the two already used, adding 48,985 labelled pairs whose strings look nothing like the VA's; (2) an auxiliary classification head that must name the target's strength from the query embedding ([Caruana 1997](#references)), aimed at the error category that every earlier model shared; (3) seven-fold cross-validation by ingredient ([Stone 1974](#references)), which yields a single out-of-fold estimate over every held-out string, a model trained on every ingredient family, and a test of whether abstention thresholds transfer across draws. A pilot on the published split found that the head's loss is fifty times the contrastive loss and that any weight above 0.01 lowers validation accuracy monotonically; the grid runs the head at 0.01. *Results and verdicts will be added as the sweeps finish; the sections are laid out below with what is already known.*

## 1. What is being tested, and against what

The first write-up left a measuring stick. Its final recipe, SapBERT ([Liu et al 2021](#references)) fine-tuned with the in-batch softmax loss of [Henderson et al 2017](#references), ingredient-matched hard negatives, and a rule that appends RxNorm-style concentrations to the VA string, reached test acc@1 0.931 on the published ingredient split and 0.907 ± 0.019 across six draws of that split, with training-seed noise of 0.001. That last pair of numbers is the reason this follow-up is shaped the way it is. [Bouthillier et al 2021](#references) measured, across a range of benchmarks, that the choice of data split contributes more variance than the choice of training seed, often by a wide margin; the first write-up found the same thing on this task, by a factor of about twenty. A lever that moves acc@1 by a point on one split has shown nothing. The only factor of the first write-up's sweep whose effect was inside the split noise, hard negatives, was re-run on all six draws and held on every one; the design here is that re-run's design, applied from the start: every arm on every draw, paired, with the sign count and a 95% interval over the units, in the manner [Dietterich 1998](#references) recommends for comparing learners on the same folds.

The three levers are §7's. Two of them are training-time interventions and run as one factorial grid; the third is a change in how the final model is trained and measured, and runs afterwards with whichever arm of the grid wins.

**More data from another source vocabulary.** The trick that built the dataset works for any RxNorm source: an atom that shares an RXCUI with a normalized clinical-drug name is a labelled pair, curated by NLM ([Nelson et al 2011](#references)). The September 2026 release carries thirteen sources, and the UMLS license sorts them into restriction categories ([Bodenreider 2004](#references); [NLM license appendix](#references)). Only category-0 content can go into a published dataset or model, which removes the commercial drug databases and SNOMED CT, and leaves exactly one large candidate, described in §3. The bet, in the spirit of [Halevy et al 2009](#references), is that four times as many strings in a different house style teach the encoder more about what a product name can look like than any tweak to the loss.

**An auxiliary strength head.** Strength was the largest error category of every model in the first write-up: three-quarters of TF-IDF's failures, six in ten of MiniLM's, and still the largest single share of the final model's 128, with a fair number of those underdetermined by the input. A bi-encoder is never told what strength is; it learns that `250MG` near `250 MG` tends to be right. Multi-task learning ([Caruana 1997](#references); [Ruder 2017](#references)) offers the obvious remedy: add a small classifier on the query embedding that has to name the target's strength string, train it jointly, and throw it away. If the shared encoder is pushed to encode strength explicitly, the retrieval head should benefit. The first write-up deferred this because the components already come free from the retrieved candidate's labels; the question here is whether they also help as a training signal.

**K-fold by ingredient, and the model that sees every ingredient family.** The published model never saw 30% of the ingredient families: 15% were held out for validation and 15% for test. A deployed mapper has no reason to be blind to them. Cross-validation ([Stone 1974](#references); [Kohavi 1995](#references)) is the standard answer: partition the ingredients into K folds, train K models each with one fold held out, pool the K held-out scores into one estimate over every string, then train a final model on everything and attach the pooled estimate to it. Two things come free. The pooled estimate has a sampling interval over roughly 12,000 strings rather than a six-draw band whose width is itself only known to a factor of two. And the K models' held-out predictions are exactly the material needed to fit the calibration layer without touching any string the final model trained on, which also answers §7's first question: chosen on six folds, does a 99%-precision threshold hold on the seventh?

## 2. Hypotheses

Written before any run of the grid had started. The pilot of §4.2 chose one number, the head's loss weight, and nothing else; it ran on the published split alone and is not part of the paired analysis.

- **H1.** MTHSPL training strings raise VA test acc@1 by a small amount, about a point, inside the split band but of one sign across the twelve (split, seed) units; and they raise accuracy on held-out MTHSPL strings by a large amount, because the VA-only model has never seen the FDA's house style.
- **H2.** The strength head lowers the VA strength-error rate and leaves ingredient and dose-form accuracy where they are; its effect on acc@1 is small and may not clear the split noise.
- **H3.** The two effects are additive: the interaction term is indistinguishable from zero.
- **H4.** The compute-matched control (VA strings only, sixteen epochs) gains less than the MTHSPL arm, and the size-matched control (MTHSPL capped at the VA row count) keeps most of the MTHSPL gain. That is, the value is in the different strings, not in the extra optimizer steps or the extra rows.
- **H5.** The pooled out-of-fold acc@1 of the recipe lands inside 0.907 ± 0.019, and a 99%-precision threshold chosen on six folds achieves at least 98% precision on the seventh, as the single-split thresholds achieved on their test set.

## 3. Data

### 3.1 The other sources

`RXNSAB` records each source's restriction category in its `SRL` column. The candidate list for a second source is short once the license and the size are both applied:

| Source | What it is | SRL | Active atoms that land on an SCD/SBD | Distinct strings | Targets covered |
|---|---|---|---|---|---|
| RXNORM | NLM's normalized names (already the target side) | 0 | 91,762 | 78,903 | 27,287 |
| **MTHSPL** | **FDA Structured Product Label names** | **0** | **101,005** | **49,036** | **16,000** |
| NDDF | FDB MedKnowledge | 3 | 30,706 | 22,541 | 9,495 |
| GS | Gold Standard | 3 | 22,593 | 21,186 | 12,927 |
| MMSL | Multum | 1 | 22,048 | 22,048 | 15,016 |
| MMX | Micromedex | 3 | 20,645 | 20,620 | 14,734 |
| VANDF | VA National Drug File (already the query side) | 0 | 17,964 | 14,931 | 8,315 |
| SNOMEDCT_US | SNOMED CT | 9 | 14,036 | 14,036 | 6,219 |
| CVX | vaccine codes | 0 | 187 | 187 | 181 |
| MTHCMSFRF | CMS formulary reference | 0 | 3 | 3 | 3 |
| ATC, DRUGBANK, USP | classifications and monographs | 0 | 0 | 0 | 0 |

MTHSPL is the drug-product names from FDA's Structured Product Labeling ([FDA SPL](#references)), which NLM ingests into RxNorm ([NLM RxNorm technical documentation](#references)). It is the largest source in the table, it is category 0, and none of its strings equals a VA string, case-insensitively, more than once in 49,036. The next three sources are bigger than the VA file but category 1 or 3, and using them would break the claim on the published dataset card that nothing restricted was used. CVX and MTHCMSFRF are category 0 and add 190 strings between them, which is not worth a third source column. So the experiment is VA versus VA plus MTHSPL.

The two house styles are far apart. A VA name is short and abbreviated (`PAROXETINE HCL 40MG TAB`, median 34 characters); an SPL name is long, spelled out, and carries the brand in brackets:

| MTHSPL string | Target |
|---|---|
| `PAROXETINE HYDROCHLORIDE 40 mg ORAL TABLET, FILM COATED [Paroxetine]` | `paroxetine hydrochloride 40 MG Oral Tablet` |
| `CAMPHOR (SYNTHETIC) 35 mg in 1 g / MENTHOL 60 mg in 1 g TOPICAL PATCH [Arcticzen]` | `camphor 0.035 MG/MG / menthol 0.06 MG/MG Medicated Patch [Arcticzen]` |

The median SPL string is 66 characters and the 99th percentile 240, against a 96-token limit that was set for the VA; the bracketed brand at the tail of the longest one or two percent is truncated, and I left the limit alone rather than confound the comparison. The `X mg in 1 mL` notation is not one the strength normalizer of the first write-up parses, so the rule is a no-op on SPL strings; that too is left as it is, for the same reason.[^normalizer]

### 3.2 Pairing and splitting

The same term-type filter applies as for the VA: `DP` (drug product) atoms only, with NLM's generated `MTH_RXN_DP` duplicates excluded as `MTH_RXN_CD` was. Deduplicated on (string, target) that gives 48,985 pairs over 47,635 distinct strings and 16,000 targets, and 21,595 active SPL strings whose concept has no SCD/SBD, which join the unmatched set and are split val/test by the same string hash.

The split needs no new rule. The first write-up assigns every ingredient to train, val, or test by an md5 of its RXCUI, and a *target* follows its ingredients; the query strings inherit the target's split. MTHSPL strings therefore fall into the six existing draws automatically, a VA test set is byte-identical whether or not the SPL strings are present, and only train-split SPL strings are ever trained on. On the published draw that is 29,505 extra training rows on top of the VA's 9,290; the other five draws have between 30,218 and 33,681.

One thing did need a rule. Eleven SPL strings across three draws (six in v3, four in v4, one in v5) map to two products whose ingredients landed in different splits, which no VA string does on any draw. Such a string can be neither trained on, which would leak a held-out ingredient, nor scored, since one of its answers is in training. The loader drops them and says so.[^straddle]

### 3.3 Folds

The published salt hashes each ingredient into a bucket from 0 to 99 and calls buckets 0–14 test and 15–29 val. Seven folds of that same hash, with fold = ⌊7 × bucket / 100⌋, make fold 0 exactly buckets 0–14: the published test set, string for string. Fold i is tested with fold i+1 as its validation set and the other five as training, so each model sees 71% of ingredient families, like the published one. The `mixed` rule is unchanged, a target whose ingredients straddle the three sets is excluded from that fold, and this is what limits the out-of-fold coverage: a combination drug whose ingredients sit in different folds is mixed in two rotations and training data in the rest, so it is never tested. The seven test sets hold 1,848, 1,688, 1,684, 1,890, 1,759, 1,533, and 1,828 VA strings, 12,230 of the 14,372 pairs. The final model's folder puts every ingredient in train, and nothing is mixed.

## 4. Method

### 4.1 The grid and its controls

One W&B grid (`sweeps/levers.yaml`): the final recipe, unchanged, times {VA only, VA + MTHSPL} times {no head, strength head} times the six draws times seeds 42 and 1, 48 runs on one Colab A100. Selection, where it is needed, is on `val/acc@1` as before; `val` and `test` keep their meaning, the VA strings of that draw, and the SPL strings of the same draw are scored alongside under their own keys, so the VA-only arm reports SPL transfer, §7's fourth item, at no cost. Epochs stay at four for both data arms; the SPL arm therefore takes about four times the optimizer steps, and two controls on the best and worst draws of the first write-up (v1 and v4) separate the confounds: VA only for sixteen epochs (the same steps, the same strings), and VA plus SPL capped at 9,300 rows (the same steps, different strings).

Analysis is paired within (split, seed): twelve units. Each main effect is the mean of the two simple contrasts per unit, the interaction is the usual double difference, and each gets a sign count and a t interval over the twelve; [Nadeau and Bengio 2003](#references) is the reminder that units sharing training data are correlated and that such intervals are optimistic, which is why the sign count is reported beside them.

### 4.2 The strength head, and the pilot that set its weight

The head is one linear layer on the query embedding, 768 inputs and one output per strength class, trained by cross-entropy against the strength string of the positive (`250 MG`, `0.025 MG/ML`, or the ` / `-joined list for a combination). Classes are the strengths with at least five training examples: 232 of them cover 79% of the VA training rows; the rest contribute no head loss rather than an OTHER class that would hold a fifth of the rows and teach a junk cluster. The head lives inside the loss module, so the trainer's optimizer updates it and `model.save()` never writes it; the saved encoder, the inference wrapper, and every published number's code path are untouched. Its initialization is seeded apart from the training stream, so the two arms of a seed see identical batches.

The weight on the head loss is the one hyper-parameter with no precedent in this project, and it matters more than I expected. A 232-way cross-entropy starts near 4 nats; the contrastive loss is near 0.09 after warm-up and 0.03 at the end. At weight 0.2 the head is the objective and the retrieval loss a regularizer, which is the wrong way round. The literature on weighting auxiliary losses ([Kendall et al 2018](#references)) and on conflicting task gradients ([Yu et al 2020](#references)) says the same in general terms; the specific number had to come from a pilot. Four weights, on the published split, seed 42, VA strings only, everything else the final recipe, and the split-v1 run of the first write-up's §5.8 (`lqf7dett`, same GPU) as the reference:

| head weight | run | val acc@1 by epoch | best val acc@1 | test acc@1 | strength | ingredient | dose form |
|---|---|---|---|---|---|---|---|
| none | `lqf7dett` | – | 0.882 | 0.931 | 0.961 | 0.983 | 0.972 |
| 0.01 | `tiotc7iy` | 0.877 → 0.876 → **0.880** → 0.880 | 0.880 | 0.928 | 0.963 | 0.985 | 0.970 |
| 0.05 | `9zw6jvov` | **0.874** → 0.870 → 0.872 → 0.872 | 0.874 | 0.923 | 0.962 | 0.983 | 0.969 |
| 0.2 | `0usjnp44` | 0.859 → **0.864** → 0.850 → 0.848 | 0.864 | 0.904 | 0.963 | 0.975 | 0.964 |
| 0.5 | `spja8f8i` | **0.849** → 0.842 → 0.821 → 0.811 | 0.850 | 0.890 | 0.957 | 0.973 | 0.956 |

A monotone dose-response. The heavier the head, the lower the validation accuracy, the earlier the best epoch, and the more ingredient and dose-form accuracy give way, while strength accuracy, the thing the head is for, does not move at any weight. At 0.01 the two losses are of a size and the run is indistinguishable from the reference. The grid tests H2 at 0.01, the largest weight that is not already worse than no head on one split; a null there is the honest answer to the hypothesis as posed, and the pilot already says how a positive would have to be read.

### 4.3 Folds and the final model

Each fold trains the winning arm of the grid, selects its epoch on its validation fold, scores its test fold once, and logs every string's top-20 candidates and cosines, matched and unmatched, as an artifact. The out-of-fold table pools the seven test folds: one acc@1 with a Wilson interval ([Wilson 1927](#references)) over 12,230 strings, beside the per-fold values and their spread, which is the six-draw band of the first write-up measured a different way. The final model trains on the all-ingredient folder for the median best epoch of the seven, since it has no validation set to stop on, and its weights are logged from Colab under a new artifact name, so that the published model's `latest` pointer does not move.

### 4.4 Calibration on out-of-fold predictions

The first write-up fit a temperature and a Platt layer ([Guo et al 2017](#references); [Platt 1999](#references)) on one validation split and chose thresholds there; the 99% target then landed at 98.2–98.8% on test, and §7 asked whether that shortfall is stable. The pooled out-of-fold predictions are the right material: every matched string was scored by a model that never saw its ingredients, and every hard unmatched string (a real drug with no clinical-drug concept) is assigned to one fold by a hash of its text so that it is counted once. Temperature and Platt are fit on that pool and shipped with the final model as its `calibration.json`, the four-number layer of the first write-up. Then the check that matters ([Varma and Simon 2006](#references)): for each fold, the 95% and 99% thresholds are chosen on the other six and applied to it, and the achieved precision and coverage are tabulated, along with the full seven-by-seven matrix of "chosen on i, applied to j". If the leave-one-fold-out precision sits where the single-split test did, the shortfall is a property of the task and can be budgeted for; if it swings by folds, the thresholds are draw-specific and a deployment should re-choose them on its own data.

## 5. Results

*Pending the sweeps.* 5.1 The grid (`outputs/levers/levers.md`, Figure 1). 5.2 The controls. 5.3 SPL transfer. 5.4 Folds and the final model (`outputs/kfold/oof.md`, Figure 2). 5.5 Calibration transfer (`outputs/kfold/calibration.md`).

## 6. Discussion

*Pending.*

## 7. Reproduction

- **Code:** the same repository as the first write-up, branch `levers` until merged; the commands are listed in its README under "The follow-up". Nothing the first write-up depends on changed: its data folder, its scripts, its figures, and its numbers are reproduced by the same commands, and the default dataset build emits the same SQL byte for byte. Every new behaviour is behind a new flag or a new configuration field with a default.
- **Runs:** sweeps `nypttmi8` (the grid), `ad9nakej` and `2v3am5xr` (cells re-run after the two failures noted in Appendix A), `2exoct4h` and `ocmpf5na` (the controls); W&B group `kfold-k7` for the folds and the final model; group `levers-pilot` for §4.2.
- **Artifacts:** `vandf-rxnorm-multi:v0` (the six draws with both sources) and `vandf-rxnorm-kfold:v0` (the folds) for the data; `vandf-rxnorm-predictions` per fold; `vandf-rxnorm-biencoder-final-all` for the all-data model. The first write-up's `vandf-rxnorm-pairs`, `vandf-rxnorm-splits`, and `vandf-rxnorm-biencoder` are untouched.

## Appendix A: Experiment log

- **Tooling** (2026-09-14, branch `levers`). A second query source in the dataset builder with the split inherited from the target; the strength-head loss; fold and all-train folders; a predictions artifact; the switches that let any run, Colab included, keep its weights under a chosen artifact name (the first write-up's sweeps kept none by policy, not necessity). Nineteen unit tests, one of which checks that the default build's SQL is byte-identical and one that the head loss's contrastive term equals the library's to the float. Smoke runs of every new path on the GTX 1070.
- **Pilot** (2026-09-14, group `levers-pilot`, GTX 1070). Table in §4.2. The order of the runs was 0.2, 0.5, 0.05, 0.01; the last two were added after the first two showed the direction.
- **First grid session** (2026-09-14, A100). The v1 VA-only cell reproduced the first write-up's sweep cell to four decimals (0.927). Twelve cells on v3, v4, v5 then died in seconds: the loader had been written to *refuse* the eleven straddling strings of §3.2 rather than drop them, and the three draws that have any were the three that failed. Two strength-head cells died on an import of a sentence-transformers internal that Colab's version of the library does not have; the loss now uses only the public model call. Two more cells were killed with the agent. A grid sweep never re-issues a failed cell, so the lost ones run again under two make-up sweeps, and the analysis script merges the three, newest finished run per cell.

## Appendix B: Decisions

| Decision | Reason |
|---|---|
| MTHSPL, and no third source | The only large category-0 source; MMSL is category 1, GS/MMX/NDDF category 3, SNOMED CT category 9; CVX and MTHCMSFRF add 190 strings between them |
| SPL strings inherit the existing draws | The split is a function of the target's ingredients; the VA test sets stay identical and every comparison is paired |
| `val/*` and `test/*` stay VA-only | Every published panel, table, and script reads those keys; the second source gets its own |
| Strings whose targets span splits are dropped at load time | Eleven SPL names on three draws; training on one leaks a held-out ingredient, scoring one is unanswerable |
| Sequence length and normalizer left at the VA settings | Changing either for the SPL arm would confound "more data" with "a different preprocessor" |
| Epochs fixed at four; a sixteen-epoch control instead | "Same recipe, more data" is the deployment question; the control separates steps from strings |
| The head lives in the loss module, not the model | The trainer optimizes the loss's parameters; `model.save()` never writes the head; inference is unchanged |
| Rare strengths get no head loss rather than an OTHER class | A class holding a fifth of the rows would teach the head a junk cluster |
| Head weight chosen by a pilot on the published split, on validation | The weight has no precedent here and the pilot showed it decides everything; the published split is the one whose reference run exists on the same GPU |
| Seven folds, contiguous buckets | 71/14/14 like the published split, and fold 0 reproduces the published test set |
| Unmatched strings assigned to one fold by hash | They are never trained on, so any fold's model scores them out-of-fold; the hash counts each once |
| Calibration fit on the pooled out-of-fold predictions | The all-data model has no held-out strings; the fold models' held-out predictions are the nearest honest substitute |
| Only the all-data model persists its weights | 48 grid checkpoints would be about 21 GB for nothing; the fold models are summarized by their predictions |
| A separate artifact name for the all-data model | `vandf-rxnorm-biencoder:latest` must keep pointing at the published model |
| Failed grid cells re-run in make-up sweeps rather than a fresh grid | The finished cells are bit-reproducible on the same GPU type; re-running them would buy nothing |

## References

- Bodenreider O (2004). "The Unified Medical Language System (UMLS): integrating biomedical terminology". *Nucleic Acids Research* 32(Database issue):D267–D270. [doi:10.1093/nar/gkh061](https://doi.org/10.1093/nar/gkh061)
- Bouthillier X, Delaunay P, Bronzi M, Trofimov A, Nichyporuk B, Szeto J, Sepah N, Raff E, Madan K, Voleti V, Kahou SE, Michalski V, Serdyuk D, Arbel T, Pal C, Varoquaux G, Vincent P (2021). "Accounting for Variance in Machine Learning Benchmarks". *Proceedings of Machine Learning and Systems* 3. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Caruana R (1997). "Multitask Learning". *Machine Learning* 28(1):41–75. [doi:10.1023/A:1007379606734](https://doi.org/10.1023/A:1007379606734)
- Dietterich TG (1998). "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms". *Neural Computation* 10(7):1895–1923. [doi:10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197)
- FDA. "Structured Product Labeling Resources". [fda.gov/…/structured-product-labeling-resources](https://www.fda.gov/industry/fda-data-standards-advisory-board/structured-product-labeling-resources)
- Guo C, Pleiss G, Sun Y, Weinberger KQ (2017). "On Calibration of Modern Neural Networks". *Proceedings of ICML 2017*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Halevy A, Norvig P, Pereira F (2009). "The Unreasonable Effectiveness of Data". *IEEE Intelligent Systems* 24(2):8–12. [doi:10.1109/MIS.2009.36](https://doi.org/10.1109/MIS.2009.36)
- Henderson M, Al-Rfou R, Strope B, Sung Y, Lukacs L, Guo R, Kumar S, Miklos B, Kurzweil R (2017). "Efficient Natural Language Response Suggestion for Smart Reply". [arXiv:1705.00652](https://arxiv.org/abs/1705.00652)
- Kendall A, Gal Y, Cipolla R (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics". *Proceedings of CVPR 2018*. [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
- Kesselman RF (2008). "Verbal Probability Expressions in National Intelligence Estimates: A Comprehensive Analysis of Trends from the Fifties through Post 9/11". Master's thesis, Mercyhurst College. [PDF](https://www.files.ethz.ch/isn/55739/kesselman_thesis_final.pdf)
- Kohavi R (1995). "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection". *Proceedings of IJCAI 1995*:1137–1143.
- Liu F, Shareghi E, Meng Z, Basaldella M, Collier N (2021). "Self-Alignment Pretraining for Biomedical Entity Representations". *Proceedings of NAACL-HLT 2021*:4228–4238. [aclanthology.org/2021.naacl-main.334](https://aclanthology.org/2021.naacl-main.334/)
- Nadeau C, Bengio Y (2003). "Inference for the Generalization Error". *Machine Learning* 52:239–281. [doi:10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366)
- National Library of Medicine. "RxNorm Technical Documentation". [nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html](https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html)
- National Library of Medicine. "UMLS Metathesaurus License Agreement Appendix". [nlm.nih.gov/…/license_agreement_appendix.html](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement_appendix.html)
- Nelson SJ, Zeng K, Kilbourne J, Powell T, Moore R (2011). "Normalized names for clinical drugs: RxNorm at 6 years". *Journal of the American Medical Informatics Association* 18(4):441–448. [doi:10.1136/amiajnl-2011-000116](https://doi.org/10.1136/amiajnl-2011-000116)
- Platt JC (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods". In Smola AJ, Bartlett P, Schölkopf B, Schuurmans D (eds.), *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74.
- Ruder S (2017). "An Overview of Multi-Task Learning in Deep Neural Networks". [arXiv:1706.05098](https://arxiv.org/abs/1706.05098)
- Stone M (1974). "Cross-Validatory Choice and Assessment of Statistical Predictions". *Journal of the Royal Statistical Society, Series B* 36(2):111–147. [doi:10.1111/j.2517-6161.1974.tb00994.x](https://doi.org/10.1111/j.2517-6161.1974.tb00994.x)
- Varma S, Simon R (2006). "Bias in error estimation when using cross-validation for model selection". *BMC Bioinformatics* 7:91. [doi:10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)
- Wilson EB (1927). "Probable Inference, the Law of Succession, and Statistical Inference". *Journal of the American Statistical Association* 22(158):209–212. [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
- Yu T, Kumar S, Gupta A, Levine S, Hausman K, Finn C (2020). "Gradient Surgery for Multi-Task Learning". *Advances in Neural Information Processing Systems* 33. [arXiv:2001.06782](https://arxiv.org/abs/2001.06782)

## Footnotes

[^conf]: Status and confidence tags follow the gwern.net convention, the confidence word from the [Kesselman 2008](#references) scale. No confidence is claimed until the runs are in; the pilot's dose-response in §4.2 is the one finding so far, and on one split it is "likely" at best.

[^normalizer]: Teaching the rule the `35 mg in 1 g` form is two lines of regex, and the transfer numbers of §5.3 will say whether it is worth doing. It is not done here because the SPL arm's question is "more strings", and a better preprocessor for those strings would be a second change riding on the first.

[^straddle]: An example is a label name that covers two products with different active ingredients under one string; NLM links the one SPL atom to both concepts. The VA never does this because its names are written per product. Dropping eleven of 49,036 strings changes no count in this write-up by a visible amount, and no VA row on any draw.
