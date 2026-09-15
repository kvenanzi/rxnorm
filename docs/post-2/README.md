# Three Interventions on a VA-to-RxNorm Drug-Name Mapper: A Second Source Vocabulary, an Auxiliary Strength Head, and Cross-Validation by Ingredient

| created | modified | status | confidence | importance |
|---|---|---|---|---|
| 2026-09-14 | 2026-09-14 | in progress | –[^conf] | 3 |

> **Abstract.** The [first report](../post/README.md) (hereafter Part I) fine-tuned a small bi-encoder to map VA National Drug File (VANDF) names to RxNorm clinical drugs and, by re-drawing the ingredient-level split five additional times, estimated its accuracy on unseen ingredients at 0.907 ± 0.019, against 0.931 on the published draw. Its concluding section listed interventions expected to change that figure. This report evaluates three of them under the same design: every arm is trained on each of the same six ingredient draws and two training seeds, and an effect is credited only if it exceeds the split-to-split spread of about two points rather than the value on a single favourable draw. The interventions are (1) a second source vocabulary from the same RxNorm release, the FDA Structured Product Label names (MTHSPL), which is the only large source at UMLS restriction level 0 other than the two already used, and which contributes 48,985 labelled pairs whose surface form differs substantially from the VA's; (2) an auxiliary classification head, trained jointly and discarded at inference, that predicts the target's strength from the query embedding ([Caruana 1997](#references)), directed at the error category shared by every earlier model; and (3) seven-fold cross-validation by ingredient ([Stone 1974](#references)), which yields a single out-of-fold estimate over every held-out string, a final model trained on every ingredient family, and a test of whether abstention thresholds chosen on one set of folds transfer to another. A pilot on the published split found that the head's loss is approximately fifty times the contrastive loss at initialization and that every weight above 0.01 lowers validation accuracy monotonically; the factorial grid therefore runs the head at 0.01. *Results and verdicts will be added as the sweeps complete; the sections below record what is already established.*

## 1. Question and design

Part I established the reference measurement. Its final recipe, SapBERT ([Liu et al 2021](#references)) fine-tuned with the in-batch softmax loss of [Henderson et al 2017](#references), ingredient-matched hard negatives, and a deterministic rule that appends RxNorm-style concentrations to the VA string, reached test acc@1 0.931 on the published ingredient split and 0.907 ± 0.019 across six draws of that split, with training-seed variation of 0.001. The ratio between those two sources of variation determines the design of the present report. [Bouthillier et al 2021](#references) found, across a range of benchmarks, that the choice of data split contributes more variance than the choice of training seed, often by a wide margin; the same held on this task, by a factor of about twenty. An intervention that changes acc@1 by one point on one split has therefore demonstrated nothing. The one factor of the earlier sweep whose effect fell within the split spread, hard negatives, was re-run on all six draws and held on each; that re-run's design is applied here from the outset. Every arm is trained on every draw, the analysis is paired within draw, and each effect is reported with its sign count over the units and a 95% interval, following [Dietterich 1998](#references) on comparing learners on common folds.

The three interventions correspond to items 4, 3, and 1 of Part I §7. Two are training-time interventions and are run as a single 2 × 2 factorial grid; the third changes how the final model is trained and evaluated, and is run afterwards using whichever arm of the grid performs best.

**A second source vocabulary.** The construction that produced the dataset applies to any RxNorm source: an atom that shares an RXCUI with a normalized clinical-drug name is a labelled pair, curated by NLM ([Nelson et al 2011](#references)). The September 2026 release carries thirteen sources, and the UMLS license assigns each a restriction category ([Bodenreider 2004](#references); [NLM license appendix](#references)). Only category-0 content may be included in a redistributed dataset or model, which excludes the commercial drug databases and SNOMED CT and leaves one large candidate, described in §3. The hypothesis, in the sense of [Halevy et al 2009](#references), is that a fourfold increase in training strings drawn from a different naming convention improves the encoder's representation of product names more than any modification to the loss.

**An auxiliary strength head.** Strength was the largest error category of every model in Part I: three-quarters of TF-IDF's errors, six in ten of MiniLM's, and the largest single share of the final model's 128, a portion of which are underdetermined by the input string. The bi-encoder receives no explicit representation of strength; it learns only that a query containing `250MG` tends to match a target containing `250 MG`. Multi-task learning ([Caruana 1997](#references); [Ruder 2017](#references)) suggests a remedy: attach a small classifier to the query embedding that must predict the target's strength string, train it jointly with the retrieval objective, and discard it at inference. If the shared encoder is thereby required to represent strength explicitly, retrieval should benefit. This was deferred in Part I because the strength, ingredient, and dose form are already available from the retrieved candidate's labels; the question here is whether they are also useful as a training signal.

**Cross-validation by ingredient, and a model trained on every ingredient family.** The published model was trained on 70% of ingredient families: 15% were held out for validation and 15% for test. A deployed mapper has no reason to exclude them. Cross-validation ([Stone 1974](#references); [Kohavi 1995](#references)) is the standard procedure: partition the ingredients into K folds, train K models each with one fold held out, pool the K held-out scores into a single estimate over every string, then train a final model on all data and report the pooled estimate for it. Two further quantities follow. The pooled estimate carries a sampling interval over roughly 12,000 strings rather than a six-draw band whose own width is known only to a factor of two. And the K models' held-out predictions are exactly the material required to fit the calibration layer on strings the final model was not trained on, which also addresses item 1 of Part I §7: whether a 99%-precision threshold chosen on six folds holds on the seventh.

## 2. Hypotheses

Recorded before any run of the grid had started. The pilot of §4.2 selected one quantity, the head's loss weight, and nothing else; it was run on the published split alone and is not part of the paired analysis.

- **H1.** MTHSPL training strings raise VA test acc@1 by a small amount, about one point, within the split band but of consistent sign across the twelve (split, seed) units; and they raise accuracy on held-out MTHSPL strings by a large amount, because the VA-only model has never encountered the FDA naming convention.
- **H2.** The strength head lowers the VA strength-error rate and does not change ingredient or dose-form accuracy; its effect on acc@1 is small and may not exceed the split spread.
- **H3.** The two effects are additive: the interaction term is indistinguishable from zero.
- **H4.** The compute-matched control (VA strings only, sixteen epochs) gains less than the MTHSPL arm, and the size-matched control (MTHSPL capped at the VA row count) retains most of the MTHSPL gain. The effect is therefore attributable to the additional strings rather than to the additional optimizer steps or the additional rows.
- **H5.** The pooled out-of-fold acc@1 of the recipe falls within 0.907 ± 0.019, and a 99%-precision threshold chosen on six folds achieves at least 98% precision on the seventh, as the single-split thresholds achieved on their test set.

## 3. Data

### 3.1 The candidate sources

`RXNSAB` records each source's restriction category in its `SRL` column. Applying the license and a size requirement together leaves a short list:

| Source | Description | SRL | Active atoms linked to an SCD/SBD | Distinct strings | Targets covered |
|---|---|---|---|---|---|
| RXNORM | NLM's normalized names (the target side) | 0 | 91,762 | 78,903 | 27,287 |
| **MTHSPL** | **FDA Structured Product Label names** | **0** | **101,005** | **49,036** | **16,000** |
| NDDF | FDB MedKnowledge | 3 | 30,706 | 22,541 | 9,495 |
| GS | Gold Standard | 3 | 22,593 | 21,186 | 12,927 |
| MMSL | Multum | 1 | 22,048 | 22,048 | 15,016 |
| MMX | Micromedex | 3 | 20,645 | 20,620 | 14,734 |
| VANDF | VA National Drug File (the query side) | 0 | 17,964 | 14,931 | 8,315 |
| SNOMEDCT_US | SNOMED CT | 9 | 14,036 | 14,036 | 6,219 |
| CVX | vaccine codes | 0 | 187 | 187 | 181 |
| MTHCMSFRF | CMS formulary reference | 0 | 3 | 3 | 3 |
| ATC, DRUGBANK, USP | classifications and monographs | 0 | 0 | 0 | 0 |

MTHSPL comprises the drug-product names from FDA Structured Product Labeling ([FDA SPL](#references)), which NLM ingests into RxNorm ([NLM RxNorm technical documentation](#references)). It is the largest source in the table, it is category 0, and exactly one of its 49,036 strings equals a VA string under case-insensitive comparison. The next three sources exceed the VA file in size but are category 1 or 3, and their inclusion would invalidate the published dataset card's statement that no restricted content was used. CVX and MTHCMSFRF are category 0 and contribute 190 strings between them, too few to justify a third source. The comparison is therefore VANDF alone against VANDF with MTHSPL.

The two naming conventions differ substantially. A VA name is short and abbreviated (`PAROXETINE HCL 40MG TAB`, median 34 characters); an SPL name is long, unabbreviated, and carries the brand in brackets:

| MTHSPL string | Target |
|---|---|
| `PAROXETINE HYDROCHLORIDE 40 mg ORAL TABLET, FILM COATED [Paroxetine]` | `paroxetine hydrochloride 40 MG Oral Tablet` |
| `CAMPHOR (SYNTHETIC) 35 mg in 1 g / MENTHOL 60 mg in 1 g TOPICAL PATCH [Arcticzen]` | `camphor 0.035 MG/MG / menthol 0.06 MG/MG Medicated Patch [Arcticzen]` |

The median SPL string is 66 characters and the 99th percentile 240, against a 96-token limit chosen for the VA; the bracketed brand at the end of the longest one to two percent of strings is truncated. The limit was left unchanged so that the comparison is not confounded. The `X mg in 1 mL` notation is not parsed by the strength normalizer, so the rule has no effect on SPL strings; that too was left unchanged, for the same reason.[^normalizer]

### 3.2 Pairing and splitting

The same term-type filter applies as for the VA: `DP` (drug product) atoms only, with NLM's generated `MTH_RXN_DP` duplicates excluded as `MTH_RXN_CD` was. After deduplication on (string, target) this yields 48,985 pairs over 47,635 distinct strings and 16,000 targets, together with 21,595 active SPL strings whose concept has no SCD/SBD; the latter join the unmatched set and are divided between validation and test by the same string hash.

No new split rule is required. The published split assigns every ingredient to train, validation, or test by an MD5 hash of its RXCUI, and a *target* follows its ingredients; query strings inherit the target's assignment. MTHSPL strings therefore fall into the six existing draws without further specification, each VA test set is byte-identical with or without the SPL strings present, and only train-split SPL strings are ever trained on. On the published draw this adds 29,505 training rows to the VA's 9,290; the other five draws add between 30,218 and 33,681.

One case required a rule. Eleven SPL strings across three draws (six in v3, four in v4, one in v5) map to two products whose ingredients were assigned to different splits, a condition no VA string meets on any draw. Such a string can be neither trained on, since that would expose a held-out ingredient, nor scored, since one of its correct answers is in the training set. The loader drops them and reports the count.[^straddle]

### 3.3 Folds

The published salt hashes each ingredient into a bucket from 0 to 99 and assigns buckets 0–14 to test and 15–29 to validation. Seven folds of the same hash, with fold = ⌊7 × bucket / 100⌋, make fold 0 exactly buckets 0–14, which is the published test set. Fold i is tested with fold i+1 as its validation set and the remaining five as training, so each model is trained on 71% of ingredient families, as the published model was. The `mixed` rule is unchanged: a target whose ingredients span the three sets is excluded from that fold. This is what limits out-of-fold coverage, since a combination drug whose ingredients sit in different folds is mixed in two rotations and training data in the rest, and is never tested. The seven test sets contain 1,848, 1,688, 1,684, 1,890, 1,759, 1,533, and 1,828 VA strings, 12,230 of the 14,372 pairs. The final model's data folder assigns every ingredient to train, and no target is mixed.

## 4. Method

### 4.1 The grid and its controls

One W&B grid (`sweeps/levers.yaml`): the final recipe, unchanged, crossed with {VA only, VA + MTHSPL}, {no head, strength head}, the six draws, and seeds 42 and 1, for 48 runs on one Colab A100. Selection, where required, is on `val/acc@1` as before. The `val` and `test` keys retain their meaning, the VA strings of that draw, and the SPL strings of the same draw are scored alongside under separate keys, so the VA-only arm reports SPL transfer, item 4 of Part I §7, without additional runs. Epochs are fixed at four for both data arms; the SPL arm therefore takes about four times as many optimizer steps. Two controls on the best and worst of the six draws (v1 and v4) separate the confounds: VA only for sixteen epochs (the same steps, the same strings), and VA plus SPL capped at 9,300 rows (the same steps, different strings).

Analysis is paired within (split, seed), giving twelve units. Each main effect is the mean of the two simple contrasts per unit, the interaction is the double difference, and each is reported with a sign count and a t interval over the twelve. [Nadeau and Bengio 2003](#references) show that units sharing training data are correlated and that such intervals are optimistic, which is why the sign count is reported alongside them.

### 4.2 The strength head, and the pilot that set its weight

The head is one linear layer on the query embedding, 768 inputs and one output per strength class, trained by cross-entropy against the strength string of the positive (`250 MG`, `0.025 MG/ML`, or the ` / `-joined list for a combination product). The classes are the strengths with at least five training examples: 232 of them cover 79% of VA training rows. The remaining rows contribute no head loss, in preference to an OTHER class that would hold a fifth of the rows and train the head toward an uninformative cluster. The head is a parameter of the loss module rather than the model, so the trainer's optimizer updates it and `model.save()` never writes it; the saved encoder, the inference wrapper, and the code path of every published number are unchanged. Its initialization is seeded independently of the training stream, so the two arms of a given seed see identical batches.

The weight on the head loss is the one hyper-parameter with no precedent in this project, and the pilot showed it to be decisive. A 232-way cross-entropy begins near 4 nats; the contrastive loss is near 0.09 after warm-up and 0.03 at the end of training. At weight 0.2 the head is the dominant objective and the retrieval loss a regularizer, an inversion of the intended roles. The literature on weighting auxiliary losses ([Kendall et al 2018](#references)) and on conflicting task gradients ([Yu et al 2020](#references)) establishes the general point; the specific value required a pilot. Four weights were run on the published split, seed 42, VA strings only, with everything else as in the final recipe, and the split-v1 run of Part I §5.8 (`lqf7dett`, same GPU) as the reference:

| head weight | run | val acc@1 by epoch | best val acc@1 | test acc@1 | strength | ingredient | dose form |
|---|---|---|---|---|---|---|---|
| none | `lqf7dett` | – | 0.882 | 0.931 | 0.961 | 0.983 | 0.972 |
| 0.01 | `tiotc7iy` | 0.877 → 0.876 → **0.880** → 0.880 | 0.880 | 0.928 | 0.963 | 0.985 | 0.970 |
| 0.05 | `9zw6jvov` | **0.874** → 0.870 → 0.872 → 0.872 | 0.874 | 0.923 | 0.962 | 0.983 | 0.969 |
| 0.2 | `0usjnp44` | 0.859 → **0.864** → 0.850 → 0.848 | 0.864 | 0.904 | 0.963 | 0.975 | 0.964 |
| 0.5 | `spja8f8i` | **0.849** → 0.842 → 0.821 → 0.811 | 0.850 | 0.890 | 0.957 | 0.973 | 0.956 |

The response is monotone. As the weight increases, validation accuracy falls, the best epoch arrives earlier, and ingredient and dose-form accuracy decline, while strength accuracy, the quantity the head targets, does not change at any weight. At 0.01 the two losses are of comparable magnitude and the run is indistinguishable from the reference. The grid tests H2 at 0.01, the largest weight that is not already inferior to no head on one split. A null result at that weight answers H2 as stated, and the pilot already constrains how a positive result would have to be interpreted.

### 4.3 Folds and the final model

Each fold trains the best-performing arm of the grid, selects its epoch on its validation fold, scores its test fold once, and logs every string's top-20 candidates and cosine similarities, matched and unmatched, as an artifact. The out-of-fold table pools the seven test folds: one acc@1 with a Wilson interval ([Wilson 1927](#references)) over 12,230 strings, alongside the per-fold values and their dispersion, which is the 0.907 ± 0.019 band measured by a different procedure. The final model trains on the all-ingredient folder for the median best epoch of the seven, since it has no validation set on which to stop, and its weights are logged from Colab under a new artifact name so that the published model's `latest` pointer is unchanged.

### 4.4 Calibration on out-of-fold predictions

Part I fit a temperature and a Platt layer ([Guo et al 2017](#references); [Platt 1999](#references)) on one validation split and chose thresholds there; the 99% target then achieved 98.2–98.8% on test, and §7 asked whether that shortfall is stable. The pooled out-of-fold predictions are the appropriate material: every matched string was scored by a model not trained on its ingredients, and every hard unmatched string (a real drug with no clinical-drug concept) is assigned to one fold by a hash of its text so that it is counted once. Temperature and Platt parameters are fit on that pool and shipped with the final model as its `calibration.json`, the same four-number layer the published model carries. The check of interest follows [Varma and Simon 2006](#references): for each fold, the 95% and 99% thresholds are chosen on the other six and applied to it, and the achieved precision and coverage are tabulated, together with the full seven-by-seven matrix of thresholds chosen on fold i and applied to fold j. If the leave-one-fold-out precision matches the single-split test result, the shortfall is a property of the task and can be budgeted for; if it varies across folds, the thresholds are draw-specific and a deployment should re-select them on its own data.

## 5. Results

*Pending the sweeps.* 5.1 The grid (`outputs/levers/levers.md`, Figure 1). 5.2 The controls. 5.3 SPL transfer. 5.4 Folds and the final model (`outputs/kfold/oof.md`, Figure 2). 5.5 Calibration transfer (`outputs/kfold/calibration.md`).

## 6. Discussion

*Pending.*

## 7. Reproduction

- **Code:** the same repository as Part I, branch `levers` until merged; the commands are listed in its README under "The follow-up". Nothing Part I depends on has changed: its data folder, scripts, figures, and numbers are reproduced by the same commands, and the default dataset build emits byte-identical SQL. Every new behaviour is controlled by a new flag or a new configuration field with a default.
- **Runs:** sweeps `nypttmi8` (the grid), `ad9nakej` and `2v3am5xr` (cells re-run after the two failures recorded in Appendix A), `2exoct4h` and `ocmpf5na` (the controls); W&B group `kfold-k7` for the folds and the final model; group `levers-pilot` for §4.2.
- **Artifacts:** `vandf-rxnorm-multi:v0` (the six draws with both sources) and `vandf-rxnorm-kfold:v0` (the folds) for the data; `vandf-rxnorm-predictions` per fold; `vandf-rxnorm-biencoder-final-all` for the all-data model. The published artifacts `vandf-rxnorm-pairs`, `vandf-rxnorm-splits`, and `vandf-rxnorm-biencoder` are unchanged.

## Appendix A: Experiment log

- **Tooling** (2026-09-14, branch `levers`). A second query source in the dataset builder, with the split inherited from the target; the strength-head loss; fold and all-train folders; a predictions artifact; and the switches that allow any run, Colab included, to retain its weights under a chosen artifact name (the earlier sweeps retained none, by policy rather than necessity). Nineteen unit tests, one of which verifies that the default build's SQL is byte-identical and one that the head loss's contrastive term equals the library's to floating-point precision. Smoke runs of every new code path on the GTX 1070.
- **Pilot** (2026-09-14, group `levers-pilot`, GTX 1070). Table in §4.2. The runs were executed in the order 0.2, 0.5, 0.05, 0.01; the last two were added after the first two indicated the direction of the effect.
- **First grid session** (2026-09-14, A100). The v1 VA-only cell reproduced the corresponding Part I sweep cell to four decimals (0.927). Twelve cells on v3, v4, and v5 then failed at load time: the loader had been written to *reject* the eleven spanning strings of §3.2 rather than drop them, and the three draws containing any were the three that failed. Two strength-head cells failed on an import of a sentence-transformers internal that Colab's version of the library lacks; the loss now uses only the public model call. Two further cells were terminated when the sweep agent was stopped. A grid sweep does not re-issue a failed cell, so the failed cells were re-issued under two make-up sweeps, and the analysis script merges the three sweeps, taking the most recently finished run per cell.

## Appendix B: Decisions

| Decision | Reason |
|---|---|
| MTHSPL, and no third source | The only large category-0 source; MMSL is category 1, GS/MMX/NDDF category 3, SNOMED CT category 9; CVX and MTHCMSFRF contribute 190 strings between them |
| SPL strings inherit the existing draws | The split is a function of the target's ingredients; the VA test sets remain identical and every comparison is paired |
| `val/*` and `test/*` remain VA-only | Every published panel, table, and script reads those keys; the second source receives its own |
| Strings whose targets span splits are dropped at load time | Eleven SPL names on three draws; training on one exposes a held-out ingredient, and scoring one has no single correct answer |
| Sequence length and normalizer left at the VA settings | Changing either for the SPL arm would confound "more data" with "a different preprocessor" |
| Epochs fixed at four; a sixteen-epoch control instead | "Same recipe, more data" is the deployment question; the control separates steps from strings |
| The head is a parameter of the loss module, not the model | The trainer optimizes the loss's parameters; `model.save()` never writes the head; inference is unchanged |
| Rare strengths receive no head loss rather than an OTHER class | A class holding a fifth of the rows would train the head toward an uninformative cluster |
| Head weight chosen by a pilot on the published split, on validation | The weight has no precedent here and the pilot showed it to be decisive; the published split is the one for which a reference run exists on the same GPU |
| Seven folds, contiguous buckets | 71/14/14 as in the published split, and fold 0 reproduces the published test set |
| Unmatched strings assigned to one fold by hash | They are never trained on, so any fold's model scores them out-of-fold; the hash counts each once |
| Calibration fit on the pooled out-of-fold predictions | The all-data model has no held-out strings; the fold models' held-out predictions are the closest unbiased substitute |
| Only the all-data model retains its weights | 48 grid checkpoints would occupy about 21 GB with no analytical use; the fold models are summarized by their predictions |
| A separate artifact name for the all-data model | `vandf-rxnorm-biencoder:latest` must continue to point at the published model |
| Failed grid cells re-run in make-up sweeps rather than a fresh grid | The finished cells are bit-reproducible on the same GPU type; re-running them would add nothing |

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

[^conf]: Status and confidence tags follow the gwern.net convention, with the confidence word taken from the [Kesselman 2008](#references) scale. No confidence is claimed until the runs are complete; the pilot's dose-response in §4.2 is the one finding so far, and on one split it is at most "likely".

[^normalizer]: Extending the rule to the `35 mg in 1 g` form requires a two-line addition to the regular expression, and the transfer results of §5.3 will indicate whether it is warranted. It is not done here because the SPL arm's question is whether additional strings help, and an improved preprocessor for those strings would be a second change confounded with the first.

[^straddle]: An example is a label name that covers two products with different active ingredients under one string; NLM links the single SPL atom to both concepts. The VA never does this, because its names are written per product. Dropping eleven of 49,036 strings changes no count in this report by a visible amount, and affects no VA row on any draw.
