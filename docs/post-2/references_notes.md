# Citation check

Each source cited in `README.md` was checked on 6 November 2026 by a Claude agent: the DOI and metadata against Crossref or the publisher, whether the source supports the sentence that cites it (with the supporting passage quoted), and whether that was judged from the full text, the abstract only, or a secondary description. The post was then corrected to match: the R5 and R10 sentences were narrowed as proposed, and Platt 1999, which could only be checked through secondary descriptions, was removed (Guo et al 2017, read in full, covers the same method). Sentences quoted under "Post" are from the draft before correction.

## Citation check results: vandf-rxnorm-interventions (R1 to R12)

Post: /home/kevin/repos/rxnorm/docs/post-2/README.md (not edited)
Checked: 2026-11-06. Downloaded copies and extracted text are in
/tmp/claude-1000/-home-kevin-repos-politics/eb377b40-d93b-4fb4-a383-7ee52ab629c0/scratchpad/refcheck/vandf-rxnorm-interventions/dl/

## Summary

| Item | Metadata OK | Support | Evidence level | Action needed |
|---|---|---|---|---|
| R1 liu2021 | Yes | Yes | Full text (ACL Anthology PDF) | None |
| R2 bouthillier2021 | Yes (see note on author list) | Yes | Full text (arXiv and MLSys PDFs), Figure 1 inspected | None. Optional: say "five" instead of "a range of" |
| R3 nelson2011 | Yes | Yes (first half; second half is the post's own framing) | Full text (PMC) | None |
| R4 caruana1997 | Yes | Yes, for the definition of multi-task learning | Abstract only (Springer blocks automated access) | None. The "discard it afterwards" design was not checked against the paper |
| R5 nlm2026b | Yes | **No** | Full page (appendix), plus the licence agreement and RxNorm technical documentation | **Reword**: the page does not say that only category 0 may be redistributed, or anything about datasets or models |
| R6 fda2026 | Yes | Yes | Full page | None |
| R7 nlm2026a | Yes | Yes | Full page | None. This page also lists every source's restriction level and could support R5 |
| R8 guo2017 | Yes (PMLR 70:1321–1330 could be added) | Yes | Full text (arXiv PDF) | None |
| R9 platt1999 | **Partly**: editors and pages correct; the publisher dates the book 2000 and titles the chapter "Probabilities for SV Machines" | Yes, from secondary descriptions | Secondary only (Guo et al 2017; Lin, Lin & Weng 2007) | **Fix metadata** (year, and either the chapter title or a note that the cited title is the preprint's) |
| R10 varma2006 | Yes | **Partly** | Full text (Europe PMC, open access) | **Narrow the sentence** (see below) |
| R11 kohavi1995 | Yes (IJCAI page says pp. 1137–1144; the printed pages are 1137–1143) | Yes | Full text (ijcai.org PDF) | None |
| R12 nadeau2003 | Yes | Yes | Abstract of the 2003 journal paper, plus full text of the 1999 NIPS conference version by the same authors (journal full text is paywalled) | None |

## Proposed rewordings

**R5.** Current sentence:

> The UMLS licence gives each RxNorm source a restriction category, and only category-0 content may be included in a redistributed dataset or model ([NLM 2026b](#references)).

Proposed:

> The UMLS licence gives each RxNorm source a restriction category. Category 0 adds no restrictions to the general licence terms, categories 1 to 4 add restrictions, and SNOMED CT has its own terms ([NLM 2026a](#references); [NLM 2026b](#references)). I used only category-0 content, so that the published dataset and model contain no restricted content.

(The next sentence, "That rules out the commercial drug databases and SNOMED CT, ...", still follows.)

**R10.** Current sentence:

> Choosing a setting and measuring its error on the same data gives an optimistic estimate ([Varma & Simon 2006](#references)), so only these held-out numbers count as the test.

Proposed:

> When a setting is chosen because it scores well on some data, the error measured on that same data tends to be lower than the true error ([Varma & Simon 2006](#references)), so only these held-out numbers count as the test.

If Kevin wants the narrowest wording, which is exactly what the paper's experiments show:

> Varma & Simon (2006) showed that when a setting is tuned by cross-validation, the error that the same cross-validation reports for it is too low ([Varma & Simon 2006](#references)), so only these held-out numbers count as the test.

**R9 (References entry, not a sentence).** Current entry:

> Platt JC (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods". In Smola AJ, Bartlett P, Schölkopf B, Schuurmans D (eds.), *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74.

Proposed (as the publisher records it):

> Platt JC (2000). "Probabilities for SV Machines". In Smola AJ, Bartlett P, Schölkopf B, Schuurmans D (eds.), *Advances in Large-Margin Classifiers*, MIT Press, pp. 61–74. [doi:10.7551/mitpress/1113.003.0008](https://doi.org/10.7551/mitpress/1113.003.0008)

If the year changes, the in-text citation "[Platt 1999](#references)" must change to "[Platt 2000](#references)" and the bib key and year must change with it, because md2qmd.py matches by surname and year. A lighter fix is to keep the familiar title and change only the year to 2000. Many papers cite it that way, for example Lin, Lin & Weng 2007.

---

## R1 `liu2021`: Liu et al 2021, SapBERT

**Metadata.** ACL Anthology page and PDF header: "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4228–4238, June 6–11, 2021". Authors in order: Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, Nigel Collier. DOI 10.18653/v1/2021.naacl-main.334. All match.

**Support: yes.** The sentence is "Part I's final recipe fine-tuned SapBERT ([Liu et al 2021](#references)), a biomedical language model, on pairs of VA names and RxNorm products."

- Abstract: "To address this challenge, we propose SAPBERT, a pretraining scheme that self-aligns the representation space of biomedical entities."
- Introduction: "To address the aforementioned issue, we propose to pretrain a Transformer-based language model on the biomedical knowledge graph of UMLS".
- Figure 1 caption: "PUBMEDBERT+SAPBERT (PUBMEDBERT further pretrained on UMLS synonyms)".

Strictly speaking, SapBERT names the pretraining scheme, and the released model is PubMedBERT further pretrained with it. "A biomedical language model" is a fair short description, and the paper uses "language model" for it.

**Evidence level.** Full text (ACL Anthology PDF).

## R2 `bouthillier2021`: Bouthillier et al 2021

**Metadata.** The MLSys proceedings page lists it in "Proceedings of Machine Learning and Systems 3 (MLSys 2021)". arXiv:2103.03098 resolves to the same paper. The post's 17 authors in its order match the author line of both the arXiv PDF and the MLSys PDF. Note: the MLSys HTML abstract page lists 16 authors: it omits Dmitriy Serdyuk and gives "Nazanin Mohammadi Sepahvand" for "Naz Sepah". The published PDF itself lists 17, as the post does, so no change is needed.

**Support: yes.** The sentence is "Across a range of benchmarks, [Bouthillier et al 2021](#references) found that the choice of data split adds more variance than the choice of training seed, and the same held in Part I."

- Section 2.2: "Figure 1 presents the individual variances due to sources from within the learning algorithms. Bootstrapping data stands out as the most important source of variance. In contrast, model initialization generally is less than 50% of the variance of bootstrap, on par with the visit order of stochastic gradient descent."
- Conclusion: "In deep learning, most evaluations focus on the effect of random weight initialization, which actually contribute a small part of the variance, on par with residual fluctuations of hyperparameter choices after their optimization but much smaller than the variance due to perturbing the split of the data in train and test sets."
- I rendered Figure 1 and inspected it. In all five case studies (Glue-RTE BERT, Glue-SST2 BERT, MHC MLP, PascalVOC ResNet, CIFAR10 VGG11), the "Data (bootstrap)" bar is the longest. "Weights init", "Data order" and "Dropout" are each shorter. Weights init comes closest in Glue-RTE.

Caveats, which do not require a change: (a) "a range of benchmarks" means five tasks. (b) The paper varies the data by bootstrap resampling of train and test, not by re-drawing a fixed partition. (c) Each seed-controlled source (initialisation, data order, dropout) is varied on its own, and the paper notes "these different contributions to the variance are not independent, the total variance cannot be obtained by simply adding them up". So "training seed" as a single combined source was not measured directly. Each of its parts is smaller than the data variance in every case study. Optional tightening: "Across five deep-learning benchmarks, ...".

**Evidence level.** Full text (arXiv PDF and MLSys PDF); Figure 1 inspected as an image.

## R3 `nelson2011`: Nelson et al 2011

**Metadata.** Crossref for 10.1136/amiajnl-2011-000116: "Normalized names for clinical drugs: RxNorm at 6 years", Stuart J Nelson, Kelly Zeng, John Kilbourne, Tammy Powell, Robin Moore; JAMIA 18(4):441–448, 2011. All match.

**Support: yes** for the part the source is meant to support. The sentence is "RxNorm links names from many sources to its own products, and each such link is a labelled training pair curated by the National Library of Medicine (NLM) ([Nelson et al 2011](#references))."

- "it provides normalized names for these clinical drugs, and related drug names, and links its names to other commonly used drug vocabularies."
- "Unique identifiers and normalized names are the primary mechanism to group together semantically equivalent terms and codes from various source vocabularies. Different names and codes can be mapped to each other if they share the same RxNorm Concept Unique Identifier (RxCUI)."
- "Second, converted source vocabularies are inserted into RxNorm using various matching algorithms. Third, human editors review all the content that was inserted, creating normalized names where needed."
- "RxNorm provides a set of names and relationships based on 11 different external source vocabularies."
- Table 1 shows MTHSPL strings grouped under an RxCUI ("309304 MTHSPL Ciprofloxacin 400 mg in 200 ml intravenous injection").

"Curated" is fair: links are proposed by matching algorithms and reviewed by NLM editors. "Labelled training pair" is the post's own use of the links, as the claims file notes.

**Evidence level.** Full text (PMC3128404).

## R4 `caruana1997`: Caruana 1997

**Metadata.** Crossref for 10.1023/A:1007379606734: "Multitask Learning", Rich Caruana, *Machine Learning* 28(1):41–75, 1997. All match.

**Support: yes**, for what it is cited for. The sentence is "Training on a related side task, called multi-task learning ([Caruana 1997](#references)), suggests a remedy: attach a small classifier that must predict the correct product's strength from the model's reading of the VA name, train it with the main task, and discard it afterwards."

- Abstract (via mlanthology.org): "Multitask Learning is an approach to inductive transfer that improves generalization by using the domain information contained in the training signals of related tasks as an inductive bias. It does this by learning tasks in parallel while using a shared representation; what is learned for each task can help other tasks be learned better."

The citation names the technique, and the abstract supports that: tasks trained together with a shared representation, where a related task helps another task. I could not check whether the paper discusses extra tasks whose outputs are thrown away after training. The sentence does not attribute that design to Caruana, so no change is needed.

**Evidence level.** Abstract only. Springer returned a JavaScript challenge and a login redirect to automated access. A third-party course-hosted copy exists, but I did not use it.

## R5 `nlm2026b`: UMLS Metathesaurus License Agreement Appendix

**Metadata.** The URL resolves to "UMLS Metathesaurus License Agreement Appendix", "APPENDIX 1 ... UMLS METATHESAURUS ® SOURCE VOCABULARIES - 2026AB Edition". It is an NLM page with no date in the citation, consulted in 2026. OK.

**Support: no.** The sentence is "The UMLS licence gives each RxNorm source a restriction category, and only category-0 content may be included in a redistributed dataset or model ([NLM 2026b](#references))."

What the page says:

- "The Categories associated with each source are defined in the general license as follows: Category 0 general terms of the License apply with no additional restrictions Category 1 in addition to the general License terms, restrictions in Section 12.1 apply Category 2 ... Section 12.2 ... Category 3 ... Section 12.3 ... Category 4 ... Section 12.4 apply"
- "Appendix 2 defines the terms of use of SNOMED CT. These terms take precedence over the general Metathesaurus license terms."
- Per-source entries: MMSL "CATEGORY 1 RESTRICTIONS APPLY"; NDDF, GS and MMX "CATEGORY 3 RESTRICTIONS APPLY"; SNOMEDCT_US "TERMS IN APPENDIX 2 OF UMLS LICENSE APPLY"; MTHSPL has no restriction line; VANDF "NOTE: Now a CATEGORY 0."
- The page never mentions redistribution, datasets or models, and has no "category 9". Category 9 is the SRL code that RXNSAB and the RxNorm technical documentation use for SNOMED CT.

The restrictions themselves are in the licence agreement (https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement.html), section 12:

- "12. 1. Category 1: LICENSEE is prohibited from translating the vocabulary source into another language or from producing other derivative works based on this single vocabulary source."
- Category 3 excludes "incorporation of material from these copyrighted sources in any publicly accessible computer-based information system ... publishing or translating or creating derivative works from material from these copyrighted sources".

So the post's rule (use only category 0 in a published dataset or model) is a reasonable cautious reading. But neither page states it, and whether a trained model counts as a "derivative work" is an interpretation. The RxNorm technical documentation (NLM 2026a, already cited) states the categories plainly: "Sources with SRL=0 follow the general terms of the license. Sources with SRL=1,2,3,4 have restrictions outlined in section 12 of the license, in addition to the general terms of the license. Appendix 2 of the license defines the terms of use for SNOMEDCT_US (SRL=9)."

Proposed sentence (each part supported by NLM 2026a and NLM 2026b):

> The UMLS licence gives each RxNorm source a restriction category. Category 0 adds no restrictions to the general licence terms, categories 1 to 4 add restrictions, and SNOMED CT has its own terms ([NLM 2026a](#references); [NLM 2026b](#references)). I used only category-0 content, so that the published dataset and model contain no restricted content.

Note for Kevin, not a wording change: even category-0 content is under the general licence. Its general terms say "LICENSEE is prohibited from distributing the UMLS Metathesaurus or subsets of it, including individual vocabulary sources within the Metathesaurus, except (a) as an integral part of computer applications developed by LICENSEE for a purpose other than redistribution of vocabulary sources ... and (b) if permitted by paragraph 12". The RxNorm technical documentation says "The majority of the content of RxNorm is available for use under the basic and open terms described in sections 1-11 and 13-16 of the Metathesaurus license." So "category 0" means "no additional restrictions", not "no restrictions". The proposed wording avoids claiming that category 0 is free to redistribute.

**Evidence level.** Full page read (appendix), plus the full licence agreement page and the RxNorm technical documentation.

## R6 `fda2026`: FDA, Structured Product Labeling Resources

**Metadata.** The URL resolves to "Structured Product Labeling Resources | FDA". OK.

**Support: yes.** The sentence is "... MTHSPL, the drug-product names from the FDA's Structured Product Labeling ([FDA 2026](#references)), which NLM brings into RxNorm ([NLM 2026a](#references))."

- "The Structured Product Labeling (SPL) is a document markup standard approved by Health Level Seven (HL7) and adopted by FDA as a mechanism for exchanging product and facility information."
- The page also lists guidance such as "SPL Standard for Content of Labeling Technical Questions and Answers" and "Indexing Structured Product Labeling".

As expected, the page does not mention MTHSPL.

**Evidence level.** Full page.

## R7 `nlm2026a`: RxNorm Technical Documentation

**Metadata.** The URL resolves to "RxNorm Technical Documentation". OK.

**Support: yes.** Same sentence as R6.

- Section 3.1 Source Vocabularies: "The following source vocabularies are included in RxNorm: SAB Source Name Source Restriction Level (SRL) ... MTHSPL FDA Structured Product Labels 0 ..."
- Section 8.0: "These UNII codes are pulled directly from the ingredient information included in the structured product labels (SPLs) on DailyMed."

Optional checks against Details A, which is not cited to this page:

- The SRL table gives GS 3, MMSL 1, MMX 3, NDDF 3, SNOMEDCT_US 9, MTHSPL 0, VANDF 0. This matches the categories the claims file gives for Details A.
- The page describes NLM's generated duplicates: "The duplicate atom's term type will be created by prepending 'MTH_RXN_' to the Base atom's term type". This is consistent with `MTH_RXN_DP` and `MTH_RXN_CD` being generated duplicates.
- The `SRL` column is documented for RXNSAB.RRF: "SRL Source Restriction Level 0,1,2,3,4 - explained in the License Agreement."
- I did not find a definition of the `DP` term type on this page.

**Evidence level.** Full page.

## R8 `guo2017`: Guo et al 2017

**Metadata.** arXiv:1706.04599 and PMLR page: "On Calibration of Modern Neural Networks", Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. PDF footer: "Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, PMLR 70, 2017". PMLR gives pages 1321–1330. OK; the pages could be added.

**Support: yes.** The sentence is "In Part I, I turned the raw score into a probability with a temperature and a Platt layer ([Guo et al 2017](#references); [Platt 1999](#references)) fit on one validation set, and chose the cut-offs there."

- "Each method requires a hold-out validation set, which in practice can be the same set used for hyperparameter tuning."
- "Temperature scaling, the simplest extension of Platt scaling, uses a single scalar parameter T > 0 for all classes. ... T is optimized with respect to NLL on the validation set."
- On Platt scaling: "Platt scaling learns scalar parameters a, b ∈ R and outputs q̂i = σ(azi + b) as the calibrated probability. Parameters a and b can be optimized using the NLL loss over the validation set."

**Evidence level.** Full text (arXiv PDF).

## R9 `platt1999`: Platt 1999

**Metadata: partly correct.**

- Crossref, for the book (10.7551/mitpress/1113.001.0001): "Advances in Large-Margin Classifiers", edited-book, editors "Alexander J. Smola, Peter Bartlett, Bernhard Schölkopf, Dale Schuurmans", issued 2000-09-29. The editors match.
- Crossref, for the chapter (10.7551/mitpress/1113.003.0008): title "Probabilities for SV Machines", author John C. Platt, pages 61–74, 2000. The pages match. The publisher's chapter title and year differ from the post.
- The title the post uses is the widely cited title of Platt's 1999 preprint. Guo et al 2017 cite it as "Platt, John et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61–74, 1999." Lin, Lin & Weng 2007 cite it as "Platt, J.: 2000, 'Probabilistic outputs for support vector machines and comparison to regularized likelihood methods'. In: A. Smola, P. Bartlett, B. Schölkopf, and D. Schuurmans (eds.): Advances in Large Margin Classifiers. Cambridge, MA, MIT Press."
- So the year of the book is 2000, not 1999. See the proposed References entry in the rewordings section.

**Support: yes, from secondary sources.** Same sentence as R8. Platt is cited for the sigmoid that maps a score to a probability.

- Lin, Lin & Weng 2007 (author's copy at csie.ntu.edu.tw): "Platt (2000) proposes approximating the posterior by a sigmoid function Pr(y = 1|x) ≈ P_A,B(f) ≡ 1 / (1 + exp(Af + B))".
- Guo et al 2017: "Platt scaling (Platt et al., 1999) is a parametric approach to calibration ... The non-probabilistic predictions of a classifier are used as features for a logistic regression model, which is trained on the validation set to return probabilities."

**Evidence level.** Secondary only. The MIT Press chapter is behind a login (direct.mit.edu returned 403). The open copies I found were course-hosted copies, which I did not use. The metadata comes from Crossref.

## R10 `varma2006`: Varma & Simon 2006 (checked with particular care)

**Metadata.** Crossref for 10.1186/1471-2105-7-91: "Bias in error estimation when using cross-validation for model selection", Sudhir Varma, Richard Simon, *BMC Bioinformatics* 7, article 91, published 2006-02-23. All match.

**Support: partly.** The sentence is "Choosing a setting and measuring its error on the same data gives an optimistic estimate ([Varma & Simon 2006](#references)), so only these held-out numbers count as the test."

What the paper shows directly (its experiments):

- Abstract, Conclusion: "We show that using CV to compute an error estimate for a classifier that has itself been tuned using CV gives a significantly biased estimate of the true error."
- Results: "Even though the mean true error is 50% (i.e. equal to randomly choosing the classes), the CV error estimate on the training set averages 37.8% for the optimized Shrunken Centroid classifier and 41.7% for the optimized SVM classifier."
- The setting is two classifiers (shrunken centroids, with Δ tuned by 10-fold CV; SVM, with C and γ tuned by LOOCV), on simulated microarray-style data with 40 training samples.

What the paper argues more generally (Conclusion):

- "Will this bias, possibly with differing magnitudes, be present for other cross validation schemes? Since the bias is caused by the variability in the estimates of prediction error for different values of the tuning parameters it is a general phenomenon."
- "If we assume that the resampling method is median unbiased, it is easy to see that the probability that the minimum of the errors is lesser than E [is] 1 − (1/2)^K ... This implies that for large K, there is a high probability that choosing the minimum resampled error will give a biased estimate of the true error."
- "The parameter selection bias, on the other hand, is always negative."
- But also: "The parameter selection bias adds to the inherent bias to form the total bias of the resampling procedure. It must be noted that inherent bias can be either negative or positive ... If the inherent bias is positive, the parameter selection bias will subtract from it and possibly bring the error estimate closer to the true error."
- Background, on the extreme case: "Testing the classifier on the same samples that were used to train it gives the re-substitution estimate of the true error, which is known to give falsely low (usually zero) error estimates for small n."
- The remedy they recommend matches what the post does: "This can be accomplished by a large validation set that is independent of data used for parameter tuning ... This is effectively two nested CV loops; the outer loop estimates the generalization error while the inner CV is used for tuning the parameters."

Assessment:

- The paper supports the mechanism behind the post's sentence, and argues that it is general. When a setting is chosen because its estimated error is lowest, the selection pulls that estimate downward. Varma & Simon call this bias "always negative", meaning optimistic.
- The general wording goes beyond the paper in two ways. First, the paper's evidence and its formal argument cover choosing a setting that minimises an estimated error. "Choosing a setting" in general is wider. The post's case is choosing a cut-off that meets a precision target on pooled out-of-fold predictions. That uses the same mechanism (picking the threshold where the noisy precision estimate looks good enough), but the paper did not study it. Second, the paper says the selection adds an optimistic bias, not that the final estimate is always optimistic, because another bias can offset it.
- So "gives an optimistic estimate" as a flat general rule is somewhat stronger than the paper. "Tends to be lower than the true error" is what the paper supports.

Proposed (general, but within what the paper argues):

> When a setting is chosen because it scores well on some data, the error measured on that same data tends to be lower than the true error ([Varma & Simon 2006](#references)), so only these held-out numbers count as the test.

Narrowest alternative (exactly what the experiments show):

> Varma & Simon (2006) showed that when a setting is tuned by cross-validation, the error that the same cross-validation reports for it is too low ([Varma & Simon 2006](#references)), so only these held-out numbers count as the test.

**Evidence level.** Full text (Europe PMC full-text XML of PMC1397873, CC BY). The BMC site returned a JavaScript challenge, which I did not try to get around.

## R11 `kohavi1995`: Kohavi 1995

**Metadata.** ijcai.org/proceedings/1995/147: "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection", Ron Kohavi, "Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence. Learning. Pages 1137-1144." The printed page numbers in the official PDF run from 1137 to 1143 (seven pages), so the post's 1137–1143 matches the paper. No DOI exists. A URL could be added: https://www.ijcai.org/proceedings/1995/147. OK.

**Support: yes.** The sentence is "Cross-validation estimates tend to be pessimistic for a model trained on more data ([Kohavi 1995](#references))."

- "Assuming that the inducer's accuracy increases as more instances are seen, the holdout method is a pessimistic estimator because only a portion of the data is given to the inducer for training."
- "The diagrams clearly show that k-fold cross-validation is pessimistically biased, especially for two and five folds. For the learning curves that have a large derivative at the measurement point, the pessimism in k-fold cross-validation for small k's is apparent. Most of the estimates are reasonably good at 10 folds and at 20 folds they are almost unbiased."
- "k-fold cross-validation with moderate k values (10-20) reduces the variance while increasing the bias."

Caveat: the evidence comes from C4.5 and Naive Bayes on small UCI datasets. The pessimism is largest for 2 to 5 folds and nearly gone at 20. The post uses 7 folds, so "tend to be" is the right strength.

**Evidence level.** Full text (ijcai.org PDF).

## R12 `nadeau2003`: Nadeau & Bengio 2003

**Metadata.** Crossref for 10.1023/A:1024068626366: "Inference for the Generalization Error", Claude Nadeau, Yoshua Bengio, *Machine Learning* 52(3):239–281, 2003. All match; the issue number (3) could be added.

**Support: yes.** The sentence is "Units that share training data are correlated, so such intervals are optimistic ([Nadeau & Bengio 2003](#references)), which is why the sign count is reported with them."

From the 2003 journal abstract (via mlanthology.org):

- "Such tests should take into account the variability due to the choice of training set and not only that due to the test examples, as is often the case. This could lead to gross underestimation of the variance of the cross-validation estimator, and to the wrong conclusion that the new algorithm is significantly better when it is not."
- "Our analysis shows that all the variance estimators that are based only on the results of the cross-validation experiment must be biased."

From the 1999 NIPS conference version by the same authors (official open PDF):

- Lemma 1: for variables with common variance δ and covariance γ, "E[S²] = δ − γ". So when the per-split estimates are positively correlated, the sample variance underestimates the variance.
- "the correlation between the μ̂j's becomes n2/n", where the correlation arises because the training sets overlap.
- "techniques currently in use have the opposite defect, that is they tend to be liberal (tests with actual size exceeding the nominal size)"
- "tests based on the t-test or resampled t-test are liberal, they reject the null hypothesis with probability greater than the prescribed α = 0.1, when the null hypothesis is true."

"Optimistic" (intervals too narrow) matches "liberal" and "underestimation of the variance".

**Evidence level.** Abstract of the 2003 journal paper, plus full text of the 1999 NIPS version (papers.nips.cc). Springer redirected to a login for the journal full text, and I did not try to get around it. The CIRANO working paper 99s-25 (the authors' 1999 technical report) was downloaded, but its text extraction was garbled, so I did not quote it.
