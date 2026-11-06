# Citation check

Each source cited in `README.md` was checked on 6 November 2026 by a Claude agent: the DOI and metadata against Crossref or the publisher, whether the source supports the sentence that cites it (with the supporting passage quoted), and whether that was judged from the full text, the abstract only, or a secondary description. The post was then corrected to match. Platt 1999 and Wilson 1927, which could only be checked through secondary descriptions, were removed; Guo et al 2017, read in full, describes Platt scaling, and the Wilson interval is still named. The bge-small comparison was replaced by a sentence that needs no source. Sentences quoted under "Post" are from the draft before correction.

## Citation check results: vandf-rxnorm-biencoder (Part I)

Post: /home/kevin/repos/rxnorm/docs/post/README.md (not edited)
Checked: 2026-11-06. Items R1 to R14 and W1 to W4, plus the uncited bge-small sentence (R7 item 4).

How I quote: passages are copied from the source text I downloaded. In PDF extracts I joined words split by line-end hyphens and wrote ligatures ("ﬁ") as plain letters; nothing else was changed. Downloaded copies are in `/tmp/claude-1000/-home-kevin-repos-politics/eb377b40-d93b-4fb4-a383-7ee52ab629c0/scratchpad/refcheck/vandf-rxnorm-biencoder/dl/`.

## Summary

| Item | Metadata OK | Support | Evidence level | Action needed |
|---|---|---|---|---|
| R1 korpela2026 | Yes | Yes (all six numbers, 22,624, quote, IN/MIN targets, LLM). The baseline is not named in the post | Full text (Europe PMC JATS XML) | Recommended: name the baseline and the dataset as the paper does |
| R2 nelson2011 | Yes | Yes for the cited sentence. The uncited "two decades" is not supported by this paper | Full text read through WebFetch (page excerpts only; direct download blocked by bot check) | Optional: cite NLM 2026a for the RXCUI/SCD sentences; reword "for two decades" |
| R3 nlm2026a | Yes | Yes (section 1). Partly for Details A: the 55% and the 40% do not measure the same thing, and the "because" clause is not from NLM | Web page, read in full | Reword the Details A sentence |
| R4 nlm2026b | Yes | Partly. The quote is exact and VANDF is listed as category 0. "So ... can be shared" does not follow. RXNSAB.SRL and category 9 are not on this page. The commercial sources are categories 1 and 3, not only 3 | Web page, read in full; the general licence page; RXNSAB.RRF from the local release | Reword both sentences |
| R5 henderson2017 | Yes | Yes. Section 4.4 is "Multiple Negatives" and describes the in-batch softmax loss. The hard negative is a sentence-transformers addition | Full text (arXiv PDF) | None. Optional clarification |
| R6 karpukhin2020 | Yes | Partly. The paper shows that one BM25 hard negative improves DPR. It does not call hard negatives "standard" | Full text (ACL Anthology PDF) | Reword |
| R7 liu2021 | Yes | Yes for items 1 and 3 (paper Table 7: 109.48M). No for item 2 (HCL), which the paper does not test. No for item 4 (bge), which the BGE card and paper do not support | Full text (ACL Anthology PDF) and model card | Reword items 2 and 4 |
| R8 reimers2019 | Yes (pages 3982–3992 per ACL Anthology and the PDF header; Crossref has 3980–3990) | Yes | Full text (ACL Anthology PDF) | None |
| R9 wang2020 | Yes (NeurIPS 33 proceedings page) | Partly. all-MiniLM-L6-v2 was made by keeping every second layer of a model that Wang et al distilled; the 6-layer model was not itself distilled by Wang et al. 22M matches the paper's 6-layer, 384-wide size and the HF page (22.7M) | Full text (arXiv PDF) and three model cards | Reword |
| R10 xiao2023 | Yes (arXiv; the arXiv comment says SIGIR 2024) | Yes, C-Pack is the paper the BGE card asks users to cite. The 33M is on the HF page (33.4M), not in the paper (the paper gives 24M for the Chinese "small") | Full text (arXiv PDF) and model card | None. Optional: add SIGIR 2024 |
| R11 guo2017 | Yes (PMLR 70:1321–1330) | Yes | Full text (arXiv PDF) | None |
| R12 platt1999 | **No.** The book chapter is "Probabilities for SV Machines", 2000, pp. 61–74, doi:10.7551/mitpress/1113.003.0008. The cited title is the preprint's. The MIT Press link returns 403 | Partly. Platt fits a sigmoid to one classifier output; the post's layer uses three inputs | Secondary only: Crossref record; Guo et al 2017 and Lin, Lin & Weng 2007 describe the method. The publisher blocks access; I could not get the chapter | Fix the reference (year, title, DOI) and the in-text key; narrow the two sentences |
| R13 naeini2015 | Yes (pp. 2901–2907 from the PDF page numbers) | Yes | Full text (AAAI OJS PDF) | None |
| R14 wilson1927 | Yes (Crossref) | Yes for the method. I recomputed ±1.2 and ±2.3 and they are correct | Secondary only (O'Neill 2021, arXiv:2109.12464). Taylor & Francis returns 403 | None |
| W1 all-MiniLM-L6-v2 card | URL resolves | Partly. "About a billion sentence pairs" is on the card. "22M" is not in the card text; the HF page shows 22.7M from safetensors metadata. "Distilled by Wang et al" needs the layer-dropping step (see R9) | Model card and HF API | Covered by the R9 rewording |
| W2 bge-small-en-v1.5 card | URL resolves | Yes for 33M (HF page metadata, 33.4M). No for the comparison with MiniLM: the card's MTEB table does not include MiniLM | Model card and HF API | See the bge sentence below |
| W3 SapBERT card | URL resolves | Yes: trained on UMLS 2020AA; 109.5M on the HF page and 109.48M in the paper | Model card and HF API | None |
| W4 MultipleNegativesRankingLoss v4.1.0 | Resolves; the GitHub link redirects from UKPLab/ to huggingface/sentence-transformers | Yes: the docstring cites Henderson et al section 4.4, accepts triplets, and uses cross-entropy with in-batch candidates (64 anchors give 128 candidates) | Source code (raw file at tag v4.1.0) | None. Optional: update the link to huggingface/ |
| bge-small sentence (R7 item 4) | n/a | **Not supported by the BGE card or the C-Pack paper alone**: neither includes MiniLM. It is supported if MTEB (Muennighoff et al 2023) is added: MiniLM-L6 scores 41.95 on retrieval there, against 51.68 for bge-small-en-v1.5 | Full text of C-Pack, the BGE card and the MTEB paper (arXiv) | Reword with no source (recommended), or add two citations |

---

## R1. korpela2026

**Metadata.** Crossref for 10.1093/jamiaopen/ooag085: title "RxMap: an LLM-assisted tool for medication normalization", *JAMIA Open*, vol. 9, issue 3, article number ooag085, published 2026-05-05. Authors Eero Korpela, Leah H Rubin, Raha M Dastgheyb, Yanxun Xu. All match. Europe PMC: PMID 42220340, PMC13221971.

**Evidence level.** Full text (JATS XML from Europe PMC), including Table 2.

**Support.**
- Numbers, dataset size and quoted phrase. From the abstract: "Evaluation on 22 624 unique medication strings from the IPUMS MEPS dataset demonstrated substantial improvements over deterministic RxNorm matching alone. The best-performing RxMap configuration improved RxCUI-level precision, recall, and F1-score from 0.865, 0.853, and 0.859 to 0.969, 0.964, and 0.966, respectively, with similar gains at the ingredient level." Yes: all six numbers, 22,624 and the quoted phrase match.
- The baseline. The 0.865/0.853/0.859 figures are for "RxNorm+", which the paper describes as "(2) RxNorm+, the deterministic matching component of RxMap". The after figures are for "RxMap (gemini-2.0-flash)". Table 2, IPUMS MEPS rows, RxCUI columns: Precision 0.865 → 0.969; Recall 0.853 → 0.964; F1 0.859 → 0.966. So the before figures come from RxMap's own deterministic matcher, without the LLM. The post does not say this. Note that the text and Table 2 of the paper disagree on the ingredient-level F1 for RxNorm+ (0.874 in the text, 0.832 in the table). This does not affect the post.
- "At the concept level." Yes. The paper's term is "RxCUI level (exact match)".
- Targets. Yes: "Raw medication strings are normalized to RxNorm ingredient-level (IN/MIN) concepts."
- "Uses a large language model." Yes: "combines deterministic RxNorm candidate generation with large language model (LLM)–assisted lexical parsing".
- "Medical Expenditure Panel Survey." The paper says only "IPUMS MEPS" and never expands MEPS. It links https://meps.ipums.org/meps/, a page titled "IPUMS MEPS" that says "Medical Expenditure Panel Survey". The expansion is correct, but it comes from that page and not from the paper.
- Uncited dependent sentences. "Published tools already match names like these reliably to the active ingredient" and "Tools that normalise drug names to the ingredient are already accurate" agree with the ingredient-level F1 of 0.972. RxMap was tested on survey strings, though, not VA names, so "names like these" goes slightly beyond the source. The section 10 sentence ("RxMap's numbers are for survey strings at the ingredient level") is supported.

**Proposed rewording (recommended, so the baseline is named):**
- Current: "RxMap, a recent tool that uses a large language model, reports precision, recall, and F1 at the concept level rising from 0.865, 0.853, and 0.859 to 0.969, 0.964, and 0.966, "with similar gains at the ingredient level", on 22,624 unique medication strings from the Medical Expenditure Panel Survey; its targets are ingredient concepts ([Korpela et al 2026](#references))."
- Proposed: "RxMap, a recent tool that uses a large language model, reports precision, recall, and F1 at the concept level rising from 0.865, 0.853, and 0.859 with RxNorm string matching alone to 0.969, 0.964, and 0.966 with its best configuration, "with similar gains at the ingredient level", on 22,624 unique medication strings from IPUMS MEPS, a version of the Medical Expenditure Panel Survey; its targets are ingredient concepts ([Korpela et al 2026](#references))."

## R2. nelson2011

**Metadata.** Crossref for 10.1136/amiajnl-2011-000116: "Normalized names for clinical drugs: RxNorm at 6 years", *JAMIA* 18(4):441–448, July 2011. Authors Stuart J Nelson, Kelly Zeng, John Kilbourne, Tammy Powell, Robin Moore. All match. PMC3128404.

**Evidence level.** I read the abstract in full from the PMC metadata. Europe PMC has no full-text XML for this article. The PMC article page shows the full text, which I read through WebFetch; that tool returns excerpts of at most 125 characters. A direct download of the page was blocked by the site's bot check, and I did not try to get around it.

**Support.**
- "RxNorm is the National Library of Medicine's (NLM's) standard vocabulary for clinical drugs." Yes. From the introduction: "RxNorm is a standard nomenclature developed by the United States National Library of Medicine (NLM)…". The title is "Normalized names for clinical drugs".
- Uncited RXCUI/SCD sentences. The paper supports them: "Different names and codes can be mapped to each other if they share the same RxNorm Concept Unique Identifier (RxCUI)." and "…representing drugs not only at the clinical drug level (SCD) {ingredient+strength+dose form}…". The NLM overview page (NLM 2026a) supports them more directly: "RxNorm groups these as synonyms into one concept."; "Every name and code provided to NLM by a source vocabulary receives an RXCUI."; "The normalized name consists of the ingredient, strength, and dose form (in that order) for fully-specified generic drugs. … The branded version of this drug uses the same format but includes the brand name in brackets at the end". You could add an NLM 2026a citation to these sentences.
- "thirteen source vocabularies". Not a citation issue, but `RXNSAB.RRF` in the 2026-09-08 release has 13 rows, and one of them is `RXNORM` itself. That makes 12 outside sources plus NLM's own names. "Collects drug names from thirteen source vocabularies" counts NLM's own names as a source. NLM's overview also treats SAB=RXNORM as "both a source and a subset", so this is defensible. "Twelve outside sources plus NLM's own names" would be more exact.
- "for two decades" (opening, uncited). Nelson 2011 does not support it. Through WebFetch, the article does not list the VA National Drug File as a source; it discusses NDF-RT ("integrated into RxNorm as a source vocabulary beginning with the RxNorm June 2010 monthly release"). NLM's VANDF source page cites Nelson SJ et al (2002), "A semantic normal form for clinical drugs in the UMLS: early experiences with the VANDF", *Proc AMIA Symp* 2002:557–61 (PubMed 12463886, PMC2244318; metadata checked on PubMed). That paper shows NLM was working with VANDF in 2002. I did not read it, so I cannot say it supports "linked … and publish the links for two decades".

**Proposed rewording (uncited opening sentence):**
- Current: "The training labels cost nothing, because the Library's staff have linked VA names to RxNorm entries for two decades and publish the links."
- Proposed: "The training labels cost nothing, because the Library's staff link VA names to RxNorm entries and publish the links."

## R3. nlm2026a

**Metadata.** The URL resolves (HTTP 200), and the page title is "RxNorm Overview". It is an NLM page with no date, so the year is the year consulted. OK.

**Evidence level.** The whole web page.

**Support.**
- Section 1 sentence. Yes. "About 60% of the drug names from RxNorm source vocabularies receive RxNorm normalized names." and "The other 40% of source vocabulary drug names do not receive RxNorm normalized names because they are either out-of-scope or their names are too ambiguous. The most common types of names that are not assigned RxNorm normalized names are medical devices, foods, and enzymes." "About 40%" and both reasons match.
- Details A sentence. Partly. NLM's 40% counts names with no RxNorm normalized name at any level. The post's 55% (17,564 of 31,936) counts names with no SCD/SBD, and that includes 819 names that do have an RxNorm concept at another level (packs, SCDG, SCDF, IN). The like-for-like figure from the post's own numbers is 16,745 of 31,936 = 52%. The "because the VA file contains more supplies and nutrition products" clause is the author's explanation. NLM's list of the most common unnamed types (devices, foods, enzymes) fits it, but NLM does not say it.

**Proposed rewording (Details A):**
- Current: "In this dataset 17,564 of the 31,936 distinct active VA CD/AB strings (55%) have no SCD/SBD, more than NLM's figure of about 40% for its sources (section 1), because the VA file contains more supplies and nutrition products than NLM's average source."
- Proposed: "In this dataset 17,564 of the 31,936 distinct active VA CD/AB strings (55%) have no SCD/SBD, and 16,745 (52%) have no RxNorm concept at all, more than NLM's figure of about 40% for its sources (section 1), probably because the VA file contains more supplies and nutrition products than NLM's average source."

## R4. nlm2026b

**Metadata.** The URL resolves (HTTP 200), and the page title is "UMLS Metathesaurus License Agreement Appendix" (2026AB edition, "Last Reviewed: November 2, 2026"). OK.

**Evidence level.** I read the whole appendix page and the general licence page (https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement.html). I also read `RXNSAB.RRF` from the local release at /home/kevin/repos/rxnorm/data/rxnorm_2026_09_08/rrf/.

**Support.**
- The quotation is exact: "Category 0 — general terms of the License apply with no additional restrictions".
- VANDF is category 0 on this page: "VANDF_2026_07_31 … NOTE: Now a CATEGORY 0." The RXNORM entry has no restriction note, which on this page means no extra restrictions. The page does not state "category 0" for RXNORM in words.
- "so a dataset built from them can be shared". **Does not follow from the cited text.** Category 0 means the general terms still apply. The general licence says: "LICENSEE is prohibited from distributing the UMLS Metathesaurus or subsets of it, including individual vocabulary sources within the Metathesaurus, except (a) as an integral part of computer applications developed by LICENSEE for a purpose other than redistribution of vocabulary sources contained in the UMLS Metathesaurus and (b) if permitted by paragraph 12 of this agreement." Whether the published dataset may be shared needs some other basis, for example the status of VA and NLM content as US government works, or NLM's RxNorm terms of use. I did not check either, and the author should.
- Details A Licence sentence. The page does not mention `RXNSAB.SRL`, and it defines only categories 0 to 4. SNOMED CT is governed by "Appendix 2" ("TERMS IN APPENDIX 2 OF UMLS LICENSE APPLY."), so "category 9" is not on this page. The local `RXNSAB.RRF` (field 14, SRL) has VANDF 0, RXNORM 0, SNOMEDCT_US 9, MMX 3, NDDF 3, GS 3, **MMSL 1**, and DRUGBANK, ATC, CVX, MTHCMSFRF, MTHSPL and USP 0. The appendix page also shows MMSL as "CATEGORY 1 RESTRICTIONS APPLY". So "the commercial drug sources (category 3)" leaves out Multum (MMSL), which is category 1.

**Proposed rewordings:**
1. Current: "Both the VA source and NLM's own names are in restriction category 0, which NLM defines as "general terms of the License apply with no additional restrictions" ([NLM 2026b](#references)), so a dataset built from them can be shared."
   Proposed: "Both the VA source and NLM's own names are in restriction category 0, which NLM defines as "general terms of the License apply with no additional restrictions" ([NLM 2026b](#references)), so no source-specific restrictions apply to a dataset built from them."
   (If the post needs to say the dataset can be shared, it needs a source that grants that right. This page does not.)
2. Current: "- **Licence:** each source's restriction category is in `RXNSAB.SRL`, where both `VANDF` and `RXNORM` are 0 ([NLM 2026b](#references)). The published dataset, model, and figures derive from those two sources alone; SNOMED CT (category 9) and the commercial drug sources (category 3) in the same release were not used."
   Proposed: "- **Licence:** each source's restriction category is in `RXNSAB.SRL`, where both `VANDF` and `RXNORM` are 0; NLM defines the categories in its licence appendix ([NLM 2026b](#references)). The published dataset, model, and figures derive from those two sources alone; SNOMED CT (category 9) and the commercial drug sources (categories 1 and 3) in the same release were not used."

## R5. henderson2017

**Metadata.** arXiv:1705.00652: "Efficient Natural Language Response Suggestion for Smart Reply". Authors Henderson, Al-Rfou, Strope, Sung, Lukacs, Guo, Kumar, Miklos, Kurzweil. Submitted 2017-05-01. All match.

**Evidence level.** Full text (arXiv PDF, 15 pages).

**Support.** Yes. Section 4.4 is titled "Multiple Negatives": "For efficiency and simplicity we use the responses of other examples in a training batch of stochastic gradient descent as negative responses. For a batch of size K, there will be K input emails x = (x1,...,xK) and their corresponding responses y = (y1,...,yK). Every reply yj is effectively treated as a negative candidate for xi if i≠j." The loss (eq. 5) is the batch mean of −[S(xi,yi) − log Σj e^S(xi,yj)], which is a softmax cross-entropy over the batch. The reference-list claim that section 4.4 is the loss sentence-transformers implements is confirmed by the W4 docstring. The paper has no explicit hard negatives; the extra negative column is a sentence-transformers addition (see W4). The sentence attributes only the loss to Henderson, so it is accurate as written.

**Optional clarification:**
- Current: "the in-batch softmax loss of [Henderson et al 2017](#references): for a batch of 64 (anchor, positive, hard negative) triplets,"
- Optional: "the in-batch softmax loss of [Henderson et al 2017](#references), with one hard negative added per anchor: for a batch of 64 (anchor, positive, hard negative) triplets,"

## R6. karpukhin2020

**Metadata.** ACL Anthology 2020.emnlp-main.550 and Crossref 10.18653/v1/2020.emnlp-main.550: *Proceedings of EMNLP 2020*, pp. 6769–6781. Authors Karpukhin, Oguz, Min, Lewis, Wu, Edunov, Chen, Yih. All match.

**Evidence level.** Full text (ACL Anthology PDF).

**Support.** Partly. The paper uses hard negatives and finds that they help: "Finally, we explore in-batch negative training with additional "hard" negative passages that have high BM25 scores given the question, but do not contain the answer string (the bottom block). … We find that adding a single BM25 negative passage improves the result substantially while adding two does not help further." It also says: "In practice, how to select negative examples is often overlooked but could be decisive for learning a high-quality encoder." The paper does not say hard negatives are "standard". It calls in-batch negatives an established "trick" and cites Henderson et al. for that. Its hard negatives are BM25 passages, which differ from the post's same-ingredient negatives; the post's previous sentence already defines its own negatives.

**Proposed rewording:**
- Current: "Hard negatives are a standard way to train retrieval models ([Karpukhin et al 2020](#references))."
- Proposed: "Adding one hard negative per question improved the dense retrieval model of [Karpukhin et al 2020](#references)."

## R7. liu2021 (and the uncited bge-small sentence)

**Metadata.** ACL Anthology 2021.naacl-main.334 and Crossref: *Proceedings of NAACL-HLT 2021*, pp. 4228–4238. Authors Liu, Shareghi, Meng, Basaldella, Collier. All match. The model URL resolves.

**Evidence level.** Full text (ACL Anthology PDF) and the HF model card.

**Support.**
1. "trained to place different names for the same medical concept close together". Yes. "We design a self-alignment objective that clusters synonyms of the same concept." Figure 1 caption: "After the self-alignment pretraining, the same concept's entity names are drawn closer to form compact clusters (right)."
2. "It already knows that `HCL` and `hydrochloride` mean the same thing." **Not supported.** The paper does not test this. It does say the UMLS training data include RxNorm: "UMLS 2020AA has 4M+ concepts and 10M+ synonyms which stem from over 150 controlled vocabularies including MeSH, SNOMED CT, RxNorm, Gene Ontology and OMIM", and "we also include tradenames of drugs as synonyms (extracted from MRREL.RRF)". If the project measured this, say so; if not, hedge.
3. "(110M …), pre-trained to pull UMLS synonyms together". Yes. Table 7: "full-model-tuning - 109.48M 100%". The HF page shows 109,482,752 parameters. Figure 1 caption: "PUBMEDBERT+SAPBERT (PUBMEDBERT further pretrained on UMLS synonyms)".
4. bge-small sentence. See the dedicated section at the end.

**Proposed rewording (item 2):**
- Current: "It already knows that `HCL` and `hydrochloride` mean the same thing."
- Proposed: "Its training names came from UMLS, which includes RxNorm, so it has probably already seen that `HCL` and `hydrochloride` mean the same thing."

## R8. reimers2019

**Metadata.** ACL Anthology D19-1410: *Proceedings of EMNLP-IJCNLP 2019*, pp. 3982–3992, DOI 10.18653/v1/D19-1410. The PDF header reads "pages 3982–3992". Crossref lists 3980–3990, which is a Crossref error; the post agrees with the Anthology and the PDF. Authors match.

**Evidence level.** Full text (ACL Anthology PDF).

**Support.** Yes for both sentences. "we present Sentence-BERT (SBERT), a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity." "The siamese network architecture enables that fixed-sized vectors for input sentences can be derived. Using a similarity measure like cosine-similarity … semantically similar sentences can be found." Footnote 1: "Code available: https://github.com/UKPLab/sentence-transformers", which confirms this is the paper behind the library. The paper does not use the word "bi-encoder"; it contrasts SBERT with BERT's "cross-encoder". The unit-vector normalisation is the project's choice, and it is equivalent to ranking by cosine.

## R9. wang2020

**Metadata.** arXiv:2002.10957, with authors and title matching. The NeurIPS proceedings page (proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) shows "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)" with the same six authors. OK.

**Evidence level.** Full text (arXiv PDF). I also read the model cards for sentence-transformers/all-MiniLM-L6-v2, nreimers/MiniLM-L6-H384-uncased and microsoft/MiniLM-L12-H384-uncased.

**Support.** Partly.
- The distillation method: "we present a simple and effective approach to compress large Transformer … based pre-trained models, termed as deep self-attention distillation."
- But the all-MiniLM-L6-v2 card says "We used the pretrained `nreimers/MiniLM-L6-H384-uncased` model", and that card says: "This is a 6 layer version of microsoft/MiniLM-L12-H384-uncased by keeping only every second layer." The microsoft card says: "MiniLM is a distilled model from the paper "MiniLM: Deep Self-Attention Distillation …"". So the 6-layer starting model is a cut-down copy of a model that Wang et al distilled. Wang et al did not distill it themselves.
- 22M. The paper's Table 3 gives "M=6; d′h=384 22M" for its own 6-layer student. The HF page reports 22,713,728 parameters (22.7M) for all-MiniLM-L6-v2. "22M" is acceptable, although strict rounding gives 23M.
- "About a billion sentence pairs": see W1.

**Proposed rewording (Details B, Starting models):**
- Current: "(22M parameters, distilled by [Wang et al 2020](#references) and further trained on about a billion sentence pairs)"
- Proposed: "(22M parameters, made by keeping every second layer of a model distilled by [Wang et al 2020](#references), then further trained on about a billion sentence pairs)"

## R10. xiao2023

**Metadata.** arXiv:2309.07597: "C-Pack: Packed Resources For General Chinese Embeddings". Authors Xiao, Liu, Zhang, Muennighoff, Lian, Nie. Submitted 2023-09-14. The arXiv comment says "SIGIR 2024", and the PDF running header reads "SIGIR '24, July 14–18, 2024, Washington, DC, USA". The citation is correct as an arXiv paper. Optional: add the venue.

**Evidence level.** Full text (arXiv PDF) and the model card.

**Support.** Yes, C-Pack is the BGE paper. The card's Citation section gives `@misc{bge_embedding, title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, … eprint={2309.07597}`, an older title and author list for the same arXiv ID. The 33M figure is not in the paper. The paper says "small (with 24M parameters)" for its Chinese models. The HF page reports 33,360,512 parameters for bge-small-en-v1.5. Because the model link sits next to the number, no change is required.

## R11. guo2017

**Metadata.** arXiv:1706.04599, authors Guo, Pleiss, Sun, Weinberger, arXiv comment "ICML 2017". PMLR v70 page: pages 1321–1330, "International Conference on Machine Learning". OK. Optional: add PMLR 70:1321–1330.

**Evidence level.** Full text (arXiv PDF).

**Support.** Yes. "Temperature scaling, the simplest extension of Platt scaling, uses a single scalar parameter T > 0 for all classes. Given the logit vector zi, the new confidence prediction is q̂i = max_k σSM(zi/T)(k). … T is optimized with respect to NLL on the validation set." The abstract adds: "temperature scaling – a single-parameter variant of Platt Scaling – is surprisingly effective at calibrating predictions."

## R12. platt1999

**Metadata. Not correct as given.** Crossref has the chapter as DOI 10.7551/mitpress/1113.003.0008: title "Probabilities for SV Machines", author John C. Platt, in *Advances in Large-Margin Classifiers*, The MIT Press, pp. 61–74, issued 2000-09-29. Open Library gives the book (ISBN 9780262194488) as published 2 October 2000, LC call number "Q325.5 .A34 2000". The long title in the post ("Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods") and the year 1999 belong to the widely circulated preprint. Lin, Lin & Weng (2007) cite the same work as "Platt, J.: 2000, … In: A. Smola, P. Bartlett, B. Schölkopf, and D. Schuurmans (eds.): Advances in Large Margin Classifiers. Cambridge, MA, MIT Press." The pages 61–74 are right for the book chapter. The MIT Press URL in the post returns HTTP 403 to automated access, and the DOI redirects to direct.mit.edu, which also returns 403.

**Evidence level.** Secondary only. I could not read the chapter: the publisher blocks automated access, and the CiteSeerX copy was rate-limited and redirected to the Internet Archive. I used Guo et al 2017 (full text): "Platt scaling (Platt et al., 1999) is a parametric approach to calibration … The non-probabilistic predictions of a classifier are used as features for a logistic regression model, which is trained on the validation set to return probabilities. More specifically, … Platt scaling learns scalar parameters a, b ∈ R and outputs q̂i = σ(azi + b) as the calibrated probability." I also used Lin, Lin & Weng (full text): "Platt (2000) proposes approximating the posterior by a sigmoid function Pr(y = 1|x) ≈ PA,B(f) ≡ 1/(1 + exp(Af + B)), where f = f(x)."

**Support.** Partly. Platt's method fits a two-parameter sigmoid to one classifier score. The post's model is a logistic layer over three inputs (cosine, margin, log-softmax). It is an extension of Platt scaling, not the method itself.

**Proposed changes:**
1. Reference list.
   - Current: "- Platt JC (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods". In Smola AJ, Bartlett P, Schölkopf B, Schuurmans D (eds.), *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74. [mitpress.mit.edu](https://mitpress.mit.edu/9780262194488/advances-in-large-margin-classifiers/)"
   - Proposed: "- Platt JC (2000). "Probabilities for SV Machines". In Smola AJ, Bartlett P, Schölkopf B, Schuurmans D (eds.), *Advances in Large-Margin Classifiers*, MIT Press, pp. 61–74. [doi:10.7551/mitpress/1113.003.0008](https://doi.org/10.7551/mitpress/1113.003.0008). Often cited under the title of its 1999 preprint, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods"."
   - Also change the bib entry `platt1999` to year 2000, the new title and the new DOI. md2qmd matches by surname and year, so both in-text links must change to `[Platt 2000](#references)`.
2. Section 7.
   - Current: "a single scaling factor on the scores ([Guo et al 2017](#references)), and a small logistic model over the scores ([Platt 1999](#references))."
   - Proposed: "a single scaling factor on the scores ([Guo et al 2017](#references)), and a small logistic model over the scores, based on Platt scaling ([Platt 2000](#references))."
3. Details B.
   - Current: "and a logistic (Platt) layer over [cosine, margin, log-softmax] ([Platt 1999](#references)), trained on validation to predict whether the top answer is right."
   - Proposed: "and a logistic layer over [cosine, margin, log-softmax], which extends Platt scaling ([Platt 2000](#references)) from one score to three, trained on validation to predict whether the top answer is right."

## R13. naeini2015

**Metadata.** Crossref 10.1609/aaai.v29i1.9602: title, authors (Mahdi Pakdaman Naeini, Gregory Cooper, Milos Hauskrecht) and vol. 29, issue 1 (2015-02-21) all match. Crossref has no pages, but the AAAI PDF has page numbers 2901 to 2907 and the footer "Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence". OK.

**Evidence level.** Full text (AAAI OJS PDF).

**Support.** Yes. "These measures are called Expected Calibration Error (ECE), and Maximum Calibration Error (MCE). In computing these measures, the predictions are sorted and partitioned into K fixed number of bins (K = 10 in our experiments). … ECE = Σ_{i=1}^{K} P(i)·|oi − ei|". Guo et al 2017 also attribute ECE to this paper: "Expected Calibration Error (Naeini et al., 2015) – or ECE".

## R14. wilson1927

**Metadata.** Crossref 10.1080/01621459.1927.10502953: "Probable Inference, the Law of Succession, and Statistical Inference", Edwin B. Wilson, *JASA* 22(158):209–212, June 1927. OK.

**Evidence level.** Secondary only. Taylor & Francis returns HTTP 403, and I did not try to get around it. I used O'Neill (2021), "Mathematical properties and finite-population correction for the Wilson score interval", arXiv:2109.12464: "Wilson (1927) introduced a form of confidence interval for this problem that has now become known by the eponymous name of the "Wilson score interval"." Its reference list gives the same JASA volume, issue and pages.

**Support.** Yes for the method. I recomputed the intervals with z = 1.96: n = 1,848 at p = 0.93 gives a half-width of 0.0117 (±1.2 points), and p = 0.51 gives 0.0228 (±2.3). The Details C intervals also check out: 98.8% of 1,460 gives 0.981–0.992, and 98.2% of 1,037 gives 0.972–0.988.

## W1. sentence-transformers/all-MiniLM-L6-v2 model card

**Resolves.** Yes.
**Evidence level.** The model card (raw README) and the HF API (safetensors parameter count).
**Support.** Partly.
- "About a billion sentence pairs": yes. The card says "We used the pretrained `nreimers/MiniLM-L6-H384-uncased` model and fine-tuned in on a 1B sentence pairs dataset." and "The total number of sentence pairs is above 1 billion sentences." The table total is 1,170,060,424.
- "22M" is not in the card text. The HF page shows 22.7M, and the MiniLM paper gives 22M for the same shape.
- For "distilled by Wang et al", see R9. The R9 rewording covers this item.

## W2. BAAI/bge-small-en-v1.5 model card

**Resolves.** Yes.
**Evidence level.** The model card (raw README) and the HF API.
**Support.**
- 33M: yes, from the HF page (33,360,512 parameters). The card text does not give a parameter count.
- The bge-small sentence: no. The card's MTEB table lists bge-small-en-v1.5 at "62.17 | 51.68" (average over 56 datasets, retrieval over 15). The comparison models are bge, gte, e5, sentence-t5-xxl, all-mpnet-base-v2 and sgpt-bloom; MiniLM is not among them. The card dates v1.5 to "09/12/2023: … release `bge-*-v1.5` embedding model".

## W3. cambridgeltl/SapBERT-from-PubMedBERT-fulltext model card

**Resolves.** Yes.
**Evidence level.** The model card and the HF API.
**Support.** Yes. "Trained with UMLS 2020AA (English only), using microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext as the base model." The HF page shows 109,482,752 parameters (110M), consistent with the paper's Table 7 (109.48M).

## W4. sentence-transformers MultipleNegativesRankingLoss (v4.1.0)

**Resolves.** Yes. github.com/UKPLab/… returns 301 to github.com/huggingface/sentence-transformers/blob/v4.1.0/…, which returns 200. The repository has moved; the old link still works. Optional: update it.
**Evidence level.** Source code (raw file at tag v4.1.0, 131 lines).
**Support.** Yes.
- Docstring: "Given a list of (anchor, positive) pairs or (anchor, positive, negative) triplets, this loss optimizes the following: 1. Given an anchor (e.g. a question), assign the highest similarity to the corresponding positive (i.e. answer) out of every single positive and negative (e.g. all answers) in the batch. If you provide the optional negatives, they will all be used as extra options from which the model must pick the correct positive."
- "This loss is also known as InfoNCE loss, SimCSE loss, Cross-Entropy Loss with in-batch negatives, or simply in-batch negatives loss."
- "References: - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf"
- Code: `candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)` and `return self.cross_entropy_loss(scores, range_labels)`. With 64 triplets, each anchor is scored against 64 × 2 = 128 candidates, as the post says.

---

## The uncited bge-small sentence (R7 item 4)

Sentence (Section 6 bullet "The starting model mattered most."): "bge-small is newer than MiniLM and scores higher on general retrieval benchmarks, yet the two could not be told apart here."

**Does the BGE model card or the C-Pack paper support it? No, not on their own.** Neither one includes MiniLM:
- BGE card MTEB table: "| [BAAI/bge-small-en-v1.5](…) | 384 | 512 | 62.17 |51.68 | …" (columns: Average (56), Retrieval (15)). The other rows are bge, gte, e5, sentence-t5-xxl, all-mpnet-base-v2 and sgpt-bloom-7b1-msmarco.
- C-Pack paper, Table 5, "Performance of English Models on MTEB": "BGE (small) 384 62.17 51.68 …". No MiniLM row.
- "Newer" can be pieced together from the cards: all-MiniLM-L6-v2 was built during the 2021 Hugging Face community week, and bge-*-v1.5 was released on 09/12/2023. Neither card states the comparison, though.

**A source that does fill the gap** (checked, but not in the post's bibliography): Muennighoff N, Tazi N, Magne L, Reimers N (2023). "MTEB: Massive Text Embedding Benchmark". *Proceedings of EACL 2023*:2014–2037. doi:10.18653/v1/2023.eacl-main.148. I checked the metadata on Crossref and the ACL Anthology, and read the full text of the arXiv version 2210.07316. Its Table 1 lists "MiniLM-L6 63.06 42.35 82.37 58.04 41.95 78.90 30.81 56.26" (Class., Clust., PairClass., Rerank., Retr. over 15 datasets, STS, Summ., Avg.). Its appendix maps "MiniLM-L6 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2". The MPNet row in that table (Retr. 43.81, Avg. 57.78) matches the all-mpnet-base-v2 row in the BGE card, which confirms that the two tables use the same benchmark. So on MTEB retrieval, bge-small-en-v1.5 scores 51.68 (BAAI's own evaluation) and all-MiniLM-L6-v2 scores 41.95 (the MTEB authors' evaluation). Caveat: the two numbers come from different groups, and I read the arXiv version of MTEB, not the EACL PDF.

**Recommended (no source needed):**
- Current: "bge-small is newer than MiniLM and scores higher on general retrieval benchmarks, yet the two could not be told apart here."
- Proposed: "bge-small and MiniLM are both small general-purpose models, yet the two could not be told apart here."

**Alternative (keeps the claim; needs one new reference, Muennighoff et al 2023, added to the reference list and bib):**
- Proposed: "bge-small is newer than MiniLM and scores higher on the retrieval tasks of the MTEB benchmark (51.7 against 42.0; [Xiao et al 2023](#references); [Muennighoff et al 2023](#references)), yet the two could not be told apart here."
