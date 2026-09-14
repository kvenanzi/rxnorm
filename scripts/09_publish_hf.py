"""Publish the final model and the dataset to Hugging Face.

Both repos are created private so the cards can be checked in place; visibility
is flipped to public in each repo's Settings.

  uv run scripts/09_publish_hf.py --dry-run     # stage under outputs/hf/, upload nothing
  uv run scripts/09_publish_hf.py               # create/update both private repos
  uv run scripts/09_publish_hf.py --skip-dataset
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "sapbert-ingredient-strength-final" / "best"
PROCESSED = ROOT / "data" / "processed"
STAGE = ROOT / "outputs" / "hf"
MODEL_REPO = "vandf-rxnorm-biencoder"
DATASET_REPO = "vandf-rxnorm-pairs"
GITHUB = "https://github.com/kvenanzi/rxnorm"
WANDB = "https://wandb.ai/kettle-labs/rxnorm-vandf"

MODEL_CARD = """---
license: apache-2.0
base_model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
library_name: sentence-transformers
pipeline_tag: sentence-similarity
language:
- en
tags:
- rxnorm
- vandf
- drug-normalization
- medication
- entity-linking
- sentence-transformers
datasets:
- __NS__/vandf-rxnorm-pairs
model-index:
- name: vandf-rxnorm-biencoder
  results:
  - task:
      type: sentence-similarity
      name: VANDF string to RxNorm clinical drug (SCD/SBD) mapping
    dataset:
      type: __NS__/vandf-rxnorm-pairs
      name: VANDF-RxNorm pairs, test split (held-out ingredients)
    metrics:
    - type: accuracy
      name: acc@1
      value: 0.931
    - type: recall
      name: recall@5
      value: 0.984
---

# VANDF → RxNorm clinical drug bi-encoder

Maps a VA National Drug File (VANDF) drug string to the RxNorm **clinical drug**
it means (SCD or SBD: ingredient, strength, *and* dose form), with a calibrated
confidence so you can auto-accept the sure cases and route the rest to review.

| Input | Output | Confidence |
|---|---|---|
| `METOPROLOL TARTRATE 12.5MG TAB` | `metoprolol tartrate 12.5 MG Oral Tablet` (RXCUI 866508) | 0.97 → accept |
| `ACETIC ACID 0.25% IRRG SOLN` | `acetic acid 2.5 MG/ML Irrigation Solution` | 0.96 → accept |
| `HYOSCYAMINE SO4 0.125MG/5ML ELIXIR` | `hyoscyamine sulfate 0.125 MG/ML Oral Solution` (wrong: truth is 0.025 MG/ML, listed 2nd) | 0.77 → **review** |
| `CATHETER,FOLEY SILICONE 22FR 5CC` | (nearest candidate, meaningless) | 0.02 → **review** |

Published tools such as RxMap normalize to the *ingredient* level (IN/MIN).
This model targets the full clinical drug, where a miss on strength or dose
form is a wrong answer. Method, hypotheses, the full sweep, and the error
analysis are in the write-up: __GITHUB__/blob/main/docs/post/README.md.

## Usage

```bash
pip install "rxnorm-vandf @ git+__GITHUB__"
```

```python
from rxnorm_vandf.infer import Mapper

mapper = Mapper.from_pretrained("__NS__/vandf-rxnorm-biencoder")   # ~450 MB download
for p in mapper.map(["METOPROLOL TARTRATE 12.5MG TAB", "CATHETER,FOLEY SILICONE 22FR 5CC"]):
    print(p.rxcui, p.name, p.tty, f"{p.confidence:.2f}", "accept" if p.accept else "review")
    # p.alternatives: the next four candidates as (rxcui, name, cosine)
```

`Mapper` loads the encoder, `train_config.json` (input preprocessing),
`calibration.json` (score → probability), and `candidates.parquet` (the 27,287
active RxNorm SCD/SBD names it searches) from this repo. The default acceptance
threshold (0.92) was chosen on validation for 99% precision on a population
that includes real drugs with no SCD/SBD; pass `threshold=` to change it.

## How it works

1. **Preprocessing.** The VA string is lowercased, punctuation is stripped, and a
   deterministic rule appends the RxNorm-style concentration
   (`0.125MG/5ML` → `0.025 mg/ml`; `0.25%` → `2.5 mg/ml`; `mg/mg` for gels and
   ointments). An embedding model can't do this arithmetic; a regex can.
2. **Retrieval.** A [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
   bi-encoder, fine-tuned with `MultipleNegativesRankingLoss` on 9,287
   (VA string, RxNorm name) pairs plus hard negatives (same ingredients,
   different strength or dose form), embeds the query and all candidates;
   top-k by cosine.
3. **Calibration.** A logistic (Platt) layer over [cosine, top-1−top-2 margin,
   temperature-scaled softmax] gives P(correct). Fit on validation only.

## Results

Test split: 1,848 VA strings whose **ingredients never appear in training**.
The candidate pool is all 27,287 active SCD/SBD in RxNorm 2026-09-08.

| Method | acc@1 | recall@5 | ingredient | strength | dose form |
|---|---|---|---|---|---|
| Exact string match | 0.000 | 0.000 | – | – | – |
| TF-IDF char 3–5-grams | 0.509 | 0.820 | 0.978 | 0.617 | 0.698 |
| MiniLM-L6 fine-tuned | 0.836 | 0.967 | 0.983 | 0.884 | 0.935 |
| + strength normalizer | 0.886 | 0.975 | 0.982 | 0.941 | 0.943 |
| **SapBERT + normalizer (this model)** | **0.931** | **0.984** | 0.983 | 0.961 | 0.972 |

Validation: acc@1 0.883, recall@5 0.961.

The 0.931 is one draw of the ingredient split. Re-drawing the split five more
times and retraining the same recipe gives test acc@1 **0.907 ± 0.019** (range
0.874–0.931; the published split is the most favorable of the six), while the
training seed moves it by 0.001. Read the headline as a band of about two
points. Recall@5 (0.974–0.988) and the component accuracies barely move.

**Abstention.** Thresholds chosen on validation, measured on test:

| Population | Signal | Auto-accepted | Precision of accepted |
|---|---|---|---|
| VA strings that have an SCD/SBD | softmax | 79.0% | 0.988 |
| + real drugs with no SCD/SBD (packs, ingredient-only) | platt (default) | 46.0% | 0.982 |

The validation target was 99%; test lands at 98.2–98.8%. Treat the achieved
number as the estimate, not the target.

An 18-run sweep (3 encoders × 3 negative strategies × normalizer on/off) found
the three effects roughly additive: domain pre-training (SapBERT vs general
encoders) +8.6 points val acc@1, the strength normalizer +4.8, ingredient-matched
hard negatives +3. The hard-negatives effect was then re-run on all six ingredient
splits (18 runs): ingredient-matched negatives beat in-batch-only on every split, by
1.6 points of test acc@1 on average (95% CI 0.8–2.4). Live charts: __WANDB__.

## Limitations

- **Trained and evaluated on VA strings only.** Other systems' drug names are a
  different distribution; accuracy there is unmeasured.
- **One split for calibration.** The abstention thresholds were chosen on one
  validation draw; the six-split experiment covers accuracy, not calibration.
  Re-choose thresholds on your own held-out data.
- **Candidates are RxNorm 2026-09-08.** RxNorm changes monthly; rebuild
  `candidates.parquet` for a newer release (`scripts/03_build_dataset.py` in the repo).
- **Not for unsupervised clinical use.** A 7% top-1 error rate on medication
  codes is a safety problem; use the confidence to route uncertain strings to a
  pharmacist, or use the top-5 as suggestions.
- Of the 128 test errors, 46 involve strength (often an underdetermined string:
  `MANNITOL 250MG/ML INJ` vs RxNorm's `50 ML mannitol 250 MG/ML Injection`),
  43 dose form, 31 ingredient, and 25 are SCD-vs-SBD twins with identical components.

## Training details

- Base: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (110M params, Apache-2.0)
- Loss: `MultipleNegativesRankingLoss`, triplets (anchor, positive, hard negative),
  `NO_DUPLICATES` batch sampler; negatives drawn from train-split candidates only
- 4 epochs, batch 64, lr 2e-5, 10% warmup, fp16, max_seq_length 96, seed 42
- Best epoch by validation acc@1; test scored once
- ~20 min on a GTX 1070, ~2 min on a Colab A100

## Data and license

Training pairs come from RxNorm itself: a VANDF atom and an RxNorm SCD/SBD atom
that share an RXCUI are the same drug by NLM's curation. Only the `VANDF` and
`RXNORM` source vocabularies were used (UMLS source restriction category 0:
"general terms of the License apply with no additional restrictions"). No PHI.
Dataset: `__NS__/vandf-rxnorm-pairs`.

Model weights: Apache-2.0, as the base model.

## References

- Liu F, Shareghi E, Meng Z, Basaldella M, Collier N (2021). "Self-Alignment
  Pretraining for Biomedical Entity Representations". *NAACL-HLT 2021*:4228–4238.
  https://aclanthology.org/2021.naacl-main.334/ (SapBERT)
- Reimers N, Gurevych I (2019). "Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks". *EMNLP-IJCNLP 2019*:3982–3992. https://aclanthology.org/D19-1410/
- Henderson M et al (2017). "Efficient Natural Language Response Suggestion for
  Smart Reply". arXiv:1705.00652. (the in-batch negatives loss)
- Guo C, Pleiss G, Sun Y, Weinberger KQ (2017). "On Calibration of Modern Neural
  Networks". *ICML 2017*. arXiv:1706.04599. (temperature scaling)
- Platt JC (1999). "Probabilistic Outputs for Support Vector Machines and
  Comparisons to Regularized Likelihood Methods". In *Advances in Large Margin
  Classifiers*, MIT Press, pp. 61–74.
- Nelson SJ, Zeng K, Kilbourne J, Powell T, Moore R (2011). "Normalized names for
  clinical drugs: RxNorm at 6 years". *JAMIA* 18(4):441–448.
  https://doi.org/10.1136/amiajnl-2011-000116
- Korpela E, Rubin LH, Dastgheyb RM, Xu Y (2026). "RxMap: an LLM-assisted tool for
  medication normalization". *JAMIA Open* 9(3):ooag085.
  https://doi.org/10.1093/jamiaopen/ooag085
- RxNorm is produced by the U.S. National Library of Medicine; VANDF by the U.S.
  Department of Veterans Affairs.
"""

DATASET_CARD = """---
license: other
license_name: umls-category-0
license_link: https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement_appendix.html
language:
- en
tags:
- rxnorm
- vandf
- drug-normalization
- medication
- entity-linking
pretty_name: VANDF-RxNorm clinical drug pairs
size_categories:
- 10K<n<100K
---

# VANDF → RxNorm clinical drug pairs

VA National Drug File (VANDF) drug strings paired with the RxNorm clinical drug
(SCD/SBD) they denote, plus the full candidate pool and the VA names that have
no clinical-drug answer. Built from the RxNorm **2026-09-08** full monthly
release by `scripts/03_build_dataset.py` in __GITHUB__.

The labels are NLM's own: a VANDF atom and an RXNORM SCD/SBD atom that share an
RXCUI are the same drug. Nothing was hand-annotated. No PHI.

## Files

| File | Rows | One row per |
|---|---|---|
| `pairs.parquet` | 14,372 | (VA string, target concept). A VA string may be both a `CD` and an `AB` name; duplicates are merged. |
| `candidates.parquet` | 27,287 | Active RxNorm SCD/SBD: the retrieval pool, with the same component labels. |
| `unmatched.parquet` | 17,564 | Active VA `CD`/`AB` names with no SCD/SBD (val/test only): abstention examples. |

### `pairs.parquet`

| Column | Meaning |
|---|---|
| `vandf_string` | The VA name, as in `RXNCONSO.STR` (e.g. `HYOSCYAMINE SO4 0.125MG/5ML ELIXIR`) |
| `vandf_ttys` | VANDF term types this string appears as (`CD` clinical drug name, `AB` print name) |
| `target_rxcui`, `target_tty`, `target_name` | The RxNorm concept (`SCD` generic / `SBD` branded) and its normalized name |
| `ingredients`, `ingredient_rxcuis` | IN-level ingredient names / RXCUIs (via SCDC `ingredient_of`) |
| `strength` | Per-ingredient strength from SCDC `RXN_STRENGTH`, joined with ` / ` |
| `dose_form` | RxNorm dose form (DF `dose_form_of`) |
| `quantity` | `RXN_QUANTITY` when present (e.g. `50 ML`) |
| `brand` | Brand name for SBDs |
| `split` | `train` / `val` / `test` / `mixed` |

`candidates.parquet` has the target/component columns (as `rxcui`, `tty`, `name`, …)
and `split`. `unmatched.parquet` has `vandf_string`, `vandf_ttys`, `rxcui`,
`rxnorm_tty` (what RxNorm has for the concept instead, e.g. `GPCK`; null for
supplies and devices), `split`.

## Split

**By ingredient**, 70/15/15: every IN RXCUI is hashed (md5 with a fixed salt)
into train/val/test, and a target inherits its ingredients' split. Combination
drugs whose ingredients fall in different splits are `mixed` and excluded from
train/val/test (training on one would leak a test ingredient). SBDs inherit
their generic SCD's ingredients so a brand and its generic share a split.

| Split | pairs | candidates | unmatched |
|---|---|---|---|
| train | 9,290 | 17,163 | – |
| val | 1,907 | 3,641 | 8,786 |
| test | 1,848 | 3,180 | 8,778 |
| mixed | 1,327 | 3,303 | – |

Test therefore measures generalization to **drugs whose ingredients were never
seen in training**. On this draw val is harder than test for every method tried
(TF-IDF 0.465 vs 0.509 acc@1), but that is a property of the draw, not the
task: over six salts of the same hash the val − test gap flips sign three times,
and the fine-tuned model's test acc@1 spans 0.874–0.931 (sd 0.019). The salt
for this release is `rxnorm-2026-09-08-v1`; the five re-draws (`-v2` … `-v6`)
are built by `scripts/03_build_dataset.py --salt` in the repo and are logged
there as the W&B artifact `vandf-rxnorm-splits`. Report val and test both.

## Known quirks

- Two VA strings map to more than one concept (an albuterol inhaler with three
  NDA-specific SCDs; a thyroid tablet with 30 and 32 MG). Any of them counts.
- Most `unmatched` names are supplies, devices, and nutrition products (easy
  rejections). ~800 are real drugs RxNorm only knows as packs (GPCK/BPCK) or at
  a vaguer level (SCDF/SCDG/IN): the hard abstention cases.

## License

Derived exclusively from the `VANDF` and `RXNORM` source vocabularies of RxNorm,
both UMLS source restriction category 0 ("general terms of the License apply
with no additional restrictions"). RxNorm is produced by the U.S. National
Library of Medicine; VANDF is produced by the U.S. Department of Veterans
Affairs. Use is subject to the UMLS Metathesaurus License general terms. No
content from restricted sources (SNOMED CT, GS, MMX, NDDF, MMSL) is included.

## References

- Nelson SJ, Zeng K, Kilbourne J, Powell T, Moore R (2011). "Normalized names for
  clinical drugs: RxNorm at 6 years". *JAMIA* 18(4):441–448.
  https://doi.org/10.1136/amiajnl-2011-000116
- Bodenreider O (2004). "The Unified Medical Language System (UMLS): integrating
  biomedical terminology". *Nucleic Acids Research* 32(Database issue):D267–D270.
  https://doi.org/10.1093/nar/gkh061
- UMLS Metathesaurus License Agreement Appendix (source restriction categories):
  https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement_appendix.html
"""


def fill(text: str, namespace: str) -> str:
    return text.replace("__NS__", namespace).replace("__GITHUB__", GITHUB).replace("__WANDB__", WANDB)


def stage_model(namespace: str) -> Path:
    dst = STAGE / "model"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(MODEL_DIR, dst, ignore=shutil.ignore_patterns("checkpoints"))
    if not (dst / "candidates.parquet").exists():
        shutil.copy(PROCESSED / "candidates.parquet", dst / "candidates.parquet")
    if not (dst / "calibration.json").exists():
        import wandb
        art = wandb.Api().artifact("kettle-labs/rxnorm-vandf/vandf-rxnorm-calibration:latest")
        shutil.copy(Path(art.download()) / "calibration.json", dst / "calibration.json")
    for required in ("train_config.json", "calibration.json", "candidates.parquet", "model.safetensors"):
        assert (dst / required).exists(), f"missing {required}"
    # The training config carries local absolute paths; the published copy needs only
    # the preprocessing and model fields, so drop the machine-specific ones.
    import json
    cfg = json.loads((dst / "train_config.json").read_text())
    cfg["data_dir"] = None
    cfg["output_dir"] = "models"
    (dst / "train_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (dst / "README.md").write_text(fill(MODEL_CARD, namespace))
    return dst


def stage_dataset(namespace: str) -> Path:
    dst = STAGE / "dataset"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("pairs.parquet", "candidates.parquet", "unmatched.parquet"):
        shutil.copy(PROCESSED / name, dst / name)
    (dst / "README.md").write_text(fill(DATASET_CARD, namespace))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="stage only; upload nothing")
    ap.add_argument("--namespace", help="HF user or org (default: the token's user)")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    namespace = args.namespace or ("<namespace>" if args.dry_run else api.whoami()["name"])

    jobs = []
    if not args.skip_model:
        jobs.append(("model", f"{namespace}/{MODEL_REPO}", stage_model(namespace)))
    if not args.skip_dataset:
        jobs.append(("dataset", f"{namespace}/{DATASET_REPO}", stage_dataset(namespace)))

    for repo_type, repo_id, folder in jobs:
        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6
        print(f"{repo_type:<8} {repo_id:<40} {size:7.1f} MB staged at {folder.relative_to(ROOT)}")
        if args.dry_run:
            continue
        api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type=repo_type,
                          commit_message="Publish from kvenanzi/rxnorm")
        print(f"         uploaded -> https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id} (private)")


if __name__ == "__main__":
    main()
