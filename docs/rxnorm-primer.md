# RxNorm Primer

The modeling in this project is fairly ordinary. The payoff comes from
understanding RxNorm's data model, which is summarized here.

## Files (RRF format)

RRF ("Rich Release Format") files are pipe-delimited, have **no header row**, and
put a **trailing pipe** on every line. Quote characters are literal data, so quote
parsing must be off.

| File | Contents |
|---|---|
| `RXNCONSO.RRF` | Every drug name string. One row per atom. The big one. |
| `RXNREL.RRF` | Relationships between concepts: which ingredient, which dose form, etc. |
| `RXNSAT.RRF` | Attributes: NDCs, strengths, prescribable flags, other metadata |
| `RXNSAB.RRF` | One row per source vocabulary, with its license restriction level |
| `RXNDOC.RRF` | Data dictionary for the codes used in the other files |

The `scripts/` folder in the release holds MySQL/Oracle loaders. I use DuckDB instead: one file, no server, reads RRF natively.

## Atoms vs. concepts

This is the single most important idea in RxNorm.

- An **atom** is one name from one source vocabulary. The VA calls a drug one thing,
  First Databank another, and FDA labels a third. Each is a separate atom with its
  own **RXAUI** (RxNorm Atom Unique Identifier).
- A **concept** is the meaning those atoms share, identified by an **RXCUI**
  (RxNorm Concept Unique Identifier). All atoms that mean the same drug collapse onto one RXCUI.

In `RXNCONSO`, one row is one atom, and many rows share an RXCUI.

**Why it matters here:** a VANDF atom and an RXNORM SCD atom with the same RXCUI
are, by NLM's curation, the same drug. That's the ground truth, and no annotation is needed.

## Key RXNCONSO fields

```
RXCUI|LAT|TS|LUI|STT|SUI|ISPREF|RXAUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF|
```

Most middle fields are UMLS holdovers that RxNorm leaves empty. The ones that matter:

| Field | Meaning |
|---|---|
| `RXCUI` | Concept ID |
| `RXAUI` | Atom ID |
| `SAB` | Source: who said this name (`VANDF`, `RXNORM`, `MTHSPL`, …) |
| `TTY` | Term type: what kind of name it is (see the ladder below) |
| `CODE` | The source's own identifier (for VANDF, links a product's CD and AB names) |
| `STR` | The name string itself |
| `SUPPRESS` | `N` = active, `O` = obsolete, `Y` = suppressed. Train on `N` only. |

## Sources in this release (`RXNSAB`)

| Restriction level | Sources |
|---|---|
| 0 (free to use) | ATC, CVX, DRUGBANK, MTHCMSFRF, MTHSPL, **RXNORM**, USP, **VANDF** |
| 1 | MMSL |
| 3 | GS, MMX, NDDF |
| 9 | SNOMEDCT_US |

Only level-0 content goes into the published dataset, model, and write-up.

## TTY: the drug ladder

RxNorm describes drugs at increasing specificity:

| TTY | Meaning | Example |
|---|---|---|
| IN | Ingredient | amoxicillin |
| PIN | Precise ingredient (the specific salt) | amoxicillin trihydrate |
| MIN | Multiple ingredients | amoxicillin / clavulanate |
| SCDC | Clinical drug component: ingredient + strength | amoxicillin 250 MG |
| SCDF | Clinical drug form: ingredient + dose form | amoxicillin Oral Capsule |
| **SCD** | **Semantic clinical drug: all three** | **amoxicillin 250 MG Oral Capsule** |
| BN | Brand name | Amoxil |
| **SBD** | **Semantic branded drug** | **amoxicillin 250 MG Oral Capsule [Amoxil]** |
| GPCK / BPCK | Generic / branded pack | Medrol Dose Pack |

**SCD and SBD are the prediction targets.** RxMap and most published tools stop at
IN/MIN, the top rungs. Getting to SCD means resolving ingredient, strength, and dose
form at once, and a miss on any one of them makes the whole answer wrong.

## VANDF term types (what's actually in the file)

| TTY | Active atoms | What it is | Use |
|---|---|---|---|
| `CD` | 21,706 | VA clinical drug name, e.g. `DOXEPIN HCL 10MG CAP` | **Input** |
| `AB` | 21,388 | VA print name (abbreviated), same product as a CD via `CODE`; often identical, sometimes very different (`LUBRICATING (PF) OPH OINT`) | **Input** |
| `IN` | 8,945 | Ingredient | Not an input; lands on IN concepts |
| `PT` | 578 | VA drug class, e.g. `PENICILLIN-G RELATED PENICILLINS` | Exclude |
| `MTH_RXN_CD` | 578 | NLM duplicate names with `_#N` suffixes, e.g. `DEXTROSE 50% INJ_#4` | Exclude |

About half of active CD/AB atoms have no SCD/SBD. Those are mostly supplies,
devices, and nutrition products that are out of RxNorm's normalization scope, and
they're a source of "should abstain" examples.

## Relationships (RXNREL)

```
RXCUI1|RXAUI1|STYPE1|REL|RXCUI2|RXAUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|DIR|RG|SUPPRESS|CVF|
```

`RELA` names the relationship. **Read each row right to left: RXCUI2 → RELA → RXCUI1.**
A row with RXCUI1 = an SCD, RELA = `dose_form_of`, and RXCUI2 = `Oral Tablet` means
"Oral Tablet is the dose form of that SCD." Reading it the other way gives labels that
are silently wrong.

What hangs off `metoprolol tartrate 25 MG Oral Tablet` (an SCD):

| RELA | TTY at the other end | Example |
|---|---|---|
| `constitutes` | SCDC | metoprolol tartrate 25 MG |
| `dose_form_of` | DF | Oral Tablet |
| `inverse_isa` | SCDF, SCDFP, SCDG | metoprolol Oral Tablet; metoprolol Pill |
| `tradename_of` | SBD | its branded versions |

### How each label is derived (`scripts/03_build_dataset.py`)

| Label | Where it comes from |
|---|---|
| Ingredient(s) | SCDC `constitutes` the SCD; IN `ingredient_of` the SCDC (one IN per SCDC) |
| Strength | The SCDC's `RXN_STRENGTH` attribute in RXNSAT, e.g. `25 MG` |
| Dose form | DF `dose_form_of` the SCD |
| Quantity | The SCD's `RXN_QUANTITY` attribute, e.g. `50 ML` (only ~3k drugs have one) |
| Brand | BN `ingredient_of` the SBD |
| SBD → generic | SCD `has_tradename` the SBD; the SBD inherits the SCD's ingredients |

Ingredients are IN, not PIN: `metoprolol`, not `metoprolol tartrate`. Whether the
salt appears in the drug's *name* depends on RxNorm's basis-of-strength rules, but
the ingredient label stays salt-free.
