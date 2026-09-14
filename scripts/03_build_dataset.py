"""Build the labeled dataset.

Writes three Parquet files to data/processed/:

  pairs.parquet       VANDF string -> RxNorm SCD/SBD, with component labels and split
  candidates.parquet  every active SCD/SBD: the pool the model retrieves from
  unmatched.parquet   active VANDF names with no SCD/SBD: future abstention examples

Run from the repo root after 01_load_duckdb.py:  uv run scripts/03_build_dataset.py

A different split (for the split-variance experiment, scripts/12_split_seeds.py)
is drawn with a different salt, and must go to its own directory:
  uv run scripts/03_build_dataset.py --salt rxnorm-2026-09-08-v2 --out-dir data/splits/v2

The follow-up experiment (scripts/13_levers.py, scripts/14_kfold.py) adds two
more ways to draw a folder, each of which also needs its own directory:
  --sources VANDF,MTHSPL      also pair FDA label names (MTHSPL, license level 0);
                              pairs.parquet gains a `source` column
  --kfold 7 --fold 2          fold 2 of a 7-fold split by ingredient (test = fold 2,
                              val = fold 3, train = the rest)
  --all-train                 every ingredient in train: the folder for the final model
"""

import argparse
import json
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "rxnorm.duckdb"
OUT_DIR = ROOT / "data" / "processed"
RXNORM_RELEASE = "2026-09-08"

# Splits are assigned per ingredient, not per row, so test drugs contain only
# ingredients the model never saw in training. md5 keeps the assignment stable
# across runs and DuckDB versions; change the salt to draw a different split.
SPLIT_SALT = "rxnorm-2026-09-08-v1"
TEST_PCT, VAL_PCT = 15, 15
SALT_RE = re.compile(r"^[A-Za-z0-9._-]+$")   # the salt is interpolated into SQL literals

# Source vocabularies whose names can be model inputs, with the term types that
# are product names (not ingredients, classes, or NLM-generated duplicates).
# Both are UMLS restriction level 0, like RXNORM itself.
SOURCES = {
    "VANDF": ("CD", "AB"),      # VA clinical drug names and print names
    "MTHSPL": ("DP",),          # FDA Structured Product Label drug products (MTH_RXN_DP are NLM duplicates)
}
DEFAULT_SOURCES = ("VANDF",)


def parse_sources(text: str | tuple[str, ...]) -> tuple[str, ...]:
    srcs = tuple(s.strip().upper() for s in (text.split(",") if isinstance(text, str) else text) if s.strip())
    unknown = [s for s in srcs if s not in SOURCES]
    if unknown or not srcs or srcs[0] != "VANDF":
        raise ValueError(f"sources must start with VANDF and be among {list(SOURCES)}: {srcs}")
    return srcs


def fold_of(bucket: int, kfold: int) -> int:
    """Which of `kfold` folds an ingredient's 0..99 bucket lands in. Contiguous
    ranges of buckets, so with 7 folds and the published salt fold 0 is exactly
    the published test set (buckets 0..14)."""
    return bucket * kfold // 100


def build_sql(salt: str, sources: tuple[str, ...] = DEFAULT_SOURCES, kfold: int | None = None,
              fold: int | None = None, all_train: bool = False) -> str:
    """The SQL that builds the pairs, candidates, and unmatched tables under one
    split salt. With the defaults the text is exactly the published build."""
    if not SALT_RE.match(salt):
        raise ValueError(f"salt must match {SALT_RE.pattern}: {salt!r}")
    sources = parse_sources(sources)
    if (kfold is None) != (fold is None):
        raise ValueError("--kfold and --fold go together")
    if kfold is not None and not 0 <= fold < kfold:
        raise ValueError(f"fold must be in 0..{kfold - 1}")
    if all_train and kfold is not None:
        raise ValueError("--all-train and --kfold are exclusive")

    if all_train:
        split_case = "'train' AS split"
    elif kfold is not None:
        split_case = (f"CASE WHEN bucket * {kfold} // 100 = {fold} THEN 'test'\n"
                      f"            WHEN bucket * {kfold} // 100 = {(fold + 1) % kfold} THEN 'val'\n"
                      "            ELSE 'train' END AS split")
    else:
        split_case = (f"CASE WHEN bucket < {TEST_PCT} THEN 'test'\n"
                      f"            WHEN bucket < {TEST_PCT + VAL_PCT} THEN 'val'\n"
                      "            ELSE 'train' END AS split")

    multi = len(sources) > 1
    src_col = "'VANDF' AS source, " if multi else ""
    v_src = "v.source, " if multi else ""
    extra_sources = "".join(
        f"\nUNION ALL\nSELECT '{src}' AS source, STR, RXCUI, list_sort(list(DISTINCT TTY))\n"
        f"FROM rxnconso\nWHERE SAB = '{src}' AND SUPPRESS = 'N' AND TTY IN ({', '.join(repr(t) for t in SOURCES[src])})\n"
        "GROUP BY STR, RXCUI"
        for src in sources[1:])
    not_paired = ("NOT EXISTS (SELECT 1 FROM pairs p WHERE p.source = v.source AND p.vandf_string = v.STR)"
                  if multi else "v.STR NOT IN (SELECT vandf_string FROM pairs)")
    # RXNREL reads right to left: RXCUI2 --RELA--> RXCUI1,
    # e.g. "Oral Tablet is dose_form_of <SCD>".
    return f"""
-- Active RxNorm normalized names we need, one row per concept.
CREATE TEMP TABLE rx AS
SELECT RXCUI, TTY, STR FROM rxnconso
WHERE SAB = 'RXNORM' AND SUPPRESS = 'N'
  AND TTY IN ('SCD', 'SBD', 'SCDC', 'IN', 'DF', 'BN');

-- Each SCD's components: SCD <- SCDC (ingredient + strength) <- IN.
CREATE TEMP TABLE scd_components AS
SELECT DISTINCT scd.RXCUI AS scd_rxcui, ing.RXCUI AS ingredient_rxcui,
       ing.STR AS ingredient, st.ATV AS strength
FROM rx scd
JOIN rxnrel c   ON c.RXCUI1 = scd.RXCUI AND c.RELA = 'constitutes' AND c.SAB = 'RXNORM'
JOIN rx scdc    ON scdc.RXCUI = c.RXCUI2 AND scdc.TTY = 'SCDC'
JOIN rxnrel i   ON i.RXCUI1 = scdc.RXCUI AND i.RELA = 'ingredient_of' AND i.SAB = 'RXNORM'
JOIN rx ing     ON ing.RXCUI = i.RXCUI2 AND ing.TTY = 'IN'
LEFT JOIN rxnsat st ON st.RXCUI = scdc.RXCUI AND st.SAB = 'RXNORM' AND st.ATN = 'RXN_STRENGTH'
WHERE scd.TTY = 'SCD';

-- An SBD inherits its generic SCD's components, which also keeps a brand
-- and its generic in the same split.
CREATE TEMP TABLE sbd_scd AS
SELECT DISTINCT r.RXCUI1 AS sbd_rxcui, r.RXCUI2 AS scd_rxcui
FROM rxnrel r
JOIN rx sbd ON sbd.RXCUI = r.RXCUI1 AND sbd.TTY = 'SBD'
JOIN rx scd ON scd.RXCUI = r.RXCUI2 AND scd.TTY = 'SCD'
WHERE r.SAB = 'RXNORM' AND r.RELA = 'has_tradename';

CREATE TEMP TABLE target_components AS
SELECT scd_rxcui AS rxcui, ingredient_rxcui, ingredient, strength FROM scd_components
UNION ALL
SELECT s.sbd_rxcui, c.ingredient_rxcui, c.ingredient, c.strength
FROM sbd_scd s JOIN scd_components c USING (scd_rxcui);

CREATE TEMP TABLE ingredient_split AS
SELECT ingredient_rxcui,
       {split_case}
FROM (
    SELECT DISTINCT ingredient_rxcui,
           md5_number(ingredient_rxcui || '{salt}') % 100 AS bucket
    FROM target_components
);

-- One row per SCD/SBD with its labels. A combination drug whose ingredients
-- fall in different splits is 'mixed': training on it would leak a test
-- ingredient, so it is excluded from train/val/test.
CREATE TEMP TABLE candidates AS
WITH comp AS (
    SELECT rxcui,
           -- A combination vaccine can list one ingredient several times with
           -- different strengths; the extra sort keys make the labels deterministic.
           list(ingredient ORDER BY ingredient, ingredient_rxcui, strength) AS ingredients,
           list(ingredient_rxcui ORDER BY ingredient, ingredient_rxcui, strength) AS ingredient_rxcuis,
           string_agg(strength, ' / ' ORDER BY ingredient, ingredient_rxcui, strength) AS strength,
           CASE WHEN count(DISTINCT split) = 1 THEN any_value(split) ELSE 'mixed' END AS split
    FROM target_components JOIN ingredient_split USING (ingredient_rxcui)
    GROUP BY rxcui
),
dose_form AS (
    SELECT DISTINCT r.RXCUI1 AS rxcui, df.STR AS dose_form
    FROM rxnrel r JOIN rx df ON df.RXCUI = r.RXCUI2 AND df.TTY = 'DF'
    WHERE r.SAB = 'RXNORM' AND r.RELA = 'dose_form_of'
),
brand AS (
    SELECT DISTINCT r.RXCUI1 AS rxcui, bn.STR AS brand
    FROM rxnrel r JOIN rx bn ON bn.RXCUI = r.RXCUI2 AND bn.TTY = 'BN'
    WHERE r.SAB = 'RXNORM' AND r.RELA = 'ingredient_of'
),
quantity AS (
    -- e.g. the "50 ML" in "50 ML mannitol 250 MG/ML Injection"
    SELECT DISTINCT RXCUI AS rxcui, ATV AS quantity
    FROM rxnsat WHERE SAB = 'RXNORM' AND ATN = 'RXN_QUANTITY'
)
SELECT t.RXCUI AS rxcui, t.TTY AS tty, t.STR AS name,
       comp.ingredients, comp.ingredient_rxcuis, comp.strength,
       dose_form.dose_form, quantity.quantity, brand.brand, comp.split
FROM rx t
LEFT JOIN comp      ON comp.rxcui = t.RXCUI
LEFT JOIN dose_form ON dose_form.rxcui = t.RXCUI
LEFT JOIN quantity  ON quantity.rxcui = t.RXCUI
LEFT JOIN brand     ON brand.rxcui = t.RXCUI AND t.TTY = 'SBD'
WHERE t.TTY IN ('SCD', 'SBD');

-- Model inputs: VA clinical drug names (CD) and print names (AB). CD and AB
-- strings are often identical, so dedupe on (string, target).
CREATE TEMP TABLE vandf AS
SELECT {src_col}STR, RXCUI, list_sort(list(DISTINCT TTY)) AS vandf_ttys
FROM rxnconso
WHERE SAB = 'VANDF' AND SUPPRESS = 'N' AND TTY IN ('CD', 'AB')
GROUP BY STR, RXCUI{extra_sources};

CREATE TEMP TABLE pairs AS
SELECT {v_src}v.STR AS vandf_string, v.vandf_ttys,
       c.rxcui AS target_rxcui, c.tty AS target_tty, c.name AS target_name,
       c.ingredients, c.ingredient_rxcuis, c.strength, c.dose_form,
       c.quantity, c.brand, c.split
FROM vandf v JOIN candidates c ON c.rxcui = v.RXCUI;

-- Active VA names with no SCD/SBD. rxnorm_tty records what, if anything,
-- RxNorm has for that concept instead (e.g. a pack), which separates hard
-- abstentions (real drugs) from easy ones (catheters). Split 50/50 so val
-- can tune the abstention threshold and test can report it.
CREATE TEMP TABLE unmatched AS
WITH rxnorm_tty AS (
    SELECT RXCUI, string_agg(DISTINCT TTY, ',' ORDER BY TTY) AS rxnorm_tty
    FROM rxnconso
    WHERE SAB = 'RXNORM' AND SUPPRESS = 'N' AND TTY NOT IN ('PSN', 'SY', 'TMSY', 'ET')
    GROUP BY RXCUI
)
SELECT {v_src}v.STR AS vandf_string, v.vandf_ttys, v.RXCUI AS rxcui, r.rxnorm_tty,
       CASE WHEN md5_number(v.STR || '{salt}') % 2 = 0 THEN 'val' ELSE 'test' END AS split
FROM vandf v
LEFT JOIN rxnorm_tty r USING (RXCUI)
WHERE v.RXCUI NOT IN (SELECT rxcui FROM candidates)
  AND {not_paired};
"""


def rel(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def check(con: duckdb.DuckDBPyConnection, sql: str, message: str) -> None:
    if not con.execute(sql).fetchone()[0]:
        raise SystemExit(f"check failed: {message}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salt", default=SPLIT_SALT, help=f"split hash salt (default: {SPLIT_SALT})")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where the three parquet files go (default: data/processed)")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help=f"input vocabularies, comma-separated, from {list(SOURCES)} (default: VANDF)")
    ap.add_argument("--kfold", type=int, default=None, help="number of folds by ingredient (with --fold)")
    ap.add_argument("--fold", type=int, default=None, help="which fold is the test set (0-based)")
    ap.add_argument("--all-train", action="store_true", help="every ingredient in train: no val, no test")
    ap.add_argument("--force", action="store_true",
                    help="allow a non-default draw to overwrite data/processed")
    args = ap.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    args.sources = parse_sources(args.sources)
    # data/processed is the published split: every model, figure, and test reads
    # it. A different draw there would silently change every downstream number.
    non_default = (args.salt != SPLIT_SALT or args.sources != DEFAULT_SOURCES
                   or args.kfold is not None or args.all_train)
    if non_default and args.out_dir == OUT_DIR.resolve() and not args.force:
        raise SystemExit(f"refusing to write a non-default draw (salt {args.salt!r}, sources {args.sources}, "
                         f"kfold {args.kfold}, all_train {args.all_train}) into {OUT_DIR.relative_to(ROOT)}: "
                         "pass --out-dir <new directory> (or --force if you really mean it)")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute(build_sql(args.salt, args.sources, args.kfold, args.fold, args.all_train))

    check(con, "SELECT count(*) = count(DISTINCT rxcui) FROM candidates",
          "each candidate appears once (no join fan-out)")
    check(con, "SELECT count(*) FILTER (WHERE split IS NULL OR dose_form IS NULL) = 0 FROM candidates",
          "every candidate has ingredients and a dose form")

    out_dir.mkdir(parents=True, exist_ok=True)
    for table in ("pairs", "candidates", "unmatched"):
        path = out_dir / f"{table}.parquet"
        con.execute(f"COPY (SELECT * FROM {table} ORDER BY ALL) TO '{path}' (FORMAT parquet)")
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"wrote {rel(path)}  ({n:,} rows)")
    # The directory describes its own split, so a dataset folder is never anonymous.
    meta = {"salt": args.salt, "test_pct": TEST_PCT, "val_pct": VAL_PCT, "rxnorm_release": RXNORM_RELEASE}
    if args.sources != DEFAULT_SOURCES:
        meta["sources"] = list(args.sources)
    if args.kfold is not None:
        meta.update({"kfold": args.kfold, "fold": args.fold, "test_pct": None, "val_pct": None})
    if args.all_train:
        meta.update({"all_train": True, "test_pct": 0, "val_pct": 0})
    (out_dir / "split.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\nRows per split")
    by_source = ", source" if len(args.sources) > 1 else ""
    con.sql(f"""
        SELECT split{by_source},
               count(*) FILTER (WHERE src = 'pairs')      AS pairs,
               count(*) FILTER (WHERE src = 'candidates') AS candidates,
               count(*) FILTER (WHERE src = 'unmatched')  AS unmatched
        FROM (SELECT 'pairs' AS src, split{by_source} FROM pairs
              UNION ALL SELECT 'candidates', split{", 'VANDF'" if by_source else ""} FROM candidates
              UNION ALL SELECT 'unmatched', split{by_source} FROM unmatched)
        GROUP BY split{by_source}
        ORDER BY array_position(['train', 'val', 'test', 'mixed'], split){by_source}
    """).show()

    print("Unmatched: what RxNorm has for those concepts instead")
    con.sql("""
        SELECT coalesce(rxnorm_tty, '(nothing)') AS rxnorm_tty, count(*) AS atoms
        FROM unmatched GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """).show()

    print("Sample pairs with their component labels")
    con.sql("""
        SELECT vandf_string, target_name, ingredients, strength, dose_form, quantity, split
        FROM pairs ORDER BY md5(vandf_string) LIMIT 8
    """).show(max_width=250, max_col_width=45)


if __name__ == "__main__":
    main()
