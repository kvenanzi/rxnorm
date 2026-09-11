"""Build the labeled dataset.

Writes three Parquet files to data/processed/:

  pairs.parquet       VANDF string -> RxNorm SCD/SBD, with component labels and split
  candidates.parquet  every active SCD/SBD: the pool the model retrieves from
  unmatched.parquet   active VANDF names with no SCD/SBD: future abstention examples

Run from the repo root after 01_load_duckdb.py:  uv run scripts/03_build_dataset.py
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "rxnorm.duckdb"
OUT_DIR = ROOT / "data" / "processed"

# Splits are assigned per ingredient, not per row, so test drugs contain only
# ingredients the model never saw in training. md5 keeps the assignment stable
# across runs and DuckDB versions; change the salt to draw a different split.
SPLIT_SALT = "rxnorm-2026-09-08-v1"
TEST_PCT, VAL_PCT = 15, 15

# RXNREL reads right to left: RXCUI2 --RELA--> RXCUI1,
# e.g. "Oral Tablet is dose_form_of <SCD>".
BUILD = f"""
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
       CASE WHEN bucket < {TEST_PCT} THEN 'test'
            WHEN bucket < {TEST_PCT + VAL_PCT} THEN 'val'
            ELSE 'train' END AS split
FROM (
    SELECT DISTINCT ingredient_rxcui,
           md5_number(ingredient_rxcui || '{SPLIT_SALT}') % 100 AS bucket
    FROM target_components
);

-- One row per SCD/SBD with its labels. A combination drug whose ingredients
-- fall in different splits is 'mixed': training on it would leak a test
-- ingredient, so it is excluded from train/val/test.
CREATE TEMP TABLE candidates AS
WITH comp AS (
    SELECT rxcui,
           list(ingredient ORDER BY ingredient) AS ingredients,
           list(ingredient_rxcui ORDER BY ingredient) AS ingredient_rxcuis,
           string_agg(strength, ' / ' ORDER BY ingredient) AS strength,
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
SELECT STR, RXCUI, list_sort(list(DISTINCT TTY)) AS vandf_ttys
FROM rxnconso
WHERE SAB = 'VANDF' AND SUPPRESS = 'N' AND TTY IN ('CD', 'AB')
GROUP BY STR, RXCUI;

CREATE TEMP TABLE pairs AS
SELECT v.STR AS vandf_string, v.vandf_ttys,
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
SELECT v.STR AS vandf_string, v.vandf_ttys, v.RXCUI AS rxcui, r.rxnorm_tty,
       CASE WHEN md5_number(v.STR || '{SPLIT_SALT}') % 2 = 0 THEN 'val' ELSE 'test' END AS split
FROM vandf v
LEFT JOIN rxnorm_tty r USING (RXCUI)
WHERE v.RXCUI NOT IN (SELECT rxcui FROM candidates)
  AND v.STR NOT IN (SELECT vandf_string FROM pairs);
"""


def check(con: duckdb.DuckDBPyConnection, sql: str, message: str) -> None:
    if not con.execute(sql).fetchone()[0]:
        raise SystemExit(f"check failed: {message}")


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute(BUILD)

    check(con, "SELECT count(*) = count(DISTINCT rxcui) FROM candidates",
          "each candidate appears once (no join fan-out)")
    check(con, "SELECT count(*) FILTER (WHERE split IS NULL OR dose_form IS NULL) = 0 FROM candidates",
          "every candidate has ingredients and a dose form")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for table in ("pairs", "candidates", "unmatched"):
        path = OUT_DIR / f"{table}.parquet"
        con.execute(f"COPY (SELECT * FROM {table} ORDER BY ALL) TO '{path}' (FORMAT parquet)")
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"wrote {path.relative_to(ROOT)}  ({n:,} rows)")

    print("\nRows per split")
    con.sql("""
        SELECT split,
               count(*) FILTER (WHERE src = 'pairs')      AS pairs,
               count(*) FILTER (WHERE src = 'candidates') AS candidates,
               count(*) FILTER (WHERE src = 'unmatched')  AS unmatched
        FROM (SELECT 'pairs' AS src, split FROM pairs
              UNION ALL SELECT 'candidates', split FROM candidates
              UNION ALL SELECT 'unmatched', split FROM unmatched)
        GROUP BY split
        ORDER BY array_position(['train', 'val', 'test', 'mixed'], split)
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
