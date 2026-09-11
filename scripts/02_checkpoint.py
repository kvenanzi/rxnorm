"""Checkpoint: how many VANDF atoms are there, and how many land on an SCD/SBD?

The labels come free from RxNorm's structure: a VANDF atom and an RXNORM
SCD/SBD atom that share an RXCUI are, by NLM's curation, the same drug.

Run from the repo root after 01_load_duckdb.py:  uv run scripts/02_checkpoint.py
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "rxnorm.duckdb"

# Active (SUPPRESS='N') VANDF atoms, left-joined to the RxNorm normalized
# clinical drug name for the same concept. No match -> target columns are NULL.
PAIRS = """
    WITH vandf AS (
        SELECT RXCUI, RXAUI, TTY, STR
        FROM rxnconso
        WHERE SAB = 'VANDF' AND SUPPRESS = 'N'
    ),
    target AS (
        SELECT RXCUI, TTY, STR
        FROM rxnconso
        WHERE SAB = 'RXNORM' AND TTY IN ('SCD', 'SBD') AND SUPPRESS = 'N'
    )
    SELECT
        v.RXCUI,
        v.RXAUI,
        v.TTY AS vandf_tty,
        v.STR AS vandf_string,
        t.TTY AS target_tty,
        t.STR AS rxnorm_target
    FROM vandf v
    LEFT JOIN target t USING (RXCUI)
"""


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute(f"CREATE TEMP VIEW pairs AS {PAIRS}")

    section("All VANDF atoms by term type and status (N=active, O=obsolete)")
    con.sql("""
        SELECT TTY, SUPPRESS, count(*) AS atoms
        FROM rxnconso WHERE SAB = 'VANDF'
        GROUP BY ALL ORDER BY atoms DESC
    """).show()

    section("Active VANDF atoms: where do they land?")
    con.sql("""
        SELECT vandf_tty,
               coalesce(target_tty, '(none)') AS lands_on,
               count(*) AS atoms,
               count(DISTINCT RXCUI) AS concepts
        FROM pairs
        GROUP BY ALL ORDER BY vandf_tty, atoms DESC
    """).show()

    section("Headline numbers")
    con.sql("""
        SELECT count(*) AS active_vandf_atoms,
               count(target_tty) AS atoms_on_scd_sbd,
               count(DISTINCT RXCUI) FILTER (WHERE target_tty IS NOT NULL) AS distinct_targets,
               -- Sanity check: each concept should have at most one SCD/SBD atom,
               -- so the join must not duplicate VANDF atoms.
               count(*) = count(DISTINCT RXAUI) AS no_fanout
        FROM pairs
    """).show()

    section("Sample pairs (VA string -> RxNorm target)")
    # Ordering by a hash gives a stable pseudo-random sample across runs.
    con.sql("""
        SELECT vandf_tty, vandf_string, target_tty, rxnorm_target
        FROM pairs
        WHERE target_tty IS NOT NULL
        ORDER BY hash(RXAUI)
        LIMIT 15
    """).show(max_width=250, max_col_width=70)

    section("Sample unmatched active VANDF atoms (future abstention examples)")
    con.sql("""
        SELECT vandf_tty, vandf_string
        FROM pairs
        WHERE target_tty IS NULL AND vandf_tty IN ('CD', 'AB')
        ORDER BY hash(RXAUI)
        LIMIT 10
    """).show(max_width=250, max_col_width=70)


if __name__ == "__main__":
    main()
