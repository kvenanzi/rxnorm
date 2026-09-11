"""Load the RxNorm RRF files into a local DuckDB database.

RRF ("Rich Release Format") is pipe-delimited with no header row, and every
line ends with a trailing pipe. A CSV reader therefore sees one extra, always
empty column per file; we name it TRAILING_PIPE and drop it.

Run from the repo root:  uv run scripts/01_load_duckdb.py
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RRF_DIR = ROOT / "data" / "rxnorm_2026_09_08" / "rrf"
DB_PATH = ROOT / "data" / "rxnorm.duckdb"

# Column layouts from the RxNorm technical documentation (also described in RXNDOC.RRF).
TABLES = {
    # One row per atom: a single drug name from a single source vocabulary.
    "rxnconso": [
        "RXCUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "RXAUI", "SAUI",
        "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF",
    ],
    # Relationships between concepts/atoms (has_ingredient, has_dose_form, ...).
    "rxnrel": [
        "RXCUI1", "RXAUI1", "STYPE1", "REL", "RXCUI2", "RXAUI2", "STYPE2",
        "RELA", "RUI", "SRUI", "SAB", "SL", "DIR", "RG", "SUPPRESS", "CVF",
    ],
    # Attributes: NDCs, strengths, prescribable flags, and other metadata.
    "rxnsat": [
        "RXCUI", "LUI", "SUI", "RXAUI", "STYPE", "CODE", "ATUI", "SATUI",
        "ATN", "SAB", "ATV", "SUPPRESS", "CVF",
    ],
    # One row per source vocabulary, including its license restriction level (SRL).
    "rxnsab": [
        "VCUI", "RCUI", "VSAB", "RSAB", "SON", "SF", "SVER", "VSTART", "VEND",
        "IMETA", "RMETA", "SLC", "SCC", "SRL", "TFR", "CFR", "CXTY", "TTYL",
        "ATNL", "LAT", "CENC", "CURVER", "SABIN", "SSN", "SCIT",
    ],
}


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def load_table(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> None:
    path = RRF_DIR / f"{table.upper()}.RRF"
    # Everything is VARCHAR: RXCUIs are identifiers, not numbers, and must join as text.
    col_spec = ", ".join(f"'{c}': 'VARCHAR'" for c in [*columns, "TRAILING_PIPE"])
    # quote='' and escape='' turn off quote handling entirely. Drug names contain
    # quote characters, and a quote-aware parser silently merges or splits rows.
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * EXCLUDE (TRAILING_PIPE)
        FROM read_csv('{path}', delim='|', header=false, quote='', escape='',
                      columns={{{col_spec}}})
    """)

    # The loud check that the parse was clean: one table row per file line.
    rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    lines = count_lines(path)
    status = "ok" if rows == lines else "MISMATCH"
    print(f"{table:<9} {rows:>11,} rows   {lines:>11,} lines   {status}")
    if rows != lines:
        raise SystemExit(f"{table}: parsed row count does not match file line count")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    for table, columns in TABLES.items():
        load_table(con, table, columns)
    con.close()
    print(f"\nWrote {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
