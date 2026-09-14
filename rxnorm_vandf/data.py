"""Load the processed dataset (output of scripts/03_build_dataset.py)."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_PUNCT = re.compile(r"[^a-z0-9%./]+")
PRIMARY_SOURCE = "VANDF"


def normalize(s: str) -> str:
    """Lowercase, turn punctuation into spaces, collapse whitespace.
    Keeps % . / because they carry strength information (0.5%, 10MG/ML)."""
    return " ".join(_PUNCT.sub(" ", s.lower()).split())


@dataclass
class Data:
    cand: pd.DataFrame            # one row per SCD/SBD; row index = candidate id
    queries: pd.DataFrame         # one row per distinct (source, string): source, vandf_string, split, targets
    valid: list[set[int]]         # per query: candidate ids that count as correct
    cand_key: list[tuple]         # per candidate: (ingredient rxcuis, strength, dose_form)
    sources: tuple[str, ...] = (PRIMARY_SOURCE,)   # distinct query sources, VANDF first

    def rows(self, split: str, source: str | None = PRIMARY_SOURCE) -> np.ndarray:
        """Query indexes belonging to one split. By default only the VA strings,
        so every published number keeps its meaning when a folder also carries
        another source's strings; `source=None` means all sources."""
        m = self.queries["split"].values == split
        if source is not None:
            m &= self.queries["source"].values == source
        return np.flatnonzero(m)


def load_data(processed_dir: Path) -> Data:
    processed_dir = Path(processed_dir)
    cand = pd.read_parquet(processed_dir / "candidates.parquet").reset_index(drop=True)
    pairs = pd.read_parquet(processed_dir / "pairs.parquet")
    if "source" not in pairs.columns:          # folders built before multi-source support
        pairs["source"] = PRIMARY_SOURCE

    rxcui_to_id = {r: i for i, r in enumerate(cand["rxcui"])}
    # A few VA strings map to more than one concept (an albuterol inhaler with
    # three NDA-specific SCDs). Any of them counts as correct.
    grouped = (
        pairs.groupby(["source", "vandf_string"], sort=False)
        .agg(split=("split", "first"), n_splits=("split", "nunique"), targets=("target_rxcui", list))
        .reset_index()
    )
    if (grouped["n_splits"] > 1).any():
        raise ValueError("a query string maps to targets in different splits")
    grouped = grouped.drop(columns="n_splits")
    valid = [{rxcui_to_id[r] for r in ts} for ts in grouped["targets"]]
    cand_key = list(zip(
        cand["ingredient_rxcuis"].map(lambda x: tuple(sorted(x))),
        cand["strength"].fillna(""),
        cand["dose_form"].fillna(""),
    ))
    present = list(dict.fromkeys(grouped["source"]))
    sources = tuple([PRIMARY_SOURCE] * (PRIMARY_SOURCE in present) + [s for s in present if s != PRIMARY_SOURCE])
    return Data(cand=cand, queries=grouped, valid=valid, cand_key=cand_key, sources=sources)
