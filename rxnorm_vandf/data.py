"""Load the processed dataset (output of scripts/03_build_dataset.py)."""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_PUNCT = re.compile(r"[^a-z0-9%./]+")


def normalize(s: str) -> str:
    """Lowercase, turn punctuation into spaces, collapse whitespace.
    Keeps % . / because they carry strength information (0.5%, 10MG/ML)."""
    return " ".join(_PUNCT.sub(" ", s.lower()).split())


@dataclass
class Data:
    cand: pd.DataFrame            # one row per SCD/SBD; row index = candidate id
    queries: pd.DataFrame         # one row per distinct VA string: vandf_string, split, targets
    valid: list[set[int]]         # per query: candidate ids that count as correct
    cand_key: list[tuple]         # per candidate: (ingredient rxcuis, strength, dose_form)

    def rows(self, split: str):
        """Query indexes belonging to one split."""
        import numpy as np
        return np.flatnonzero(self.queries["split"].values == split)


def load_data(processed_dir: Path) -> Data:
    processed_dir = Path(processed_dir)
    cand = pd.read_parquet(processed_dir / "candidates.parquet").reset_index(drop=True)
    pairs = pd.read_parquet(processed_dir / "pairs.parquet")

    rxcui_to_id = {r: i for i, r in enumerate(cand["rxcui"])}
    # A few VA strings map to more than one concept (an albuterol inhaler with
    # three NDA-specific SCDs). Any of them counts as correct.
    grouped = (
        pairs.groupby("vandf_string", sort=False)
        .agg(split=("split", "first"), targets=("target_rxcui", list))
        .reset_index()
    )
    valid = [{rxcui_to_id[r] for r in ts} for ts in grouped["targets"]]
    cand_key = list(zip(
        cand["ingredient_rxcuis"].map(lambda x: tuple(sorted(x))),
        cand["strength"].fillna(""),
        cand["dose_form"].fillna(""),
    ))
    return Data(cand=cand, queries=grouped, valid=valid, cand_key=cand_key)
