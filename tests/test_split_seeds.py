"""The split-variance tooling, without the DuckDB or a GPU: the salt reaches the
SQL, the interlock protects data/processed, and the statistics helper is right."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(script: str):
    """Import a numbered script (its name is not an identifier) as a module."""
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[script] = mod
    spec.loader.exec_module(mod)
    return mod


build = load("03_build_dataset")
seeds = load("12_split_seeds")


def test_salt_reaches_both_hashes_and_nothing_else_changes():
    a, b = build.build_sql("rxnorm-2026-09-08-v1"), build.build_sql("rxnorm-2026-09-08-v2")
    assert a.count("'rxnorm-2026-09-08-v1'") == 2      # ingredient buckets and the unmatched 50/50
    assert b.count("'rxnorm-2026-09-08-v2'") == 2
    assert a.replace("v1'", "vX'") == b.replace("v2'", "vX'")


def test_salt_is_validated_before_it_meets_sql():
    with pytest.raises(ValueError):
        build.build_sql("x'; DROP TABLE rxnconso; --")


def test_refuses_new_salt_into_published_directory():
    with pytest.raises(SystemExit):
        build.parse_args(["--salt", "rxnorm-2026-09-08-v2"])
    args = build.parse_args(["--salt", "rxnorm-2026-09-08-v2", "--out-dir", "data/splits/v2"])
    assert args.out_dir == (ROOT / "data" / "splits" / "v2").resolve()
    assert build.parse_args([]).salt == build.SPLIT_SALT                      # the published default
    assert build.parse_args(["--salt", "rxnorm-2026-09-08-v2", "--force"]).force


def test_job_matrix():
    assert len(seeds.JOBS) == 8
    assert seeds.SPLIT_JOBS == [f"split-v{i}" for i in range(1, 7)]
    assert set(seeds.SEED_JOBS) == {"split-v1", "seed-1", "seed-2"}
    assert seeds.SPLITS["v1"] == build.SPLIT_SALT


def test_describe():
    d = seeds.describe([0.92, 0.94, 0.93], [1000, 1000, 1000])
    assert d["n_runs"] == 3 and d["min"] == 0.92 and d["max"] == 0.94
    assert math.isclose(d["mean"], 0.93) and math.isclose(d["sd"], 0.01) and math.isclose(d["range"], 0.02)
    # Binomial sampling sd at p≈0.93, n=1000 is about 0.008; the excess is what is left.
    assert math.isclose(d["sampling_sd"], math.sqrt(sum(p * (1 - p) / 1000 for p in (0.92, 0.94, 0.93)) / 3))
    assert math.isclose(d["excess_sd"], math.sqrt(0.01 ** 2 - d["sampling_sd"] ** 2))
    one = seeds.describe([0.5])
    assert one["sd"] == 0.0 and "excess_sd" not in one
    # Observed spread below the sampling floor clamps to zero rather than going imaginary.
    assert seeds.describe([0.930, 0.931], [1000, 1000])["excess_sd"] == 0.0
