"""The follow-up tooling (docs/post-2) without the DuckDB, W&B, or a GPU: the
dataset builder's new draws, the multi-source data layer, the strength-head
loss, the k-fold plumbing, and the statistics."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from test_split_seeds import ROOT, load

build = load("03_build_dataset")
levers = load("13_levers")
kfold = load("14_kfold")


# ------------------------------------------------------------- 03_build_dataset

def test_default_build_has_no_source_column_and_two_salt_literals():
    sql = build.build_sql("rxnorm-2026-09-08-v1")
    assert sql.count("'rxnorm-2026-09-08-v1'") == 2
    assert "source" not in sql and "MTHSPL" not in sql
    assert "bucket < 15 THEN 'test'" in sql


def test_sources_add_an_mthspl_block_and_a_source_column():
    sql = build.build_sql("rxnorm-2026-09-08-v1", ("VANDF", "MTHSPL"))
    assert sql.count("'rxnorm-2026-09-08-v1'") == 2
    assert "SAB = 'MTHSPL' AND SUPPRESS = 'N' AND TTY IN ('DP')" in sql     # never MTH_RXN_DP
    assert "'VANDF' AS source" in sql and "'MTHSPL' AS source" in sql
    assert "p.source = v.source AND p.vandf_string = v.STR" in sql          # unmatched is per source
    assert build.parse_sources("vandf, mthspl") == ("VANDF", "MTHSPL")
    for bad in ("MTHSPL", "VANDF,SNOMEDCT_US", ""):
        with pytest.raises(ValueError):
            build.parse_sources(bad)


def test_kfold_case_and_wraparound():
    sql = build.build_sql("s", kfold=7, fold=0)
    assert "bucket * 7 // 100 = 0 THEN 'test'" in sql and "bucket * 7 // 100 = 1 THEN 'val'" in sql
    sql6 = build.build_sql("s", kfold=7, fold=6)
    assert "= 6 THEN 'test'" in sql6 and "= 0 THEN 'val'" in sql6
    head = build.build_sql("s", all_train=True)
    assert "'train' AS split" in head and "THEN 'test'" not in head.split("-- One row per SCD")[0]
    with pytest.raises(ValueError):
        build.build_sql("s", kfold=7)
    with pytest.raises(ValueError):
        build.build_sql("s", kfold=7, fold=7)
    with pytest.raises(ValueError):
        build.build_sql("s", kfold=7, fold=0, all_train=True)


def test_fold_of_matches_the_published_test_buckets():
    assert [build.fold_of(b, 7) for b in range(15)] == [0] * 15          # fold 0 = the published test set
    assert build.fold_of(15, 7) == 1 and build.fold_of(99, 7) == 6
    sizes = [sum(build.fold_of(b, 7) == f for b in range(100)) for f in range(7)]
    assert sizes == [15, 14, 14, 15, 14, 14, 14]


def test_interlock_covers_every_non_default_draw():
    for argv in (["--sources", "VANDF,MTHSPL"], ["--kfold", "7", "--fold", "0"], ["--all-train"]):
        with pytest.raises(SystemExit):
            build.parse_args(argv)
        assert build.parse_args(argv + ["--out-dir", "data/x"]).out_dir == (ROOT / "data" / "x").resolve()
    assert build.parse_args([]).sources == ("VANDF",)


# ----------------------------------------------------------------- data layer

def _write_folder(tmp_path: Path, with_source: bool) -> Path:
    cand = pd.DataFrame({"rxcui": ["1", "2", "3"], "tty": ["SCD"] * 3, "name": ["a 1 MG", "a 2 MG", "b 1 MG"],
                         "ingredients": [["a"], ["a"], ["b"]], "ingredient_rxcuis": [["10"], ["10"], ["20"]],
                         "strength": ["1 MG", "2 MG", "1 MG"], "dose_form": ["Oral Tablet"] * 3,
                         "quantity": [None] * 3, "brand": [None] * 3, "split": ["train", "train", "test"]})
    pairs = pd.DataFrame({"vandf_string": ["A 1MG TAB", "A 2MG TAB", "B 1MG TAB", "A 1 mg [x]"],
                          "target_rxcui": ["1", "2", "3", "1"], "split": ["train", "train", "test", "train"],
                          "strength": ["1 MG", "2 MG", "1 MG", "1 MG"]})
    if with_source:
        pairs["source"] = ["VANDF", "VANDF", "VANDF", "MTHSPL"]
    else:
        pairs = pairs.iloc[:3]
    cand.to_parquet(tmp_path / "candidates.parquet")
    pairs.to_parquet(tmp_path / "pairs.parquet")
    return tmp_path


def test_load_data_defaults_old_folders_to_vandf(tmp_path):
    from rxnorm_vandf.data import load_data
    d = load_data(_write_folder(tmp_path, with_source=False))
    assert d.sources == ("VANDF",) and list(d.queries["source"].unique()) == ["VANDF"]
    assert list(d.rows("train")) == list(d.rows("train", None)) == [0, 1]
    assert list(d.rows("test")) == [2]


def test_load_data_keeps_val_test_keys_vandf_only(tmp_path):
    from rxnorm_vandf.data import load_data
    from rxnorm_vandf.eval import Retrieval, evaluate_splits
    d = load_data(_write_folder(tmp_path, with_source=True))
    assert d.sources == ("VANDF", "MTHSPL")
    assert list(d.rows("train")) == [0, 1] and list(d.rows("train", "MTHSPL")) == [3]
    assert list(d.rows("train", None)) == [0, 1, 3]
    top = np.array([[0, 1, 2, 2, 2], [1, 0, 2, 2, 2], [0, 2, 1, 1, 1], [0, 1, 2, 2, 2]])
    ret = Retrieval(top, np.ones_like(top, dtype=np.float32))
    res = evaluate_splits(d, ret, ret.scores[:, 0])
    assert set(res) == {"train", "test"}            # no val rows at all, no MTHSPL val/test rows: no keys for them
    assert res["test"]["metrics"]["n"] == 1 and res["test"]["metrics"]["acc@1"] == 0.0


def test_load_data_drops_strings_whose_targets_span_splits(tmp_path, capsys):
    from rxnorm_vandf.data import load_data
    _write_folder(tmp_path, with_source=True)
    pairs = pd.read_parquet(tmp_path / "pairs.parquet")
    # the MTHSPL string also maps to the test-split product: ambiguous, dropped
    extra = pd.DataFrame({"vandf_string": ["A 1 mg [x]"], "target_rxcui": ["3"], "split": ["test"], "strength": ["1 MG"], "source": ["MTHSPL"]})
    pd.concat([pairs, extra]).to_parquet(tmp_path / "pairs.parquet")
    d = load_data(tmp_path)
    assert len(d.queries) == 3 and "MTHSPL" not in set(d.queries["source"])
    assert "dropping 1 query strings" in capsys.readouterr().out
    assert d.sources == ("VANDF",)


def test_train_config_back_compat_and_sources_normalization():
    from rxnorm_vandf.train import TrainConfig, auto_name
    old = json.loads((ROOT / "tests" / "fixtures" / "train_config_v3.json").read_text())
    cfg = TrainConfig(**old)
    assert cfg.train_sources == ("VANDF",) and cfg.aux == "none" and cfg.model_artifact == "vandf-rxnorm-biencoder"
    assert auto_name(cfg) == "sapbert-from-pubmedbert-fulltext-ingredient-strength"
    assert TrainConfig(train_sources="VANDF,MTHSPL").train_sources == ("VANDF", "MTHSPL")
    assert TrainConfig(train_sources=["vandf", "mthspl"]).train_sources == ("VANDF", "MTHSPL")
    c = TrainConfig(train_sources="VANDF,MTHSPL", aux="strength", seed=1, dataset_subdir="v3", max_extra_pairs="9300")
    assert c.max_extra_pairs == 9300
    assert auto_name(c) == "all-minilm-l6-v2-ingredient-mthspl-cap9300-aux-strength-v3-s1"


def test_training_rows_and_labels(tmp_path):
    import random
    from rxnorm_vandf.data import load_data
    from rxnorm_vandf.train import TrainConfig, build_triplets
    d = load_data(_write_folder(tmp_path, with_source=True))
    cols, info = build_triplets(d, TrainConfig(negatives="none", train_sources="VANDF,MTHSPL", aux="strength",
                                               aux_min_count=2), random.Random(0))
    assert info["n_train_rows"] == 3 and info["n_train_rows_mthspl"] == 1
    assert cols["label"] == [0, -100, 0]             # "1 MG" twice -> class 0; "2 MG" once -> ignored
    assert info["aux/num_labels"] == 1 and math.isclose(info["aux/label_coverage"], 2 / 3)
    cols, info = build_triplets(d, TrainConfig(negatives="none", train_sources="VANDF,MTHSPL", max_extra_pairs=0),
                                random.Random(0))
    assert info["n_train_rows"] == 2 and "label" not in cols


# ----------------------------------------------------------------------- loss

def test_strength_head_loss_owns_its_parameters():
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import StaticEmbedding
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from rxnorm_vandf.losses import IGNORE_LABEL, MNRLWithStrengthHead

    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(["[UNK]", "a", "b", "c", "1", "2", "mg"])}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    model = SentenceTransformer(modules=[StaticEmbedding(tok, embedding_dim=8)], device="cpu")
    loss = MNRLWithStrengthHead(model, num_labels=3, aux_weight=0.5, seed=42)
    assert any(n.startswith("head.") for n, _ in loss.named_parameters())
    feats = [model.tokenize(["a 1 mg", "b 2 mg"]), model.tokenize(["a 1 mg", "b 2 mg"]), model.tokenize(["c 1 mg", "a 2 mg"])]
    out = loss(feats, torch.tensor([0, 2]))
    assert set(out) == {"mnrl", "strength"} and all(torch.isfinite(v) for v in out.values())
    # the contrastive term is the library's MultipleNegativesRankingLoss, to the float
    from sentence_transformers import losses
    ref = losses.MultipleNegativesRankingLoss(model)(feats, None)
    assert torch.isclose(out["mnrl"], ref, atol=1e-6), (out["mnrl"], ref)
    out2 = loss(feats, torch.tensor([IGNORE_LABEL, IGNORE_LABEL]))
    assert float(out2["strength"]) == 0.0 and torch.isfinite(out2["mnrl"])
    # same seed, same head; the head's init does not disturb the global stream
    torch.manual_seed(0); before = torch.rand(1)
    torch.manual_seed(0); MNRLWithStrengthHead(model, 3, 0.5, seed=42); after = torch.rand(1)
    assert torch.equal(before, after)
    assert torch.equal(MNRLWithStrengthHead(model, 3, 0.5, seed=7).head.weight, MNRLWithStrengthHead(model, 3, 0.5, seed=7).head.weight)


# ---------------------------------------------------------------------- stats

def test_paired_diffs_and_wilson():
    from rxnorm_vandf.stats import T_975, paired_diffs, paired_stats, wilson
    d = paired_diffs({"u1": 0.02, "u2": 0.01, "u3": 0.03})
    assert d["n_pos"] == 3 and math.isclose(d["mean"], 0.02) and math.isclose(d["ci95"][1] - d["mean"], 4.303 * d["se"])
    assert T_975[11] == 2.201                          # twelve (split, seed) units
    lo, hi = wilson(930, 1000)
    assert 0.912 < lo < 0.915 and 0.944 < hi < 0.946
    assert all(math.isnan(v) for v in wilson(0, 0))
    cells = {(("v1", 42), "base"): {"m": 0.90}, (("v1", 42), "mthspl"): {"m": 0.92},
             (("v1", 1), "base"): {"m": 0.91}, (("v1", 1), "mthspl"): {"m": 0.92}}
    st = paired_stats(cells, ["m"], [("mthspl", "base")])
    assert st["mthspl-base"]["m"]["n_runs"] == 2 and st["mthspl-base"]["m"]["n_pos"] == 2


def test_levers_analysis_main_effects_and_interaction():
    cells = {}
    for u, (b, m, a, both) in {("v1", 42): (0.90, 0.92, 0.91, 0.94), ("v2", 42): (0.88, 0.89, 0.89, 0.90)}.items():
        for arm, v in zip(("base", "mthspl", "aux", "both"), (b, m, a, both)):
            cells[(u, arm)] = {k: v for k in levers.METRICS} | {"test/n": 1, "val/n": 1, "test_mthspl/n": 1}
    st = levers.analyse(cells)
    main = st["main"]["test/acc@1"]
    assert math.isclose(main["data"]["diffs"]["v1/s42"], ((0.92 - 0.90) + (0.94 - 0.91)) / 2)
    assert math.isclose(main["head"]["diffs"]["v2/s42"], ((0.89 - 0.88) + (0.90 - 0.89)) / 2)
    assert math.isclose(main["interaction"]["diffs"]["v1/s42"], 0.94 - 0.92 - 0.91 + 0.90)
    assert st["units"] == ["v1/s42", "v2/s42"]
    json.dumps(st)                                     # every key is JSON-safe: summarize writes this
    assert levers.arm_of({"train_sources": ["VANDF", "MTHSPL"], "aux": "strength"}) == "both"
    assert levers.arm_of({"train_sources": "VANDF"}) == "base"


def test_kfold_constants_and_unmatched_assignment():
    assert kfold.K == 7 and kfold.FOLDS[0] == "fold0" and kfold.SALT == build.SPLIT_SALT
    folds = {kfold.unmatched_fold(s) for s in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")}
    assert folds <= set(range(7)) and len(folds) > 1
    assert kfold.unmatched_fold("CATHETER") == kfold.unmatched_fold("CATHETER")
    assert kfold.run_name("fold3").endswith("kfold-fold3") and kfold.run_name("all") == kfold.FINAL_RUN


def test_correct_pos_and_features_from_predictions():
    from rxnorm_vandf.calibrate import Calibrator
    df = pd.DataFrame({"top_rxcuis": [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]],
                       "truth_rxcuis": [["2"], [], ["7"]],
                       "scores": [[0.9, 0.8, 0.1], [0.5, 0.4, 0.3], [0.95, 0.2, 0.1]]})
    assert kfold.correct_pos(df).tolist() == [1, -1, 0]
    f = kfold.features(df, Calibrator(temperature=0.05))
    assert set(f) == {"cosine", "margin", "softmax"} and math.isclose(f["margin"][0], 0.1, abs_tol=1e-6)


# ---------------------------------------------------------------------- sweeps

def test_levers_sweeps_are_well_formed():
    sweep = load("08_sweep")
    for name in ("levers", "levers_control_steps", "levers_control_size", "levers_rerun", "levers_rerun_head",
                 "levers_rerun_head_v1s1", "levers_rerun_head_v2s42"):
        cfg = yaml.safe_load((ROOT / "sweeps" / f"{name}.yaml").read_text())
        assert isinstance(cfg["parameters"]["lr"]["value"], float)
        srcs = cfg["parameters"]["train_sources"]
        assert all(isinstance(v, str) for v in srcs.get("values", [srcs.get("value")]))
        sm = sweep.smoke_config(cfg)
        assert sm["parameters"]["smoke"] == {"value": True}
    grid = yaml.safe_load((ROOT / "sweeps" / "levers.yaml").read_text())["parameters"]
    n = 1
    for spec in grid.values():
        n *= len(spec.get("values", [1]))
    assert n == 48


def test_levers_head_makeups_cover_the_lost_cells_with_the_grid_settings():
    """nypttmi8 lost six head cells (v1/s42, v1/s1, v2/s42, both data arms); the
    make-up grids re-issue exactly those, every fixed setting equal to the grid's."""
    import itertools
    grid = yaml.safe_load((ROOT / "sweeps" / "levers.yaml").read_text())["parameters"]
    cells = set()
    for name in ("levers_rerun_head", "levers_rerun_head_v1s1", "levers_rerun_head_v2s42"):
        p = yaml.safe_load((ROOT / "sweeps" / f"{name}.yaml").read_text())["parameters"]
        assert set(p) == set(grid), name
        for k, spec in grid.items():
            if "value" in spec:
                assert p[k] == spec, (name, k)
        vals = lambda k: p[k].get("values", [p[k].get("value")])
        cells |= set(itertools.product(vals("dataset_subdir"), vals("seed"), vals("train_sources"), vals("aux")))
    assert cells == {(k, s, src, "strength") for k, s in (("v1", 42), ("v1", 1), ("v2", 42))
                     for src in ("VANDF", "VANDF,MTHSPL")}


def test_kfold_head_weight_defaults_to_the_grid_weight():
    import argparse
    p = argparse.ArgumentParser()
    kfold.add_recipe_flags(p)
    grid = yaml.safe_load((ROOT / "sweeps" / "levers.yaml").read_text())["parameters"]
    assert p.parse_args([]).aux_weight == grid["aux_weight"]["value"]
