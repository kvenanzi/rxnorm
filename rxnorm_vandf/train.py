"""Fine-tune a bi-encoder to map VA strings to RxNorm clinical drugs.

The same `train(cfg)` runs on a local GPU (scripts/06_train.py) and in Colab
(notebooks/01_train_biencoder.ipynb). Evaluation reuses rxnorm_vandf.eval so the
run is directly comparable with the baselines in the W&B runs table.
"""

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import DATASET_ARTIFACT, MODEL_ARTIFACT, WANDB_PROJECT
from .data import Data, load_data, normalize
from .eval import TOP_K, Retrieval, evaluate, evaluate_splits, log_results, print_summary
from .strength import normalize_strength
from .tfidf import tfidf_retrieve


@dataclass
class TrainConfig:
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    epochs: int = 4
    batch_size: int = 64
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 96
    # Where the third item of each training triplet comes from:
    #   ingredient  a train-split product with the same ingredients (different strength/form)
    #   tfidf       the highest-scoring wrong train-split candidate by char n-gram TF-IDF
    #   none        in-batch negatives only
    negatives: str = "ingredient"
    normalize_text: bool = True       # lowercase / strip punctuation, as the baselines do
    normalize_strength: bool = False  # append RxNorm-style strength to VA strings (see strength.py)
    log_model: bool = True            # sweeps set False: one artifact per trial is storage for nothing
    seed: int = 42
    smoke: bool = False               # 256 pairs, 1 epoch: proves the pipeline end to end
    data_dir: str | None = None       # local data/processed; None = download the W&B artifact
    output_dir: str = "models"
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    group: str | None = None          # W&B run group, e.g. "split-seeds"; None = ungrouped
    # Which dataset artifact a run consumes (lineage when data_dir is set; the
    # download when it is not), and an optional folder inside it.
    dataset_artifact: str = f"{DATASET_ARTIFACT}:latest"
    dataset_subdir: str | None = None


def auto_name(cfg: "TrainConfig") -> str:
    return (f"{cfg.base_model.split('/')[-1].lower()}-{cfg.negatives}"
            + ("-strength" if cfg.normalize_strength else "")
            + ("-smoke" if cfg.smoke else ""))


def preprocess(s: str, cfg: "TrainConfig", query: bool) -> str:
    """Text as the model sees it. Strength normalization applies to VA strings
    (queries) only; candidate names are already in RxNorm form."""
    if query and cfg.normalize_strength:
        s = normalize_strength(s)
    return normalize(s) if cfg.normalize_text else s


# ------------------------------------------------------------------ training data

def build_triplets(data: Data, cfg: TrainConfig, rng: random.Random) -> dict[str, list[str]]:
    """Training rows: anchor (VA string), positive (its RxNorm name), and, unless
    negatives == 'none', one hard negative drawn only from train-split candidates."""
    text = lambda s: preprocess(s, cfg, query=False)
    rows = data.rows("train")
    if cfg.smoke:
        rows = rows[:256]

    train_ids = np.flatnonzero(data.cand["split"].values == "train")
    train_pool = set(train_ids.tolist())
    names = data.cand["name"]

    # Candidates grouped by ingredient set, restricted to the train pool.
    by_ingredients: dict[tuple, list[int]] = {}
    for i in train_ids:
        by_ingredients.setdefault(data.cand_key[i][0], []).append(int(i))

    anchors, positives, negatives = [], [], []
    tfidf_hits = None
    if cfg.negatives in ("tfidf", "ingredient"):
        # TF-IDF over the train pool: the primary miner for 'tfidf', the fallback
        # for 'ingredient' when a drug is the only product with those ingredients.
        queries = [data.queries["vandf_string"][q] for q in rows]
        ret = tfidf_retrieve([names[i] for i in train_ids], queries, top_k=10)
        tfidf_hits = train_ids[ret.top]      # map pool-relative ids back to candidate ids

    n_fallback = 0
    for k, q in enumerate(rows):
        valid = data.valid[q]
        pos = rng.choice(sorted(valid))
        anchors.append(preprocess(data.queries["vandf_string"][q], cfg, query=True))
        positives.append(text(names[pos]))
        if cfg.negatives == "none":
            continue
        neg = None
        if cfg.negatives == "ingredient":
            same = [i for i in by_ingredients.get(data.cand_key[pos][0], []) if i not in valid]
            if same:
                neg = rng.choice(same)
        if neg is None:
            wrong = [int(i) for i in tfidf_hits[k] if i not in valid and i in train_pool]
            neg = wrong[0] if wrong else rng.choice(sorted(train_pool - valid))
            n_fallback += cfg.negatives == "ingredient"
        negatives.append(text(names[neg]))

    out = {"anchor": anchors, "positive": positives}
    if negatives:
        out["negative"] = negatives
    print(f"{len(anchors):,} training triplets; negatives={cfg.negatives}"
          + (f" ({n_fallback:,} fell back to tfidf)" if cfg.negatives == "ingredient" else ""))
    print(f"example anchor: {anchors[0]!r}")
    return out


# --------------------------------------------------------------------- retrieval

@torch.no_grad()
def encode(model, texts: list[str], cfg: TrainConfig, query: bool, batch_size: int = 256) -> torch.Tensor:
    return model.encode([preprocess(t, cfg, query) for t in texts], batch_size=batch_size,
                        convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)


@torch.no_grad()
def retrieve_texts(model, data: Data, texts: list[str], cfg: TrainConfig, top_k: int = TOP_K,
                   cand_emb: torch.Tensor | None = None) -> Retrieval:
    """Top-k candidates for arbitrary VA strings, by cosine (dot product of unit
    vectors). Pass `cand_emb` to reuse candidate embeddings across calls."""
    C = cand_emb if cand_emb is not None else encode(model, data.cand["name"].tolist(), cfg, query=False)
    Q = encode(model, texts, cfg, query=True)
    top = np.empty((len(texts), top_k), dtype=np.int64)
    scores = np.empty((len(texts), top_k), dtype=np.float32)
    for start in range(0, len(Q), 4096):
        s, i = torch.topk(Q[start : start + 4096] @ C.T, k=top_k, dim=1)
        top[start : start + 4096] = i.cpu().numpy()
        scores[start : start + 4096] = s.float().cpu().numpy()
    return Retrieval(top, scores)


@torch.no_grad()
def retrieve(model, data: Data, cfg: TrainConfig, rows: np.ndarray | None = None,
             top_k: int = TOP_K, cand_emb: torch.Tensor | None = None) -> Retrieval:
    """Retrieval for the dataset's queries. Arrays are always indexed by global
    query id; with `rows`, only those queries are computed and the rest stay
    empty (-1 / 0)."""
    all_queries = data.queries["vandf_string"].tolist()
    if rows is None:
        rows = np.arange(len(all_queries))
    part = retrieve_texts(model, data, [all_queries[i] for i in rows], cfg, top_k, cand_emb)
    top = np.full((len(all_queries), top_k), -1, dtype=np.int64)
    scores = np.zeros((len(all_queries), top_k), dtype=np.float32)
    top[rows], scores[rows] = part.top, part.scores
    return Retrieval(top, scores)


# ---------------------------------------------------------------------- training

def train(cfg: TrainConfig, sweep: bool = False) -> Path:
    """Train, evaluate, and log one run. With `sweep=True` the W&B agent has
    already started the run and `wandb.config` carries this trial's parameters."""
    import wandb
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback

    offline = os.environ.get("WANDB_MODE") == "offline"
    tags = cfg.tags + (["smoke"] if cfg.smoke else [])
    if sweep:
        run = wandb.init(job_type="train", tags=tags)
        for k, v in dict(wandb.config).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        run.name = cfg.run_name or auto_name(cfg)
        run.config.update(asdict(cfg), allow_val_change=True)
    else:
        run = wandb.init(project=WANDB_PROJECT, job_type="train", name=cfg.run_name or auto_name(cfg),
                         config=asdict(cfg), tags=tags, group=cfg.group)

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    # Data: the local parquet files, or the versioned artifact (Colab).
    if cfg.data_dir:
        data_dir = Path(cfg.data_dir)
        if not offline:
            run.use_artifact(cfg.dataset_artifact)     # lineage only
    else:
        data_dir = Path(run.use_artifact(cfg.dataset_artifact).download())
    if cfg.dataset_subdir:
        data_dir = data_dir / cfg.dataset_subdir
    data = load_data(data_dir)
    split_meta = data_dir / "split.json"     # written by 03_build_dataset.py; absent in older folders
    if split_meta.exists():
        run.config.update({"split_salt": json.loads(split_meta.read_text())["salt"]}, allow_val_change=True)
    print(f"{len(data.queries):,} VA strings, {len(data.cand):,} candidates; "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    train_ds = Dataset.from_dict(build_triplets(data, cfg, rng))

    model = SentenceTransformer(cfg.base_model)
    model.max_seq_length = cfg.max_seq_length
    # Each anchor is scored against its own positive, its hard negative, and every
    # other row's positive and negative in the batch; cross-entropy over those.
    loss = losses.MultipleNegativesRankingLoss(model)

    out_dir = Path(cfg.output_dir) / run.name
    best_dir = out_dir / "best"
    val_rows = data.rows("val")
    if cfg.smoke:
        val_rows = val_rows[:200]
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    class ValEval(TrainerCallback):
        """After each epoch: full-pool retrieval on val, log it, keep the best model."""
        best = -1.0

        def on_epoch_end(self, args, state, control, **kwargs):
            ret = retrieve(model, data, cfg, val_rows)
            ev = evaluate(data, ret, val_rows, ret.scores[val_rows, 0])
            m = ev["metrics"]
            wandb.log({"epoch": state.epoch, **{f"val/{k}": v for k, v in m.items() if k != "n"}})
            print(f"epoch {state.epoch:.0f}: val acc@1 {m['acc@1']:.3f}  recall@5 {m['recall@5']:.3f}")
            if m["acc@1"] > self.best:
                self.best = m["acc@1"]
                model.save(str(best_dir))
                # Inference must preprocess queries the way training did.
                (best_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    import transformers
    # transformers v5 expresses a warmup *ratio* as a float warmup_steps; v4 (Colab
    # may still ship it) keeps the warmup_ratio argument.
    warmup = ({"warmup_steps": cfg.warmup_ratio} if int(transformers.__version__.split(".")[0]) >= 5
              else {"warmup_ratio": cfg.warmup_ratio})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1 if cfg.smoke else cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        **warmup,
        fp16=torch.cuda.is_available(),
        # Never put two rows with the same text in one batch: the other row's
        # positive would be treated as a negative for this anchor.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=10,
        save_strategy="no",
        report_to=["wandb"],
        run_name=run.name,
        seed=cfg.seed,
        dataloader_drop_last=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds,
                                         loss=loss, callbacks=[ValEval()])
    trainer.train()

    # Final evaluation with the best epoch, logged exactly like the baselines.
    best = SentenceTransformer(str(best_dir))
    best.max_seq_length = cfg.max_seq_length
    ret = retrieve(best, data, cfg)
    results = evaluate_splits(data, ret, ret.scores[:, 0], ret.scores[:, 0] - ret.scores[:, 1])
    log_results(run, run.name, data, ret, results)
    print_summary(run.name, results)

    if cfg.log_model:
        artifact = wandb.Artifact(MODEL_ARTIFACT, type="model", metadata={
            **asdict(cfg), "test/acc@1": results["test"]["metrics"]["acc@1"],
            "test/recall@5": results["test"]["metrics"]["recall@5"]})
        artifact.add_dir(str(best_dir))
        run.log_artifact(artifact)
    run.finish()
    return best_dir
