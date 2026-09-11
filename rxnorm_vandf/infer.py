"""Map VA drug strings to RxNorm clinical drugs with a published model.

    from rxnorm_vandf.infer import Mapper
    mapper = Mapper.from_pretrained("kvenanzi/vandf-rxnorm-biencoder")
    for p in mapper.map(["METOPROLOL TARTRATE 12.5MG TAB", "CATHETER,FOLEY SILICONE 22FR 5CC"]):
        print(p.rxcui, p.name, f"{p.confidence:.2f}", "accept" if p.accept else "review")

A model directory (local or a Hugging Face repo) holds the sentence-transformers
model plus three files written at publish time: `train_config.json` (how to
preprocess inputs), `calibration.json` (score -> probability), and
`candidates.parquet` (the 27,287 RxNorm SCD/SBD names to search).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .calibrate import softmax_top1
from .train import TrainConfig, encode, preprocess


@dataclass
class Prediction:
    query: str
    rxcui: str
    name: str
    tty: str                        # SCD (generic) or SBD (branded)
    confidence: float               # calibrated P(correct), 0-1
    accept: bool                    # confidence >= threshold
    alternatives: list[tuple[str, str, float]]   # (rxcui, name, cosine) for ranks 2..k


class Mapper:
    def __init__(self, model, cfg: TrainConfig, calibration: dict, candidates: pd.DataFrame,
                 top_k_scores: int = 20):
        self.model = model
        self.cfg = cfg
        self.calibration = calibration
        self.candidates = candidates.reset_index(drop=True)
        self.top_k_scores = top_k_scores      # what the softmax was fit over
        self.cand_emb = encode(model, self.candidates["name"].tolist(), cfg, query=False)

    @classmethod
    def from_pretrained(cls, path_or_repo: str, device: str | None = None) -> "Mapper":
        from sentence_transformers import SentenceTransformer

        path = Path(path_or_repo)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(path_or_repo))
        cfg = TrainConfig(**json.loads((path / "train_config.json").read_text()))
        model = SentenceTransformer(str(path), device=device)
        model.max_seq_length = cfg.max_seq_length
        return cls(model, cfg, json.loads((path / "calibration.json").read_text()),
                   pd.read_parquet(path / "candidates.parquet"))

    @property
    def default_threshold(self) -> float:
        # Chosen on validation for 99% precision on strings that may have no answer.
        return self.calibration["thresholds"]["+hard"]["p99"]

    def _confidence(self, scores: np.ndarray) -> np.ndarray:
        """Platt layer from calibration.json: sigmoid(w . [cosine, margin, log softmax] + b)."""
        cosine = scores[:, 0]
        margin = scores[:, 0] - scores[:, 1]
        log_softmax = np.log(softmax_top1(scores, self.calibration["temperature"]) + 1e-12)
        w = np.asarray(self.calibration["platt"]["coef"])
        z = np.column_stack([cosine, margin, log_softmax]) @ w + self.calibration["platt"]["intercept"]
        return 1.0 / (1.0 + np.exp(-z))

    @torch.no_grad()
    def map(self, strings: list[str], top_k: int = 5, threshold: float | None = None) -> list[Prediction]:
        threshold = self.default_threshold if threshold is None else threshold
        Q = encode(self.model, list(strings), self.cfg, query=True)
        k = max(top_k, self.top_k_scores)
        scores, idx = torch.topk(Q @ self.cand_emb.T, k=k, dim=1)
        scores, idx = scores.float().cpu().numpy(), idx.cpu().numpy()
        confidence = self._confidence(scores)
        out = []
        for q, s in enumerate(strings):
            rows = self.candidates.iloc[idx[q, :top_k]]
            best = rows.iloc[0]
            out.append(Prediction(
                query=s, rxcui=best["rxcui"], name=best["name"], tty=best["tty"],
                confidence=float(confidence[q]), accept=bool(confidence[q] >= threshold),
                alternatives=[(r["rxcui"], r["name"], float(scores[q, j + 1]))
                              for j, (_, r) in enumerate(rows.iloc[1:].iterrows())]))
        return out
