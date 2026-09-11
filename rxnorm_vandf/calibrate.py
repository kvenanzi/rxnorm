"""Turn retrieval scores into a confidence you can set a threshold on.

A cosine of 0.84 is not "84% likely to be right". Calibration learns the map
from raw scores to P(correct) on the validation split, so that a threshold
chosen there behaves the same way on test and in deployment.

Signals compared:
  cosine   top-1 similarity, as is
  margin   top-1 minus top-2: how clear the winner is
  softmax  P(top-1) from a temperature-scaled softmax over the top-k scores
  platt    logistic regression on [cosine, margin, log softmax] -> P(correct)
"""

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .eval import Retrieval

SIGNALS = ["cosine", "margin", "softmax", "platt"]
TARGETS = [0.95, 0.99]


def softmax_top1(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)          # max-subtraction: no overflow
    e = np.exp(z)
    return e[:, 0] / e.sum(axis=1)


def fit_temperature(scores: np.ndarray, correct_pos: np.ndarray) -> float:
    """Temperature that minimizes NLL of the correct candidate's softmax
    probability, over rows whose correct answer is somewhere in the top-k."""
    keep = correct_pos >= 0
    s, pos = scores[keep], correct_pos[keep]

    def nll(t):
        z = s / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(pos)), pos].mean()

    return float(minimize_scalar(nll, bounds=(1e-3, 1.0), method="bounded").x)


@dataclass
class Calibrator:
    temperature: float
    platt: LogisticRegression | None = None
    thresholds: dict = field(default_factory=dict)   # population -> {"p95": t, "p99": t}

    def features(self, ret: Retrieval, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
        s = ret.scores if rows is None else ret.scores[rows]
        cosine = s[:, 0]
        margin = s[:, 0] - s[:, 1]
        softmax = softmax_top1(s, self.temperature)
        out = {"cosine": cosine, "margin": margin, "softmax": softmax}
        if self.platt is not None:
            out["platt"] = self.platt.predict_proba(_platt_x(out))[:, 1]
        return out

    def fit_platt(self, feats: dict[str, np.ndarray], correct: np.ndarray) -> None:
        self.platt = LogisticRegression(C=1.0).fit(_platt_x(feats), correct)

    def to_json(self) -> dict:
        return {
            "temperature": self.temperature,
            "platt": {"features": ["cosine", "margin", "log_softmax"],
                      "coef": self.platt.coef_[0].tolist(),
                      "intercept": float(self.platt.intercept_[0])},
            "thresholds": self.thresholds,
        }


def _platt_x(feats: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([feats["cosine"], feats["margin"], np.log(feats["softmax"] + 1e-12)])


# ------------------------------------------------------------- metrics

def correct_position(ret: Retrieval, valid: list[set[int]], rows: np.ndarray) -> np.ndarray:
    """Index of the first correct candidate in each row's top-k, or -1."""
    pos = np.full(len(rows), -1, dtype=np.int64)
    for i, q in enumerate(rows):
        for j, t in enumerate(ret.top[q]):
            if t in valid[q]:
                pos[i] = j
                break
    return pos


def ece(p: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: average |confidence - accuracy| across bins,
    weighted by how many predictions land in each bin. 0 = perfectly calibrated."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - correct[m].mean())
    return float(total)


def reliability_table(p: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[list]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append([f"{edges[b]:.1f}-{edges[b+1]:.1f}", float(p[m].mean()),
                         float(correct[m].mean()), int(m.sum())])
    return rows


def select_threshold(p: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest confidence at which everything at or above it is >= target precise.
    Returns a value above 1 when no threshold achieves the target."""
    order = np.argsort(-p, kind="stable")
    cum_precision = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(cum_precision >= target)
    return float(p[order][ok[-1]]) if len(ok) else 1.01


def at_threshold(p: np.ndarray, correct: np.ndarray, thr: float) -> tuple[float, float]:
    """(coverage, precision) when accepting predictions with confidence >= thr."""
    accept = p >= thr
    coverage = float(accept.mean())
    precision = float(correct[accept].mean()) if accept.any() else float("nan")
    return coverage, precision


def auroc(p: np.ndarray, correct: np.ndarray) -> float:
    if correct.all() or not correct.any():
        return float("nan")
    return float(roc_auc_score(correct, p))


def save_json(cal: Calibrator, path) -> None:
    with open(path, "w") as f:
        json.dump(cal.to_json(), f, indent=2)
