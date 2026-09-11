"""Character n-gram TF-IDF retrieval: the baseline, and the fallback hard-negative miner."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import normalize
from .eval import TOP_K, Retrieval


def tfidf_retrieve(cand_names: list[str], queries: list[str], top_k: int = TOP_K,
                   ngram_range=(3, 5), batch: int = 2000) -> Retrieval:
    """Top-k candidates per query by cosine similarity of char n-gram TF-IDF vectors.
    Candidate ids in the result index into `cand_names`."""
    # char_wb builds n-grams inside word boundaries, so "12.5mg" and "125mg"
    # share fewer fragments than plain char n-grams would give them.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True)
    C = vec.fit_transform([normalize(s) for s in cand_names])   # rows are L2-normalized
    Q = vec.transform([normalize(s) for s in queries])
    top = np.empty((len(queries), top_k), dtype=np.int64)
    scores = np.empty((len(queries), top_k), dtype=np.float32)
    for start in range(0, len(queries), batch):
        # Rows are unit vectors, so the dot product is the cosine similarity.
        sim = (Q[start : start + batch] @ C.T).toarray()
        part = np.argpartition(-sim, top_k - 1, axis=1)[:, :top_k]
        part_scores = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        top[start : start + batch] = np.take_along_axis(part, order, axis=1)
        scores[start : start + batch] = np.take_along_axis(part_scores, order, axis=1)
    return Retrieval(top, scores)
