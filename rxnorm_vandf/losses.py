"""Training losses beyond the plain contrastive one.

`MNRLWithStrengthHead` is the auxiliary-head arm of the follow-up experiment:
the usual MultipleNegativesRankingLoss on (anchor, positive[, negative]) plus
a linear classifier on the anchor's embedding that has to name the target's
strength ("250 MG", "0.025 MG/ML"). The classifier is a training signal only.
It lives inside the loss module, so the trainer's optimizer updates it (the
trainer builds its parameter groups from the loss), and `model.save()` never
writes it: the saved encoder and the inference path are exactly as before.

Only the public `model(features)` call is used, so the loss runs on whichever
sentence-transformers version Colab ships; the contrastive term is the same
arithmetic as the library's MultipleNegativesRankingLoss (tests check it).
"""

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL = -100   # rows whose strength is too rare for a class: no head loss


def mnrl(anchors: torch.Tensor, candidates: torch.Tensor, scale: float) -> torch.Tensor:
    """In-batch softmax loss (Henderson et al 2017): each anchor against every
    candidate in the batch (positives first, then hard negatives), scaled
    cosine, cross-entropy with its own positive as the target."""
    scores = scale * F.normalize(anchors, dim=-1) @ F.normalize(candidates, dim=-1).T
    return F.cross_entropy(scores, torch.arange(len(anchors), device=scores.device))


class MNRLWithStrengthHead(nn.Module):
    def __init__(self, model, num_labels: int, aux_weight: float, seed: int, scale: float = 20.0):
        super().__init__()
        self.model = model
        self.scale = scale
        # renamed in newer sentence-transformers; Colab may ship either
        dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        dim = dim()
        # The head's random init must not shift the global RNG stream, so the
        # aux=off and aux=on arms of the same seed see the same batches.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Linear(dim, num_labels)
        self.head.to(model.device)
        self.aux_weight = aux_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    def forward(self, sentence_features, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]   # anchor, positive, (negative)
        main = mnrl(embeddings[0], torch.cat(embeddings[1:]), self.scale)
        labels = labels.view(-1).long().to(embeddings[0].device)
        if (labels != IGNORE_LABEL).any():
            aux = self.ce(self.head(embeddings[0]).float(), labels)
        else:
            aux = embeddings[0].sum() * 0.0          # keeps the graph and the head in the optimizer step
        # A dict: the trainer sums the values and logs train/mnrl and train/strength separately.
        return {"mnrl": main, "strength": self.aux_weight * aux}

    def get_config_dict(self) -> dict:
        return {"num_labels": self.head.out_features, "aux_weight": self.aux_weight, "scale": self.scale}
