"""Training losses beyond the plain contrastive one.

`MNRLWithStrengthHead` is the auxiliary-head arm of the follow-up experiment:
the usual MultipleNegativesRankingLoss on (anchor, positive[, negative]) plus
a linear classifier on the anchor's embedding that has to name the target's
strength ("250 MG", "0.025 MG/ML"). The classifier is a training signal only.
It lives inside the loss module, so the trainer's optimizer updates it (the
trainer builds its parameter groups from the loss), and `model.save()` never
writes it: the saved encoder and the inference path are exactly as before.
"""

import torch
from torch import nn

IGNORE_LABEL = -100   # rows whose strength is too rare for a class: no head loss


class MNRLWithStrengthHead(nn.Module):
    def __init__(self, model, num_labels: int, aux_weight: float, seed: int, scale: float = 20.0):
        from sentence_transformers import losses

        super().__init__()
        self.model = model
        self.mnrl = losses.MultipleNegativesRankingLoss(model, scale=scale)
        dim = model.get_sentence_embedding_dimension()
        # The head's random init must not shift the global RNG stream, so the
        # aux=off and aux=on arms of the same seed see the same batches.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Linear(dim, num_labels)
        self.head.to(model.device)
        self.aux_weight = aux_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    def forward(self, sentence_features, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        from sentence_transformers.base.losses.merged_forward import embed_columns

        embeddings = embed_columns(self.model, sentence_features)   # [anchor, positive, (negative)]
        main = self.mnrl.compute_loss_from_embeddings(embeddings, None)
        labels = labels.view(-1).long().to(embeddings[0].device)
        if (labels != IGNORE_LABEL).any():
            aux = self.ce(self.head(embeddings[0]).float(), labels)
        else:
            aux = embeddings[0].sum() * 0.0          # keeps the graph and the head in the optimizer step
        # A dict: the trainer sums the values and logs train/mnrl and train/strength separately.
        return {"mnrl": main, "strength": self.aux_weight * aux}

    def get_config_dict(self) -> dict:
        return {"num_labels": self.head.out_features, "aux_weight": self.aux_weight, "scale": self.mnrl.scale}
