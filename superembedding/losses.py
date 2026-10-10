"""A small training primitive for attribute-labelled image triplets."""

import torch
from torch.nn import functional as F


def attribute_triplet_loss(anchor, positive, negative, attribute_ids, margin=0.2):
    """Cosine margin loss on one labelled attribute per object triplet.

    Inputs are normalized slots [B,K,d], and integer attribute_ids [B].
    The dataset must supply a positive sharing the selected attribute value
    and a negative differing in that value. Other slots receive no direct loss.
    """
    if anchor.ndim != 3 or positive.shape != anchor.shape or negative.shape != anchor.shape:
        raise ValueError("anchor, positive and negative must have matching [B,K,d] shapes")
    if any(size == 0 for size in anchor.shape):
        raise ValueError("triplet batches and slot dimensions must be nonempty")
    if attribute_ids.shape != (anchor.shape[0],) or attribute_ids.dtype != torch.long:
        raise ValueError("attribute_ids must be an int64 tensor of shape [B]")
    if not 0 < margin <= 2:
        raise ValueError("cosine margin must be in (0, 2]")
    if (attribute_ids < 0).any() or (attribute_ids >= anchor.shape[1]).any():
        raise ValueError("attribute_ids contains an invalid slot index")
    ids = attribute_ids.to(anchor.device)
    rows = torch.arange(anchor.shape[0], device=anchor.device)
    a, p, n = anchor[rows, ids], positive[rows, ids], negative[rows, ids]
    pos_sim = F.cosine_similarity(a, p, dim=-1)
    neg_sim = F.cosine_similarity(a, n, dim=-1)
    return F.relu(margin + neg_sim - pos_sim).mean()
