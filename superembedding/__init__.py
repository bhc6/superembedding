"""Attribute embeddings read from OmniJev's option-conditioned states."""

from .heads import AttributeBasis, EmbeddingOutput, OptionEmbeddingHead
from .losses import attribute_triplet_loss

__all__ = [
    "AttributeBasis",
    "EmbeddingOutput",
    "OptionEmbeddingHead",
    "attribute_triplet_loss",
]
