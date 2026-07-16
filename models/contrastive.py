"""Shared contrastive-learning primitives for stages 2 and 3."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_image_text_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float | None = None,
    logit_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Compute the bidirectional image-to-text/text-to-image InfoNCE loss.

    Stage 2 passes a learnable ``logit_scale``. Stage 3 passes a fixed
    ``temperature``. Exactly one of the two scaling options must be supplied.
    Positive pairs are expected to occupy matching positions in the batch.
    """

    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("image and text embeddings must both be rank-2 tensors")
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            "image and text embeddings must have identical shapes, got "
            f"{tuple(image_embeddings.shape)} and {tuple(text_embeddings.shape)}"
        )
    if (temperature is None) == (logit_scale is None):
        raise ValueError("provide exactly one of temperature or logit_scale")
    if temperature is not None and temperature <= 0:
        raise ValueError("temperature must be positive")

    image_features = F.normalize(image_embeddings, p=2, dim=1)
    text_features = F.normalize(text_embeddings, p=2, dim=1)
    scale = logit_scale if logit_scale is not None else 1.0 / temperature
    logits = scale * image_features @ text_features.T
    labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)

    image_to_text = F.cross_entropy(logits, labels)
    text_to_image = F.cross_entropy(logits.T, labels)
    return (image_to_text + text_to_image) / 2
