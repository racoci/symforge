"""Differentiable Contrastive Perceptual Loss (InfoNCE) engine for SymForge synthesis."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_infonce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute InfoNCE contrastive perceptual loss between query, positive target, and negative distractors.

    Formula:
        L = -log( exp(sim(q, pos) / tau) / (exp(sim(q, pos) / tau) + sum_j exp(sim(q, neg_j) / tau)) )

    Args:
        query: Candidate embedding tensor of shape (D,) or (B, D).
        positive: Ground-truth positive target embedding of shape (D,) or (B, D).
        negatives: Negative distractor embeddings of shape (K, D) or (B, K, D).
        temperature: Softmax temperature parameter tau > 0.
        normalize: If True, L2-normalizes embeddings along the feature dimension before dot product.

    Returns:
        Scalar tensor containing the contrastive InfoNCE loss.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")

    # Standardize dimensions to 2D for batch/feature: (B, D)
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if positive.ndim == 1:
        positive = positive.unsqueeze(0)

    # Standardize negatives to (B, K, D) or (K, D)
    if negatives.ndim == 1:
        negatives = negatives.unsqueeze(0)

    # Normalize if requested
    if normalize:
        q = F.normalize(query, p=2, dim=-1, eps=1e-8)
        pos = F.normalize(positive, p=2, dim=-1, eps=1e-8)
        negs = F.normalize(negatives, p=2, dim=-1, eps=1e-8)
    else:
        q, pos, negs = query, positive, negatives

    # Positive cosine/dot similarity: (B, 1)
    sim_pos = torch.sum(q * pos, dim=-1, keepdim=True) / temperature

    # Negative similarities: (B, K)
    if negs.ndim == 2:
        # negs is (K, D), broadcast across batch B
        sim_neg = torch.matmul(q, negs.transpose(-2, -1)) / temperature
    elif negs.ndim == 3:
        # negs is (B, K, D)
        sim_neg = torch.bmm(q.unsqueeze(1), negs.transpose(-2, -1)).squeeze(1) / temperature
    else:
        raise ValueError(f"Unsupported negative distractors shape: {negatives.shape}")

    # Concatenate positive logit (index 0) with negative logits: (B, 1 + K)
    logits = torch.cat([sim_pos, sim_neg], dim=-1)

    # Target class index is always 0 (the positive target)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


class ContrastivePerceptualLoss(nn.Module):
    """PyTorch module implementing the InfoNCE contrastive perceptual loss."""

    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss."""
        return compute_infonce_loss(
            query=query,
            positive=positive,
            negatives=negatives,
            temperature=self.temperature,
            normalize=self.normalize,
        )


class CompositeSynthesisObjective(nn.Module):
    """Composite optimization objective combining contrastive perceptual loss, reconstruction loss, and hardware cost.

    J(G, theta) = lambda_contrast * L_contrast + lambda_rec * L_rec + lambda_cost * C_H(G)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_contrast: float = 1.0,
        lambda_rec: float = 0.5,
        lambda_cost: float = 0.01,
        normalize: bool = True,
    ):
        super().__init__()
        self.contrastive_loss = ContrastivePerceptualLoss(
            temperature=temperature, normalize=normalize
        )
        self.lambda_contrast = lambda_contrast
        self.lambda_rec = lambda_rec
        self.lambda_cost = lambda_cost

    def forward(
        self,
        query_latent: torch.Tensor,
        target_latent: torch.Tensor,
        negatives_latent: torch.Tensor,
        reconstruction_loss: Optional[torch.Tensor] = None,
        hardware_cost: float = 0.0,
    ) -> torch.Tensor:
        """Evaluate total composite loss."""
        loss = self.lambda_contrast * self.contrastive_loss(
            query_latent, target_latent, negatives_latent
        )
        if reconstruction_loss is not None and self.lambda_rec > 0.0:
            loss = loss + self.lambda_rec * reconstruction_loss
        if hardware_cost > 0.0 and self.lambda_cost > 0.0:
            loss = loss + self.lambda_cost * hardware_cost
        return loss
