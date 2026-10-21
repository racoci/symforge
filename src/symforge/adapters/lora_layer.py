"""Modular Low-Rank Adaptation (LoRA) layer implementation for PyTorch modules."""

from __future__ import annotations
import abc
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from symforge.adapters.exceptions import AdapterMismatchError

logger = logging.getLogger(__name__)


class LoRAModule(nn.Module, abc.ABC):
    """Abstract interface for all adapted neural modules."""

    @abc.abstractmethod
    def attach_adapter(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        rank: int,
        alpha: float,
        adapter_name: Optional[str] = None,
    ) -> None:
        """Attach adapter weight matrices to the module."""

    @abc.abstractmethod
    def detach_adapter(self) -> None:
        """Detach active adapter, restoring standard unadapted base forward pass."""

    @abc.abstractmethod
    def hot_swap_adapter(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        rank: int,
        alpha: float,
        adapter_name: Optional[str] = None,
    ) -> None:
        """Hot-swap active adapter weights in-place without rebuilding the layer."""


class LoRALinearLayer(LoRAModule):
    """Wraps PyTorch nn.Linear layer to apply modular low-rank adaptation:

        y = W_0 x + (alpha / r) * B (A x)

    Base weights W_0 (and bias b_0) remain strictly frozen (requires_grad=False).
    Adapter weights A and B can be attached, detached, or hot-swapped dynamically.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        adapter_name: Optional[str] = None,
    ) -> None:
        """Initialize LoRALinearLayer wrapping a frozen base linear layer."""
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)}")

        # Freeze base layer parameters
        base_layer.weight.requires_grad = False
        if base_layer.bias is not None:
            base_layer.bias.requires_grad = False

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.adapter_name: Optional[str] = adapter_name

        self._is_adapted: bool = False
        self._is_enabled: bool = True

        # Buffers for LoRA weights (A: r x in_features, B: out_features x r)
        self.register_buffer("lora_A", None)
        self.register_buffer("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        adapter_name: Optional[str] = None,
    ) -> LoRALinearLayer:
        """Factory method constructing LoRALinearLayer from an existing nn.Linear."""
        return cls(base_layer=base_linear, rank=rank, alpha=alpha, adapter_name=adapter_name)

    @property
    def is_adapted(self) -> bool:
        """Return True if adapter weights are currently attached."""
        return self._is_adapted and self.lora_A is not None and self.lora_B is not None

    @property
    def is_enabled(self) -> bool:
        """Return True if attached adapter delta is actively computed."""
        return self._is_enabled

    @property
    def scaling(self) -> float:
        """Return the LoRA scaling coefficient gamma = alpha / r."""
        if self.rank <= 0:
            return 1.0
        return self.alpha / float(self.rank)

    def enable_adapter(self, enabled: bool = True) -> None:
        """Enable or disable adapter evaluation without detaching weights."""
        self._is_enabled = bool(enabled)

    def _validate_adapter_shapes(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        rank: int,
    ) -> None:
        """Validate shapes of incoming LoRA A and B tensor matrices."""
        expected_a = (rank, self.in_features)
        expected_b = (self.out_features, rank)

        if tuple(lora_a.shape) != expected_a:
            raise AdapterMismatchError(
                f"LoRA A shape mismatch for layer with in_features={self.in_features}: "
                f"expected {expected_a}, got {tuple(lora_a.shape)}"
            )
        if tuple(lora_b.shape) != expected_b:
            raise AdapterMismatchError(
                f"LoRA B shape mismatch for layer with out_features={self.out_features}: "
                f"expected {expected_b}, got {tuple(lora_b.shape)}"
            )

    def attach_adapter(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        rank: int,
        alpha: float,
        adapter_name: Optional[str] = None,
    ) -> None:
        """Attach LoRA weights A and B to the linear layer."""
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0.0, got {alpha}")

        self._validate_adapter_shapes(lora_a, lora_b, rank)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.adapter_name = adapter_name

        # Register detached tensors as buffers to avoid memory leaks
        target_device = self.base_layer.weight.device
        target_dtype = self.base_layer.weight.dtype

        self.register_buffer("lora_A", lora_a.detach().to(device=target_device, dtype=target_dtype))
        self.register_buffer("lora_B", lora_b.detach().to(device=target_device, dtype=target_dtype))
        self._is_adapted = True
        self._is_enabled = True

    def detach_adapter(self) -> None:
        """Detach adapter and explicitly sever tensor references to prevent memory accumulation."""
        self.lora_A = None
        self.lora_B = None
        self._is_adapted = False
        self.adapter_name = None

    def hot_swap_adapter(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        rank: int,
        alpha: float,
        adapter_name: Optional[str] = None,
    ) -> None:
        """Hot-swap active adapter matrices in-place without rebuilding the layer."""
        self.attach_adapter(
            lora_a=lora_a,
            lora_b=lora_b,
            rank=rank,
            alpha=alpha,
            adapter_name=adapter_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = W_0 x + (alpha / r) * B (A x) if adapted and enabled."""
        base_out = self.base_layer(x)

        if not self.is_adapted or not self._is_enabled:
            return base_out

        # F.linear(x, A) computes x @ A^T -> shape (*, r)
        # F.linear(..., B) computes (...) @ B^T -> shape (*, out_features)
        lora_delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_delta

    def extra_repr(self) -> str:
        """Return descriptive string representation."""
        status = "adapted" if self.is_adapted else "unadapted"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"status={status}, adapter_name={self.adapter_name}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )
