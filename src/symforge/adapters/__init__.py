"""Modular LoRA Adapter Registry and Local CPU Inference Loader package."""

from __future__ import annotations

from symforge.adapters.exceptions import (
    AdapterAlreadyExistsError,
    AdapterError,
    AdapterInjectionError,
    AdapterMismatchError,
    AdapterNotFoundError,
    AdapterVerificationError,
)
from symforge.adapters.loader import LoRAAdapterLoader
from symforge.adapters.lora_layer import LoRALinearLayer, LoRAModule
from symforge.adapters.registry import (
    DEFAULT_COLLECTIONS,
    AdapterMetadata,
    LoRAAdapterRegistry,
)

__all__ = [
    "AdapterError",
    "AdapterVerificationError",
    "AdapterNotFoundError",
    "AdapterAlreadyExistsError",
    "AdapterMismatchError",
    "AdapterInjectionError",
    "LoRAModule",
    "LoRALinearLayer",
    "LoRAAdapterRegistry",
    "AdapterMetadata",
    "DEFAULT_COLLECTIONS",
    "LoRAAdapterLoader",
]
