"""Modular LoRA Adapter Loader for dynamic in-place attachment, detaching, and hot-swapping."""

from __future__ import annotations
import hashlib
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import torch
import torch.nn as nn

from symforge.adapters.exceptions import (
    AdapterInjectionError,
    AdapterMismatchError,
    AdapterVerificationError,
)
from symforge.adapters.lora_layer import LoRALinearLayer
from symforge.adapters.registry import LoRAAdapterRegistry
from symforge.compute.contracts import AdapterArtifact

logger = logging.getLogger(__name__)


class LoRAAdapterLoader:
    """Orchestrates injection of LoRA layers and high-speed in-place hot-swapping on CPU."""

    @classmethod
    def _split_parent_child(cls, module_path: str) -> Tuple[str, str]:
        """Split a dotted module path into parent path and child attribute name."""
        parts = module_path.split(".")
        if len(parts) == 1:
            return "", parts[0]
        return ".".join(parts[:-1]), parts[-1]

    @classmethod
    def _find_module_name(
        cls,
        base_model: nn.Module,
        target_name: str,
    ) -> Optional[str]:
        """Resolve full module name in base_model matching target_name exactly or by suffix."""
        named = dict(base_model.named_modules())
        if target_name in named:
            return target_name

        target_leaf = target_name.split(".")[-1]
        for name, mod in named.items():
            if name.endswith(target_name) or name.split(".")[-1] == target_leaf:
                if isinstance(mod, (nn.Linear, LoRALinearLayer)):
                    return name
        return None

    @classmethod
    def inject_adapters(
        cls,
        base_model: nn.Module,
        target_module_names: Sequence[str],
        rank: int = 8,
        alpha: float = 16.0,
    ) -> Dict[str, LoRALinearLayer]:
        """Convert target linear layers into frozen LoRALinearLayers ready for adapter attachment.

        Args:
            base_model: PyTorch nn.Module instance.
            target_module_names: Names or paths of linear layers to adapt.
            rank: Default LoRA rank dimension.
            alpha: Default LoRA scaling factor.

        Returns:
            Dictionary mapping module path to injected LoRALinearLayer.
        """
        injected: Dict[str, LoRALinearLayer] = {}

        for target in target_module_names:
            resolved_name = cls._find_module_name(base_model, target)
            if resolved_name is None:
                logger.error("Target module '%s' not found in base model.", target)
                raise AdapterInjectionError(
                    f"Target module '{target}' could not be resolved in base model."
                )

            current_mod = base_model.get_submodule(resolved_name)
            if isinstance(current_mod, LoRALinearLayer):
                injected[resolved_name] = current_mod
                continue

            if not isinstance(current_mod, nn.Linear):
                logger.error("Module '%s' is not nn.Linear (got %s)", resolved_name, type(current_mod))
                raise AdapterInjectionError(
                    f"Module '{resolved_name}' must be nn.Linear to inject LoRA, got {type(current_mod)}."
                )

            lora_layer = LoRALinearLayer.from_linear(current_mod, rank=rank, alpha=alpha)
            parent_path, child_name = cls._split_parent_child(resolved_name)
            parent_mod = base_model.get_submodule(parent_path) if parent_path else base_model

            setattr(parent_mod, child_name, lora_layer)
            injected[resolved_name] = lora_layer
            logger.debug("Injected LoRALinearLayer at '%s' (r=%d, alpha=%.1f)", resolved_name, rank, alpha)

        return injected

    @classmethod
    def _extract_lora_pair(
        cls,
        state_dict: Dict[str, torch.Tensor],
        resolved_name: str,
        target_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Locate and return LoRA A and B weight tensors from state_dict."""
        candidates = [resolved_name, target_name, resolved_name.split(".")[-1], target_name.split(".")[-1]]

        for cand in candidates:
            for suffix_a, suffix_b in [
                (".lora_A.weight", ".lora_B.weight"),
                (".lora_A", ".lora_B"),
            ]:
                key_a = f"{cand}{suffix_a}"
                key_b = f"{cand}{suffix_b}"
                if key_a in state_dict and key_b in state_dict:
                    return state_dict[key_a], state_dict[key_b]

        # Check if state_dict has un-prefixed keys
        for key_a, key_b in [
            ("lora_A.weight", "lora_B.weight"),
            ("lora_A", "lora_B"),
        ]:
            if key_a in state_dict and key_b in state_dict:
                return state_dict[key_a], state_dict[key_b]

        logger.error(
            "LoRA weights missing in state_dict for module '%s' (candidates=%s)",
            resolved_name,
            candidates,
        )
        raise AdapterMismatchError(
            f"Could not locate LoRA weight tensors for module '{resolved_name}' in state_dict."
        )

    @classmethod
    def _load_weights_from_artifact(
        cls,
        artifact: AdapterArtifact,
        registry: Optional[LoRAAdapterRegistry] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load and cryptographically verify weights from registry or artifact directly."""
        if registry is not None:
            return registry.load_weights(artifact.adapter_id)

        raw_bytes = artifact.weights_bytes
        if raw_bytes is None and artifact.weights_uri is not None:
            parsed = urlparse(artifact.weights_uri)
            file_path = Path(parsed.path if parsed.scheme == "file" else artifact.weights_uri)
            if not file_path.exists():
                logger.error("Weights file missing at uri: %s", artifact.weights_uri)
                raise AdapterVerificationError(f"Weights file missing: {file_path}")
            raw_bytes = file_path.read_bytes()

        if raw_bytes is None:
            logger.error("No weights bytes found in artifact: %s", artifact.adapter_id)
            raise AdapterVerificationError(f"No binary weights in artifact '{artifact.adapter_id}'.")

        computed_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
        expected_hash = artifact.weights_checksum.lower()
        if computed_hash != expected_hash:
            logger.error(
                "Checksum mismatch for artifact %s: computed=%s, expected=%s",
                artifact.adapter_id,
                computed_hash,
                expected_hash,
            )
            raise AdapterVerificationError(
                f"Checksum mismatch for adapter '{artifact.adapter_id}': "
                f"expected {expected_hash}, computed {computed_hash}."
            )

        try:
            return torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.error("Failed to deserialize weights for %s: %s", artifact.adapter_id, exc)
            raise AdapterVerificationError(
                f"Failed to deserialize weights for '{artifact.adapter_id}': {exc}"
            ) from exc

    @classmethod
    def load_and_attach(
        cls,
        base_model: nn.Module,
        adapter_artifact: AdapterArtifact,
        registry: Optional[LoRAAdapterRegistry] = None,
    ) -> Dict[str, LoRALinearLayer]:
        """Load weights and attach them to base_model in < 5 ms on CPU.

        Args:
            base_model: PyTorch nn.Module instance.
            adapter_artifact: Specification of LoRA weights and target modules.
            registry: Optional registry providing LRU-cached verified weights.

        Returns:
            Dictionary of adapted layers.
        """
        # Ensure target modules are injected
        injected = cls.inject_adapters(
            base_model=base_model,
            target_module_names=adapter_artifact.target_modules,
            rank=adapter_artifact.lora_rank,
            alpha=adapter_artifact.lora_alpha,
        )

        state_dict = cls._load_weights_from_artifact(adapter_artifact, registry=registry)

        for target in adapter_artifact.target_modules:
            resolved_name = cls._find_module_name(base_model, target)
            if resolved_name is None:
                continue

            lora_layer = injected[resolved_name]
            lora_a, lora_b = cls._extract_lora_pair(state_dict, resolved_name, target)

            lora_layer.attach_adapter(
                lora_a=lora_a,
                lora_b=lora_b,
                rank=adapter_artifact.lora_rank,
                alpha=adapter_artifact.lora_alpha,
                adapter_name=adapter_artifact.adapter_id,
            )

        logger.info(
            "Attached adapter '%s' to %d layers in base model.",
            adapter_artifact.adapter_id,
            len(injected),
        )
        return injected

    @classmethod
    def detach(
        cls,
        base_model: nn.Module,
        adapter_name: Optional[str] = None,
    ) -> None:
        """Detach adapter and restore unadapted base forward pass across all LoRA layers."""
        for _, mod in base_model.named_modules():
            if isinstance(mod, LoRALinearLayer):
                if adapter_name is None or mod.adapter_name == adapter_name:
                    mod.detach_adapter()
        logger.info("Detached adapter (name filter=%s) from base model.", adapter_name)

    @classmethod
    def hot_swap(
        cls,
        base_model: nn.Module,
        new_adapter_artifact: AdapterArtifact,
        registry: Optional[LoRAAdapterRegistry] = None,
    ) -> None:
        """In-place replace active adapter weights without rebuilding the base model in < 5 ms."""
        state_dict = cls._load_weights_from_artifact(new_adapter_artifact, registry=registry)

        # Inject any newly targeted modules if not already injected
        injected = cls.inject_adapters(
            base_model=base_model,
            target_module_names=new_adapter_artifact.target_modules,
            rank=new_adapter_artifact.lora_rank,
            alpha=new_adapter_artifact.lora_alpha,
        )

        for target in new_adapter_artifact.target_modules:
            resolved_name = cls._find_module_name(base_model, target)
            if resolved_name is None:
                continue

            lora_layer = injected[resolved_name]
            lora_a, lora_b = cls._extract_lora_pair(state_dict, resolved_name, target)

            lora_layer.hot_swap_adapter(
                lora_a=lora_a,
                lora_b=lora_b,
                rank=new_adapter_artifact.lora_rank,
                alpha=new_adapter_artifact.lora_alpha,
                adapter_name=new_adapter_artifact.adapter_id,
            )

        logger.debug("Hot-swapped adapter in-place to '%s'", new_adapter_artifact.adapter_id)

    @classmethod
    def get_adapted_modules(cls, base_model: nn.Module) -> Dict[str, LoRALinearLayer]:
        """Return all LoRALinearLayer modules present in base_model."""
        return {
            name: mod
            for name, mod in base_model.named_modules()
            if isinstance(mod, LoRALinearLayer)
        }

    @classmethod
    def is_adapted(cls, base_model: nn.Module) -> bool:
        """Check if any LoRALinearLayer in base_model currently has an active adapter attached."""
        for _, mod in base_model.named_modules():
            if isinstance(mod, LoRALinearLayer) and mod.is_adapted:
                return True
        return False
