"""Modular LoRA Adapter Registry managing domain-specific collections, LRU caching, and cryptographic verification."""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import torch

from symforge.adapters.exceptions import (
    AdapterAlreadyExistsError,
    AdapterNotFoundError,
    AdapterVerificationError,
)
from symforge.compute.contracts import AdapterArtifact, ComputeDomain

logger = logging.getLogger(__name__)

# Standard domain-specific adapter collections
DEFAULT_COLLECTIONS: Tuple[str, ...] = (
    "audio_modulators",
    "audio_filters",
    "sdf_csg",
    "sdf_blends",
    "hw_opt",
)

MAX_ADAPTER_FOOTPRINT_BYTES: int = 15 * 1024 * 1024  # 15 MB limit


@dataclass(frozen=True)
class AdapterMetadata:
    """Read-only metadata tracking adapter version, domain, parameters, and memory footprint."""

    adapter_id: str
    domain: ComputeDomain
    collection: str
    version: str
    tags: Tuple[str, ...]
    lora_rank: int
    lora_alpha: float
    target_modules: Tuple[str, ...]
    parameter_count: int
    memory_footprint_bytes: int
    checksum: str
    registered_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoRAAdapterRegistry:
    """Manages domain-specific LoRA adapter collections with bounded LRU caching and cryptographic verification."""

    def __init__(
        self,
        max_cached_adapters: int = 8,
        max_footprint_bytes: int = MAX_ADAPTER_FOOTPRINT_BYTES,
    ) -> None:
        """Initialize registry with configurable LRU cache size and memory constraints."""
        if max_cached_adapters <= 0:
            raise ValueError(f"max_cached_adapters must be > 0, got {max_cached_adapters}")

        self._max_cached = int(max_cached_adapters)
        self._max_footprint_bytes = int(max_footprint_bytes)

        self._lock = threading.RLock()
        self._artifacts: Dict[str, AdapterArtifact] = {}
        self._metadata: Dict[str, AdapterMetadata] = {}
        self._collections: Dict[str, Set[str]] = {col: set() for col in DEFAULT_COLLECTIONS}
        self._cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()

    @property
    def max_cached_adapters(self) -> int:
        """Return the maximum number of in-memory cached adapter weight sets."""
        return self._max_cached

    def _infer_collection(self, artifact: AdapterArtifact, tags: Tuple[str, ...]) -> str:
        """Infer default domain collection if none explicitly provided."""
        tag_str = " ".join(t.lower() for t in tags)
        mod_str = " ".join(m.lower() for m in artifact.target_modules)

        if artifact.domain == ComputeDomain.AUDIO:
            if "filter" in tag_str or "filter" in mod_str:
                return "audio_filters"
            return "audio_modulators"

        if artifact.domain == ComputeDomain.SDF:
            if "blend" in tag_str or "blend" in mod_str:
                return "sdf_blends"
            return "sdf_csg"

        if artifact.domain == ComputeDomain.HARDWARE:
            return "hw_opt"

        return f"{artifact.domain.value}_general"

    def _extract_and_verify_bytes(self, artifact: AdapterArtifact) -> bytes:
        """Extract raw binary payload and enforce cryptographic SHA-256 verification."""
        raw_bytes = artifact.weights_bytes

        if raw_bytes is None and artifact.weights_uri is not None:
            parsed = urlparse(artifact.weights_uri)
            file_path = Path(parsed.path if parsed.scheme == "file" else artifact.weights_uri)
            if not file_path.exists():
                logger.error(
                    "Weights file not found at uri: id=%s, uri=%s",
                    artifact.adapter_id,
                    artifact.weights_uri,
                )
                raise AdapterVerificationError(
                    f"Weights file not found for adapter '{artifact.adapter_id}': {file_path}"
                )
            raw_bytes = file_path.read_bytes()

        if raw_bytes is None:
            logger.error("No weights bytes or valid URI for adapter: %s", artifact.adapter_id)
            raise AdapterVerificationError(
                f"Adapter '{artifact.adapter_id}' contains no readable binary weights payload."
            )

        # Enforce canonical SHA-256 checksum verification
        computed_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
        expected_hash = artifact.weights_checksum.lower()

        if computed_hash != expected_hash:
            logger.error(
                "Cryptographic checksum mismatch for adapter %s: computed=%s, expected=%s",
                artifact.adapter_id,
                computed_hash,
                expected_hash,
            )
            raise AdapterVerificationError(
                f"Cryptographic checksum verification failed for adapter '{artifact.adapter_id}': "
                f"expected {expected_hash}, computed {computed_hash}. Weights may be corrupted."
            )

        return raw_bytes

    def register(
        self,
        artifact: AdapterArtifact,
        collection: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        allow_overwrite: bool = False,
    ) -> AdapterMetadata:
        """Register an adapter artifact with cryptographic verification and metadata tracking."""
        with self._lock:
            adapter_id = artifact.adapter_id

            if adapter_id in self._artifacts and not allow_overwrite:
                logger.warning("Attempted duplicate registration for adapter: %s", adapter_id)
                raise AdapterAlreadyExistsError(
                    f"Adapter '{adapter_id}' is already registered. Set allow_overwrite=True to update."
                )

            # Enforce cryptographic integrity
            raw_bytes = self._extract_and_verify_bytes(artifact)
            footprint = len(raw_bytes)

            if footprint > self._max_footprint_bytes:
                logger.error(
                    "Adapter footprint exceeds limit: id=%s, size=%d, max=%d",
                    adapter_id,
                    footprint,
                    self._max_footprint_bytes,
                )
                raise AdapterVerificationError(
                    f"Adapter '{adapter_id}' memory footprint ({footprint} bytes) "
                    f"exceeds maximum threshold ({self._max_footprint_bytes} bytes / 15 MB)."
                )

            # Deserialize weights safely to inspect parameter count
            try:
                state_dict = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
            except Exception as exc:
                logger.error("Failed to deserialize weights for %s: %s", adapter_id, exc)
                raise AdapterVerificationError(f"Failed to deserialize weights for '{adapter_id}': {exc}") from exc

            param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))

            clean_tags = tuple(tags) if tags is not None else tuple()
            col = collection or self._infer_collection(artifact, clean_tags)
            ver = version or artifact.metadata.get("version", "1.0.0")

            meta = AdapterMetadata(
                adapter_id=adapter_id,
                domain=artifact.domain,
                collection=col,
                version=str(ver),
                tags=clean_tags,
                lora_rank=artifact.lora_rank,
                lora_alpha=artifact.lora_alpha,
                target_modules=tuple(artifact.target_modules),
                parameter_count=param_count,
                memory_footprint_bytes=footprint,
                checksum=artifact.weights_checksum.lower(),
                registered_at=datetime.now(timezone.utc),
                metadata=dict(artifact.metadata),
            )

            # If overwriting, remove from old collections and evict cache
            if adapter_id in self._artifacts:
                old_col = self._metadata[adapter_id].collection
                if old_col in self._collections:
                    self._collections[old_col].discard(adapter_id)
                self._evict_from_cache(adapter_id)

            self._artifacts[adapter_id] = artifact
            self._metadata[adapter_id] = meta

            if col not in self._collections:
                self._collections[col] = set()
            self._collections[col].add(adapter_id)

            logger.info("Successfully registered adapter '%s' in collection '%s'", adapter_id, col)
            return meta

    def get_artifact(self, adapter_id: str) -> AdapterArtifact:
        """Retrieve the registered AdapterArtifact by identifier."""
        with self._lock:
            if adapter_id not in self._artifacts:
                raise AdapterNotFoundError(f"Adapter '{adapter_id}' is not registered.")
            return self._artifacts[adapter_id]

    def get_metadata(self, adapter_id: str) -> AdapterMetadata:
        """Retrieve the registered AdapterMetadata by identifier."""
        with self._lock:
            if adapter_id not in self._metadata:
                raise AdapterNotFoundError(f"Adapter '{adapter_id}' is not registered.")
            return self._metadata[adapter_id]

    def load_weights(self, adapter_id: str) -> Dict[str, torch.Tensor]:
        """Load and return LoRA weights state dict, served from bounded LRU cache or verified bytes."""
        with self._lock:
            if adapter_id in self._cache:
                self._cache.move_to_end(adapter_id)
                return self._cache[adapter_id]

            artifact = self.get_artifact(adapter_id)
            raw_bytes = self._extract_and_verify_bytes(artifact)

            try:
                state_dict = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
            except Exception as exc:
                logger.error("Failed to load weights into memory for %s: %s", adapter_id, exc)
                raise AdapterVerificationError(f"Failed to load weights for '{adapter_id}': {exc}") from exc

            # Evict oldest entry if LRU cache capacity reached
            while len(self._cache) >= self._max_cached:
                evicted_id, _ = self._cache.popitem(last=False)
                logger.debug("LRU eviction severed cached weights for adapter '%s'", evicted_id)

            self._cache[adapter_id] = state_dict
            return state_dict

    def _evict_from_cache(self, adapter_id: str) -> None:
        """Explicitly sever cached tensor references for a given adapter."""
        if adapter_id in self._cache:
            tensors = self._cache.pop(adapter_id)
            tensors.clear()

    def unregister(self, adapter_id: str) -> bool:
        """Unregister an adapter, removing artifacts, metadata, and cached tensors."""
        with self._lock:
            if adapter_id not in self._artifacts:
                return False

            meta = self._metadata.pop(adapter_id, None)
            if meta and meta.collection in self._collections:
                self._collections[meta.collection].discard(adapter_id)

            self._artifacts.pop(adapter_id, None)
            self._evict_from_cache(adapter_id)
            return True

    def list_adapters(
        self,
        domain: Optional[ComputeDomain | str] = None,
        collection: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[AdapterMetadata]:
        """List metadata for registered adapters matching optional domain, collection, or tag filters."""
        with self._lock:
            results: List[AdapterMetadata] = []
            target_domain = ComputeDomain(domain) if isinstance(domain, str) else domain

            for meta in self._metadata.values():
                if target_domain is not None and meta.domain != target_domain:
                    continue
                if collection is not None and meta.collection != collection:
                    continue
                if tag is not None and tag not in meta.tags:
                    continue
                results.append(meta)

            return results

    def list_collections(self) -> List[str]:
        """Return list of all registered collection names."""
        with self._lock:
            return sorted(self._collections.keys())

    def clear_cache(self) -> None:
        """Clear all in-memory cached tensors and sever references."""
        with self._lock:
            for tensors in self._cache.values():
                tensors.clear()
            self._cache.clear()

    def cached_adapter_ids(self) -> List[str]:
        """Return IDs of currently cached adapters in LRU order."""
        with self._lock:
            return list(self._cache.keys())
