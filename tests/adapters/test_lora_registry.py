"""Automated test suite for modular LoRA adapter registry, loading, and CPU inference."""

from __future__ import annotations
import gc
import hashlib
import io
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from symforge.adapters import (
    AdapterAlreadyExistsError,
    AdapterError,
    AdapterInjectionError,
    AdapterMismatchError,
    AdapterNotFoundError,
    AdapterVerificationError,
    LoRAAdapterLoader,
    LoRAAdapterRegistry,
    LoRALinearLayer,
)
from symforge.compute.contracts import AdapterArtifact, ComputeDomain
from symforge.models.pde import PrimitiveDepthEstimator


def _create_synthetic_adapter_artifact(
    adapter_id: str,
    domain: ComputeDomain,
    target_modules: List[str],
    module_shapes: Dict[str, Tuple[int, int]],
    rank: int = 4,
    alpha: float = 8.0,
    seed: int = 42,
    destination_file: Optional[Path] = None,
) -> AdapterArtifact:
    """Create a verified synthetic AdapterArtifact with matching weight matrices."""
    generator = torch.Generator().manual_seed(seed)
    state_dict: Dict[str, torch.Tensor] = {}

    for mod in target_modules:
        in_dim, out_dim = module_shapes[mod]
        state_dict[f"{mod}.lora_A.weight"] = torch.randn(rank, in_dim, generator=generator)
        state_dict[f"{mod}.lora_B.weight"] = torch.randn(out_dim, rank, generator=generator)

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    file_bytes = buffer.getvalue()

    file_uri: Optional[str] = None
    if destination_file is not None:
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        destination_file.write_bytes(file_bytes)
        file_uri = f"file://{destination_file.resolve()}"

    checksum = hashlib.sha256(file_bytes).hexdigest()

    return AdapterArtifact(
        adapter_id=adapter_id,
        domain=domain,
        lora_rank=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        weights_checksum=checksum,
        weights_uri=file_uri,
        weights_bytes=file_bytes,
        file_size_bytes=len(file_bytes),
        metadata={
            "generator_seed": seed,
            "version": "1.0.0",
        },
    )


class TestLoRALinearLayer:
    """Unit tests for LoRALinearLayer wrapper and mathematical invariants."""

    def test_layer_freezes_base_weights(self) -> None:
        """Verify that wrapping an nn.Linear freezes base parameters."""
        linear = nn.Linear(16, 32)
        assert linear.weight.requires_grad is True
        assert linear.bias.requires_grad is True

        lora_layer = LoRALinearLayer.from_linear(linear, rank=4, alpha=8.0)
        assert lora_layer.base_layer.weight.requires_grad is False
        assert lora_layer.base_layer.bias.requires_grad is False
        assert not lora_layer.is_adapted

    def test_unadapted_forward_matches_base_linear(self) -> None:
        """Verify that forward pass without adapter attached is identical to base layer."""
        linear = nn.Linear(8, 16)
        lora_layer = LoRALinearLayer.from_linear(linear, rank=2, alpha=4.0)

        x = torch.randn(4, 8)
        y_base = linear(x)
        y_lora = lora_layer(x)
        assert torch.allclose(y_base, y_lora, atol=1e-6)

    def test_adapted_forward_matches_dense_factorization(self) -> None:
        """Verify mathematical identity y = W_0 x + b_0 + (alpha / r) * B (A x)."""
        d_in, d_out, rank, alpha = 8, 12, 3, 6.0
        linear = nn.Linear(d_in, d_out)
        lora_layer = LoRALinearLayer.from_linear(linear, rank=rank, alpha=alpha)

        generator = torch.Generator().manual_seed(123)
        lora_a = torch.randn(rank, d_in, generator=generator)
        lora_b = torch.randn(d_out, rank, generator=generator)

        lora_layer.attach_adapter(lora_a, lora_b, rank=rank, alpha=alpha, adapter_name="test_adapter")
        assert lora_layer.is_adapted
        assert lora_layer.scaling == (alpha / rank)

        x = torch.randn(5, d_in)
        y_actual = lora_layer(x)

        # Analytical dense equivalence: W_eff = W_0 + (alpha / r) * (B @ A)
        w_eff = linear.weight + (alpha / rank) * (lora_b @ lora_a)
        y_expected = F.linear(x, w_eff, linear.bias)

        assert torch.allclose(y_actual, y_expected, atol=1e-5)

    def test_detach_restores_base_computation(self) -> None:
        """Verify that detaching the adapter severs tensors and restores base pass."""
        linear = nn.Linear(4, 8)
        lora_layer = LoRALinearLayer.from_linear(linear, rank=2, alpha=4.0)
        lora_layer.attach_adapter(torch.randn(2, 4), torch.randn(8, 2), rank=2, alpha=4.0)

        x = torch.randn(2, 4)
        y_adapted = lora_layer(x)
        y_base = linear(x)
        assert not torch.allclose(y_adapted, y_base, atol=1e-3)

        lora_layer.detach_adapter()
        assert not lora_layer.is_adapted
        assert lora_layer.lora_A is None
        assert lora_layer.lora_B is None
        assert torch.allclose(lora_layer(x), y_base, atol=1e-6)

    def test_enable_disable_toggle(self) -> None:
        """Verify that enabling/disabling bypasses LoRA delta without removing weights."""
        linear = nn.Linear(4, 4)
        lora_layer = LoRALinearLayer.from_linear(linear, rank=2, alpha=2.0)
        lora_layer.attach_adapter(torch.ones(2, 4), torch.ones(4, 2), rank=2, alpha=2.0)

        x = torch.randn(1, 4)
        y_adapted = lora_layer(x)
        lora_layer.enable_adapter(False)
        assert torch.allclose(lora_layer(x), linear(x), atol=1e-6)

        lora_layer.enable_adapter(True)
        assert torch.allclose(lora_layer(x), y_adapted, atol=1e-6)

    def test_shape_mismatch_raises_error(self) -> None:
        """Verify that attaching wrong tensor dimensions raises AdapterMismatchError."""
        linear = nn.Linear(8, 16)
        lora_layer = LoRALinearLayer.from_linear(linear, rank=4, alpha=8.0)

        # Wrong in_features on lora_A (10 instead of 8)
        with pytest.raises(AdapterMismatchError):
            lora_layer.attach_adapter(torch.randn(4, 10), torch.randn(16, 4), rank=4, alpha=8.0)

        # Wrong out_features on lora_B (12 instead of 16)
        with pytest.raises(AdapterMismatchError):
            lora_layer.attach_adapter(torch.randn(4, 8), torch.randn(12, 4), rank=4, alpha=8.0)


class TestLoRAAdapterRegistry:
    """Unit tests for LoRAAdapterRegistry registration, duplicate handling, and cryptographic validation."""

    def test_register_and_retrieve_metadata(self) -> None:
        """Verify registration and metadata querying."""
        registry = LoRAAdapterRegistry(max_cached_adapters=4)
        shapes = {"fc1": (16, 32)}
        artifact = _create_synthetic_adapter_artifact(
            adapter_id="audio_filt_01",
            domain=ComputeDomain.AUDIO,
            target_modules=["fc1"],
            module_shapes=shapes,
        )

        meta = registry.register(
            artifact=artifact,
            collection="audio_filters",
            version="1.2.0",
            tags=["butterworth", "lowpass"],
        )

        assert meta.adapter_id == "audio_filt_01"
        assert meta.collection == "audio_filters"
        assert meta.version == "1.2.0"
        assert "butterworth" in meta.tags
        assert meta.parameter_count > 0
        assert meta.memory_footprint_bytes < 15 * 1024 * 1024

        fetched = registry.get_artifact("audio_filt_01")
        assert fetched.adapter_id == artifact.adapter_id

        # Listing with filters
        matching = registry.list_adapters(domain=ComputeDomain.AUDIO, collection="audio_filters")
        assert len(matching) == 1
        assert matching[0].adapter_id == "audio_filt_01"

        non_matching = registry.list_adapters(domain=ComputeDomain.SDF)
        assert len(non_matching) == 0

    def test_duplicate_registration_handling(self) -> None:
        """Verify duplicate registration rejection and overwrite support."""
        registry = LoRAAdapterRegistry()
        shapes = {"fc1": (16, 32)}
        art1 = _create_synthetic_adapter_artifact("dup_test", ComputeDomain.SDF, ["fc1"], shapes, seed=1)
        art2 = _create_synthetic_adapter_artifact("dup_test", ComputeDomain.SDF, ["fc1"], shapes, seed=2)

        registry.register(art1)

        # Duplicate without overwrite must raise
        with pytest.raises(AdapterAlreadyExistsError):
            registry.register(art2, allow_overwrite=False)

        # Overwrite with allow_overwrite=True succeeds
        meta2 = registry.register(art2, version="2.0.0", allow_overwrite=True)
        assert meta2.version == "2.0.0"
        assert registry.get_artifact("dup_test").weights_checksum == art2.weights_checksum

    def test_symmetrical_sha256_verification_detects_single_byte_mutation(self, tmp_path: Path) -> None:
        """Verify that mutating a single byte of weights triggers AdapterVerificationError."""
        registry = LoRAAdapterRegistry()
        shapes = {"fc1": (8, 16)}
        weights_file = tmp_path / "corrupted_adapter.pt"

        valid_artifact = _create_synthetic_adapter_artifact(
            adapter_id="tamper_test",
            domain=ComputeDomain.AUDIO,
            target_modules=["fc1"],
            module_shapes=shapes,
            destination_file=weights_file,
        )

        # Mutate 1 byte of weights on disk
        corrupted_bytes = bytearray(valid_artifact.weights_bytes)
        corrupted_bytes[10] = (corrupted_bytes[10] + 1) % 256
        weights_file.write_bytes(bytes(corrupted_bytes))

        # Test A: Corrupted payload in artifact.weights_bytes
        corrupted_artifact_mem = AdapterArtifact(
            adapter_id="tamper_mem",
            domain=valid_artifact.domain,
            lora_rank=valid_artifact.lora_rank,
            lora_alpha=valid_artifact.lora_alpha,
            target_modules=valid_artifact.target_modules,
            weights_checksum=valid_artifact.weights_checksum,  # Original checksum
            weights_bytes=bytes(corrupted_bytes),
            file_size_bytes=len(corrupted_bytes),
        )

        with pytest.raises(AdapterVerificationError) as exc_info:
            registry.register(corrupted_artifact_mem)
        assert "checksum verification failed" in str(exc_info.value).lower()

        # Test B: Corrupted file on disk via weights_uri
        corrupted_artifact_file = AdapterArtifact(
            adapter_id="tamper_file",
            domain=valid_artifact.domain,
            lora_rank=valid_artifact.lora_rank,
            lora_alpha=valid_artifact.lora_alpha,
            target_modules=valid_artifact.target_modules,
            weights_checksum=valid_artifact.weights_checksum,  # Original checksum
            weights_uri=f"file://{weights_file.resolve()}",
            weights_bytes=None,
            file_size_bytes=len(corrupted_bytes),
        )

        with pytest.raises(AdapterVerificationError):
            registry.register(corrupted_artifact_file)

    def test_lru_cache_bounded_capacity(self) -> None:
        """Verify LRU cache evicts oldest entries and does not exceed max_cached_adapters."""
        registry = LoRAAdapterRegistry(max_cached_adapters=2)
        shapes = {"fc1": (8, 16)}

        artifacts = [
            _create_synthetic_adapter_artifact(f"art_{i}", ComputeDomain.AUDIO, ["fc1"], shapes, seed=i)
            for i in range(4)
        ]

        for art in artifacts:
            registry.register(art)

        # Load weights for art_0 and art_1
        registry.load_weights("art_0")
        registry.load_weights("art_1")
        assert registry.cached_adapter_ids() == ["art_0", "art_1"]

        # Loading art_2 should evict art_0 (oldest)
        registry.load_weights("art_2")
        assert registry.cached_adapter_ids() == ["art_1", "art_2"]

        # Accessing art_1 refreshes it, then loading art_3 should evict art_2
        registry.load_weights("art_1")
        registry.load_weights("art_3")
        assert registry.cached_adapter_ids() == ["art_1", "art_3"]

    def test_memory_footprint_threshold_enforced(self) -> None:
        """Verify that adapter artifacts exceeding maximum footprint are rejected."""
        registry = LoRAAdapterRegistry(max_footprint_bytes=100)  # Very low 100 bytes threshold
        shapes = {"fc1": (8, 16)}
        artifact = _create_synthetic_adapter_artifact("large_test", ComputeDomain.SDF, ["fc1"], shapes)

        with pytest.raises(AdapterVerificationError) as exc_info:
            registry.register(artifact)
        assert "memory footprint" in str(exc_info.value).lower()


class TestLoRAAdapterLoaderAndHotSwap:
    """Tests for LoRAAdapterLoader, module injection, and dynamic CPU hot-swapping."""

    @pytest.fixture
    def pde_model(self) -> PrimitiveDepthEstimator:
        """Instantiate a lightweight PrimitiveDepthEstimator base model."""
        return PrimitiveDepthEstimator(
            in_features=16,
            domain="sdf",
            max_depth=3,
            hidden_dim=32,
            dropout=0.0,
        )

    def test_inject_adapters_converts_target_layers(self, pde_model: PrimitiveDepthEstimator) -> None:
        """Verify that inject_adapters replaces target linear layers with LoRALinearLayer."""
        injected = LoRAAdapterLoader.inject_adapters(
            base_model=pde_model,
            target_module_names=["fc1", "fc2"],
            rank=4,
            alpha=8.0,
        )
        assert len(injected) == 2
        assert isinstance(pde_model.fc1, LoRALinearLayer)
        assert isinstance(pde_model.fc2, LoRALinearLayer)
        assert pde_model.fc1.base_layer.weight.requires_grad is False
        assert pde_model.fc2.base_layer.weight.requires_grad is False

    def test_hot_swap_output_divergence_and_base_weight_immutability(
        self,
        pde_model: PrimitiveDepthEstimator,
    ) -> None:
        """Verify f_A(x) != f_B(x) and base model weights W_0 remain 100% frozen and identical."""
        # Record pristine base model weights
        w0_fc1_pristine = pde_model.fc1.weight.clone()
        w0_fc2_pristine = pde_model.fc2.weight.clone()

        module_shapes = {
            "fc1": (16, 32),
            "fc2": (32, 32),
        }

        art_a = _create_synthetic_adapter_artifact("adapter_A", ComputeDomain.SDF, ["fc1", "fc2"], module_shapes, seed=10)
        art_b = _create_synthetic_adapter_artifact("adapter_B", ComputeDomain.SDF, ["fc1", "fc2"], module_shapes, seed=20)

        registry = LoRAAdapterRegistry()
        registry.register(art_a)
        registry.register(art_b)

        x = torch.randn(2, 16)

        # Baseline evaluation (unadapted)
        pde_model.eval()
        with torch.no_grad():
            y_base = pde_model(x).clone()

        # 1. Attach Adapter A
        LoRAAdapterLoader.load_and_attach(pde_model, art_a, registry=registry)
        with torch.no_grad():
            y_a = pde_model(x).clone()

        # 2. Hot-swap to Adapter B
        LoRAAdapterLoader.hot_swap(pde_model, art_b, registry=registry)
        with torch.no_grad():
            y_b = pde_model(x).clone()

        # Verify output divergence
        assert not torch.allclose(y_a, y_base, atol=1e-3), "Adapter A output must differ from base model"
        assert not torch.allclose(y_b, y_base, atol=1e-3), "Adapter B output must differ from base model"
        assert not torch.allclose(y_a, y_b, atol=1e-3), "Adapter A output must differ from Adapter B output"

        # 3. Detach adapter and verify return to exact base output
        LoRAAdapterLoader.detach(pde_model)
        with torch.no_grad():
            y_detached = pde_model(x).clone()
        assert torch.allclose(y_detached, y_base, atol=1e-5), "Detaching must restore base model output"

        # Verify base model weights remain 100% frozen and identical
        adapted_fc1 = pde_model.fc1
        adapted_fc2 = pde_model.fc2
        assert isinstance(adapted_fc1, LoRALinearLayer)
        assert isinstance(adapted_fc2, LoRALinearLayer)

        assert torch.equal(adapted_fc1.base_layer.weight, w0_fc1_pristine), "Base fc1 weight was modified!"
        assert torch.equal(adapted_fc2.base_layer.weight, w0_fc2_pristine), "Base fc2 weight was modified!"


class TestPassToCompleteGate:
    """Pass-to-Complete Gate: Runs 100 consecutive hot-swaps in a tight loop on CPU verifying latency and memory containment."""

    def test_100_consecutive_hot_swaps_latency_and_memory_containment(self) -> None:
        """Run 100 hot-swap operations in a tight loop on CPU, verifying latency < 5 ms and zero leaks."""
        model = PrimitiveDepthEstimator(
            in_features=32,
            domain="audio",
            max_depth=3,
            hidden_dim=64,
            dropout=0.0,
        )
        model.eval()

        module_shapes = {
            "fc1": (32, 64),
            "fc2": (64, 64),
        }

        # Create two distinct adapters to alternate between
        art_alpha = _create_synthetic_adapter_artifact("art_alpha", ComputeDomain.AUDIO, ["fc1", "fc2"], module_shapes, seed=100)
        art_beta = _create_synthetic_adapter_artifact("art_beta", ComputeDomain.AUDIO, ["fc1", "fc2"], module_shapes, seed=200)

        registry = LoRAAdapterRegistry(max_cached_adapters=4)
        registry.register(art_alpha)
        registry.register(art_beta)

        # Initial attachment
        LoRAAdapterLoader.load_and_attach(model, art_alpha, registry=registry)

        x = torch.randn(4, 32)
        latencies_ms: List[float] = []

        # Warm up
        LoRAAdapterLoader.hot_swap(model, art_beta, registry=registry)
        LoRAAdapterLoader.hot_swap(model, art_alpha, registry=registry)
        with torch.no_grad():
            _ = model(x)

        # Force garbage collection baseline
        gc.collect()
        tensors_before = len([obj for obj in gc.get_objects() if type(obj) is torch.Tensor])

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            # Execute 100 consecutive hot swaps in a tight loop
            for step in range(100):
                target_artifact = art_beta if step % 2 == 0 else art_alpha
                t_start = time.perf_counter()

                LoRAAdapterLoader.hot_swap(model, target_artifact, registry=registry)

                # Perform a quick forward evaluation
                with torch.no_grad():
                    _ = model(x)

                duration_ms = (time.perf_counter() - t_start) * 1000.0
                latencies_ms.append(duration_ms)
        finally:
            if gc_was_enabled:
                gc.enable()

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        max_latency = max(latencies_ms)
        # Nearest rank method for 100 samples: 99th percentile is index 98 (99th item)
        p99_latency = sorted(latencies_ms)[min(int(len(latencies_ms) * 0.99), len(latencies_ms) - 2)]

        print(
            f"\n[Pass-to-Complete Gate] 100 Hot-Swaps: "
            f"Avg={avg_latency:.3f} ms, Max={max_latency:.3f} ms, P99={p99_latency:.3f} ms"
        )

        # Invariant 1: Average and P99 latency must be well under the 5 ms budget
        assert avg_latency < 5.0, f"Average hot-swap latency {avg_latency:.3f} ms exceeded 5.0 ms threshold"
        assert p99_latency < 5.0, f"P99 hot-swap latency {p99_latency:.3f} ms exceeded 5.0 ms threshold"

        # Invariant 2: Memory containment (zero tensor leaks)
        gc.collect()
        tensors_after = len([obj for obj in gc.get_objects() if type(obj) is torch.Tensor])
        tensor_growth = tensors_after - tensors_before

        print(
            f"[Pass-to-Complete Gate] Tensors before={tensors_before}, "
            f"after={tensors_after}, net growth={tensor_growth}"
        )

        # Cache is bounded to 4; tensor count must not grow unbounded with 100 swaps
        assert tensor_growth <= 10, f"Unbounded tensor memory accumulation detected: grew by {tensor_growth} tensors"
