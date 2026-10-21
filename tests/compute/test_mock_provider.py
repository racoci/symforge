"""Unit, lifecycle, and benchmark tests for MockComputeProvider."""

from __future__ import annotations
import asyncio
import hashlib
from pathlib import Path
import time
from typing import Dict
import pytest
import torch

from symforge.compute import (
    AdapterArtifact,
    ArtifactRetrievalError,
    ComputeDomain,
    ComputeProvider,
    JobExecutionError,
    JobStatus,
    JobStatusResponse,
    MockComputeProvider,
    MockExecutionMode,
    RemoteJobConfig,
    TrainingPayload,
    verify_compute_provider_conformance,
)


def _create_sample_payload(
    domain: ComputeDomain = ComputeDomain.AUDIO,
    trajectories_count: int = 100,
) -> TrainingPayload:
    """Helper to create valid TrainingPayload instances for testing."""
    return TrainingPayload(
        trajectories_count=trajectories_count,
        domain=domain,
        dataset_uri="s3://symforge-test/dataset.json",
        base_model_name="pde-audio-v1",
        model_type="pde",
        hyperparameters={"lr": 0.001, "batch_size": 16},
    )


def _create_sample_config(
    mock_mode: str | None = None,
    steps_to_complete: int | None = None,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> RemoteJobConfig:
    """Helper to create valid RemoteJobConfig instances with mock tags."""
    tags: Dict[str, str] = {}
    if mock_mode is not None:
        tags["mock_mode"] = mock_mode
    if steps_to_complete is not None:
        tags["steps_to_complete"] = str(steps_to_complete)

    return RemoteJobConfig(
        provider_name="mock",
        hardware_accelerator="CPU",
        timeout_seconds=30.0,
        max_cost_usd=0.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        tags=tags,
    )


# =====================================================================
# 1. Protocol Conformance & Type Safety
# =====================================================================

def test_mock_compute_provider_conformance() -> None:
    """Verify runtime checkability and conformance verification for MockComputeProvider."""
    provider = MockComputeProvider()
    assert isinstance(provider, ComputeProvider)
    assert verify_compute_provider_conformance(provider) is True


def test_mock_execution_mode_values() -> None:
    """Verify MockExecutionMode enum representations."""
    assert MockExecutionMode.INSTANT == "instant"
    assert MockExecutionMode.SIMULATED_STEPS == "simulated_steps"
    assert MockExecutionMode.FAILING == "failing"


# =====================================================================
# 2. Instant Mode Lifecycle & Artifact Persistence
# =====================================================================

def test_mock_provider_instant_lifecycle(tmp_path: Path) -> None:
    """Verify immediate completion, artifact serialization, and PyTorch tensor loading."""
    async def _run() -> None:
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        payload = _create_sample_payload(domain=ComputeDomain.AUDIO)
        config = _create_sample_config(lora_rank=4, lora_alpha=8.0)

        # 1. Submit job
        job_id = await asyncio.wait_for(
            provider.submit_training_job(payload, config),
            timeout=5.0,
        )
        assert job_id.startswith("mock-job-")

        # 2. Poll job status -> instantly COMPLETED
        status_resp: JobStatusResponse = await asyncio.wait_for(
            provider.poll_job_status(job_id),
            timeout=5.0,
        )
        assert status_resp.status == JobStatus.COMPLETED
        assert status_resp.progress_pct == 100.0
        assert status_resp.estimated_cost_usd == 0.0
        assert "train_loss" in status_resp.metrics
        assert "val_f1" in status_resp.metrics

        # 3. Fetch artifacts
        artifact: AdapterArtifact = await asyncio.wait_for(
            provider.fetch_job_artifacts(job_id, tmp_path),
            timeout=5.0,
        )
        assert artifact.adapter_id == f"adapter-{job_id}"
        assert artifact.domain == ComputeDomain.AUDIO
        assert artifact.lora_rank == 4
        assert artifact.lora_alpha == 8.0
        assert len(artifact.target_modules) >= 1

        # 4. Symmetrical checksum validation on persisted file
        persisted_file = tmp_path / f"{job_id}_adapter.pt"
        assert persisted_file.exists()
        file_bytes = persisted_file.read_bytes()
        calculated_sha = hashlib.sha256(file_bytes).hexdigest()
        assert calculated_sha == artifact.weights_checksum
        assert len(file_bytes) == artifact.file_size_bytes

        # 5. Load weights with torch.load and inspect tensors
        state_dict = torch.load(persisted_file, weights_only=True)
        for mod in artifact.target_modules:
            a_key = f"{mod}.lora_A.weight"
            b_key = f"{mod}.lora_B.weight"
            assert a_key in state_dict
            assert b_key in state_dict
            assert state_dict[a_key].shape == (4, 32)
            assert state_dict[b_key].shape == (32, 4)

    asyncio.run(_run())


# =====================================================================
# 3. Simulated Steps Progression & Monotonic Metric Dynamics
# =====================================================================

def test_mock_provider_simulated_steps_lifecycle(tmp_path: Path) -> None:
    """Verify progression across discrete steps (PENDING -> RUNNING -> COMPLETED)."""
    async def _run() -> None:
        steps_total = 4
        provider = MockComputeProvider(
            default_mode=MockExecutionMode.SIMULATED_STEPS,
            steps_to_complete=steps_total,
        )
        payload = _create_sample_payload(domain=ComputeDomain.SDF)
        config = _create_sample_config(lora_rank=16)

        job_id = await asyncio.wait_for(
            provider.submit_training_job(payload, config),
            timeout=5.0,
        )

        losses = []
        f1_scores = []

        # Poll 1: 1/4 = 25% -> RUNNING
        poll1 = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert poll1.status == JobStatus.RUNNING
        assert poll1.progress_pct == 25.0
        losses.append(poll1.metrics["train_loss"])
        f1_scores.append(poll1.metrics["val_f1"])

        # Poll 2: 2/4 = 50% -> RUNNING
        poll2 = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert poll2.status == JobStatus.RUNNING
        assert poll2.progress_pct == 50.0
        losses.append(poll2.metrics["train_loss"])
        f1_scores.append(poll2.metrics["val_f1"])

        # Poll 3: 3/4 = 75% -> RUNNING
        poll3 = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert poll3.status == JobStatus.RUNNING
        assert poll3.progress_pct == 75.0
        losses.append(poll3.metrics["train_loss"])
        f1_scores.append(poll3.metrics["val_f1"])

        # Attempt to fetch artifacts before completion must fail
        with pytest.raises(ArtifactRetrievalError) as exc_info:
            await asyncio.wait_for(provider.fetch_job_artifacts(job_id, tmp_path), timeout=5.0)
        assert "not COMPLETED" in str(exc_info.value)

        # Poll 4: 4/4 = 100% -> COMPLETED
        poll4 = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert poll4.status == JobStatus.COMPLETED
        assert poll4.progress_pct == 100.0
        losses.append(poll4.metrics["train_loss"])
        f1_scores.append(poll4.metrics["val_f1"])

        # Verify telemetry dynamics: losses decrease and F1 increases monotonically
        assert losses == sorted(losses, reverse=True)
        assert f1_scores == sorted(f1_scores)

        # Fetch artifacts now succeeds
        artifact = await asyncio.wait_for(
            provider.fetch_job_artifacts(job_id, tmp_path),
            timeout=5.0,
        )
        assert artifact.domain == ComputeDomain.SDF
        assert artifact.lora_rank == 16

    asyncio.run(_run())


# =====================================================================
# 4. Simulated Failure Simulation
# =====================================================================

def test_mock_provider_failure_mode(tmp_path: Path) -> None:
    """Verify failing mode reports FAILED with detailed error message and blocks fetch."""
    async def _run() -> None:
        custom_error = "CUDA out of memory in backward pass (allocating 16GB)"
        provider = MockComputeProvider(
            default_mode=MockExecutionMode.FAILING,
            failure_message=custom_error,
        )
        payload = _create_sample_payload()
        config = _create_sample_config()

        job_id = await asyncio.wait_for(
            provider.submit_training_job(payload, config),
            timeout=5.0,
        )

        status_resp = await asyncio.wait_for(
            provider.poll_job_status(job_id),
            timeout=5.0,
        )
        assert status_resp.status == JobStatus.FAILED
        assert status_resp.error_message == custom_error
        assert status_resp.completed_at is not None

        # Subsequent polls remain FAILED (immutable terminal state)
        poll_again = await asyncio.wait_for(
            provider.poll_job_status(job_id),
            timeout=5.0,
        )
        assert poll_again.status == JobStatus.FAILED

        # Fetching artifacts for a failed job must raise ArtifactRetrievalError
        with pytest.raises(ArtifactRetrievalError) as exc_info:
            await asyncio.wait_for(provider.fetch_job_artifacts(job_id, tmp_path), timeout=5.0)
        assert "FAILED" in str(exc_info.value)
        assert custom_error in str(exc_info.value)

    asyncio.run(_run())


# =====================================================================
# 5. Cancellation Behavior & Guardrails
# =====================================================================

def test_mock_provider_cancellation_lifecycle(tmp_path: Path) -> None:
    """Verify cancellation marks state CANCELLED and blocks artifact retrieval."""
    async def _run() -> None:
        provider = MockComputeProvider(
            default_mode=MockExecutionMode.SIMULATED_STEPS,
            steps_to_complete=10,
        )
        payload = _create_sample_payload()
        config = _create_sample_config()

        job_id = await asyncio.wait_for(
            provider.submit_training_job(payload, config),
            timeout=5.0,
        )

        # Poll once -> RUNNING
        status = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert status.status == JobStatus.RUNNING

        # Cancel job
        cancelled = await asyncio.wait_for(provider.cancel_job(job_id), timeout=5.0)
        assert cancelled is True

        # Polling status reports CANCELLED
        status_cancelled = await asyncio.wait_for(
            provider.poll_job_status(job_id),
            timeout=5.0,
        )
        assert status_cancelled.status == JobStatus.CANCELLED

        # Subsequent cancel call returns False (cannot re-cancel)
        assert await asyncio.wait_for(provider.cancel_job(job_id), timeout=5.0) is False

        # Attempt to fetch artifacts must be rejected
        with pytest.raises(ArtifactRetrievalError) as exc_info:
            await asyncio.wait_for(provider.fetch_job_artifacts(job_id, tmp_path), timeout=5.0)
        assert "CANCELLED" in str(exc_info.value)

    asyncio.run(_run())


def test_mock_provider_cancel_completed_job(tmp_path: Path) -> None:
    """Verify cancelling an already COMPLETED job returns False."""
    async def _run() -> None:
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        payload = _create_sample_payload()
        config = _create_sample_config()

        job_id = await asyncio.wait_for(provider.submit_training_job(payload, config), timeout=5.0)
        # Job is already completed in instant mode
        assert await asyncio.wait_for(provider.cancel_job(job_id), timeout=5.0) is False

    asyncio.run(_run())


# =====================================================================
# 6. Tag Overrides, History Eviction & Unknown Job Errors
# =====================================================================

def test_mock_provider_tag_overrides_and_domains(tmp_path: Path) -> None:
    """Verify per-job tag overrides for mode and steps, and multi-domain target modules."""
    async def _run() -> None:
        # Default is instant, but tag overrides to simulated_steps
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        payload = _create_sample_payload(domain=ComputeDomain.HARDWARE)
        config = _create_sample_config(
            mock_mode="simulated_steps",
            steps_to_complete=2,
            lora_rank=8,
        )

        job_id = await asyncio.wait_for(provider.submit_training_job(payload, config), timeout=5.0)

        # Poll 1
        s1 = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert s1.status == JobStatus.RUNNING
        assert s1.progress_pct == 50.0

        # Poll 2
        s2 = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert s2.status == JobStatus.COMPLETED
        assert s2.progress_pct == 100.0

        artifact = await asyncio.wait_for(
            provider.fetch_job_artifacts(job_id, tmp_path),
            timeout=5.0,
        )
        assert artifact.domain == ComputeDomain.HARDWARE
        assert "hw_opt.alu_proj" in artifact.target_modules

    asyncio.run(_run())


def test_mock_provider_unknown_job_errors(tmp_path: Path) -> None:
    """Verify appropriate errors when querying or fetching unknown job IDs."""
    async def _run() -> None:
        provider = MockComputeProvider()

        with pytest.raises(JobExecutionError) as exc_info:
            await asyncio.wait_for(provider.poll_job_status("nonexistent-job"), timeout=5.0)
        assert "not found" in str(exc_info.value)

        with pytest.raises(ArtifactRetrievalError) as exc_info2:
            await asyncio.wait_for(
                provider.fetch_job_artifacts("nonexistent-job", tmp_path),
                timeout=5.0,
            )
        assert "not found" in str(exc_info2.value)

        assert await asyncio.wait_for(provider.cancel_job("nonexistent-job"), timeout=5.0) is False

    asyncio.run(_run())


def test_mock_provider_reset_and_eviction() -> None:
    """Verify cache reset and history capacity eviction."""
    async def _run() -> None:
        provider = MockComputeProvider(max_history=5)
        payload = _create_sample_payload()
        config = _create_sample_config()

        job_ids = []
        for _ in range(6):
            jid = await asyncio.wait_for(provider.submit_training_job(payload, config), timeout=5.0)
            job_ids.append(jid)

        # First job must have been evicted
        with pytest.raises(JobExecutionError):
            await asyncio.wait_for(provider.poll_job_status(job_ids[0]), timeout=5.0)

        # Latest job exists
        status = await asyncio.wait_for(provider.poll_job_status(job_ids[-1]), timeout=5.0)
        assert status.status == JobStatus.COMPLETED

        # Reset clears everything
        provider.reset()
        with pytest.raises(JobExecutionError):
            await asyncio.wait_for(provider.poll_job_status(job_ids[-1]), timeout=5.0)

    asyncio.run(_run())


# =====================================================================
# 7. Pass-to-Complete Gate: High-Precision Latency Benchmark (< 50ms)
# =====================================================================

def test_mock_provider_pass_to_complete_latency_gate(tmp_path: Path) -> None:
    """Pass-to-Complete Gate: Verify complete lifecycle completes in < 50ms (typically < 10ms)."""
    async def _run_benchmark() -> None:
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        payload = _create_sample_payload(domain=ComputeDomain.AUDIO)
        config = _create_sample_config(lora_rank=8)

        # Warm up PyTorch tensor serialization mechanism once
        warmup_id = await asyncio.wait_for(
            provider.submit_training_job(payload, config),
            timeout=5.0,
        )
        await asyncio.wait_for(
            provider.fetch_job_artifacts(warmup_id, tmp_path / "warmup"),
            timeout=5.0,
        )

        # Benchmark 10 iterations of the full submit -> poll -> fetch cycle
        latencies_ms = []
        for i in range(10):
            out_dir = tmp_path / f"bench_{i}"
            t_start = time.perf_counter()

            job_id = await asyncio.wait_for(
                provider.submit_training_job(payload, config),
                timeout=5.0,
            )
            status_resp = await asyncio.wait_for(
                provider.poll_job_status(job_id),
                timeout=5.0,
            )
            assert status_resp.status == JobStatus.COMPLETED

            artifact = await asyncio.wait_for(
                provider.fetch_job_artifacts(job_id, out_dir),
                timeout=5.0,
            )
            assert artifact.adapter_id == f"adapter-{job_id}"

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            latencies_ms.append(t_elapsed_ms)

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        max_latency = max(latencies_ms)

        # Enforce Pass-to-Complete Gate requirements
        # Average must be well under 10ms; max must strictly not exceed 50ms
        assert max_latency < 50.0, (
            f"Pass-to-Complete Gate Failed: Max latency {max_latency:.2f}ms exceeded 50.0ms limit."
        )
        assert avg_latency < 15.0, (
            f"Pass-to-Complete Gate Failed: Average latency {avg_latency:.2f}ms exceeded 15.0ms limit."
        )

    asyncio.run(_run_benchmark())
