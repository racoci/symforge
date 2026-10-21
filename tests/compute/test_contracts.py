"""Unit and integration tests for ComputeProvider protocol and Pydantic data contracts."""

from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List
import pytest
from pydantic import ValidationError

from symforge.compute import (
    AdapterArtifact,
    ArtifactRetrievalError,
    ComputeDomain,
    ComputeProvider,
    ComputeProviderError,
    JobExecutionError,
    JobStatus,
    JobStatusResponse,
    JobSubmissionError,
    JobTimeoutError,
    RemoteJobConfig,
    TrainingPayload,
    verify_compute_provider_conformance,
)


# =====================================================================
# Enum Verification
# =====================================================================

def test_job_status_enum_values() -> None:
    """Verify all required JobStatus enum members exist and match string representation."""
    assert JobStatus.PENDING == "PENDING"
    assert JobStatus.RUNNING == "RUNNING"
    assert JobStatus.COMPLETED == "COMPLETED"
    assert JobStatus.FAILED == "FAILED"
    assert JobStatus.CANCELLED == "CANCELLED"
    assert set(JobStatus) == {
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }


def test_compute_domain_enum_values() -> None:
    """Verify all required ComputeDomain enum members exist."""
    assert ComputeDomain.AUDIO == "audio"
    assert ComputeDomain.SDF == "sdf"
    assert ComputeDomain.HARDWARE == "hardware"
    assert set(ComputeDomain) == {
        ComputeDomain.AUDIO,
        ComputeDomain.SDF,
        ComputeDomain.HARDWARE,
    }


# =====================================================================
# TrainingPayload Schema and Validation
# =====================================================================

def test_training_payload_roundtrip_uri() -> None:
    """Test JSON serialization round-trip for TrainingPayload with remote URI."""
    created = datetime(2026, 10, 21, 12, 0, 0, tzinfo=timezone.utc)
    payload = TrainingPayload(
        trajectories_count=500,
        domain=ComputeDomain.AUDIO,
        dataset_uri="s3://symforge-datasets/audio/level3_batch.json",
        base_model_name="symforge-audio-pde-v1",
        model_type="pde",
        hyperparameters={"lr": 0.001, "batch_size": 32, "epochs": 50},
        created_at=created,
    )

    json_str = payload.model_dump_json()
    restored = TrainingPayload.model_validate_json(json_str)

    assert restored == payload
    assert restored.domain == ComputeDomain.AUDIO
    assert restored.trajectories_count == 500
    assert restored.dataset_uri == "s3://symforge-datasets/audio/level3_batch.json"
    assert restored.dataset_payload is None
    assert restored.hyperparameters["lr"] == 0.001


def test_training_payload_roundtrip_inline_payload() -> None:
    """Test JSON serialization round-trip for TrainingPayload with inline payload."""
    payload = TrainingPayload(
        trajectories_count=20,
        domain=ComputeDomain.SDF,
        dataset_payload={"samples": [{"sdf_type": "sphere", "radius": 1.5}]},
        base_model_name="symforge-sdf-tgc-base",
        model_type="tgc",
        hyperparameters={"beam_size": 8},
    )

    json_str = payload.model_dump_json()
    restored = TrainingPayload.model_validate_json(json_str)

    assert restored == payload
    assert restored.domain == ComputeDomain.SDF
    assert restored.dataset_payload == {"samples": [{"sdf_type": "sphere", "radius": 1.5}]}
    assert restored.dataset_uri is None


def test_training_payload_validation_errors() -> None:
    """Test validation rejection for invalid TrainingPayload configurations."""
    # Rejection: Neither dataset_uri nor dataset_payload provided
    with pytest.raises(ValidationError) as exc_info:
        TrainingPayload(
            trajectories_count=100,
            domain=ComputeDomain.AUDIO,
            base_model_name="base-v1",
            model_type="pde",
        )
    assert "At least one of 'dataset_uri' or 'dataset_payload' must be provided" in str(exc_info.value)

    # Rejection: trajectories_count < 1
    with pytest.raises(ValidationError):
        TrainingPayload(
            trajectories_count=0,
            domain=ComputeDomain.AUDIO,
            dataset_uri="file:///data.json",
            base_model_name="base-v1",
            model_type="pde",
        )

    # Rejection: Invalid domain
    with pytest.raises(ValidationError):
        TrainingPayload(
            trajectories_count=10,
            domain="video",  # type: ignore
            dataset_uri="file:///data.json",
            base_model_name="base-v1",
            model_type="pde",
        )

    # Rejection: Extra forbidden field
    with pytest.raises(ValidationError):
        TrainingPayload(
            trajectories_count=10,
            domain=ComputeDomain.AUDIO,
            dataset_uri="file:///data.json",
            base_model_name="base-v1",
            model_type="pde",
            unexpected_field="disallowed",  # type: ignore
        )


# =====================================================================
# RemoteJobConfig Schema and Validation
# =====================================================================

def test_remote_job_config_defaults_and_roundtrip() -> None:
    """Test RemoteJobConfig default values and JSON round-trip."""
    config = RemoteJobConfig(provider_name="modal")

    assert config.hardware_accelerator == "CPU"
    assert config.timeout_seconds == 600.0
    assert config.max_retries == 3
    assert config.max_cost_usd == 5.0
    assert config.lora_rank == 8
    assert config.lora_alpha == 16.0
    assert config.lora_dropout == 0.05
    assert config.tags == {}

    json_str = config.model_dump_json()
    restored = RemoteJobConfig.model_validate_json(json_str)
    assert restored == config


def test_remote_job_config_custom_values() -> None:
    """Test RemoteJobConfig with custom parameters."""
    config = RemoteJobConfig(
        provider_name="sagemaker",
        hardware_accelerator="A10G",
        timeout_seconds=1200.0,
        max_retries=5,
        max_cost_usd=25.50,
        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        tags={"experiment": "phase4_lora", "stage": "benchmark"},
    )

    json_str = config.model_dump_json()
    restored = RemoteJobConfig.model_validate_json(json_str)
    assert restored == config
    assert restored.hardware_accelerator == "A10G"
    assert restored.lora_rank == 16
    assert restored.tags["experiment"] == "phase4_lora"


def test_remote_job_config_validation_errors() -> None:
    """Test validation rejection for invalid RemoteJobConfig bounds."""
    # Rejection: timeout_seconds <= 0
    with pytest.raises(ValidationError):
        RemoteJobConfig(provider_name="mock", timeout_seconds=0.0)

    with pytest.raises(ValidationError):
        RemoteJobConfig(provider_name="mock", timeout_seconds=-50.0)

    # Rejection: max_retries < 0
    with pytest.raises(ValidationError):
        RemoteJobConfig(provider_name="mock", max_retries=-1)

    # Rejection: max_cost_usd < 0
    with pytest.raises(ValidationError):
        RemoteJobConfig(provider_name="mock", max_cost_usd=-0.01)

    # Rejection: lora_rank < 1
    with pytest.raises(ValidationError):
        RemoteJobConfig(provider_name="mock", lora_rank=0)

    # Rejection: lora_alpha <= 0
    with pytest.raises(ValidationError):
        RemoteJobConfig(provider_name="mock", lora_alpha=0.0)

    # Rejection: lora_dropout < 0.0 or > 1.0
    with pytest.raises(ValidationError):
        RemoteJobConfig(provider_name="mock", lora_dropout=-0.1)

    with pytest.raises(ValidationError):
        RemoteJobConfig(provider_name="mock", lora_dropout=1.05)


# =====================================================================
# JobStatusResponse Schema and Validation
# =====================================================================

def test_job_status_response_roundtrip() -> None:
    """Test JSON serialization round-trip for JobStatusResponse across states."""
    t_start = datetime(2026, 10, 21, 10, 0, 0, tzinfo=timezone.utc)
    t_done = datetime(2026, 10, 21, 10, 15, 30, tzinfo=timezone.utc)

    # COMPLETED status
    response = JobStatusResponse(
        job_id="job-12345-abc",
        status=JobStatus.COMPLETED,
        progress_pct=100.0,
        started_at=t_start,
        completed_at=t_done,
        estimated_cost_usd=1.25,
        metrics={"train_loss": 0.024, "val_loss": 0.031, "val_f1": 0.945},
    )

    json_str = response.model_dump_json()
    restored = JobStatusResponse.model_validate_json(json_str)
    assert restored == response
    assert restored.status == JobStatus.COMPLETED
    assert restored.progress_pct == 100.0
    assert restored.metrics["val_f1"] == 0.945
    assert restored.error_message is None

    # FAILED status
    fail_response = JobStatusResponse(
        job_id="job-fail-999",
        status=JobStatus.FAILED,
        progress_pct=42.5,
        error_message="CUDA out of memory exception during epoch 12",
        estimated_cost_usd=0.35,
    )
    fail_json = fail_response.model_dump_json()
    restored_fail = JobStatusResponse.model_validate_json(fail_json)
    assert restored_fail == fail_response
    assert restored_fail.status == JobStatus.FAILED
    assert restored_fail.error_message == "CUDA out of memory exception during epoch 12"


def test_job_status_response_validation_errors() -> None:
    """Test validation errors for invalid JobStatusResponse fields."""
    # Rejection: progress_pct < 0 or > 100
    with pytest.raises(ValidationError):
        JobStatusResponse(
            job_id="job-1",
            status=JobStatus.RUNNING,
            progress_pct=-0.5,
        )

    with pytest.raises(ValidationError):
        JobStatusResponse(
            job_id="job-1",
            status=JobStatus.RUNNING,
            progress_pct=100.01,
        )

    # Rejection: estimated_cost_usd < 0
    with pytest.raises(ValidationError):
        JobStatusResponse(
            job_id="job-1",
            status=JobStatus.PENDING,
            estimated_cost_usd=-1.0,
        )

    # Rejection: invalid status
    with pytest.raises(ValidationError):
        JobStatusResponse(
            job_id="job-1",
            status="UNKNOWN_STATUS",  # type: ignore
        )


# =====================================================================
# AdapterArtifact Schema, Checksum & Base64
# =====================================================================

def test_adapter_artifact_roundtrip_bytes_and_checksum() -> None:
    """Test AdapterArtifact with binary weights, SHA-256 calculation, and base64 sync."""
    dummy_weights = b"BINARY_LORA_TENSORS_DUMMY_DATA_XYZ_1234567890"
    checksum = hashlib.sha256(dummy_weights).hexdigest()

    artifact = AdapterArtifact(
        adapter_id="adapter-audio-fm-pair-v1",
        domain=ComputeDomain.AUDIO,
        lora_rank=8,
        lora_alpha=16.0,
        target_modules=["audio_dsp.carrier_proj", "audio_dsp.modulator_proj"],
        weights_checksum=checksum.upper(),  # Check uppercase normalization
        weights_bytes=dummy_weights,
        file_size_bytes=len(dummy_weights),
        metadata={"target_preset": "dexed_bell", "trained_epochs": 30},
    )

    # Checksum was normalized to lowercase
    assert artifact.weights_checksum == checksum.lower()
    # base64 representation automatically synchronized
    assert artifact.weights_base64 is not None

    json_str = artifact.model_dump_json()
    restored = AdapterArtifact.model_validate_json(json_str)

    assert restored == artifact
    assert restored.weights_bytes == dummy_weights
    assert restored.domain == ComputeDomain.AUDIO
    assert restored.file_size_bytes == len(dummy_weights)


def test_adapter_artifact_roundtrip_uri() -> None:
    """Test AdapterArtifact with remote storage URI."""
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    artifact = AdapterArtifact(
        adapter_id="adapter-sdf-csg-blend",
        domain=ComputeDomain.SDF,
        lora_rank=4,
        lora_alpha=8.0,
        target_modules=["sdf_net.fc1", "sdf_net.fc2"],
        weights_checksum=sha,
        weights_uri="s3://symforge-adapters/sdf/adapter_csg_blend.pt",
        file_size_bytes=1048576,
    )

    json_str = artifact.model_dump_json()
    restored = AdapterArtifact.model_validate_json(json_str)

    assert restored == artifact
    assert restored.weights_uri == "s3://symforge-adapters/sdf/adapter_csg_blend.pt"


def test_adapter_artifact_validation_errors() -> None:
    """Test rejection on missing weights sources or invalid SHA-256 checksum."""
    valid_sha = "a" * 64

    # Rejection: Missing all of weights_uri, weights_bytes, weights_base64
    with pytest.raises(ValidationError) as exc_info:
        AdapterArtifact(
            adapter_id="test",
            domain=ComputeDomain.AUDIO,
            lora_rank=4,
            lora_alpha=8.0,
            target_modules=["mod1"],
            weights_checksum=valid_sha,
            file_size_bytes=100,
        )
    assert "At least one of 'weights_uri', 'weights_bytes', or 'weights_base64' must be provided" in str(
        exc_info.value
    )

    # Rejection: Checksum too short (< 64 chars)
    with pytest.raises(ValidationError) as exc_info:
        AdapterArtifact(
            adapter_id="test",
            domain=ComputeDomain.AUDIO,
            lora_rank=4,
            lora_alpha=8.0,
            target_modules=["mod1"],
            weights_checksum="abcd1234",
            weights_uri="file:///w.pt",
            file_size_bytes=100,
        )
    assert "not a valid 64-character SHA-256 hex string" in str(exc_info.value)

    # Rejection: Checksum contains non-hex characters
    with pytest.raises(ValidationError):
        AdapterArtifact(
            adapter_id="test",
            domain=ComputeDomain.AUDIO,
            lora_rank=4,
            lora_alpha=8.0,
            target_modules=["mod1"],
            weights_checksum="z" * 64,
            weights_uri="file:///w.pt",
            file_size_bytes=100,
        )

    # Rejection: file_size_bytes < 0
    with pytest.raises(ValidationError):
        AdapterArtifact(
            adapter_id="test",
            domain=ComputeDomain.AUDIO,
            lora_rank=4,
            lora_alpha=8.0,
            target_modules=["mod1"],
            weights_checksum=valid_sha,
            weights_uri="file:///w.pt",
            file_size_bytes=-1,
        )


# =====================================================================
# ComputeProvider Protocol Conformance
# =====================================================================

class MockValidComputeProvider:
    """Mock implementation conforming to the asynchronous ComputeProvider protocol."""

    def __init__(self) -> None:
        self.jobs: Dict[str, JobStatusResponse] = {}

    async def submit_training_job(
        self,
        payload: TrainingPayload,
        config: RemoteJobConfig,
    ) -> str:
        job_id = f"mock-job-{len(self.jobs) + 1}"
        self.jobs[job_id] = JobStatusResponse(
            job_id=job_id,
            status=JobStatus.RUNNING,
            progress_pct=10.0,
        )
        return job_id

    async def poll_job_status(self, job_id: str) -> JobStatusResponse:
        if job_id not in self.jobs:
            raise JobExecutionError(f"Job {job_id} not found.")
        return self.jobs[job_id]

    async def fetch_job_artifacts(
        self,
        job_id: str,
        destination_dir: Path,
    ) -> AdapterArtifact:
        if job_id not in self.jobs:
            raise ArtifactRetrievalError(f"Job {job_id} not found.")
        dummy_bytes = b"PEFT_LORA_WEIGHTS"
        checksum = hashlib.sha256(dummy_bytes).hexdigest()
        return AdapterArtifact(
            adapter_id=f"adapter-{job_id}",
            domain=ComputeDomain.AUDIO,
            lora_rank=8,
            lora_alpha=16.0,
            target_modules=["audio.fc"],
            weights_checksum=checksum,
            weights_bytes=dummy_bytes,
            file_size_bytes=len(dummy_bytes),
        )

    async def cancel_job(self, job_id: str) -> bool:
        if job_id in self.jobs:
            self.jobs[job_id] = JobStatusResponse(
                job_id=job_id,
                status=JobStatus.CANCELLED,
                progress_pct=0.0,
            )
            return True
        return False


class MockIncompleteComputeProvider:
    """Mock implementation missing cancel_job method."""

    async def submit_training_job(self, payload: TrainingPayload, config: RemoteJobConfig) -> str:
        return "incomplete-job"

    async def poll_job_status(self, job_id: str) -> JobStatusResponse:
        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)

    async def fetch_job_artifacts(self, job_id: str, destination_dir: Path) -> AdapterArtifact:
        raise NotImplementedError()


class MockSynchronousComputeProvider:
    """Mock implementation where methods are sync rather than async coroutines."""

    def submit_training_job(self, payload: TrainingPayload, config: RemoteJobConfig) -> str:
        return "sync-job"

    def poll_job_status(self, job_id: str) -> JobStatusResponse:
        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)

    def fetch_job_artifacts(self, job_id: str, destination_dir: Path) -> AdapterArtifact:
        raise NotImplementedError()

    def cancel_job(self, job_id: str) -> bool:
        return True


def test_compute_provider_conformance_valid() -> None:
    """Verify runtime checkability and conformance verification for valid provider."""
    provider = MockValidComputeProvider()
    assert isinstance(provider, ComputeProvider)
    assert verify_compute_provider_conformance(provider) is True


def test_compute_provider_conformance_incomplete() -> None:
    """Verify failure for provider missing required methods."""
    incomplete = MockIncompleteComputeProvider()
    assert not isinstance(incomplete, ComputeProvider)
    with pytest.raises(TypeError) as exc_info:
        verify_compute_provider_conformance(incomplete)
    assert "does not implement required method 'cancel_job'" in str(exc_info.value)


def test_compute_provider_conformance_synchronous() -> None:
    """Verify failure for provider with synchronous rather than asynchronous methods."""
    sync_provider = MockSynchronousComputeProvider()
    # Protocol runtime checkable checks attribute existence, but conformance verifies coroutine signatures
    with pytest.raises(TypeError) as exc_info:
        verify_compute_provider_conformance(sync_provider)
    assert "must be an asynchronous coroutine function" in str(exc_info.value)


def test_compute_provider_async_execution_flow(tmp_path: Path) -> None:
    """Test full asynchronous execution lifecycle on a mock conforming provider."""
    async def _run_lifecycle() -> None:
        provider = MockValidComputeProvider()
        payload = TrainingPayload(
            trajectories_count=100,
            domain=ComputeDomain.AUDIO,
            dataset_uri="s3://symforge-test/audio.json",
            base_model_name="pde-audio-v1",
            model_type="pde",
        )
        config = RemoteJobConfig(provider_name="mock", timeout_seconds=10.0)

        # 1. Submit job with timeout guardrail
        job_id = await asyncio.wait_for(
            provider.submit_training_job(payload, config),
            timeout=5.0,
        )
        assert job_id == "mock-job-1"

        # 2. Poll job status
        status = await asyncio.wait_for(
            provider.poll_job_status(job_id),
            timeout=5.0,
        )
        assert status.status == JobStatus.RUNNING
        assert status.progress_pct == 10.0

        # 3. Fetch artifacts
        artifact = await asyncio.wait_for(
            provider.fetch_job_artifacts(job_id, tmp_path),
            timeout=5.0,
        )
        assert artifact.adapter_id == "adapter-mock-job-1"
        assert artifact.domain == ComputeDomain.AUDIO
        assert artifact.weights_bytes == b"PEFT_LORA_WEIGHTS"

        # 4. Cancel job
        cancelled = await asyncio.wait_for(
            provider.cancel_job(job_id),
            timeout=5.0,
        )
        assert cancelled is True

        # Verify updated status
        status_after = await asyncio.wait_for(
            provider.poll_job_status(job_id),
            timeout=5.0,
        )
        assert status_after.status == JobStatus.CANCELLED

    asyncio.run(_run_lifecycle())


def test_compute_provider_exceptions_hierarchy() -> None:
    """Verify that domain exceptions inherit properly from ComputeProviderError."""
    assert issubclass(JobSubmissionError, ComputeProviderError)
    assert issubclass(JobTimeoutError, ComputeProviderError)
    assert issubclass(ArtifactRetrievalError, ComputeProviderError)
    assert issubclass(JobExecutionError, ComputeProviderError)
