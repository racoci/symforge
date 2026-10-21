"""MockComputeProvider: Local zero-cost in-memory testbed conforming to ComputeProvider protocol."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import torch

from symforge.compute.contracts import (
    AdapterArtifact,
    ArtifactRetrievalError,
    ComputeDomain,
    JobExecutionError,
    JobStatus,
    JobStatusResponse,
    JobSubmissionError,
    RemoteJobConfig,
    TrainingPayload,
)
from symforge.compute.protocol import ComputeProvider

logger = logging.getLogger(__name__)


class MockExecutionMode(str, Enum):
    """Execution mode for MockComputeProvider job simulation."""

    INSTANT = "instant"
    SIMULATED_STEPS = "simulated_steps"
    FAILING = "failing"


@dataclass
class _MockJobRecord:
    """Internal state record tracking mock compute job execution."""

    job_id: str
    payload: TrainingPayload
    config: RemoteJobConfig
    mode: MockExecutionMode
    status: JobStatus
    progress_pct: float
    poll_count: int
    steps_to_complete: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class MockComputeProvider:
    """Zero-cost local in-memory simulation backend conforming to ComputeProvider."""

    def __init__(
        self,
        default_mode: MockExecutionMode | str = MockExecutionMode.INSTANT,
        steps_to_complete: int = 2,
        failure_message: str = "Simulated remote compute failure: out of GPU memory",
        max_history: int = 1000,
    ) -> None:
        """Initialize MockComputeProvider.

        Args:
            default_mode: Execution mode ('instant', 'simulated_steps', 'failing').
            steps_to_complete: Number of poll steps before completing in simulated_steps mode.
            failure_message: Error string reported on simulated job failure.
            max_history: Maximum number of job records retained in memory.
        """
        if isinstance(default_mode, str):
            default_mode = MockExecutionMode(default_mode)
        self._default_mode = default_mode
        self._steps_to_complete = max(1, steps_to_complete)
        self._failure_message = failure_message
        self._max_history = max(1, max_history)
        self._jobs: Dict[str, _MockJobRecord] = {}
        self._job_counter: int = 0

    @property
    def default_mode(self) -> MockExecutionMode:
        """Return the default execution mode."""
        return self._default_mode

    @property
    def steps_to_complete(self) -> int:
        """Return configured steps to complete in simulated_steps mode."""
        return self._steps_to_complete

    def reset(self) -> None:
        """Clear all in-memory job records and reset job counter."""
        self._jobs.clear()
        self._job_counter = 0

    def _evict_oldest_if_needed(self) -> None:
        """Evict oldest jobs if memory history exceeds capacity."""
        if len(self._jobs) >= self._max_history:
            oldest_key = next(iter(self._jobs))
            del self._jobs[oldest_key]

    async def submit_training_job(
        self,
        payload: TrainingPayload,
        config: RemoteJobConfig,
    ) -> str:
        """Submit a mock training job and register in memory with zero cost.

        Args:
            payload: Training specification and dataset source.
            config: Remote execution environment parameters.

        Returns:
            Unique mock job identifier string.
        """
        self._evict_oldest_if_needed()
        self._job_counter += 1
        job_id = f"mock-job-{uuid.uuid4().hex[:8]}-{self._job_counter}"

        # Resolve mode: tag override > constructor default
        mode_str = config.tags.get("mock_mode", self._default_mode.value)
        try:
            mode = MockExecutionMode(mode_str)
        except ValueError:
            logger.warning(
                "Invalid mock_mode tag '%s', falling back to default '%s'",
                mode_str,
                self._default_mode.value,
            )
            mode = self._default_mode

        # Resolve steps_to_complete
        steps_tag = config.tags.get("steps_to_complete")
        steps = int(steps_tag) if steps_tag and steps_tag.isdigit() else self._steps_to_complete

        now = datetime.now(timezone.utc)
        record = self._create_initial_record(
            job_id=job_id,
            payload=payload,
            config=config,
            mode=mode,
            steps=steps,
            created_at=now,
        )
        self._jobs[job_id] = record
        return job_id

    def _create_initial_record(
        self,
        job_id: str,
        payload: TrainingPayload,
        config: RemoteJobConfig,
        mode: MockExecutionMode,
        steps: int,
        created_at: datetime,
    ) -> _MockJobRecord:
        """Construct the initial job record according to the execution mode."""
        if mode == MockExecutionMode.INSTANT:
            return _MockJobRecord(
                job_id=job_id,
                payload=payload,
                config=config,
                mode=mode,
                status=JobStatus.COMPLETED,
                progress_pct=100.0,
                poll_count=0,
                steps_to_complete=steps,
                created_at=created_at,
                started_at=created_at,
                completed_at=created_at,
                metrics={"train_loss": 0.042, "val_loss": 0.051, "val_f1": 0.965},
            )

        if mode == MockExecutionMode.SIMULATED_STEPS:
            return _MockJobRecord(
                job_id=job_id,
                payload=payload,
                config=config,
                mode=mode,
                status=JobStatus.PENDING,
                progress_pct=0.0,
                poll_count=0,
                steps_to_complete=steps,
                created_at=created_at,
                started_at=created_at,
                metrics={"train_loss": 0.850, "val_loss": 0.920, "val_f1": 0.300},
            )

        # MockExecutionMode.FAILING
        return _MockJobRecord(
            job_id=job_id,
            payload=payload,
            config=config,
            mode=mode,
            status=JobStatus.RUNNING,
            progress_pct=25.0,
            poll_count=0,
            steps_to_complete=steps,
            created_at=created_at,
            started_at=created_at,
            metrics={"train_loss": 0.720, "val_loss": 0.810, "val_f1": 0.350},
        )

    async def poll_job_status(self, job_id: str) -> JobStatusResponse:
        """Query execution status and update step progression.

        Args:
            job_id: Unique job identifier.

        Returns:
            JobStatusResponse with current status, metrics, and progress.

        Raises:
            JobExecutionError: If job_id is unknown.
        """
        record = self._jobs.get(job_id)
        if record is None:
            raise JobExecutionError(f"Job '{job_id}' not found in mock provider registry.")

        # Terminal states are immutable
        if record.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return self._build_status_response(record)

        if record.mode == MockExecutionMode.FAILING:
            record.status = JobStatus.FAILED
            record.error_message = self._failure_message
            record.completed_at = datetime.now(timezone.utc)
            return self._build_status_response(record)

        if record.mode == MockExecutionMode.SIMULATED_STEPS:
            record.poll_count += 1
            if record.poll_count >= record.steps_to_complete:
                record.status = JobStatus.COMPLETED
                record.progress_pct = 100.0
                record.completed_at = datetime.now(timezone.utc)
                record.metrics = {
                    "train_loss": 0.038,
                    "val_loss": 0.045,
                    "val_f1": 0.975,
                }
            else:
                record.status = JobStatus.RUNNING
                frac = record.poll_count / record.steps_to_complete
                record.progress_pct = round(frac * 100.0, 1)
                train_loss = round(0.85 * (1.0 - 0.9 * frac) + 0.02, 4)
                val_loss = round(0.92 * (1.0 - 0.85 * frac) + 0.03, 4)
                val_f1 = round(0.30 + 0.65 * frac, 4)
                record.metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                }
            return self._build_status_response(record)

        return self._build_status_response(record)

    def _build_status_response(self, record: _MockJobRecord) -> JobStatusResponse:
        """Convert an internal job record into a JobStatusResponse contract."""
        return JobStatusResponse(
            job_id=record.job_id,
            status=record.status,
            progress_pct=record.progress_pct,
            error_message=record.error_message,
            started_at=record.started_at,
            completed_at=record.completed_at,
            estimated_cost_usd=0.0,
            metrics=dict(record.metrics),
        )

    async def fetch_job_artifacts(
        self,
        job_id: str,
        destination_dir: Path,
    ) -> AdapterArtifact:
        """Synthesize and serialize LoRA adapter weight tensors to disk.

        Args:
            job_id: Unique job identifier.
            destination_dir: Directory where the adapter weights file will be written.

        Returns:
            AdapterArtifact metadata describing the stored weights.

        Raises:
            ArtifactRetrievalError: If job does not exist, was cancelled, failed, or has not completed.
        """
        record = self._jobs.get(job_id)
        if record is None:
            raise ArtifactRetrievalError(
                f"Cannot retrieve artifacts: Job '{job_id}' not found."
            )

        if record.status == JobStatus.CANCELLED:
            raise ArtifactRetrievalError(
                f"Cannot retrieve artifacts: Job '{job_id}' was CANCELLED."
            )

        if record.status == JobStatus.FAILED:
            raise ArtifactRetrievalError(
                f"Cannot retrieve artifacts: Job '{job_id}' FAILED: {record.error_message}"
            )

        if record.status != JobStatus.COMPLETED:
            raise ArtifactRetrievalError(
                f"Cannot retrieve artifacts: Job '{job_id}' is not COMPLETED (status={record.status.value})."
            )

        # Determine target modules by domain
        target_modules = self._resolve_target_modules(record.payload.domain)
        rank = record.config.lora_rank
        alpha = record.config.lora_alpha

        # Synthesize PyTorch LoRA weight tensors
        state_dict = self._synthesize_lora_weights(
            job_id=job_id,
            target_modules=target_modules,
            lora_rank=rank,
        )

        # Serialize into destination directory
        destination_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = destination_dir / f"{job_id}_adapter.pt"

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        file_bytes = buffer.getvalue()
        artifact_file.write_bytes(file_bytes)

        checksum = hashlib.sha256(file_bytes).hexdigest()

        return AdapterArtifact(
            adapter_id=f"adapter-{job_id}",
            domain=record.payload.domain,
            lora_rank=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            weights_checksum=checksum,
            weights_uri=f"file://{artifact_file.resolve()}",
            weights_bytes=file_bytes,
            file_size_bytes=len(file_bytes),
            metadata={
                "job_id": job_id,
                "mode": record.mode.value,
                "base_model": record.payload.base_model_name,
                "model_type": record.payload.model_type,
                "trajectories_count": record.payload.trajectories_count,
                "target_file": str(artifact_file.resolve()),
            },
        )

    @staticmethod
    def _resolve_target_modules(domain: ComputeDomain) -> List[str]:
        """Map compute domain to standard neural target module names."""
        if domain == ComputeDomain.AUDIO:
            return ["audio_dsp.carrier_proj", "audio_dsp.modulator_proj"]
        if domain == ComputeDomain.SDF:
            return ["sdf_net.fc1", "sdf_net.fc2"]
        return ["hw_opt.alu_proj", "hw_opt.latency_proj"]

    @staticmethod
    def _synthesize_lora_weights(
        job_id: str,
        target_modules: List[str],
        lora_rank: int,
    ) -> Dict[str, torch.Tensor]:
        """Synthesize deterministic LoRA weight matrices matching rank and dimension."""
        seed = int(hashlib.md5(job_id.encode("utf-8")).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed)
        state_dict: Dict[str, torch.Tensor] = {}

        for mod in target_modules:
            state_dict[f"{mod}.lora_A.weight"] = torch.randn(
                lora_rank, 32, generator=generator
            )
            state_dict[f"{mod}.lora_B.weight"] = torch.zeros(
                32, lora_rank
            )

        return state_dict

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active or pending job.

        Args:
            job_id: Unique job identifier.

        Returns:
            True if job was cancelled, False if job not found or already terminal.
        """
        record = self._jobs.get(job_id)
        if record is None:
            return False

        if record.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        record.status = JobStatus.CANCELLED
        record.completed_at = datetime.now(timezone.utc)
        return True
