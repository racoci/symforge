"""Data contracts and schemas for asynchronous decoupled compute execution."""

from __future__ import annotations
import base64
from datetime import datetime, timezone
from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_SHA256_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")


class JobStatus(str, Enum):
    """Execution status of an asynchronous remote training job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ComputeDomain(str, Enum):
    """Problem domain category for compute dispatch and adapter specialization."""

    AUDIO = "audio"
    SDF = "sdf"
    HARDWARE = "hardware"


class TrainingPayload(BaseModel):
    """Specification of training data and hyperparameter configurations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trajectories_count: int = Field(
        ...,
        ge=1,
        description="Number of trajectories or samples in the dataset.",
    )
    domain: ComputeDomain = Field(
        ...,
        description="Compute domain (audio, sdf, hardware).",
    )
    dataset_uri: Optional[str] = Field(
        default=None,
        description="Remote URI pointing to training dataset (e.g. s3://, https://, file://).",
    )
    dataset_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Inline dataset payload dictionary if not using remote URI.",
    )
    base_model_name: str = Field(
        ...,
        min_length=1,
        description="Base foundation model or checkpoint name.",
    )
    model_type: str = Field(
        ...,
        min_length=1,
        description="Model architecture category (e.g., 'pde', 'tgc', 'lora').",
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary hyperparameter dictionary for training loop.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of payload creation.",
    )

    @model_validator(mode="after")
    def validate_dataset_source(self) -> TrainingPayload:
        """Ensure at least one dataset source (URI or inline payload) is provided."""
        if not self.dataset_uri and self.dataset_payload is None:
            raise ValueError(
                "At least one of 'dataset_uri' or 'dataset_payload' must be provided."
            )
        return self


class RemoteJobConfig(BaseModel):
    """Configuration parameters for remote compute resource provisioning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_name: str = Field(
        ...,
        min_length=1,
        description="Target provider name (e.g. 'mock', 'modal', 'sagemaker').",
    )
    hardware_accelerator: str = Field(
        default="CPU",
        description="Hardware accelerator tier (e.g., 'CPU', 'T4', 'A10G', 'A100').",
    )
    timeout_seconds: float = Field(
        default=600.0,
        gt=0.0,
        description="Strict execution timeout in seconds (> 0.0).",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts on transient network or infrastructure failure.",
    )
    max_cost_usd: float = Field(
        default=5.0,
        ge=0.0,
        description="Maximum budget cap in USD for job execution (>= 0.0).",
    )
    lora_rank: int = Field(
        default=8,
        ge=1,
        description="LoRA adapter rank r (>= 1).",
    )
    lora_alpha: float = Field(
        default=16.0,
        gt=0.0,
        description="LoRA scaling coefficient alpha (> 0.0).",
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="LoRA dropout rate in [0.0, 1.0].",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Descriptive metadata tags for billing and tracking.",
    )


class JobStatusResponse(BaseModel):
    """Response payload containing current execution status and telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the training job.",
    )
    status: JobStatus = Field(
        ...,
        description="Current execution status.",
    )
    progress_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Job completion percentage in [0.0, 100.0].",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details or failure reason if status is FAILED.",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when job execution began.",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when job execution completed or failed.",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Accrued cost in USD for the compute session (>= 0.0).",
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Training and validation metrics (e.g. train_loss, val_loss).",
    )


class AdapterArtifact(BaseModel):
    """Trained LoRA adapter weights and verification artifact metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    adapter_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the adapter artifact.",
    )
    domain: ComputeDomain = Field(
        ...,
        description="Domain specialization of the adapter.",
    )
    lora_rank: int = Field(
        ...,
        ge=1,
        description="LoRA adapter rank dimension.",
    )
    lora_alpha: float = Field(
        ...,
        gt=0.0,
        description="LoRA scaling factor.",
    )
    target_modules: List[str] = Field(
        ...,
        min_length=1,
        description="Target module paths adapted by LoRA tensors.",
    )
    weights_checksum: str = Field(
        ...,
        description="Canonical SHA-256 hex checksum (64 characters).",
    )
    weights_uri: Optional[str] = Field(
        default=None,
        description="Remote storage URI pointing to adapter weights file.",
    )
    weights_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw binary payload of the adapter weights.",
    )
    weights_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded string of adapter weights.",
    )
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of adapter weights file in bytes (>= 0).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary adapter metadata and hyperparameters.",
    )

    @field_validator("weights_checksum")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        """Validate that weights_checksum is a valid 64-character SHA-256 hex string."""
        if not isinstance(value, str):
            raise ValueError("weights_checksum must be a string.")
        stripped = value.strip()
        if not _SHA256_REGEX.match(stripped):
            raise ValueError(
                f"weights_checksum '{stripped}' is not a valid 64-character SHA-256 hex string."
            )
        return stripped.lower()

    @model_validator(mode="before")
    @classmethod
    def synchronize_weights_representation(cls, data: Any) -> Any:
        """Synchronize weights_bytes and weights_base64 representations."""
        if not isinstance(data, dict):
            return data

        raw_bytes = data.get("weights_bytes")
        b64_str = data.get("weights_base64")

        # Decode base64 if bytes missing
        if raw_bytes is None and b64_str is not None:
            try:
                decoded = base64.b64decode(b64_str.encode("utf-8"))
                data = dict(data)
                data["weights_bytes"] = decoded
            except Exception as exc:
                prefix = b64_str[:32] if isinstance(b64_str, str) else type(b64_str)
                logger.error(
                    "Failed to decode base64 weights_base64: prefix=%s, exc=%s",
                    prefix,
                    exc,
                )
                raise ValueError(f"Invalid base64 encoding in weights_base64: {exc}") from exc

        # Encode bytes if base64 missing
        elif raw_bytes is not None and b64_str is None:
            if isinstance(raw_bytes, bytes):
                data = dict(data)
                data["weights_base64"] = base64.b64encode(raw_bytes).decode("utf-8")

        return data

    @model_validator(mode="after")
    def validate_weights_source(self) -> AdapterArtifact:
        """Ensure at least one source for weights (URI, bytes, or base64) is provided."""
        if not self.weights_uri and self.weights_bytes is None and self.weights_base64 is None:
            raise ValueError(
                "At least one of 'weights_uri', 'weights_bytes', or 'weights_base64' must be provided."
            )
        return self


# =====================================================================
# Domain Exceptions for Compute Provider Failures
# =====================================================================

class ComputeProviderError(Exception):
    """Base exception for all compute provider failures."""


class JobSubmissionError(ComputeProviderError):
    """Raised when remote job submission is rejected or fails."""


class JobTimeoutError(ComputeProviderError):
    """Raised when job execution or polling exceeds allowed timeout."""


class ArtifactRetrievalError(ComputeProviderError):
    """Raised when fetching or verifying adapter artifacts fails."""


class JobExecutionError(ComputeProviderError):
    """Raised when a remote training job fails during execution."""
