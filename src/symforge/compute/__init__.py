"""Decoupled Compute Provider and Data Contracts for Remote Training."""

from symforge.compute.contracts import (
    AdapterArtifact,
    ArtifactRetrievalError,
    ComputeDomain,
    ComputeProviderError,
    JobExecutionError,
    JobStatus,
    JobStatusResponse,
    JobSubmissionError,
    JobTimeoutError,
    RemoteJobConfig,
    TrainingPayload,
)
from symforge.compute.mock_provider import (
    MockComputeProvider,
    MockExecutionMode,
)
from symforge.compute.protocol import (
    ComputeProvider,
    verify_compute_provider_conformance,
)

__all__ = [
    "AdapterArtifact",
    "ArtifactRetrievalError",
    "ComputeDomain",
    "ComputeProvider",
    "ComputeProviderError",
    "JobExecutionError",
    "JobStatus",
    "JobStatusResponse",
    "JobSubmissionError",
    "JobTimeoutError",
    "MockComputeProvider",
    "MockExecutionMode",
    "RemoteJobConfig",
    "TrainingPayload",
    "verify_compute_provider_conformance",
]
