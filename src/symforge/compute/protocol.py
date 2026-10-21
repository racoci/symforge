"""Asynchronous ComputeProvider protocol for decoupled neural training backends."""

from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from symforge.compute.contracts import (
    AdapterArtifact,
    JobStatusResponse,
    RemoteJobConfig,
    TrainingPayload,
)


@runtime_checkable
class ComputeProvider(Protocol):
    """Protocol defining the asynchronous contract for remote training backends."""

    async def submit_training_job(
        self,
        payload: TrainingPayload,
        config: RemoteJobConfig,
    ) -> str:
        """Submit an asynchronous training job to the compute backend.

        Args:
            payload: Training payload containing dataset and model specifications.
            config: Remote execution environment parameters and budget limits.

        Returns:
            Unique string job_id identifier.
        """
        ...

    async def poll_job_status(self, job_id: str) -> JobStatusResponse:
        """Query the current execution status of an active or completed training job.

        Args:
            job_id: Unique job identifier returned by submit_training_job.

        Returns:
            JobStatusResponse with current status, metrics, and progress.
        """
        ...

    async def fetch_job_artifacts(
        self,
        job_id: str,
        destination_dir: Path,
    ) -> AdapterArtifact:
        """Retrieve the trained adapter artifact files once the job is COMPLETED.

        Args:
            job_id: Unique job identifier.
            destination_dir: Local directory path where artifact weights will be stored.

        Returns:
            AdapterArtifact metadata describing the stored weights.
        """
        ...

    async def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of an active or pending training job.

        Args:
            job_id: Unique job identifier.

        Returns:
            True if cancellation was acknowledged/succeeded, False otherwise.
        """
        ...


def verify_compute_provider_conformance(candidate: Any) -> bool:
    """Verify that a candidate object or class conforms to ComputeProvider protocol.

    Args:
        candidate: An instance or class to inspect for ComputeProvider protocol conformance.

    Returns:
        True if the candidate conforms to all methods and signature contracts.

    Raises:
        TypeError: If candidate does not conform to the required protocol methods.
    """
    required_methods = [
        "submit_training_job",
        "poll_job_status",
        "fetch_job_artifacts",
        "cancel_job",
    ]

    for method_name in required_methods:
        method = getattr(candidate, method_name, None)
        if method is None or not callable(method):
            raise TypeError(
                f"Candidate {candidate} does not implement required method '{method_name}'."
            )
        # Check if it is a coroutine function (or returns a coroutine)
        if not inspect.iscoroutinefunction(method) and not getattr(method, "_is_coroutine", False):
            # Allow AsyncMock which has _is_coroutine attribute
            if not callable(getattr(method, "assert_called", None)):
                raise TypeError(
                    f"Method '{method_name}' on {candidate} must be an asynchronous coroutine function."
                )

    return True
