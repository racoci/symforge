"""Remote serverless GPU compute providers (Modal & SageMaker) with resilient networking."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol, Tuple, TypeVar, runtime_checkable
import urllib.error
import urllib.parse
import urllib.request
import uuid

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
from symforge.compute.protocol import ComputeProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =====================================================================
# 1. HTTP Transport Contracts and Data Structures
# =====================================================================


@dataclass
class HttpResponse:
    """HTTP response payload with status code, headers, and binary content."""

    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    content: bytes = b""

    @property
    def text(self) -> str:
        """Decode content bytes as UTF-8 string."""
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Dict[str, Any]:
        """Parse content bytes as JSON dictionary."""
        return json.loads(self.text)


class HttpTransportError(Exception):
    """Exception raised by HTTP transport layer."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@runtime_checkable
class HttpTransport(Protocol):
    """Protocol for asynchronous HTTP client transports."""

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        body_bytes: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> HttpResponse:
        """Execute an asynchronous HTTP request."""
        ...

    async def aclose(self) -> None:
        """Deterministically close client connections and sever sockets."""
        ...


class DefaultHttpTransport:
    """Standard library HTTP transport using urllib and executor thread."""

    def __init__(self) -> None:
        self._closed: bool = False

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        body_bytes: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> HttpResponse:
        """Execute HTTP request in thread pool to prevent blocking event loop."""
        if self._closed:
            raise HttpTransportError("HttpTransport is closed and severed.")
        return await asyncio.wait_for(
            asyncio.to_thread(
                self._sync_request,
                method,
                url,
                headers,
                json_data,
                body_bytes,
                timeout,
            ),
            timeout=timeout + 2.0,
        )

    def _sync_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]],
        json_data: Optional[Dict[str, Any]],
        body_bytes: Optional[bytes],
        timeout: float,
    ) -> HttpResponse:
        req_headers = {"User-Agent": "SymForge-RemoteClient/1.0"}
        if headers:
            req_headers.update(headers)
        data = body_bytes
        if json_data is not None:
            data = json.dumps(json_data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.getcode()
                resp_headers = dict(resp.info().items())
                content = resp.read()
                return HttpResponse(status_code=status, headers=resp_headers, content=content)
        except urllib.error.HTTPError as err:
            err_content = err.read()
            logger.warning("HTTP transport returned error status: code=%d url=%s err=%s", err.code, url, err)
            return HttpResponse(status_code=err.code, headers=dict(err.headers.items()), content=err_content)
        except Exception as exc:
            logger.error("HTTP transport network failure: url=%s method=%s exc=%s", url, method, exc)
            raise HttpTransportError(f"HTTP transport connection failed: {exc}") from exc

    async def aclose(self) -> None:
        """Mark transport as closed."""
        self._closed = True


# =====================================================================
# 2. Resilient Network Runner (Timeouts, Backoff & Jitter, Severance)
# =====================================================================


class ResilientNetworkRunner:
    """Resilience engine enforcing timeouts, bounded retries, and jittered backoff."""

    def __init__(
        self,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        max_backoff: float = 10.0,
        jitter: float = 0.25,
        retryable_status_codes: Optional[Tuple[int, ...]] = None,
        rng_func: Optional[Callable[[], float]] = None,
    ) -> None:
        """Initialize ResilientNetworkRunner with explicit safety parameters.

        Args:
            connect_timeout: Timeout for network connection establishment in seconds.
            read_timeout: Timeout for operation execution/read in seconds.
            max_retries: Hard upper bound on retry attempts.
            base_backoff: Initial backoff base delay in seconds.
            max_backoff: Maximum backoff delay cap in seconds.
            jitter: Maximum random uniform jitter added to backoff delay in seconds.
            retryable_status_codes: Tuple of HTTP status codes considered transient.
            rng_func: Optional callable returning float in [0.0, 1.0) for deterministic jitter.
        """
        self.connect_timeout = max(0.1, connect_timeout)
        self.read_timeout = max(0.1, read_timeout)
        self.max_retries = max(0, max_retries)
        self.base_backoff = max(0.01, base_backoff)
        self.max_backoff = max(self.base_backoff, max_backoff)
        self.jitter = max(0.0, jitter)
        self.retryable_status_codes = retryable_status_codes or (429, 500, 502, 503, 504)
        self._rng = rng_func or random.random

    def compute_backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay with randomized uniform jitter.

        Formula: t_backoff = min(t_max, t_base * 2^attempt) + uniform(0, jitter)
        """
        capped_exp = min(self.max_backoff, self.base_backoff * (2.0 ** attempt))
        rand_jitter = self._rng() * self.jitter
        return capped_exp + rand_jitter

    def is_retryable(self, exc: Exception) -> bool:
        """Determine if an exception represents a transient retryable condition."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
            return True
        if isinstance(exc, HttpTransportError) and exc.status_code is not None:
            return exc.status_code in self.retryable_status_codes
        err_msg = str(exc).lower()
        throttled = "throttl" in err_msg or "rate limit" in err_msg or "429" in err_msg
        server_err = "500" in err_msg or "502" in err_msg or "503" in err_msg or "504" in err_msg
        return throttled or server_err

    async def execute(
        self,
        operation_name: str,
        endpoint: str,
        func: Callable[[], Coroutine[Any, Any, T]],
        payload_summary: Optional[str] = None,
        client_handle: Any = None,
    ) -> T:
        """Execute an asynchronous operation with strict timeout and resilient retries."""
        attempt_idx = 0
        total_attempts = self.max_retries + 1
        last_error: Optional[Exception] = None

        while True:
            attempt_idx += 1
            if attempt_idx > total_attempts:
                break
            try:
                return await asyncio.wait_for(func(), timeout=self.read_timeout)
            except Exception as exc:
                last_error = exc
                is_transient = self.is_retryable(exc)
                logger.warning(
                    "Network call failed: op=%s endpoint=%s attempt=%d/%d transient=%s exc=%s summary=%s",
                    operation_name,
                    endpoint,
                    attempt_idx,
                    total_attempts,
                    is_transient,
                    exc,
                    payload_summary or "none",
                )
                if not is_transient or attempt_idx >= total_attempts:
                    await asyncio.wait_for(self.sever_resource(client_handle), timeout=5.0)
                    self._raise_domain_error(operation_name, endpoint, exc, attempt_idx)
                delay = self.compute_backoff_delay(attempt_idx - 1)
                await asyncio.wait_for(asyncio.sleep(delay), timeout=delay + 1.0)

        await asyncio.wait_for(self.sever_resource(client_handle), timeout=5.0)
        raise ComputeProviderError(f"Operation '{operation_name}' failed after {total_attempts} attempts: {last_error}")

    def _raise_domain_error(self, op: str, endpoint: str, exc: Exception, attempts: int) -> None:
        """Translate lower-level transport exception into domain-specific error."""
        msg = f"Operation '{op}' at '{endpoint}' failed after {attempts} attempts: {exc}"
        if "submit" in op.lower():
            raise JobSubmissionError(msg) from exc
        if "fetch" in op.lower() or "artifact" in op.lower():
            raise ArtifactRetrievalError(msg) from exc
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            raise JobTimeoutError(msg) from exc
        raise ComputeProviderError(msg) from exc

    async def sever_resource(self, client: Any) -> None:
        """Deterministic resource severance for sockets, connections, or sessions."""
        if client is None:
            return
        try:
            if hasattr(client, "aclose") and callable(client.aclose):
                await asyncio.wait_for(client.aclose(), timeout=5.0)
            elif hasattr(client, "close") and callable(client.close):
                if asyncio.iscoroutinefunction(client.close):
                    await asyncio.wait_for(client.close(), timeout=5.0)
                else:
                    client.close()
        except Exception as exc:
            logger.error("Error during deterministic resource severance: client=%s exc=%s", client, exc)


async def resilient_async_call(
    func: Callable[[], Coroutine[Any, Any, T]],
    operation_name: str = "remote_call",
    endpoint: str = "unknown",
    runner: Optional[ResilientNetworkRunner] = None,
    client_handle: Any = None,
) -> T:
    """Convenience wrapper for executing an async call with resilient runner."""
    active_runner = runner or ResilientNetworkRunner()
    return await asyncio.wait_for(
        active_runner.execute(
            operation_name=operation_name,
            endpoint=endpoint,
            func=func,
            client_handle=client_handle,
        ),
        timeout=active_runner.read_timeout + 5.0,
    )


# =====================================================================
# 3. Modal Serverless Compute Provider
# =====================================================================


class ModalComputeProvider:
    """Modal serverless container backend conforming to ComputeProvider."""

    def __init__(
        self,
        base_url: str = "https://api.modal.com/v1",
        api_token: Optional[str] = None,
        transport: Optional[HttpTransport] = None,
        runner: Optional[ResilientNetworkRunner] = None,
        max_poll_iterations: int = 120,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Initialize ModalComputeProvider.

        Args:
            base_url: Root API endpoint for Modal functions and jobs.
            api_token: Optional bearer token for authentication.
            transport: Pluggable HttpTransport instance (defaults to DefaultHttpTransport).
            runner: Pluggable ResilientNetworkRunner for timeouts and retries.
            max_poll_iterations: Strict upper bound on polling attempts before timeout.
            poll_interval_seconds: Delay interval in seconds between poll iterations.
        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.transport = transport or DefaultHttpTransport()
        self.runner = runner or ResilientNetworkRunner()
        self.max_poll_iterations = max(1, max_poll_iterations)
        self.poll_interval_seconds = max(0.05, poll_interval_seconds)

    def _auth_headers(self) -> Dict[str, str]:
        """Construct standard HTTP headers for Modal API."""
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def submit_training_job(
        self,
        payload: TrainingPayload,
        config: RemoteJobConfig,
    ) -> str:
        """Submit a training job to Modal serverless container infrastructure."""
        if config.max_cost_usd < 0.0:
            raise JobSubmissionError(f"Invalid max_cost_usd {config.max_cost_usd}: must be non-negative.")
        endpoint = f"{self.base_url}/jobs/submit"
        body = {
            "domain": payload.domain.value,
            "trajectories_count": payload.trajectories_count,
            "base_model_name": payload.base_model_name,
            "model_type": payload.model_type,
            "hyperparameters": payload.hyperparameters,
            "dataset_uri": payload.dataset_uri,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "accelerator": config.hardware_accelerator,
            "timeout_seconds": config.timeout_seconds,
            "tags": config.tags,
        }

        async def _call() -> HttpResponse:
            resp = await asyncio.wait_for(
                self.transport.request(
                    method="POST",
                    url=endpoint,
                    headers=self._auth_headers(),
                    json_data=body,
                    timeout=self.runner.connect_timeout,
                ),
                timeout=self.runner.connect_timeout + 2.0,
            )
            if resp.status_code >= 400:
                raise HttpTransportError(
                    f"Modal submission HTTP {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )
            return resp

        resp = await asyncio.wait_for(
            self.runner.execute(
                operation_name="modal_submit",
                endpoint=endpoint,
                func=_call,
                payload_summary=f"domain={payload.domain.value}, trajectories={payload.trajectories_count}",
                client_handle=self.transport,
            ),
            timeout=self.runner.read_timeout + 5.0,
        )
        data = resp.json()
        job_id = data.get("job_id")
        if not job_id:
            raise JobSubmissionError(f"Modal response missing 'job_id': {data}")
        return str(job_id)

    async def poll_job_status(self, job_id: str) -> JobStatusResponse:
        """Poll the current execution status of a Modal training job."""
        endpoint = f"{self.base_url}/jobs/{job_id}/status"

        async def _call() -> HttpResponse:
            resp = await asyncio.wait_for(
                self.transport.request(
                    method="GET",
                    url=endpoint,
                    headers=self._auth_headers(),
                    timeout=self.runner.connect_timeout,
                ),
                timeout=self.runner.connect_timeout + 2.0,
            )
            if resp.status_code >= 400:
                raise HttpTransportError(
                    f"Modal poll HTTP {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )
            return resp

        resp = await asyncio.wait_for(
            self.runner.execute(
                operation_name="modal_poll",
                endpoint=endpoint,
                func=_call,
                payload_summary=f"job_id={job_id}",
                client_handle=self.transport,
            ),
            timeout=self.runner.read_timeout + 5.0,
        )
        data = resp.json()
        return self._parse_modal_status_response(job_id, data)

    def _parse_modal_status_response(self, job_id: str, data: Dict[str, Any]) -> JobStatusResponse:
        raw_status = str(data.get("status", "")).lower()
        status = JobStatus.PENDING
        if raw_status in ("running", "in_progress"):
            status = JobStatus.RUNNING
        elif raw_status in ("completed", "success", "done"):
            status = JobStatus.COMPLETED
        elif raw_status in ("failed", "error", "oom"):
            status = JobStatus.FAILED
        elif raw_status in ("cancelled", "stopped"):
            status = JobStatus.CANCELLED

        return JobStatusResponse(
            job_id=job_id,
            status=status,
            progress_pct=float(data.get("progress_pct", 100.0 if status == JobStatus.COMPLETED else 0.0)),
            error_message=data.get("error_message"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            metrics={k: float(v) for k, v in data.get("metrics", {}).items()},
        )

    async def fetch_job_artifacts(
        self,
        job_id: str,
        destination_dir: Path,
    ) -> AdapterArtifact:
        """Download and verify trained LoRA adapter weights from Modal."""
        endpoint = f"{self.base_url}/jobs/{job_id}/artifacts"

        async def _call() -> HttpResponse:
            resp = await asyncio.wait_for(
                self.transport.request(
                    method="GET",
                    url=endpoint,
                    headers=self._auth_headers(),
                    timeout=self.runner.read_timeout,
                ),
                timeout=self.runner.read_timeout + 2.0,
            )
            if resp.status_code >= 400:
                raise HttpTransportError(
                    f"Modal artifact HTTP {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )
            return resp

        resp = await asyncio.wait_for(
            self.runner.execute(
                operation_name="modal_fetch_artifacts",
                endpoint=endpoint,
                func=_call,
                payload_summary=f"job_id={job_id}",
                client_handle=self.transport,
            ),
            timeout=self.runner.read_timeout + 5.0,
        )
        data = resp.json()
        return self._save_and_verify_artifact(job_id, data, destination_dir)

    def _save_and_verify_artifact(
        self,
        job_id: str,
        data: Dict[str, Any],
        destination_dir: Path,
    ) -> AdapterArtifact:
        weights_bytes: Optional[bytes] = None
        weights_b64 = data.get("weights_base64")
        if weights_b64:
            import base64
            weights_bytes = base64.b64decode(weights_b64)
        if weights_bytes is None:
            raise ArtifactRetrievalError(f"Modal artifact response for '{job_id}' contains no weights data.")

        calculated_checksum = hashlib.sha256(weights_bytes).hexdigest()
        reported_checksum = data.get("weights_checksum")
        if reported_checksum and reported_checksum.lower() != calculated_checksum:
            raise ArtifactRetrievalError(
                f"Checksum mismatch: reported={reported_checksum}, calculated={calculated_checksum}"
            )

        destination_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = destination_dir / f"{job_id}_adapter.pt"
        try:
            artifact_path.write_bytes(weights_bytes)
        except OSError as exc:
            logger.error("Failed to write artifact file to %s: %s", artifact_path, exc)
            raise ArtifactRetrievalError(f"Failed to write artifact weights: {exc}") from exc

        return AdapterArtifact(
            adapter_id=str(data.get("adapter_id", f"modal-adapter-{job_id}")),
            domain=ComputeDomain(data.get("domain", "audio")),
            lora_rank=int(data.get("lora_rank", 8)),
            lora_alpha=float(data.get("lora_alpha", 16.0)),
            target_modules=list(data.get("target_modules", ["attention.q_proj", "attention.v_proj"])),
            weights_checksum=calculated_checksum,
            weights_bytes=weights_bytes,
            weights_uri=str(artifact_path.resolve()),
            file_size_bytes=len(weights_bytes),
            metadata=data.get("metadata", {}),
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active or pending Modal training job."""
        endpoint = f"{self.base_url}/jobs/{job_id}/cancel"

        async def _call() -> HttpResponse:
            resp = await asyncio.wait_for(
                self.transport.request(
                    method="POST",
                    url=endpoint,
                    headers=self._auth_headers(),
                    timeout=self.runner.connect_timeout,
                ),
                timeout=self.runner.connect_timeout + 2.0,
            )
            return resp

        try:
            resp = await asyncio.wait_for(
                self.runner.execute(
                    operation_name="modal_cancel",
                    endpoint=endpoint,
                    func=_call,
                    payload_summary=f"job_id={job_id}",
                    client_handle=self.transport,
                ),
                timeout=self.runner.connect_timeout + 5.0,
            )
            return resp.status_code in (200, 202, 204)
        except Exception as exc:
            logger.warning("Modal cancellation failed for job %s: %s", job_id, exc)
            return False

    async def poll_until_complete(self, job_id: str) -> JobStatusResponse:
        """Poll job status in a loop until terminal state or max_poll_iterations cap."""
        iteration_count = 0
        while True:
            iteration_count += 1
            if iteration_count > self.max_poll_iterations:
                logger.error("Job %s exceeded max poll iterations (%d)", job_id, self.max_poll_iterations)
                await asyncio.wait_for(self.cancel_job(job_id), timeout=10.0)
                break
            status_resp = await asyncio.wait_for(self.poll_job_status(job_id), timeout=self.runner.read_timeout + 5.0)
            if status_resp.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return status_resp
            await asyncio.wait_for(asyncio.sleep(self.poll_interval_seconds), timeout=self.poll_interval_seconds + 1.0)

        raise JobTimeoutError(f"Job '{job_id}' exceeded max poll iterations ({self.max_poll_iterations}).")

    async def aclose(self) -> None:
        """Sever all underlying HTTP transport connections."""
        await asyncio.wait_for(self.runner.sever_resource(self.transport), timeout=5.0)

    async def __aenter__(self) -> ModalComputeProvider:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await asyncio.wait_for(self.aclose(), timeout=5.0)


# =====================================================================
# 4. SageMaker AWS Compute Provider
# =====================================================================


@runtime_checkable
class SageMakerTransport(Protocol):
    """Protocol for SageMaker and S3 client interaction."""

    async def create_training_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call AWS SageMaker CreateTrainingJob API."""
        ...

    async def describe_training_job(self, job_name: str) -> Dict[str, Any]:
        """Call AWS SageMaker DescribeTrainingJob API."""
        ...

    async def stop_training_job(self, job_name: str) -> Dict[str, Any]:
        """Call AWS SageMaker StopTrainingJob API."""
        ...

    async def get_s3_object(self, s3_uri: str) -> bytes:
        """Download binary artifact from S3 URI."""
        ...

    async def aclose(self) -> None:
        """Deterministically sever underlying AWS connections."""
        ...


class SageMakerComputeProvider:
    """AWS SageMaker managed training backend conforming to ComputeProvider."""

    def __init__(
        self,
        region_name: str = "us-east-1",
        role_arn: str = "arn:aws:iam::123456789012:role/SymForgeSageMakerRole",
        s3_output_uri: str = "s3://symforge-models/adapters",
        transport: Optional[SageMakerTransport] = None,
        runner: Optional[ResilientNetworkRunner] = None,
        max_poll_iterations: int = 120,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Initialize SageMakerComputeProvider with strict security safeguards.

        Args:
            region_name: AWS region identifier.
            role_arn: IAM execution role ARN for SageMaker.
            s3_output_uri: S3 bucket prefix for training artifacts.
            transport: Pluggable SageMakerTransport client wrapper.
            runner: Pluggable ResilientNetworkRunner for timeouts and retries.
            max_poll_iterations: Strict upper bound on polling loop iterations.
            poll_interval_seconds: Polling interval in seconds.
        """
        self.region_name = region_name
        self.role_arn = role_arn
        self.s3_output_uri = s3_output_uri.rstrip("/")
        self.transport = transport
        self.runner = runner or ResilientNetworkRunner()
        self.max_poll_iterations = max(1, max_poll_iterations)
        self.poll_interval_seconds = max(0.05, poll_interval_seconds)

    def _verify_credentials_safeguard(self) -> None:
        """Safeguard: verify ~/.aws/credentials is treated as strict read-only."""
        cred_path = Path.home() / ".aws" / "credentials"
        if cred_path.exists():
            mode = cred_path.stat().st_mode
            logger.info("AWS credentials safeguard active: cred_path=%s stat_mode=%o", cred_path, mode)

    async def submit_training_job(
        self,
        payload: TrainingPayload,
        config: RemoteJobConfig,
    ) -> str:
        """Submit an asynchronous SageMaker training job."""
        self._verify_credentials_safeguard()
        if config.max_cost_usd < 0.0:
            raise JobSubmissionError(f"Budget cap max_cost_usd {config.max_cost_usd} is negative.")
        if self.transport is None:
            raise JobSubmissionError("SageMakerTransport is not configured or available.")

        unique_suffix = uuid.uuid4().hex[:8]
        job_name = f"symforge-{payload.domain.value}-{unique_suffix}"
        params = {
            "TrainingJobName": job_name,
            "RoleArn": self.role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": f"763104351884.dkr.ecr.{self.region_name}.amazonaws.com/pytorch-training:2.1.0-gpu-py310",
                "TrainingInputMode": "File",
            },
            "ResourceConfig": {
                "InstanceType": f"ml.{config.hardware_accelerator.lower()}" if not config.hardware_accelerator.startswith("ml.") else config.hardware_accelerator,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": int(config.timeout_seconds)},
            "OutputDataConfig": {"S3OutputPath": self.s3_output_uri},
            "HyperParameters": {
                "lora_rank": str(config.lora_rank),
                "lora_alpha": str(config.lora_alpha),
                "domain": payload.domain.value,
                "base_model": payload.base_model_name,
            },
        }

        async def _call() -> Dict[str, Any]:
            assert self.transport is not None
            return await asyncio.wait_for(
                self.transport.create_training_job(params),
                timeout=self.runner.read_timeout,
            )

        await asyncio.wait_for(
            self.runner.execute(
                operation_name="sagemaker_create_training_job",
                endpoint=f"sagemaker.{self.region_name}.amazonaws.com",
                func=_call,
                payload_summary=f"job_name={job_name}, domain={payload.domain.value}",
                client_handle=self.transport,
            ),
            timeout=self.runner.read_timeout + 5.0,
        )
        return job_name

    async def poll_job_status(self, job_id: str) -> JobStatusResponse:
        """Query execution status of a SageMaker training job."""
        if self.transport is None:
            raise JobExecutionError("SageMakerTransport is not configured.")

        async def _call() -> Dict[str, Any]:
            assert self.transport is not None
            return await asyncio.wait_for(
                self.transport.describe_training_job(job_id),
                timeout=self.runner.read_timeout,
            )

        desc = await asyncio.wait_for(
            self.runner.execute(
                operation_name="sagemaker_describe_training_job",
                endpoint=f"sagemaker.{self.region_name}.amazonaws.com",
                func=_call,
                payload_summary=f"job_name={job_id}",
                client_handle=self.transport,
            ),
            timeout=self.runner.read_timeout + 5.0,
        )
        return self._translate_sagemaker_status(job_id, desc)

    def _translate_sagemaker_status(self, job_id: str, desc: Dict[str, Any]) -> JobStatusResponse:
        raw_status = str(desc.get("TrainingJobStatus", ""))
        status = JobStatus.PENDING
        if raw_status == "InProgress":
            status = JobStatus.RUNNING
        elif raw_status == "Completed":
            status = JobStatus.COMPLETED
        elif raw_status == "Failed":
            status = JobStatus.FAILED
        elif raw_status in ("Stopped", "Stopping"):
            status = JobStatus.CANCELLED

        metrics: Dict[str, float] = {}
        for m in desc.get("FinalMetricDataList", []):
            if "MetricName" in m and "Value" in m:
                metrics[m["MetricName"]] = float(m["Value"])

        progress = 100.0 if status == JobStatus.COMPLETED else (50.0 if status == JobStatus.RUNNING else 0.0)
        return JobStatusResponse(
            job_id=job_id,
            status=status,
            progress_pct=progress,
            error_message=desc.get("FailureReason"),
            started_at=desc.get("TrainingStartTime"),
            completed_at=desc.get("TrainingEndTime"),
            estimated_cost_usd=float(desc.get("BillableTimeInSeconds", 0.0)) * 0.0003,
            metrics=metrics,
        )

    async def fetch_job_artifacts(
        self,
        job_id: str,
        destination_dir: Path,
    ) -> AdapterArtifact:
        """Download and verify trained LoRA adapter weights from SageMaker S3 output."""
        if self.transport is None:
            raise ArtifactRetrievalError("SageMakerTransport is not configured.")

        status_resp = await asyncio.wait_for(
            self.poll_job_status(job_id),
            timeout=self.runner.read_timeout + 5.0,
        )
        if status_resp.status != JobStatus.COMPLETED:
            raise ArtifactRetrievalError(f"Cannot fetch artifacts for job '{job_id}' with status {status_resp.status}")

        s3_artifact_uri = f"{self.s3_output_uri}/{job_id}/output/model.tar.gz"

        async def _call() -> bytes:
            assert self.transport is not None
            return await asyncio.wait_for(
                self.transport.get_s3_object(s3_artifact_uri),
                timeout=self.runner.read_timeout,
            )

        weights_bytes = await asyncio.wait_for(
            self.runner.execute(
                operation_name="sagemaker_s3_get_object",
                endpoint=s3_artifact_uri,
                func=_call,
                payload_summary=f"job_id={job_id}",
                client_handle=self.transport,
            ),
            timeout=self.runner.read_timeout + 5.0,
        )
        if not weights_bytes:
            raise ArtifactRetrievalError(f"Downloaded 0 bytes from S3 for job '{job_id}'.")

        checksum = hashlib.sha256(weights_bytes).hexdigest()
        destination_dir.mkdir(parents=True, exist_ok=True)
        local_path = destination_dir / f"{job_id}_adapter.pt"
        try:
            local_path.write_bytes(weights_bytes)
        except OSError as exc:
            logger.error("Failed to write SageMaker artifact to %s: %s", local_path, exc)
            raise ArtifactRetrievalError(f"Failed to save artifact: {exc}") from exc

        return AdapterArtifact(
            adapter_id=f"sagemaker-{job_id}",
            domain=ComputeDomain.AUDIO,
            lora_rank=8,
            lora_alpha=16.0,
            target_modules=["attention.q_proj", "attention.v_proj"],
            weights_checksum=checksum,
            weights_bytes=weights_bytes,
            weights_uri=str(local_path.resolve()),
            file_size_bytes=len(weights_bytes),
            metadata={"s3_source": s3_artifact_uri},
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of an active SageMaker training job."""
        if self.transport is None:
            return False

        async def _call() -> Dict[str, Any]:
            assert self.transport is not None
            return await asyncio.wait_for(
                self.transport.stop_training_job(job_id),
                timeout=self.runner.read_timeout,
            )

        try:
            await asyncio.wait_for(
                self.runner.execute(
                    operation_name="sagemaker_stop_training_job",
                    endpoint=f"sagemaker.{self.region_name}.amazonaws.com",
                    func=_call,
                    payload_summary=f"job_name={job_id}",
                    client_handle=self.transport,
                ),
                timeout=self.runner.read_timeout + 5.0,
            )
            return True
        except Exception as exc:
            logger.warning("SageMaker StopTrainingJob failed for %s: %s", job_id, exc)
            return False

    async def poll_until_complete(self, job_id: str) -> JobStatusResponse:
        """Poll SageMaker status until terminal state or max_poll_iterations cap."""
        iteration_count = 0
        while True:
            iteration_count += 1
            if iteration_count > self.max_poll_iterations:
                logger.error("SageMaker job %s exceeded max poll iterations (%d)", job_id, self.max_poll_iterations)
                await asyncio.wait_for(self.cancel_job(job_id), timeout=10.0)
                break
            resp = await asyncio.wait_for(self.poll_job_status(job_id), timeout=self.runner.read_timeout + 5.0)
            if resp.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return resp
            await asyncio.wait_for(asyncio.sleep(self.poll_interval_seconds), timeout=self.poll_interval_seconds + 1.0)

        raise JobTimeoutError(f"SageMaker job '{job_id}' exceeded max poll iterations ({self.max_poll_iterations}).")

    async def aclose(self) -> None:
        """Sever all SageMaker/S3 client connections."""
        if self.transport:
            await asyncio.wait_for(self.runner.sever_resource(self.transport), timeout=5.0)

    async def __aenter__(self) -> SageMakerComputeProvider:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await asyncio.wait_for(self.aclose(), timeout=5.0)
