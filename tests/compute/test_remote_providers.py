"""Unit tests for remote serverless GPU providers (Modal & SageMaker) with resilient networking."""

from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from symforge.compute import (
    AdapterArtifact,
    ArtifactRetrievalError,
    ComputeDomain,
    ComputeProvider,
    ComputeProviderError,
    DefaultHttpTransport,
    HttpResponse,
    HttpTransport,
    HttpTransportError,
    JobExecutionError,
    JobStatus,
    JobStatusResponse,
    JobSubmissionError,
    JobTimeoutError,
    ModalComputeProvider,
    RemoteJobConfig,
    ResilientNetworkRunner,
    SageMakerComputeProvider,
    SageMakerTransport,
    TrainingPayload,
    resilient_async_call,
    verify_compute_provider_conformance,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Mock Transports for Hermetic Offline Testing
# =====================================================================


class MockHttpTransport:
    """In-memory mock HTTP transport recording calls and returning canned responses."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.responses: Dict[str, HttpResponse] = {}
        self.url_queues: Dict[str, List[Any]] = {}
        self.queued_responses: List[Any] = []
        self.closed: bool = False
        self.hang_seconds: float = 0.0

    def set_response(self, method: str, url: str, response: HttpResponse) -> None:
        """Register a static response for method and url."""
        self.responses[f"{method.upper()}:{url}"] = response

    def queue_for_url(self, method: str, url: str, item: Any) -> None:
        """Queue a sequential response for a specific method and url."""
        key = f"{method.upper()}:{url}"
        self.url_queues.setdefault(key, []).append(item)

    def queue_response(self, item: Any) -> None:
        """Queue a generic response or exception to be yielded sequentially."""
        self.queued_responses.append(item)

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        body_bytes: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> HttpResponse:
        """Simulate HTTP request handling."""
        self.calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "json_data": json_data,
            "body_bytes": body_bytes,
            "timeout": timeout,
        })
        if self.hang_seconds > 0:
            await asyncio.wait_for(asyncio.sleep(self.hang_seconds), timeout=self.hang_seconds + 1.0)

        key = f"{method.upper()}:{url}"
        if key in self.url_queues and self.url_queues[key]:
            item = self.url_queues[key].pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        if key in self.responses:
            return self.responses[key]

        if self.queued_responses:
            item = self.queued_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return HttpResponse(status_code=404, content=b'{"error": "Not Found"}')

    async def aclose(self) -> None:
        """Record deterministic resource severance."""
        self.closed = True


class MockSageMakerTransport:
    """In-memory mock SageMaker/S3 transport recording calls and returning canned data."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.create_training_job_response: Dict[str, Any] = {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:job"}
        self.describe_responses: List[Dict[str, Any]] = []
        self.stop_response: Dict[str, Any] = {"ResponseMetadata": {"HTTPStatusCode": 200}}
        self.s3_objects: Dict[str, bytes] = {}
        self.closed: bool = False
        self.hang_seconds: float = 0.0
        self.create_job_error: Optional[Exception] = None

    async def create_training_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"operation": "CreateTrainingJob", "params": params})
        if self.hang_seconds > 0:
            await asyncio.wait_for(asyncio.sleep(self.hang_seconds), timeout=self.hang_seconds + 1.0)
        if self.create_job_error:
            raise self.create_job_error
        return self.create_training_job_response

    async def describe_training_job(self, job_name: str) -> Dict[str, Any]:
        self.calls.append({"operation": "DescribeTrainingJob", "job_name": job_name})
        if self.describe_responses:
            return self.describe_responses.pop(0)
        return {
            "TrainingJobName": job_name,
            "TrainingJobStatus": "Completed",
            "BillableTimeInSeconds": 120,
            "FinalMetricDataList": [{"MetricName": "val_loss", "Value": 0.042}],
        }

    async def stop_training_job(self, job_name: str) -> Dict[str, Any]:
        self.calls.append({"operation": "StopTrainingJob", "job_name": job_name})
        return self.stop_response

    async def get_s3_object(self, s3_uri: str) -> bytes:
        self.calls.append({"operation": "GetObject", "s3_uri": s3_uri})
        if s3_uri in self.s3_objects:
            return self.s3_objects[s3_uri]
        return b"MOCK_LORA_WEIGHTS_BINARY_TENSORS"

    async def aclose(self) -> None:
        self.closed = True


def _sample_payload() -> TrainingPayload:
    return TrainingPayload(
        trajectories_count=50,
        domain=ComputeDomain.AUDIO,
        dataset_uri="s3://symforge-test/dataset.json",
        base_model_name="pde-audio-base",
        model_type="pde",
        hyperparameters={"lr": 0.0005},
    )


def _sample_config(provider_name: str = "modal") -> RemoteJobConfig:
    return RemoteJobConfig(
        provider_name=provider_name,
        hardware_accelerator="A10G",
        timeout_seconds=300.0,
        max_cost_usd=2.0,
        lora_rank=8,
        lora_alpha=16.0,
        tags={"track": "curriculum_202505"},
    )


# =====================================================================
# 1. Protocol Conformance & Type Safety
# =====================================================================


def test_protocol_conformance_modal_and_sagemaker() -> None:
    """Verify both ModalComputeProvider and SageMakerComputeProvider conform to ComputeProvider."""
    modal_provider = ModalComputeProvider(transport=MockHttpTransport())
    sagemaker_provider = SageMakerComputeProvider(transport=MockSageMakerTransport())

    assert verify_compute_provider_conformance(modal_provider) is True
    assert verify_compute_provider_conformance(sagemaker_provider) is True
    assert isinstance(modal_provider, ComputeProvider)
    assert isinstance(sagemaker_provider, ComputeProvider)


# =====================================================================
# 2. Resilient Network Runner: Exponential Backoff & Jitter
# =====================================================================


def test_exponential_backoff_with_deterministic_jitter() -> None:
    """Verify backoff formula t = min(t_max, t_base * 2^attempt) + uniform(0, jitter)."""
    runner = ResilientNetworkRunner(
        base_backoff=0.5,
        max_backoff=4.0,
        jitter=0.2,
        rng_func=lambda: 0.5,  # deterministic jitter: 0.5 * 0.2 = 0.1
    )

    # attempt 0: min(4.0, 0.5 * 1) + 0.1 = 0.6
    assert abs(runner.compute_backoff_delay(0) - 0.6) < 1e-6
    # attempt 1: min(4.0, 0.5 * 2) + 0.1 = 1.1
    assert abs(runner.compute_backoff_delay(1) - 1.1) < 1e-6
    # attempt 2: min(4.0, 0.5 * 4) + 0.1 = 2.1
    assert abs(runner.compute_backoff_delay(2) - 2.1) < 1e-6
    # attempt 3: min(4.0, 0.5 * 8) + 0.1 = 4.1 (capped at max_backoff=4.0 + 0.1)
    assert abs(runner.compute_backoff_delay(3) - 4.1) < 1e-6
    # attempt 4: min(4.0, 0.5 * 16) + 0.1 = 4.1 (capped)
    assert abs(runner.compute_backoff_delay(4) - 4.1) < 1e-6


def test_exponential_backoff_jitter_bounds() -> None:
    """Verify random jitter stays strictly in [0, jitter] across many samples."""
    runner = ResilientNetworkRunner(base_backoff=1.0, max_backoff=10.0, jitter=0.5)
    for _ in range(100):
        d = runner.compute_backoff_delay(attempt=1)
        base = 2.0  # 1.0 * 2^1
        assert base <= d <= base + 0.5


# =====================================================================
# 3. Retries on Transient 5xx Errors and AWS Throttling
# =====================================================================


def test_resilient_retry_on_transient_http_500() -> None:
    """Verify runner retries on transient 500 error and succeeds on 3rd attempt."""
    async def _run() -> None:
        transport = MockHttpTransport()
        # Queue two 500 errors, then a 200 success
        transport.queue_response(HttpResponse(status_code=500, content=b"Internal Server Error"))
        transport.queue_response(HttpResponse(status_code=503, content=b"Service Unavailable"))
        transport.queue_response(HttpResponse(status_code=200, content=b'{"job_id": "modal-job-retry"}'))

        runner = ResilientNetworkRunner(base_backoff=0.01, jitter=0.01, max_retries=3)
        provider = ModalComputeProvider(transport=transport, runner=runner)

        payload = _sample_payload()
        config = _sample_config()

        job_id = await asyncio.wait_for(provider.submit_training_job(payload, config), timeout=5.0)
        assert job_id == "modal-job-retry"
        assert len(transport.calls) == 3

    asyncio.run(_run())


def test_resilient_non_retryable_400_fails_immediately() -> None:
    """Verify non-retryable 400 Bad Request raises JobSubmissionError without retry."""
    async def _run() -> None:
        transport = MockHttpTransport()
        transport.queue_response(HttpResponse(status_code=400, content=b'{"error": "Invalid Hyperparameter"}'))

        runner = ResilientNetworkRunner(base_backoff=0.01, jitter=0.01, max_retries=3)
        provider = ModalComputeProvider(transport=transport, runner=runner)

        payload = _sample_payload()
        config = _sample_config()

        with pytest.raises(JobSubmissionError) as exc_info:
            await asyncio.wait_for(provider.submit_training_job(payload, config), timeout=5.0)
        logger.info("Expected non-retryable error caught: %s", exc_info.value)
        assert len(transport.calls) == 1
        assert transport.closed is True

    asyncio.run(_run())


# =====================================================================
# 4. ModalComputeProvider Full Lifecycle (submit -> poll -> fetch -> cancel)
# =====================================================================


def test_modal_provider_full_lifecycle(tmp_path: Path) -> None:
    """Verify complete Modal compute lifecycle from submission to artifact download."""
    async def _run() -> None:
        transport = MockHttpTransport()
        runner = ResilientNetworkRunner(base_backoff=0.01, jitter=0.01)
        provider = ModalComputeProvider(transport=transport, runner=runner)

        # 1. Mock submit response
        transport.set_response(
            "POST",
            "https://api.modal.com/v1/jobs/submit",
            HttpResponse(status_code=200, content=b'{"job_id": "modal-job-12345"}'),
        )

        # 2. Mock poll responses via url queue
        transport.queue_for_url(
            "GET",
            "https://api.modal.com/v1/jobs/modal-job-12345/status",
            HttpResponse(
                status_code=200,
                content=json.dumps({
                    "status": "running",
                    "progress_pct": 50.0,
                    "metrics": {"train_loss": 0.45},
                }).encode("utf-8"),
            ),
        )
        transport.queue_for_url(
            "GET",
            "https://api.modal.com/v1/jobs/modal-job-12345/status",
            HttpResponse(
                status_code=200,
                content=json.dumps({
                    "status": "completed",
                    "progress_pct": 100.0,
                    "metrics": {"train_loss": 0.038, "val_loss": 0.041},
                    "estimated_cost_usd": 0.12,
                }).encode("utf-8"),
            ),
        )

        # 3. Mock artifact response
        dummy_weights = b"PYTORCH_LORA_WEIGHTS_TEST_TENSOR_BYTES_12345"
        checksum = hashlib.sha256(dummy_weights).hexdigest()
        weights_b64 = base64.b64encode(dummy_weights).decode("utf-8")
        transport.set_response(
            "GET",
            "https://api.modal.com/v1/jobs/modal-job-12345/artifacts",
            HttpResponse(
                status_code=200,
                content=json.dumps({
                    "adapter_id": "modal-adapter-12345",
                    "domain": "audio",
                    "lora_rank": 8,
                    "lora_alpha": 16.0,
                    "target_modules": ["attention.q_proj", "attention.v_proj"],
                    "weights_checksum": checksum,
                    "weights_base64": weights_b64,
                    "file_size_bytes": len(dummy_weights),
                }).encode("utf-8"),
            ),
        )

        # 4. Mock cancel response
        transport.set_response(
            "POST",
            "https://api.modal.com/v1/jobs/modal-job-12345/cancel",
            HttpResponse(status_code=200, content=b'{"cancelled": true}'),
        )

        # Execution
        job_id = await asyncio.wait_for(
            provider.submit_training_job(_sample_payload(), _sample_config()),
            timeout=5.0,
        )
        assert job_id == "modal-job-12345"

        poll_1 = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert poll_1.status == JobStatus.RUNNING
        assert poll_1.progress_pct == 50.0

        poll_2 = await asyncio.wait_for(provider.poll_job_status(job_id), timeout=5.0)
        assert poll_2.status == JobStatus.COMPLETED
        assert poll_2.progress_pct == 100.0
        assert poll_2.metrics["val_loss"] == 0.041

        artifact = await asyncio.wait_for(provider.fetch_job_artifacts(job_id, tmp_path), timeout=5.0)
        assert artifact.adapter_id == "modal-adapter-12345"
        assert artifact.weights_checksum == checksum
        assert artifact.weights_bytes == dummy_weights

        saved_file = tmp_path / f"{job_id}_adapter.pt"
        assert saved_file.exists()
        assert saved_file.read_bytes() == dummy_weights

        cancelled = await asyncio.wait_for(provider.cancel_job(job_id), timeout=5.0)
        assert cancelled is True

    asyncio.run(_run())


# =====================================================================
# 5. SageMakerComputeProvider Full Lifecycle
# =====================================================================


def test_sagemaker_provider_full_lifecycle(tmp_path: Path) -> None:
    """Verify complete SageMaker compute lifecycle and status translation."""
    async def _run() -> None:
        transport = MockSageMakerTransport()
        runner = ResilientNetworkRunner(base_backoff=0.01, jitter=0.01)
        provider = SageMakerComputeProvider(transport=transport, runner=runner)

        transport.describe_responses = [
            {"TrainingJobName": "test-job", "TrainingJobStatus": "InProgress", "BillableTimeInSeconds": 60},
            {
                "TrainingJobName": "test-job",
                "TrainingJobStatus": "Completed",
                "BillableTimeInSeconds": 180,
                "FinalMetricDataList": [{"MetricName": "val_loss", "Value": 0.035}],
            },
        ]
        dummy_weights = b"SAGEMAKER_S3_TAR_GZ_TENSORS"
        transport.s3_objects["s3://symforge-models/adapters/test-job/output/model.tar.gz"] = dummy_weights

        payload = _sample_payload()
        config = _sample_config(provider_name="sagemaker")

        job_id = await asyncio.wait_for(provider.submit_training_job(payload, config), timeout=5.0)
        assert job_id.startswith("symforge-audio-")

        poll_1 = await asyncio.wait_for(provider.poll_job_status("test-job"), timeout=5.0)
        assert poll_1.status == JobStatus.RUNNING
        assert poll_1.progress_pct == 50.0

        poll_2 = await asyncio.wait_for(provider.poll_job_status("test-job"), timeout=5.0)
        assert poll_2.status == JobStatus.COMPLETED
        assert poll_2.progress_pct == 100.0
        assert poll_2.metrics["val_loss"] == 0.035

        artifact = await asyncio.wait_for(provider.fetch_job_artifacts("test-job", tmp_path), timeout=5.0)
        assert artifact.adapter_id == "sagemaker-test-job"
        assert artifact.weights_checksum == hashlib.sha256(dummy_weights).hexdigest()

        cancelled = await asyncio.wait_for(provider.cancel_job("test-job"), timeout=5.0)
        assert cancelled is True

    asyncio.run(_run())


# =====================================================================
# 6. Network Timeout and Deterministic Resource Severance
# =====================================================================


def test_simulated_network_hang_and_resource_severance() -> None:
    """Verify simulated network freeze triggers JobTimeoutError and severs socket connection."""
    async def _run() -> None:
        transport = MockHttpTransport()
        transport.hang_seconds = 2.0  # longer than read_timeout

        runner = ResilientNetworkRunner(
            read_timeout=0.1,
            connect_timeout=0.1,
            base_backoff=0.01,
            jitter=0.01,
            max_retries=1,
        )
        provider = ModalComputeProvider(transport=transport, runner=runner)

        payload = _sample_payload()
        config = _sample_config()

        with pytest.raises(JobSubmissionError) as exc_info:
            await asyncio.wait_for(provider.submit_training_job(payload, config), timeout=5.0)
        logger.info("Simulated hang error caught: %s", exc_info.value)
        assert transport.closed is True

    asyncio.run(_run())


# =====================================================================
# 7. Max Poll Iteration Cap
# =====================================================================


def test_max_poll_iterations_cap_exceeded() -> None:
    """Verify exceeding max_poll_iterations raises JobTimeoutError and triggers cancel."""
    async def _run() -> None:
        transport = MockHttpTransport()
        runner = ResilientNetworkRunner(base_backoff=0.01, jitter=0.01)
        provider = ModalComputeProvider(
            transport=transport,
            runner=runner,
            max_poll_iterations=2,
            poll_interval_seconds=0.01,
        )

        # Job always returns running
        transport.set_response(
            "GET",
            "https://api.modal.com/v1/jobs/stuck-job/status",
            HttpResponse(status_code=200, content=b'{"status": "running"}'),
        )
        transport.set_response(
            "POST",
            "https://api.modal.com/v1/jobs/stuck-job/cancel",
            HttpResponse(status_code=200, content=b'{"cancelled": true}'),
        )

        with pytest.raises(JobTimeoutError) as exc_info:
            await asyncio.wait_for(provider.poll_until_complete("stuck-job"), timeout=5.0)
        logger.info("Max poll iteration cap error caught: %s", exc_info.value)
        assert "exceeded max poll iterations" in str(exc_info.value)

    asyncio.run(_run())


# =====================================================================
# 8. Checksum Verification Failure on Corrupted Download
# =====================================================================


def test_corrupted_artifact_checksum_mismatch(tmp_path: Path) -> None:
    """Verify corrupted artifact with wrong SHA-256 raises ArtifactRetrievalError."""
    async def _run() -> None:
        transport = MockHttpTransport()
        provider = ModalComputeProvider(transport=transport)

        corrupted_bytes = b"CORRUPTED_WEIGHTS"
        bogus_checksum = "0" * 64
        transport.set_response(
            "GET",
            "https://api.modal.com/v1/jobs/corrupted-job/artifacts",
            HttpResponse(
                status_code=200,
                content=json.dumps({
                    "adapter_id": "corrupted-adapter",
                    "domain": "audio",
                    "weights_checksum": bogus_checksum,
                    "weights_base64": base64.b64encode(corrupted_bytes).decode("utf-8"),
                }).encode("utf-8"),
            ),
        )

        with pytest.raises(ArtifactRetrievalError) as exc_info:
            await asyncio.wait_for(provider.fetch_job_artifacts("corrupted-job", tmp_path), timeout=5.0)
        logger.info("Checksum mismatch error caught: %s", exc_info.value)
        assert "Checksum mismatch" in str(exc_info.value)

    asyncio.run(_run())


# =====================================================================
# 9. Context Manager Clean Severance
# =====================================================================


def test_context_manager_clean_severance() -> None:
    """Verify async with statement deterministically closes transports."""
    async def _run() -> None:
        transport = MockHttpTransport()
        async with ModalComputeProvider(transport=transport) as provider:
            assert provider.transport is transport
            assert transport.closed is False
        assert transport.closed is True

        s3_transport = MockSageMakerTransport()
        async with SageMakerComputeProvider(transport=s3_transport) as provider:
            assert s3_transport.closed is False
        assert s3_transport.closed is True

    asyncio.run(_run())


# =====================================================================
# 10. Additional Edge Cases & Resilience Safeguards
# =====================================================================


def test_resilient_async_call_wrapper() -> None:
    """Verify convenient resilient_async_call executes and returns result."""
    async def _run() -> None:
        async def _sample_call() -> str:
            return "SUCCESS_VALUE"

        runner = ResilientNetworkRunner(base_backoff=0.01, jitter=0.01)
        res = await asyncio.wait_for(
            resilient_async_call(_sample_call, operation_name="test_call", runner=runner),
            timeout=5.0,
        )
        assert res == "SUCCESS_VALUE"

    asyncio.run(_run())


def test_sagemaker_negative_budget_and_unconfigured_transport() -> None:
    """Verify SageMaker catches budget violations and missing transports."""
    async def _run() -> None:
        provider = SageMakerComputeProvider(transport=None)
        payload = _sample_payload()
        config = _sample_config()

        # Unconfigured transport raises JobSubmissionError
        with pytest.raises(JobSubmissionError) as exc_info:
            await asyncio.wait_for(provider.submit_training_job(payload, config), timeout=5.0)
        logger.info("Unconfigured transport error: %s", exc_info.value)

        # Unconfigured poll raises JobExecutionError
        with pytest.raises(JobExecutionError) as exc_info:
            await asyncio.wait_for(provider.poll_job_status("job-123"), timeout=5.0)
        logger.info("Unconfigured poll error: %s", exc_info.value)

    asyncio.run(_run())


def test_sagemaker_fetch_before_completion(tmp_path: Path) -> None:
    """Verify fetching artifacts before SageMaker job completes raises ArtifactRetrievalError."""
    async def _run() -> None:
        transport = MockSageMakerTransport()
        transport.describe_responses = [{"TrainingJobName": "job-run", "TrainingJobStatus": "InProgress"}]
        provider = SageMakerComputeProvider(transport=transport)

        with pytest.raises(ArtifactRetrievalError) as exc_info:
            await asyncio.wait_for(provider.fetch_job_artifacts("job-run", tmp_path), timeout=5.0)
        logger.info("Early fetch error: %s", exc_info.value)
        assert "Cannot fetch artifacts" in str(exc_info.value)

    asyncio.run(_run())


def test_default_http_transport_closed_error() -> None:
    """Verify DefaultHttpTransport rejects requests when closed."""
    async def _run() -> None:
        transport = DefaultHttpTransport()
        await asyncio.wait_for(transport.aclose(), timeout=5.0)
        with pytest.raises(HttpTransportError) as exc_info:
            await asyncio.wait_for(transport.request("GET", "https://example.com"), timeout=5.0)
        logger.info("Closed transport error: %s", exc_info.value)
        assert "closed and severed" in str(exc_info.value)

    asyncio.run(_run())
