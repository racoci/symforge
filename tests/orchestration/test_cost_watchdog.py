"""Automated test suite for Task 6.2: Financial Pre-Flight Cost Watchdog & Safety Gate."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from symforge.compute.contracts import (
    AdapterArtifact,
    ComputeDomain,
    JobStatus,
    JobStatusResponse,
    RemoteJobConfig,
    TrainingPayload,
)
from symforge.compute.mock_provider import MockComputeProvider
from symforge.compute.protocol import ComputeProvider
from symforge.core.ast import FreeParam, MixNode, SineNode
from symforge.orchestration.cost_watchdog import (
    BudgetExceededError,
    CostEstimate,
    CostWatchdog,
    CostWatchdogConfig,
    DEFAULT_RATE_CARDS,
    DispatchDecision,
)
from symforge.orchestration.meta_controller import (
    MetaController,
    MetaControllerConfig,
)


def _make_dummy_payload(
    trajectories_count: int = 10,
    duration_seconds: Optional[float] = None,
) -> TrainingPayload:
    """Create a valid TrainingPayload with optional explicit duration."""
    hparams: Dict[str, Any] = {"lora_rank": 8}
    if duration_seconds is not None:
        hparams["duration_seconds"] = duration_seconds

    return TrainingPayload(
        trajectories_count=trajectories_count,
        domain=ComputeDomain.AUDIO,
        base_model_name="symforge-base",
        model_type="lora",
        dataset_payload={
            "trajectories": [{"ast": "dummy", "level": "L1", "domain": "audio"}],
            "count": trajectories_count,
        },
        hyperparameters=hparams,
    )


class RecordingRealComputeProvider:
    """Mock representing a billable cloud provider to record external calls."""

    def __init__(self, provider_name: str = "modal") -> None:
        self.provider_name = provider_name
        self.submission_calls: List[Tuple[TrainingPayload, RemoteJobConfig]] = []
        self._job_counter: int = 0

    async def submit_training_job(
        self, payload: TrainingPayload, config: RemoteJobConfig
    ) -> str:
        self.submission_calls.append((payload, config))
        self._job_counter += 1
        return f"{self.provider_name}-real-job-{self._job_counter}"

    async def poll_job_status(self, job_id: str) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress_pct=100.0,
            estimated_cost_usd=0.20,
        )

    async def fetch_job_artifacts(
        self, job_id: str, destination_dir: Path
    ) -> AdapterArtifact:
        mock_p = MockComputeProvider()
        return await asyncio.wait_for(
            mock_p.fetch_job_artifacts("mock-0", destination_dir),
            timeout=10.0,
        )

    async def cancel_job(self, job_id: str) -> bool:
        return True


# =====================================================================
# 1. Arithmetic Accuracy & Rate Cards
# =====================================================================


class TestCostWatchdogArithmetic:
    """Validate arithmetic calculation, 25% safety buffer, and rate cards."""

    def test_modal_rate_cards_and_25pct_buffer(self) -> None:
        """Invariant: Buffered cost is exactly raw_cost * 1.25 across Modal rate card."""
        watchdog = CostWatchdog(CostWatchdogConfig(safety_buffer_pct=0.25))

        # Modal A10G: 0.000305 USD/sec, 100 seconds
        payload = _make_dummy_payload(duration_seconds=100.0)
        config = RemoteJobConfig(
            provider_name="modal",
            hardware_accelerator="A10G",
        )

        estimate = watchdog.estimate_job_cost(payload, config)
        assert estimate.provider_name == "modal"
        assert estimate.hardware_accelerator == "A10G"
        assert estimate.estimated_duration_seconds == 100.0
        assert estimate.base_rate_per_second_usd == pytest.approx(0.000305, rel=1e-5)

        expected_raw = 0.000305 * 100.0  # 0.0305
        expected_buffered = expected_raw * 1.25  # 0.038125
        assert estimate.raw_estimated_cost_usd == pytest.approx(expected_raw, rel=1e-5)
        assert estimate.buffered_estimated_cost_usd == pytest.approx(
            expected_buffered, rel=1e-5
        )

        # Modal T4: 0.000164 USD/sec, 100 seconds
        config_t4 = RemoteJobConfig(
            provider_name="modal",
            hardware_accelerator="T4",
        )
        estimate_t4 = watchdog.estimate_job_cost(payload, config_t4)
        assert estimate_t4.base_rate_per_second_usd == pytest.approx(0.000164, rel=1e-5)
        expected_t4_buffered = (0.000164 * 100.0) * 1.25
        assert estimate_t4.buffered_estimated_cost_usd == pytest.approx(
            expected_t4_buffered, rel=1e-5
        )

    def test_sagemaker_rate_cards_and_25pct_buffer(self) -> None:
        """Invariant: Buffered cost is exactly raw_cost * 1.25 across SageMaker rate card."""
        watchdog = CostWatchdog(CostWatchdogConfig(safety_buffer_pct=0.25))
        payload = _make_dummy_payload(duration_seconds=200.0)

        # SageMaker A10G: 0.000336 USD/sec
        config_sm = RemoteJobConfig(
            provider_name="sagemaker",
            hardware_accelerator="A10G",
        )
        est = watchdog.estimate_job_cost(payload, config_sm)
        assert est.base_rate_per_second_usd == pytest.approx(0.000336, rel=1e-5)
        expected_raw = 0.000336 * 200.0
        expected_buffered = expected_raw * 1.25
        assert est.raw_estimated_cost_usd == pytest.approx(expected_raw, rel=1e-5)
        assert est.buffered_estimated_cost_usd == pytest.approx(
            expected_buffered, rel=1e-5
        )

        # SageMaker T4: 0.000205 USD/sec
        config_sm_t4 = RemoteJobConfig(
            provider_name="sagemaker",
            hardware_accelerator="T4",
        )
        est_sm_t4 = watchdog.estimate_job_cost(payload, config_sm_t4)
        assert est_sm_t4.base_rate_per_second_usd == pytest.approx(0.000205, rel=1e-5)
        assert est_sm_t4.buffered_estimated_cost_usd == pytest.approx(
            (0.000205 * 200.0) * 1.25, rel=1e-5
        )

    def test_zero_cost_for_mock_provider(self) -> None:
        """Invariant: MockComputeProvider always results in exactly $0.00 USD."""
        watchdog = CostWatchdog()
        payload = _make_dummy_payload(duration_seconds=1000.0)
        config = RemoteJobConfig(provider_name="mock", hardware_accelerator="A100")

        estimate = watchdog.estimate_job_cost(payload, config)
        assert estimate.base_rate_per_second_usd == 0.0
        assert estimate.raw_estimated_cost_usd == 0.0
        assert estimate.buffered_estimated_cost_usd == 0.0
        assert estimate.is_within_budget is True

    def test_custom_rate_card_override(self) -> None:
        """Invariant: Custom rate cards override default pricing values."""
        custom = {("custom_cloud", "H100"): 0.002000}
        cfg = CostWatchdogConfig(
            custom_rate_cards=custom,
            safety_buffer_pct=0.20,
        )
        watchdog = CostWatchdog(cfg)

        payload = _make_dummy_payload(duration_seconds=50.0)
        job_cfg = RemoteJobConfig(
            provider_name="custom_cloud",
            hardware_accelerator="H100",
        )
        est = watchdog.estimate_job_cost(payload, job_cfg)
        assert est.base_rate_per_second_usd == 0.002000
        assert est.raw_estimated_cost_usd == pytest.approx(0.1000, rel=1e-5)
        assert est.buffered_estimated_cost_usd == pytest.approx(0.1200, rel=1e-5)


# =====================================================================
# 2. Budget Reservation & Reconciliation Lifecycle
# =====================================================================


class TestBudgetReservationAndReconciliation:
    """Validate atomic reservation, balance tracking, and post-flight reconciliation."""

    def test_authorize_and_reserve_decrements_remaining_budget(self) -> None:
        """Invariant: Authorizing a job immediately encumbers buffered budget."""
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=2.00))
        assert watchdog.get_remaining_budget() == pytest.approx(2.00)
        assert watchdog.get_total_spent() == pytest.approx(0.00)

        payload = _make_dummy_payload(duration_seconds=1000.0)
        # Modal A10G: 0.000305 * 1000 = 0.305 -> buffered = 0.38125
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        authorized, estimate = watchdog.authorize_and_reserve(payload, config)
        assert authorized is True
        assert estimate.reservation_id is not None
        assert estimate.buffered_estimated_cost_usd == pytest.approx(0.38125, rel=1e-5)

        # Remaining budget must decrease by exactly buffered cost
        expected_remaining = 2.00 - 0.38125
        assert watchdog.get_remaining_budget() == pytest.approx(
            expected_remaining, rel=1e-5
        )
        assert watchdog.get_total_spent() == pytest.approx(0.38125, rel=1e-5)

    def test_record_actual_cost_reconciles_and_refunds_unspent_buffer(self) -> None:
        """Invariant: Reporting lower actual cost refunds unneeded safety buffer."""
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=2.00))
        payload = _make_dummy_payload(duration_seconds=1000.0)
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        authorized, estimate = watchdog.authorize_and_reserve(payload, config)
        assert authorized is True
        res_id = estimate.reservation_id
        assert res_id is not None

        # Actual cost reported is $0.2500 (lower than $0.38125 reserved)
        watchdog.record_actual_cost(res_id, 0.2500)

        # Remaining budget should be exactly 2.00 - 0.25 = 1.75
        assert watchdog.get_remaining_budget() == pytest.approx(1.75, rel=1e-5)
        assert watchdog.get_total_spent() == pytest.approx(0.25, rel=1e-5)

    def test_record_actual_cost_handles_overrun_symmetrically(self) -> None:
        """Invariant: Reporting higher actual cost deducts extra amount symmetrically."""
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=2.00))
        payload = _make_dummy_payload(duration_seconds=1000.0)
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        _, estimate = watchdog.authorize_and_reserve(payload, config)
        res_id = estimate.reservation_id
        assert res_id is not None

        # Actual cost was $0.4500 (higher than $0.38125 reserved)
        watchdog.record_actual_cost(res_id, 0.4500)

        expected_remaining = 2.00 - 0.4500
        assert watchdog.get_remaining_budget() == pytest.approx(
            expected_remaining, rel=1e-5
        )

    def test_reset_session_restores_clean_initial_state(self) -> None:
        """Invariant: reset_session clears all reservations and restores initial cap."""
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=2.00))
        payload = _make_dummy_payload(duration_seconds=1000.0)
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        watchdog.authorize_and_reserve(payload, config)
        assert watchdog.get_remaining_budget() < 2.00

        watchdog.reset_session()
        assert watchdog.get_remaining_budget() == pytest.approx(2.00)
        assert watchdog.get_total_spent() == pytest.approx(0.00)


# =====================================================================
# 3. Hard Cap & Multi-Job Exhaustion Rejection
# =====================================================================


class TestBudgetExhaustionAndRejection:
    """Validate rejection of single-job hard caps and sequential exhaustion."""

    def test_single_job_exceeding_hard_cap_rejected(self) -> None:
        """Invariant: Any job with projected cost > $2.00 cap is strictly rejected."""
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=2.00))
        # Modal A10G: 0.000305 USD/sec; 6000 seconds -> raw = 1.83 -> buffered = 2.2875 > 2.00
        payload = _make_dummy_payload(duration_seconds=6000.0)
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        decision = watchdog.evaluate_dispatch(payload, config)
        assert decision.authorized is False
        assert decision.use_mock_fallback is True
        assert "exceeds remaining budget" in decision.reason

        authorized, estimate = watchdog.authorize_and_reserve(payload, config)
        assert authorized is False
        assert estimate.is_within_budget is False
        assert estimate.reservation_id is None
        # Budget remains completely untouched
        assert watchdog.get_remaining_budget() == pytest.approx(2.00)

    def test_fallback_disabled_raises_budget_exceeded_error(self) -> None:
        """Invariant: If fallback is disabled, unauthorized dispatch raises BudgetExceededError."""
        cfg = CostWatchdogConfig(
            session_budget_cap_usd=2.00,
            fallback_to_mock_on_exhaustion=False,
        )
        watchdog = CostWatchdog(cfg)
        payload = _make_dummy_payload(duration_seconds=8000.0)
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")
        real_provider = RecordingRealComputeProvider()

        with pytest.raises(BudgetExceededError) as exc_info:
            asyncio.run(watchdog.intercept_or_fallback(real_provider, payload, config))

        assert "Budget exceeded" in str(exc_info.value)
        assert len(real_provider.submission_calls) == 0

    def test_multi_job_accumulation_blocks_further_dispatches(self) -> None:
        """Invariant: Successive jobs drain budget until cap is reached, blocking subsequent calls."""
        # Cap = $1.00. Job costs $0.38125.
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=1.00))
        payload = _make_dummy_payload(duration_seconds=1000.0)
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        # Job 1: $0.38125 -> Remaining $0.61875
        ok1, est1 = watchdog.authorize_and_reserve(payload, config)
        assert ok1 is True

        # Job 2: $0.38125 -> Remaining $0.23750
        ok2, est2 = watchdog.authorize_and_reserve(payload, config)
        assert ok2 is True

        # Job 3: Needs $0.38125, but remaining is $0.23750 -> REJECTED
        ok3, est3 = watchdog.authorize_and_reserve(payload, config)
        assert ok3 is False
        assert est3.is_within_budget is False


# =====================================================================
# 4. Mock Fallback Interception & Zero External Calls
# =====================================================================


class TestMockFallbackInterception:
    """Validate that exhausted budget intercepts dispatches and calls zero external APIs."""

    def test_intercept_or_fallback_switches_to_mock_on_exhaustion(self) -> None:
        """Invariant: When budget is exhausted, real provider receives 0 calls and mock is returned."""
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=0.20))
        # Job costs $0.38125 > $0.20
        payload = _make_dummy_payload(duration_seconds=1000.0)
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        real_provider = RecordingRealComputeProvider()
        mock_provider = MockComputeProvider()

        job_id, was_mocked = asyncio.run(
            watchdog.intercept_or_fallback(
                provider=real_provider,
                payload=payload,
                config=config,
                mock_provider=mock_provider,
            )
        )

        assert was_mocked is True
        assert job_id.startswith("mock-")
        # Real provider was NEVER called!
        assert len(real_provider.submission_calls) == 0

    def test_intercept_or_fallback_permits_authorized_real_dispatches(self) -> None:
        """Invariant: When within budget, real provider is invoked and was_mocked is False."""
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=2.00))
        payload = _make_dummy_payload(duration_seconds=100.0)  # ~$0.038
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        real_provider = RecordingRealComputeProvider()

        job_id, was_mocked = asyncio.run(
            watchdog.intercept_or_fallback(
                provider=real_provider,
                payload=payload,
                config=config,
            )
        )

        assert was_mocked is False
        assert job_id.startswith("modal-real-job-")
        assert len(real_provider.submission_calls) == 1


# =====================================================================
# 5. Pass-to-Complete Gate: Zero Unauthorized External API Calls
# =====================================================================


class TestPassToCompleteGate:
    """Pass-to-Complete Gate: Verify zero unauthorized external API calls across 10 dispatches."""

    def test_strict_pass_gate_zero_unauthorized_external_calls(self) -> None:
        """Gate: With $1.00 budget and $0.38125 jobs, exactly 2 real calls are made across 10 dispatches.

        The remaining 8 jobs are intercepted and redirected to MockComputeProvider.
        Under no circumstances may the real provider be called more than 2 times.
        """
        # Budget cap = $1.00.
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=1.00))
        real_provider = RecordingRealComputeProvider(provider_name="modal")
        mock_provider = MockComputeProvider()

        # Each job: Modal A10G for 1000s -> raw $0.3050 -> buffered $0.38125
        payload = _make_dummy_payload(duration_seconds=1000.0)
        config = RemoteJobConfig(provider_name="modal", hardware_accelerator="A10G")

        dispatched_results: List[Tuple[str, bool]] = []

        async def _run_10_dispatches() -> None:
            for _ in range(10):
                res = await asyncio.wait_for(
                    watchdog.intercept_or_fallback(
                        provider=real_provider,
                        payload=payload,
                        config=config,
                        mock_provider=mock_provider,
                    ),
                    timeout=15.0,
                )
                dispatched_results.append(res)

        asyncio.run(_run_10_dispatches())

        assert len(dispatched_results) == 10

        # Jobs 1 and 2: Authorized (2 * 0.38125 = 0.76250 <= 1.00)
        assert dispatched_results[0][1] is False  # real
        assert dispatched_results[1][1] is False  # real

        # Jobs 3 through 10: Blocked by financial watchdog and seamlessly mocked
        for i in range(2, 10):
            assert (
                dispatched_results[i][1] is True
            ), f"Job {i+1} must be mocked due to budget exhaustion"
            assert dispatched_results[i][0].startswith("mock-")

        # STRICT PASS-TO-COMPLETE CHECK:
        # Real provider received EXACTLY 2 calls. Zero unauthorized calls made!
        assert len(real_provider.submission_calls) == 2


# =====================================================================
# 6. MetaController Integration with CostWatchdog
# =====================================================================


class TestMetaControllerCostWatchdogIntegration:
    """Validate MetaController seamless degradation under tight budgets."""

    def test_meta_controller_switches_cleanly_to_mock_under_tight_budget(
        self, tmp_path: Path
    ) -> None:
        """Invariant: MetaController completes wake-sleep cycles without error when budget exhausts."""
        # Ultra-tight budget of $0.05
        watchdog = CostWatchdog(CostWatchdogConfig(session_budget_cap_usd=0.05))
        real_provider = RecordingRealComputeProvider(provider_name="modal")

        ctrl_config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=2,
            remote_dispatch_enabled=True,
            hardware_accelerator="A10G",
            artifacts_dir=tmp_path,
        )

        controller = MetaController(
            config=ctrl_config,
            compute_provider=real_provider,
            cost_watchdog=watchdog,
            artifacts_dir=tmp_path,
        )

        # Run 3 consecutive cycles
        telemetries = controller.run_cycles(3)

        assert len(telemetries) == 3
        assert all(t.wake_tasks_attempted == 4 for t in telemetries)

        # Budget exhaustion check:
        # Jobs could not all go to real provider; mock fallback handled exhausted jobs seamlessly.
        # Controller completed all 3 cycles without unhandled exceptions or stalls.
        assert controller.active_adapter_id is not None or len(controller.replay_buffer) > 0
