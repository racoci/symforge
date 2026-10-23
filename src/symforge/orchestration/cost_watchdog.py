"""Financial Pre-Flight Cost Watchdog & Safety Gate for remote compute dispatch.

Enforces session budget caps, calculates 25% safety over-estimation buffers,
performs zero-cost local arithmetic pre-flight checks, and intercepts unauthorized
cloud calls with graceful degradation to MockComputeProvider ($0.00 USD).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import logging
from typing import Any, Dict, Optional, Tuple
import uuid

from symforge.compute.contracts import (
    ComputeProviderError,
    RemoteJobConfig,
    TrainingPayload,
)
from symforge.compute.mock_provider import MockComputeProvider
from symforge.compute.protocol import ComputeProvider

logger = logging.getLogger(__name__)


# =====================================================================
# Exceptions
# =====================================================================


class CostWatchdogError(ComputeProviderError):
    """Base exception for all CostWatchdog errors."""


class BudgetExceededError(CostWatchdogError):
    """Raised when remote job cost exceeds session budget and fallback is disabled."""


# =====================================================================
# Rate Cards and Pricing Constants
# =====================================================================

# Standard published provider rate cards: (provider_normalized, accelerator_upper) -> USD/sec
DEFAULT_RATE_CARDS: Dict[Tuple[str, str], float] = {
    # Modal serverless container rates
    ("modal", "CPU"): 0.000028,
    ("modal", "T4"): 0.000164,
    ("modal", "A10G"): 0.000305,
    ("modal", "A100"): 0.001111,
    # AWS SageMaker managed training rates
    ("sagemaker", "CPU"): 0.000064,
    ("sagemaker", "T4"): 0.000205,
    ("sagemaker", "A10G"): 0.000336,
    ("sagemaker", "A100"): 0.009000,
    # Mock zero-cost provider
    ("mock", "CPU"): 0.0,
    ("mock", "T4"): 0.0,
    ("mock", "A10G"): 0.0,
    ("mock", "A100"): 0.0,
}

# Conservative fallback rates when provider is not recognized
FALLBACK_RATES: Dict[str, float] = {
    "CPU": 0.000050,
    "T4": 0.000200,
    "A10G": 0.000350,
    "A100": 0.001500,
}


# =====================================================================
# Data Structures
# =====================================================================


@dataclass(frozen=True)
class CostWatchdogConfig:
    """Configuration parameters for Financial Cost Watchdog and safety gate.

    Attributes:
        session_budget_cap_usd: Hard cap for total session expenditures in USD (default $2.00).
        safety_buffer_pct: Over-estimation safety buffer percentage (default 0.25 = 25%).
        fallback_to_mock_on_exhaustion: Whether to intercept blocked dispatches and route to mock.
        alert_threshold_pct: Expenditure threshold fraction (e.g. 0.80 = 80%) triggering alert logs.
        min_billing_duration_seconds: Minimum billable duration floor in seconds.
        custom_rate_cards: Optional user-defined mapping of (provider, accelerator) -> USD/sec.
        use_timeout_for_duration: Whether to conservatively assume config.timeout_seconds as duration.
    """

    session_budget_cap_usd: float = 2.00
    safety_buffer_pct: float = 0.25
    fallback_to_mock_on_exhaustion: bool = True
    alert_threshold_pct: float = 0.80
    min_billing_duration_seconds: float = 1.0
    custom_rate_cards: Optional[Dict[Tuple[str, str], float]] = None
    use_timeout_for_duration: bool = False

    def __post_init__(self) -> None:
        """Validate invariant bounds for watchdog configuration."""
        if self.session_budget_cap_usd < 0.0:
            raise ValueError(
                f"session_budget_cap_usd must be non-negative, got {self.session_budget_cap_usd}"
            )
        if self.safety_buffer_pct < 0.0:
            raise ValueError(
                f"safety_buffer_pct must be non-negative, got {self.safety_buffer_pct}"
            )
        if not 0.0 < self.alert_threshold_pct <= 1.0:
            raise ValueError(
                f"alert_threshold_pct must be in (0.0, 1.0], got {self.alert_threshold_pct}"
            )
        if self.min_billing_duration_seconds <= 0.0:
            raise ValueError(
                f"min_billing_duration_seconds must be > 0.0, got {self.min_billing_duration_seconds}"
            )


@dataclass(frozen=True)
class CostEstimate:
    """Pre-flight financial cost estimate for a single training job dispatch.

    Attributes:
        provider_name: Target compute provider name.
        hardware_accelerator: Target hardware accelerator tier (e.g. 'CPU', 'A10G').
        estimated_duration_seconds: Projected job execution duration in seconds.
        base_rate_per_second_usd: Base unbuffered rate in USD per second.
        raw_estimated_cost_usd: Unbuffered baseline cost projection.
        buffered_estimated_cost_usd: Final buffered cost with safety factor and granularity applied.
        is_within_budget: Whether the projected buffered cost is within remaining budget.
        remaining_budget_before_usd: Session budget available prior to this dispatch.
        remaining_budget_after_usd: Session budget remaining if this dispatch is authorized.
        reservation_id: Optional UUID identifying this reservation if authorized.
    """

    provider_name: str
    hardware_accelerator: str
    estimated_duration_seconds: float
    base_rate_per_second_usd: float
    raw_estimated_cost_usd: float
    buffered_estimated_cost_usd: float
    is_within_budget: bool
    remaining_budget_before_usd: float
    remaining_budget_after_usd: float
    reservation_id: Optional[str] = None


@dataclass(frozen=True)
class DispatchDecision:
    """Pre-flight evaluation verdict for dispatching a remote training job.

    Attributes:
        authorized: Whether the dispatch is permitted to execute on the real provider.
        use_mock_fallback: Whether to gracefully degrade and route to MockComputeProvider.
        estimate: Detailed CostEstimate calculation.
        reason: Diagnostic post-mortem string explaining the authorization or blocking decision.
    """

    authorized: bool
    use_mock_fallback: bool
    estimate: CostEstimate
    reason: str


# =====================================================================
# CostWatchdog Class
# =====================================================================


class CostWatchdog:
    """Financial Pre-Flight Cost Watchdog & Safety Gate."""

    def __init__(
        self,
        config: Optional[CostWatchdogConfig] = None,
    ) -> None:
        """Initialize CostWatchdog with budget cap and rate card configurations."""
        self._config = config or CostWatchdogConfig()
        self._rate_cards: Dict[Tuple[str, str], float] = dict(DEFAULT_RATE_CARDS)
        if self._config.custom_rate_cards:
            self._rate_cards.update(self._config.custom_rate_cards)

        self._total_actual_spent: float = 0.0
        self._total_reserved: float = 0.0
        self._active_reservations: Dict[str, float] = {}
        self._job_to_reservation: Dict[str, str] = {}
        self._alert_emitted: bool = False

    @property
    def config(self) -> CostWatchdogConfig:
        """Return active CostWatchdog configuration."""
        return self._config

    def _normalize_provider_name(self, provider_name: str) -> str:
        """Normalize provider class or string name to canonical rate-card key."""
        clean = provider_name.lower().replace("computeprovider", "").strip()
        if "mock" in clean:
            return "mock"
        if "modal" in clean:
            return "modal"
        if "sagemaker" in clean:
            return "sagemaker"
        return clean

    def resolve_rate_per_second(
        self, provider_name: str, accelerator: str
    ) -> float:
        """Resolve base USD/sec rate for a given provider and accelerator."""
        p_key = self._normalize_provider_name(provider_name)
        a_key = accelerator.upper()

        if p_key == "mock":
            return 0.0

        lookup_key = (p_key, a_key)
        if lookup_key in self._rate_cards:
            return self._rate_cards[lookup_key]

        logger.info(
            "Rate card not found for (%s, %s); applying fallback rate.",
            p_key,
            a_key,
        )
        return FALLBACK_RATES.get(a_key, 0.000100)

    def estimate_duration_seconds(
        self, payload: TrainingPayload, config: RemoteJobConfig
    ) -> float:
        """Estimate job duration in seconds with cold-start overhead and safety bounds."""
        if "duration_seconds" in payload.hyperparameters:
            return max(
                self._config.min_billing_duration_seconds,
                float(payload.hyperparameters["duration_seconds"]),
            )
        if "estimated_duration_seconds" in payload.hyperparameters:
            return max(
                self._config.min_billing_duration_seconds,
                float(payload.hyperparameters["estimated_duration_seconds"]),
            )
        if "duration_seconds" in config.tags:
            return max(
                self._config.min_billing_duration_seconds,
                float(config.tags["duration_seconds"]),
            )

        if self._config.use_timeout_for_duration:
            return max(
                self._config.min_billing_duration_seconds,
                config.timeout_seconds,
            )

        overhead = 10.0
        accel = config.hardware_accelerator.upper()
        speed_per_traj = {
            "A100": 0.02,
            "A10G": 0.05,
            "T4": 0.10,
            "CPU": 0.50,
        }.get(accel, 0.10)

        computed = overhead + (payload.trajectories_count * speed_per_traj)
        clamped = min(computed, config.timeout_seconds)
        return max(self._config.min_billing_duration_seconds, clamped)

    def get_remaining_budget(self) -> float:
        """Return currently available session budget in USD."""
        encumbered = self._total_actual_spent + self._total_reserved
        return max(0.0, self._config.session_budget_cap_usd - encumbered)

    def get_total_spent(self) -> float:
        """Return cumulative encumbered session expenditure (actual + reserved)."""
        return self._total_actual_spent + self._total_reserved

    def reset_session(self) -> None:
        """Reset all session reservations and expenditure counters to zero."""
        self._total_actual_spent = 0.0
        self._total_reserved = 0.0
        self._active_reservations.clear()
        self._job_to_reservation.clear()
        self._alert_emitted = False

    def estimate_job_cost(
        self, payload: TrainingPayload, config: RemoteJobConfig
    ) -> CostEstimate:
        """Calculate pre-flight unbuffered and buffered cost estimates."""
        p_name = self._normalize_provider_name(config.provider_name)
        accel = config.hardware_accelerator.upper()

        rate = self.resolve_rate_per_second(p_name, accel)
        duration = self.estimate_duration_seconds(payload, config)

        raw_cost = rate * duration
        buffered_cost = raw_cost * (1.0 + self._config.safety_buffer_pct)

        # For zero-cost providers, guarantee exactly 0.0
        if rate == 0.0 or p_name == "mock":
            raw_cost = 0.0
            buffered_cost = 0.0

        remaining_before = self.get_remaining_budget()
        is_within_budget = buffered_cost <= remaining_before
        remaining_after = (
            max(0.0, remaining_before - buffered_cost)
            if is_within_budget
            else remaining_before
        )

        return CostEstimate(
            provider_name=config.provider_name,
            hardware_accelerator=config.hardware_accelerator,
            estimated_duration_seconds=duration,
            base_rate_per_second_usd=rate,
            raw_estimated_cost_usd=raw_cost,
            buffered_estimated_cost_usd=buffered_cost,
            is_within_budget=is_within_budget,
            remaining_budget_before_usd=remaining_before,
            remaining_budget_after_usd=remaining_after,
        )

    def evaluate_dispatch(
        self, payload: TrainingPayload, config: RemoteJobConfig
    ) -> DispatchDecision:
        """Evaluate pre-flight dispatch feasibility against remaining budget."""
        estimate = self.estimate_job_cost(payload, config)

        if estimate.is_within_budget:
            reason = (
                f"Authorized: Buffered cost ${estimate.buffered_estimated_cost_usd:.4f} "
                f"is within remaining budget ${estimate.remaining_budget_before_usd:.4f}."
            )
            return DispatchDecision(
                authorized=True,
                use_mock_fallback=False,
                estimate=estimate,
                reason=reason,
            )

        reason = (
            f"Blocked: Buffered cost ${estimate.buffered_estimated_cost_usd:.4f} "
            f"exceeds remaining budget ${estimate.remaining_budget_before_usd:.4f} "
            f"(Session cap: ${self._config.session_budget_cap_usd:.2f})."
        )
        return DispatchDecision(
            authorized=False,
            use_mock_fallback=self._config.fallback_to_mock_on_exhaustion,
            estimate=estimate,
            reason=reason,
        )

    def authorize_and_reserve(
        self, payload: TrainingPayload, config: RemoteJobConfig
    ) -> Tuple[bool, CostEstimate]:
        """Atomically evaluate dispatch and encumber estimated budget if authorized."""
        decision = self.evaluate_dispatch(payload, config)
        estimate = decision.estimate

        if not decision.authorized:
            logger.warning(
                "CostWatchdog Rejected Dispatch: %s | Provider=%s, Accel=%s, Trajectories=%d, "
                "BufferedCost=$%.4f, RemainingBudget=$%.4f, Cap=$%.2f",
                decision.reason,
                config.provider_name,
                config.hardware_accelerator,
                payload.trajectories_count,
                estimate.buffered_estimated_cost_usd,
                estimate.remaining_budget_before_usd,
                self._config.session_budget_cap_usd,
            )
            return False, estimate

        res_id = str(uuid.uuid4())
        self._active_reservations[res_id] = estimate.buffered_estimated_cost_usd
        self._total_reserved += estimate.buffered_estimated_cost_usd

        estimate_with_res = replace(estimate, reservation_id=res_id)

        # Check alert threshold
        spent_fraction = (
            self.get_total_spent() / max(0.001, self._config.session_budget_cap_usd)
        )
        if spent_fraction >= self._config.alert_threshold_pct and not self._alert_emitted:
            logger.warning(
                "CostWatchdog Alert: Session budget consumption reached %.1f%% "
                "($%.2f / $%.2f). Remaining budget: $%.4f",
                spent_fraction * 100.0,
                self.get_total_spent(),
                self._config.session_budget_cap_usd,
                self.get_remaining_budget(),
            )
            self._alert_emitted = True

        return True, estimate_with_res

    def record_actual_cost(
        self, reservation_or_job_id: str, actual_cost_usd: float
    ) -> None:
        """Reconcile reserved budget against actual reported cost upon job completion."""
        if actual_cost_usd < 0.0:
            raise ValueError(
                f"actual_cost_usd must be non-negative, got {actual_cost_usd}"
            )

        res_id = self._job_to_reservation.pop(
            reservation_or_job_id, reservation_or_job_id
        )

        if res_id in self._active_reservations:
            reserved_amount = self._active_reservations.pop(res_id)
            self._total_reserved = max(0.0, self._total_reserved - reserved_amount)
            self._total_actual_spent += actual_cost_usd
            logger.info(
                "CostWatchdog Reconciled: id=%s, reserved=$%.4f, actual=$%.4f, "
                "released_diff=$%.4f, remaining_budget=$%.4f",
                res_id,
                reserved_amount,
                actual_cost_usd,
                reserved_amount - actual_cost_usd,
                self.get_remaining_budget(),
            )
        else:
            self._total_actual_spent += actual_cost_usd
            logger.info(
                "CostWatchdog Direct Spend Recorded: id=%s, actual=$%.4f, remaining_budget=$%.4f",
                res_id,
                actual_cost_usd,
                self.get_remaining_budget(),
            )

    async def intercept_or_fallback(
        self,
        provider: ComputeProvider,
        payload: TrainingPayload,
        config: RemoteJobConfig,
        mock_provider: Optional[ComputeProvider] = None,
    ) -> Tuple[str, bool]:
        """Pre-flight safety gate intercepting dispatches or falling back to mock provider.

        Args:
            provider: Real or primary target ComputeProvider.
            payload: Training dataset and parameters payload.
            config: Remote execution environment configuration.
            mock_provider: Optional fallback mock provider (defaults to MockComputeProvider).

        Returns:
            Tuple of (job_id, was_mocked).

        Raises:
            BudgetExceededError: If budget cap is exceeded and fallback_to_mock is disabled.
        """
        authorized, estimate = self.authorize_and_reserve(payload, config)

        if authorized:
            try:
                job_id = await asyncio.wait_for(
                    provider.submit_training_job(payload, config),
                    timeout=15.0,
                )
                if estimate.reservation_id:
                    self._job_to_reservation[job_id] = estimate.reservation_id
                return job_id, False
            except Exception as exc:
                logger.error(
                    "CostWatchdog: Real provider dispatch failed for %s (%s): %s. Releasing reservation.",
                    config.provider_name,
                    config.hardware_accelerator,
                    exc,
                )
                if estimate.reservation_id:
                    self.record_actual_cost(estimate.reservation_id, 0.0)
                raise

        # Unauthorized: Budget exceeded
        if not self._config.fallback_to_mock_on_exhaustion:
            raise BudgetExceededError(
                f"Budget exceeded: Projected cost ${estimate.buffered_estimated_cost_usd:.4f} "
                f"exceeds remaining budget ${self.get_remaining_budget():.4f} and fallback is disabled."
            )

        active_mock = mock_provider or MockComputeProvider()
        mock_config = config.model_copy(update={"provider_name": "mock"})

        try:
            job_id = await asyncio.wait_for(
                active_mock.submit_training_job(payload, mock_config),
                timeout=10.0,
            )
        except Exception as exc:
            logger.error("CostWatchdog: Fallback mock submission failed: %s", exc)
            raise

        logger.warning(
            "CostWatchdog Safety Gate Intercepted: Dispatched job to zero-cost MockComputeProvider ($0.00). "
            "Original target: %s (%s), Projected buffered cost: $%.4f, Remaining budget: $%.4f. "
            "Session cap: $%.2f. Assigned mock job_id=%s",
            config.provider_name,
            config.hardware_accelerator,
            estimate.buffered_estimated_cost_usd,
            self.get_remaining_budget(),
            self._config.session_budget_cap_usd,
            job_id,
        )
        return job_id, True
