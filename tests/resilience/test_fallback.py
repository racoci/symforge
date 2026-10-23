"""Tests for OfflineCircuitBreaker and OfflineTrajectoryQueue."""

import asyncio
import time
import pytest

from symforge.compute.mock_provider import MockComputeProvider
from symforge.core.ast import SineNode, SphereNode, FreeParam
from symforge.curriculum.frontier import CurriculumLevel
from symforge.orchestration.meta_controller import SolvedTrajectory
from symforge.resilience.fallback import (
    CircuitBreakerOpenError,
    CircuitState,
    OfflineCircuitBreaker,
    OfflineTrajectoryQueue,
)


class TestOfflineCircuitBreaker:
    """Test circuit breaker state machine transitions and execution guarding."""

    def test_initial_state_closed(self):
        cb = OfflineCircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.can_execute() is True

    def test_trip_on_consecutive_failures(self):
        cb = OfflineCircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True
        assert cb.can_execute() is False

    def test_open_state_raises_circuit_breaker_open_error(self):
        cb = OfflineCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.is_open is True

        with pytest.raises(CircuitBreakerOpenError):
            with cb:
                pass

        with pytest.raises(CircuitBreakerOpenError):
            cb.call_sync(lambda: 42)

    def test_half_open_transition_after_timeout(self):
        cb = OfflineCircuitBreaker(failure_threshold=1, recovery_timeout=0.05, success_threshold=1)
        cb.record_failure()
        assert cb.is_open is True

        # Wait for recovery timeout
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_half_open is True
        assert cb.can_execute() is True

        # Successful probe closes breaker
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True

    def test_half_open_failure_re_opens_breaker(self):
        cb = OfflineCircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN

        # Failed probe resets to OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestOfflineTrajectoryQueueGate:
    """Test offline trajectory queue buffering and flush gate."""

    def test_queue_enqueue_and_capacity(self):
        queue = OfflineTrajectoryQueue(capacity=150)
        assert queue.is_empty() is True

        # Enqueue 100 trajectories
        for i in range(100):
            traj = SolvedTrajectory(
                ast=SphereNode(FreeParam(f"r_{i}", 0.1, 5.0, 1.0)),
                level=CurriculumLevel.L1,
                domain="sdf",
                score=0.95,
            )
            queue.enqueue_trajectory(traj)

        assert len(queue) == 100
        assert queue.is_empty() is False

        # Peek
        peeked = queue.peek(5)
        assert len(peeked) == 5

    def test_gate_queues_100_trajectories_and_flushes_upon_reconnect(self, tmp_path):
        async def _test():
            cache_file = tmp_path / "offline_cache.jsonl"
            queue = OfflineTrajectoryQueue(capacity=500, cache_path=cache_file)
            cb = OfflineCircuitBreaker(failure_threshold=2, recovery_timeout=0.05)

            # 1. Simulate network failure (trip breaker to OPEN)
            cb.record_failure()
            cb.record_failure()
            assert cb.is_open is True

            # 2. Synthesis continues offline, enqueuing 100 trajectories
            for i in range(100):
                traj = SolvedTrajectory(
                    ast=SineNode(FreeParam(f"freq_{i}", 100.0, 1000.0, 440.0)),
                    level=CurriculumLevel.L1,
                    domain="audio",
                    score=0.90,
                )
                success = queue.enqueue_trajectory(traj)
                assert success is True

            # Verify all 100 trajectories are buffered and persisted to cache
            assert len(queue) == 100
            assert cache_file.exists()
            lines = cache_file.read_text().strip().split("\n")
            assert len(lines) == 100

            # 3. Network recovers: timeout expires and probe succeeds
            await asyncio.wait_for(asyncio.sleep(0.06), timeout=2.0)
            assert cb.state == CircuitState.HALF_OPEN
            cb.record_success()
            assert cb.is_closed is True

            # 4. Flush queued trajectories to provider
            mock_provider = MockComputeProvider()
            flushed_count = await asyncio.wait_for(
                queue.flush_queued_trajectories_async(
                    provider=mock_provider,
                    domain="audio",
                ),
                timeout=10.0,
            )

            assert flushed_count == 100
            assert len(queue) == 0
            assert queue.is_empty() is True

        asyncio.run(_test())
