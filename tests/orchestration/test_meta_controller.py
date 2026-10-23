"""Automated test suite for Task 6.1: Meta-Controller Regime Loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import time
from typing import List, Optional, Tuple
import pytest

from symforge.abstraction.grammar_expander import DynamicGrammarLibrary
from symforge.adapters.registry import LoRAAdapterRegistry
from symforge.compute.contracts import JobStatus
from symforge.compute.mock_provider import MockComputeProvider, MockExecutionMode
from symforge.core.ast import ASTNode, FreeParam, MixNode, SineNode
from symforge.core.curriculum import CurriculumLevel
from symforge.curriculum.frontier import ZPDSampler
from symforge.orchestration.meta_controller import (
    CycleTelemetry,
    MetaController,
    MetaControllerConfig,
    SolvedTrajectory,
)


def _make_dummy_mix_ast() -> ASTNode:
    """Create a recurrent audio mix AST for predictable macro discovery."""
    f = FreeParam(id="f", min_val=20.0, max_val=2000.0, default_val=440.0)
    p = FreeParam(id="p", min_val=0.0, max_val=6.28, default_val=0.0)
    r = FreeParam(id="r", min_val=0.0, max_val=1.0, default_val=0.5)
    return MixNode(ratio=r, input1=SineNode(f, p), input2=SineNode(f, p))


class TestMetaControllerLifecycle:
    """Validate full state machine lifecycle and individual Wake-Sleep phases."""

    def test_single_step_lifecycle_execution(self, tmp_path: Path):
        """Invariant: Single step executes wake synthesis, sleep abstraction, and dispatch."""
        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=2,
            sleep_mdl_threshold=5.0,  # permissive threshold for test macro acceptance
            remote_dispatch_enabled=True,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        telemetry = controller.step()

        assert isinstance(telemetry, CycleTelemetry)
        assert telemetry.cycle_index == 0
        assert telemetry.wake_tasks_attempted == 4
        assert telemetry.wake_tasks_solved > 0
        assert 0.0 <= telemetry.wake_success_rate <= 1.0
        assert isinstance(telemetry.active_frontier_level, CurriculumLevel)
        assert telemetry.cycle_duration_seconds > 0.0

        # Trajectories harvested into replay buffer
        assert len(controller.replay_buffer) == telemetry.wake_tasks_solved
        assert all(isinstance(t, SolvedTrajectory) for t in controller.replay_buffer)

        # Sleep dispatch submitted a background job
        assert controller.active_job_id is not None
        assert telemetry.background_job_id == controller.active_job_id

    def test_double_buffering_non_blocking_invariant(self, tmp_path: Path):
        """Invariant: Local synthesis never blocks on remote training; adapter hot-swaps on next cycle."""
        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=2,
            remote_dispatch_enabled=True,
            artifacts_dir=tmp_path,
        )
        # 2 steps to complete ensures job remains RUNNING across poll 1
        provider = MockComputeProvider(
            default_mode=MockExecutionMode.SIMULATED_STEPS,
            steps_to_complete=2,
        )
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        # Cycle 0: Starts with no adapter; submits Job 1 in background sleep phase
        t0 = controller.step()
        job_1_id = t0.background_job_id
        assert job_1_id is not None
        assert controller.active_adapter_id is None
        assert t0.active_adapter_id is None

        # Cycle 1: Pre-wake polls Job 1 (poll_count=1 < 2 -> RUNNING).
        # Synthesis executes immediately with active_adapter_id still None (NO BLOCKING)
        t1 = controller.step()
        assert t1.active_adapter_id is None
        assert controller.active_job_id == job_1_id

        # Cycle 2: Pre-wake polls Job 1 (poll_count=2 == steps_to_complete -> COMPLETED).
        # Seamlessly downloads artifact, checks SHA-256, and hot-swaps adapter!
        t2 = controller.step()
        assert controller.active_adapter_id is not None
        assert controller.active_adapter_id.startswith("adapter-")
        assert t2.active_adapter_id == controller.active_adapter_id

        # Verify weights loaded in registry
        weights = controller.adapter_registry.load_weights(controller.active_adapter_id)
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_multi_cycle_execution_and_pass_gate(self, tmp_path: Path):
        """Pass-to-Complete Gate: 5 full wake-sleep iterations run in < 2.0 seconds on CPU."""
        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=3,
            remote_dispatch_enabled=True,
            fpo_steps=3,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        start_time = time.perf_counter()
        telemetries = controller.run_cycles(5)
        total_duration = time.perf_counter() - start_time

        assert len(telemetries) == 5
        assert [t.cycle_index for t in telemetries] == [0, 1, 2, 3, 4]
        assert len(controller.telemetry_history) == 5

        # Check total wall-clock duration against the < 2.0 seconds gate
        assert total_duration < 2.0, f"Pass-to-Complete Gate exceeded: {total_duration:.3f}s >= 2.0s"

        # Check adapter hot-swapping occurred across cycles
        assert controller.active_adapter_id is not None

    def test_remote_dispatch_disabled(self, tmp_path: Path):
        """Invariant: When remote dispatch is disabled, no background jobs are dispatched."""
        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=1,
            remote_dispatch_enabled=False,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        telemetry = controller.step()

        assert telemetry.background_job_id is None
        assert controller.active_job_id is None
        assert controller.active_adapter_id is None

    def test_min_solutions_for_sleep_gating(self, tmp_path: Path):
        """Invariant: Sleep phase is bypassed if replay buffer lacks sufficient solutions."""
        # Custom synthesizer that never succeeds
        def failing_synthesizer(target: ASTNode, level: CurriculumLevel, domain: str) -> Tuple[Optional[ASTNode], bool]:
            return None, False

        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=5,
            remote_dispatch_enabled=True,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        controller = MetaController(
            config=config,
            compute_provider=provider,
            synthesizer=failing_synthesizer,
            artifacts_dir=tmp_path,
        )

        telemetry = controller.step()

        assert telemetry.wake_tasks_solved == 0
        assert telemetry.wake_success_rate == 0.0
        assert telemetry.abstractions_discovered == 0
        assert telemetry.background_job_id is None
        assert controller.active_job_id is None

    def test_replay_buffer_bounded_capacity(self, tmp_path: Path):
        """Invariant: Replay buffer adheres strictly to max_replay_buffer_size capacity."""
        max_cap = 5
        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=2,
            max_replay_buffer_size=max_cap,
            remote_dispatch_enabled=False,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        controller.run_cycles(3)  # Attempts 12 tasks

        assert len(controller.replay_buffer) <= max_cap

    def test_failing_remote_job_resilience(self, tmp_path: Path):
        """Invariant: Simulated remote training failure is handled gracefully without crashing."""
        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=2,
            remote_dispatch_enabled=True,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.FAILING)
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        # Cycle 0 dispatches failing job 1
        t0 = controller.step()
        failed_job_id = t0.background_job_id
        assert failed_job_id is not None

        # Cycle 1 polls failing job: status is FAILED, logged, job cleared, and new job dispatched in sleep
        t1 = controller.step()
        assert controller.active_adapter_id is None
        assert t1.background_job_id != failed_job_id

        # Explicit poll clears active failed job without crashing
        controller.check_background_training()
        assert controller.active_job_id is None
        assert controller.active_adapter_id is None

    def test_sdf_domain_synthesis(self, tmp_path: Path):
        """Invariant: MetaController operates symmetrically across the SDF geometry domain."""
        config = MetaControllerConfig(
            domain="sdf",
            wake_batch_size=3,
            min_solutions_for_sleep=2,
            remote_dispatch_enabled=True,
            fpo_steps=3,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        telemetry = controller.step()

        assert telemetry.wake_tasks_attempted == 3
        assert telemetry.wake_tasks_solved > 0
        assert all(t.domain == "sdf" for t in controller.replay_buffer)

    def test_abstraction_discovery_and_grammar_expansion(self, tmp_path: Path):
        """Invariant: Recurrent solution structures are discovered as macros and registered."""
        # Inject custom synthesizer returning recurrent mix AST
        def recurrent_synthesizer(target: ASTNode, level: CurriculumLevel, domain: str) -> Tuple[Optional[ASTNode], bool]:
            return _make_dummy_mix_ast(), True

        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=3,
            sleep_mdl_threshold=100.0,  # High threshold ensures acceptance for test
            min_pattern_size=2,
            min_pattern_frequency=2,
            remote_dispatch_enabled=False,
            artifacts_dir=tmp_path,
        )
        library = DynamicGrammarLibrary()
        controller = MetaController(
            config=config,
            grammar_library=library,
            synthesizer=recurrent_synthesizer,
            artifacts_dir=tmp_path,
        )

        initial_macros = len(library.macros)
        telemetry = controller.step()

        # At least one new macro should be discovered and registered
        assert telemetry.abstractions_discovered > 0
        assert len(library.macros) > initial_macros

    def test_manual_check_background_training_call(self, tmp_path: Path):
        """Invariant: Manual call to check_background_training returns hot-swapped adapter ID."""
        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=4,
            min_solutions_for_sleep=2,
            remote_dispatch_enabled=True,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        controller.step()  # Dispatches job in cycle 0
        assert controller.active_job_id is not None

        # Explicit manual invocation
        new_adapter = controller.check_background_training()

        assert new_adapter is not None
        assert new_adapter == controller.active_adapter_id
        assert controller.active_job_id is None

    def test_async_execution_api(self, tmp_path: Path):
        """Invariant: Async API methods execute symmetrically with hard timeouts."""
        config = MetaControllerConfig(
            domain="audio",
            wake_batch_size=2,
            min_solutions_for_sleep=2,
            fpo_steps=2,
            artifacts_dir=tmp_path,
        )
        provider = MockComputeProvider(default_mode=MockExecutionMode.INSTANT)
        controller = MetaController(config=config, compute_provider=provider, artifacts_dir=tmp_path)

        telemetries = asyncio.run(
            asyncio.wait_for(
                controller.run_cycles_async(2),
                timeout=10.0,
            )
        )

        assert len(telemetries) == 2
        assert telemetries[0].cycle_index == 0
        assert telemetries[1].cycle_index == 1
