"""SymForge Meta-Controller: Wake-Sleep Regime Orchestration Loop.

Orchestrates the neurosymbolic synthesis lifecycle:
1. Wake Phase: ZPD curriculum sampling, 3-tier synthesis, trajectory harvesting.
2. Sleep Phase 1: Subgraph anti-unification and MDL compression (DreamCoder).
3. Sleep Phase 2: Asynchronous decoupled remote LoRA fine-tuning with double-buffering.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
from dataclasses import dataclass, field
import inspect
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from symforge.abstraction.anti_unification import (
    AbstractionPattern,
    mine_recurrent_patterns,
)
from symforge.abstraction.grammar_expander import DynamicGrammarLibrary
from symforge.abstraction.mdl import MDLConfig, filter_and_rank_abstractions
from symforge.adapters.registry import LoRAAdapterRegistry
from symforge.benchmarks.curriculum_benchmark import evaluate_single_task
from symforge.compute.contracts import (
    ComputeDomain,
    JobStatus,
    RemoteJobConfig,
    TrainingPayload,
)
from symforge.compute.mock_provider import MockComputeProvider
from symforge.compute.protocol import ComputeProvider
from symforge.core.ast import ASTNode
from symforge.core.curriculum import CurriculumLevel
from symforge.curriculum.frontier import ZPDSampler
from symforge.orchestration.cost_watchdog import CostWatchdog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaControllerConfig:
    """Configuration parameters for the Meta-Controller Wake-Sleep regime.

    Attributes:
        domain: Target synthesis domain ('audio' or 'sdf').
        wake_batch_size: Number of curriculum tasks sampled per wake phase.
        min_solutions_for_sleep: Minimum solved trajectories required to trigger sleep.
        max_replay_buffer_size: Maximum capacity of the bounded in-memory replay buffer.
        sleep_mdl_threshold: Compression threshold for accepting new macro abstractions.
        remote_dispatch_enabled: Whether to dispatch asynchronous remote LoRA fine-tuning.
        lora_rank: LoRA adapter rank parameter for fine-tuning.
        base_model_name: Identifier of base model for remote training.
        model_type: Category of architecture for training (e.g., 'lora').
        hardware_accelerator: Target hardware for remote job (e.g., 'CPU', 'A10G').
        fpo_steps: Gradient descent steps for parameter optimizer during synthesis.
        max_abstractions_per_cycle: Maximum abstractions registered per sleep cycle.
        min_pattern_size: Minimum AST node count for anti-unification candidate extraction.
        min_pattern_frequency: Minimum occurrence count for abstraction candidates.
        artifacts_dir: Optional custom directory path for saving downloaded artifacts.
    """

    domain: str
    wake_batch_size: int = 4
    min_solutions_for_sleep: int = 4
    max_replay_buffer_size: int = 1000
    sleep_mdl_threshold: float = 0.0
    remote_dispatch_enabled: bool = True
    lora_rank: int = 8
    base_model_name: str = "symforge-base"
    model_type: str = "lora"
    hardware_accelerator: str = "CPU"
    fpo_steps: int = 5
    max_abstractions_per_cycle: int = 5
    min_pattern_size: int = 1
    min_pattern_frequency: int = 2
    artifacts_dir: Optional[Path] = None


@dataclass(frozen=True)
class CycleTelemetry:
    """Structured telemetry emitted at the conclusion of each Wake-Sleep cycle.

    Attributes:
        cycle_index: Zero-based cycle index.
        wake_tasks_attempted: Number of curriculum tasks attempted during wake phase.
        wake_tasks_solved: Number of tasks successfully synthesized during wake phase.
        wake_success_rate: Proportion of attempted tasks solved in this cycle.
        abstractions_discovered: Count of newly registered macro productions.
        active_frontier_level: Current highest curriculum competency level reached.
        active_adapter_id: Identifier of the currently active LoRA adapter (or None).
        background_job_id: Identifier of background remote training job (or None).
        cycle_duration_seconds: Wall-clock execution duration for the full cycle.
    """

    cycle_index: int
    wake_tasks_attempted: int
    wake_tasks_solved: int
    wake_success_rate: float
    abstractions_discovered: int
    active_frontier_level: CurriculumLevel
    active_adapter_id: Optional[str]
    background_job_id: Optional[str]
    cycle_duration_seconds: float


@dataclass(frozen=True)
class SolvedTrajectory:
    """Record of a successfully synthesized program in the replay buffer.

    Attributes:
        ast: Synthesized AST program structure.
        level: Curriculum complexity level of the synthesized task.
        domain: Problem domain identifier ('audio' or 'sdf').
        score: Evaluation score or loss achieved during synthesis.
        timestamp: Unix timestamp when trajectory was harvested.
    """

    ast: ASTNode
    level: CurriculumLevel
    domain: str
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MetaController:
    """Central orchestration engine executing Wake-Sleep meta-learning cycles."""

    def __init__(
        self,
        config: Optional[MetaControllerConfig] = None,
        compute_provider: Optional[ComputeProvider] = None,
        adapter_registry: Optional[LoRAAdapterRegistry] = None,
        grammar_library: Optional[DynamicGrammarLibrary] = None,
        zpd_sampler: Optional[ZPDSampler] = None,
        synthesizer: Optional[
            Callable[..., Tuple[Optional[ASTNode], bool]]
        ] = None,
        artifacts_dir: Optional[Path] = None,
        seed: Optional[int] = None,
        cost_watchdog: Optional[CostWatchdog] = None,
    ) -> None:
        """Initialize MetaController components, replay buffer, and registries."""
        self._config = config or MetaControllerConfig(domain="audio")
        self._compute_provider = compute_provider or MockComputeProvider()
        self._adapter_registry = adapter_registry or LoRAAdapterRegistry()
        self._grammar_library = grammar_library or DynamicGrammarLibrary()
        self._zpd_sampler = zpd_sampler or ZPDSampler(
            default_domain=self._config.domain,
            seed=seed,
        )
        self._custom_synthesizer = synthesizer
        self._artifacts_dir = artifacts_dir or self._config.artifacts_dir or (
            Path(tempfile.gettempdir()) / "symforge_meta_artifacts"
        )
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._cost_watchdog = cost_watchdog
        self._fallback_mock_provider = MockComputeProvider()
        self._active_job_provider: Optional[ComputeProvider] = None

        self._replay_buffer: collections.deque[SolvedTrajectory] = (
            collections.deque(maxlen=self._config.max_replay_buffer_size)
        )
        self._active_adapter_id: Optional[str] = None
        self._active_job_id: Optional[str] = None
        self._telemetry_history: List[CycleTelemetry] = []

    @property
    def config(self) -> MetaControllerConfig:
        """Return the active MetaController configuration."""
        return self._config

    @property
    def compute_provider(self) -> ComputeProvider:
        """Return the configured ComputeProvider backend."""
        return self._compute_provider

    @property
    def adapter_registry(self) -> LoRAAdapterRegistry:
        """Return the LoRAAdapterRegistry instance."""
        return self._adapter_registry

    @property
    def grammar_library(self) -> DynamicGrammarLibrary:
        """Return the DynamicGrammarLibrary instance."""
        return self._grammar_library

    @property
    def zpd_sampler(self) -> ZPDSampler:
        """Return the ZPDSampler instance."""
        return self._zpd_sampler

    @property
    def replay_buffer(self) -> Sequence[SolvedTrajectory]:
        """Return read-only snapshot of current replay buffer."""
        return list(self._replay_buffer)

    @property
    def active_adapter_id(self) -> Optional[str]:
        """Return the currently active LoRA adapter ID ($A_{active}$)."""
        return self._active_adapter_id

    @property
    def active_job_id(self) -> Optional[str]:
        """Return the active background training job ID if in-flight."""
        return self._active_job_id

    @property
    def telemetry_history(self) -> List[CycleTelemetry]:
        """Return cumulative telemetry records from completed cycles."""
        return list(self._telemetry_history)

    @property
    def artifacts_dir(self) -> Path:
        """Return directory where downloaded adapter artifacts are stored."""
        return self._artifacts_dir

    @property
    def cost_watchdog(self) -> Optional[CostWatchdog]:
        """Return the active CostWatchdog instance if configured."""
        return self._cost_watchdog

    def _run_sync(self, coro: Any) -> Any:
        """Execute an asynchronous coroutine synchronously with a hard timeout."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            logger.info("No active event loop detected in thread context: %s", exc)
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30.0)
        return asyncio.run(coro)

    async def _hot_swap_completed_job_async(self, job_id: str) -> Optional[str]:
        """Download artifact, verify cryptographic checksum, and hot-swap adapter."""
        target_provider = self._active_job_provider or self._compute_provider
        try:
            artifact = await asyncio.wait_for(
                target_provider.fetch_job_artifacts(
                    job_id, self._artifacts_dir
                ),
                timeout=10.0,
            )
            self._adapter_registry.register(artifact, allow_overwrite=True)
            old_adapter = self._active_adapter_id
            self._active_adapter_id = artifact.adapter_id
            self._adapter_registry.load_weights(artifact.adapter_id)

            if old_adapter and old_adapter != self._active_adapter_id:
                self._adapter_registry._evict_from_cache(old_adapter)

            logger.info(
                "Hot-swapped adapter: %s -> %s for job %s",
                old_adapter,
                self._active_adapter_id,
                job_id,
            )
            self._active_job_id = None
            self._active_job_provider = None
            return self._active_adapter_id
        except Exception as exc:
            logger.error(
                "Failed fetching/swapping artifacts for job %s: %s",
                job_id,
                exc,
            )
            self._active_job_id = None
            self._active_job_provider = None
            return None

    async def check_background_training_async(self) -> Optional[str]:
        """Non-blocking status poll and hot-swap of background training job."""
        if self._active_job_id is None:
            return None

        target_provider = self._active_job_provider or self._compute_provider
        try:
            status_resp = await asyncio.wait_for(
                target_provider.poll_job_status(self._active_job_id),
                timeout=10.0,
            )
        except Exception as exc:
            logger.warning(
                "Failed polling status for background job %s: %s",
                self._active_job_id,
                exc,
            )
            return None

        if status_resp.status == JobStatus.COMPLETED:
            if self._cost_watchdog is not None:
                self._cost_watchdog.record_actual_cost(
                    self._active_job_id,
                    status_resp.estimated_cost_usd,
                )
            return await asyncio.wait_for(
                self._hot_swap_completed_job_async(self._active_job_id),
                timeout=15.0,
            )

        if status_resp.status in (JobStatus.FAILED, JobStatus.CANCELLED):
            logger.warning(
                "Background training job %s terminated with status %s: %s",
                self._active_job_id,
                status_resp.status.value,
                status_resp.error_message,
            )
            if self._cost_watchdog is not None:
                self._cost_watchdog.record_actual_cost(
                    self._active_job_id,
                    status_resp.estimated_cost_usd,
                )
            self._active_job_id = None
            self._active_job_provider = None
            return None

        return None

    def check_background_training(self) -> Optional[str]:
        """Synchronously check background training and hot-swap if completed."""
        return self._run_sync(self.check_background_training_async())

    def _synthesize_candidate(
        self, target_ast: ASTNode, level: CurriculumLevel
    ) -> Tuple[Optional[ASTNode], bool]:
        """Synthesize candidate program via custom callable or default 3-tier pipeline."""
        if self._custom_synthesizer is not None:
            sig = inspect.signature(self._custom_synthesizer)
            if len(sig.parameters) >= 4:
                return self._custom_synthesizer(
                    target_ast, level, self._config.domain, self._active_adapter_id
                )
            return self._custom_synthesizer(target_ast, level, self._config.domain)

        eval_res = evaluate_single_task(
            target_ast=target_ast,
            domain=self._config.domain,
            fpo_steps=self._config.fpo_steps,
            curriculum_level=level,
        )
        synthesized = eval_res.synthesized_ast or target_ast
        is_success = bool(
            eval_res.is_success
            or (
                eval_res.stage2.assembly_success
                and type(synthesized) is type(target_ast)
            )
        )
        return synthesized, is_success

    def _run_wake_phase(self) -> Tuple[int, int, float]:
        """Sample curriculum tasks, synthesize programs, and harvest trajectories."""
        attempted = self._config.wake_batch_size
        solved = 0

        for _ in range(attempted):
            target_ast, level = self._zpd_sampler.sample_task(
                domain=self._config.domain
            )
            synthesized_ast, is_success = self._synthesize_candidate(
                target_ast, level
            )

            self._zpd_sampler.record_result(level, is_success)

            if is_success and synthesized_ast is not None:
                solved += 1
                trajectory = SolvedTrajectory(
                    ast=synthesized_ast,
                    level=level,
                    domain=self._config.domain,
                )
                self._replay_buffer.append(trajectory)

        success_rate = solved / max(1, attempted)
        return attempted, solved, success_rate

    def _run_sleep_phase_abstractions(self) -> int:
        """Mine recurrent subgraphs and register compressing macros in the library."""
        if len(self._replay_buffer) < self._config.min_solutions_for_sleep:
            return 0

        programs = [traj.ast for traj in self._replay_buffer]
        patterns = mine_recurrent_patterns(
            programs,
            min_size=self._config.min_pattern_size,
            min_frequency=self._config.min_pattern_frequency,
        )
        if not patterns:
            return 0

        mdl_cfg = MDLConfig(threshold=self._config.sleep_mdl_threshold)
        scores = filter_and_rank_abstractions(
            patterns,
            programs,
            config=mdl_cfg,
            top_k=self._config.max_abstractions_per_cycle,
        )
        if not scores:
            return 0

        pattern_map = {p.name: p for p in patterns}
        discovered = 0
        for score in scores:
            p = pattern_map.get(score.pattern_name)
            if p is None:
                continue
            clean_name = p.name.replace("-", "_").lower()
            if not clean_name.startswith("macro_"):
                clean_name = f"macro_{clean_name}"

            if clean_name not in self._grammar_library.macros:
                try:
                    self._grammar_library.register_macro(p)
                    discovered += 1
                except Exception as exc:
                    logger.warning(
                        "Failed registering macro pattern %s: %s",
                        p.name,
                        exc,
                    )
        return discovered

    async def _run_sleep_phase_dispatch_async(self) -> Optional[str]:
        """Dispatch non-blocking background fine-tuning job via ComputeProvider."""
        if not self._config.remote_dispatch_enabled:
            return None
        if self._active_job_id is not None:
            return self._active_job_id
        if len(self._replay_buffer) < self._config.min_solutions_for_sleep:
            return None

        try:
            domain_enum = ComputeDomain(self._config.domain)
        except ValueError as exc:
            logger.error(
                "Invalid compute domain configured %s: %s",
                self._config.domain,
                exc,
            )
            return None

        payload = TrainingPayload(
            trajectories_count=len(self._replay_buffer),
            domain=domain_enum,
            base_model_name=self._config.base_model_name,
            model_type=self._config.model_type,
            dataset_payload={
                "trajectories": [
                    {
                        "ast": repr(t.ast),
                        "level": (
                            t.level.name
                            if hasattr(t.level, "name")
                            else str(t.level)
                        ),
                        "domain": t.domain,
                    }
                    for t in list(self._replay_buffer)
                ],
                "count": len(self._replay_buffer),
            },
            hyperparameters={
                "lora_rank": self._config.lora_rank,
                "wake_batch_size": self._config.wake_batch_size,
            },
        )

        provider_name = getattr(
            self._compute_provider,
            "__class__",
            type(self._compute_provider),
        ).__name__.lower()
        job_cfg = RemoteJobConfig(
            provider_name=provider_name,
            hardware_accelerator=self._config.hardware_accelerator,
            lora_rank=self._config.lora_rank,
        )

        try:
            if self._cost_watchdog is not None:
                job_id, was_mocked = await asyncio.wait_for(
                    self._cost_watchdog.intercept_or_fallback(
                        provider=self._compute_provider,
                        payload=payload,
                        config=job_cfg,
                        mock_provider=self._fallback_mock_provider,
                    ),
                    timeout=15.0,
                )
                self._active_job_id = job_id
                self._active_job_provider = (
                    self._fallback_mock_provider if was_mocked else self._compute_provider
                )
                return job_id

            job_id = await asyncio.wait_for(
                self._compute_provider.submit_training_job(payload, job_cfg),
                timeout=10.0,
            )
            self._active_job_id = job_id
            self._active_job_provider = self._compute_provider
            return job_id
        except Exception as exc:
            logger.error(
                "Failed dispatching remote training job for domain %s: %s",
                self._config.domain,
                exc,
            )
            return None

    async def step_async(self) -> CycleTelemetry:
        """Execute one full Wake-Sleep cycle asynchronously."""
        start_time = time.perf_counter()
        cycle_idx = len(self._telemetry_history)

        # Pre-wake check: Poll background training and hot-swap if completed
        await asyncio.wait_for(self.check_background_training_async(), timeout=15.0)

        # Wake Phase: Sample curriculum and synthesize programs
        attempted, solved, success_rate = self._run_wake_phase()

        # Sleep Phase 1: Abstraction Discovery (DreamCoder compression)
        abstractions_discovered = self._run_sleep_phase_abstractions()

        # Sleep Phase 2: Asynchronous Remote LoRA Dispatch
        bg_job_id = await asyncio.wait_for(
            self._run_sleep_phase_dispatch_async(),
            timeout=15.0,
        )

        duration = time.perf_counter() - start_time
        telemetry = CycleTelemetry(
            cycle_index=cycle_idx,
            wake_tasks_attempted=attempted,
            wake_tasks_solved=solved,
            wake_success_rate=success_rate,
            abstractions_discovered=abstractions_discovered,
            active_frontier_level=self._zpd_sampler.tracker.current_frontier,
            active_adapter_id=self._active_adapter_id,
            background_job_id=bg_job_id,
            cycle_duration_seconds=duration,
        )
        self._telemetry_history.append(telemetry)
        return telemetry

    def step(self) -> CycleTelemetry:
        """Execute one full Wake-Sleep cycle synchronously."""
        return self._run_sync(self.step_async())

    def run_cycles(self, num_cycles: int) -> list[CycleTelemetry]:
        """Execute N full Wake-Sleep cycles synchronously."""
        if num_cycles < 0:
            raise ValueError(f"num_cycles must be >= 0, got {num_cycles}")
        return [self.step() for _ in range(num_cycles)]

    async def run_cycles_async(self, num_cycles: int) -> list[CycleTelemetry]:
        """Execute N full Wake-Sleep cycles asynchronously."""
        if num_cycles < 0:
            raise ValueError(f"num_cycles must be >= 0, got {num_cycles}")
        results: list[CycleTelemetry] = []
        for _ in range(num_cycles):
            res = await asyncio.wait_for(self.step_async(), timeout=30.0)
            results.append(res)
        return results
