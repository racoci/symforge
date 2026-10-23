"""Competency Frontier Tracker and Zone of Proximal Development (ZPD) Sampler.

Implements adaptive curriculum sequencing based on Vygotsky's Zone of Proximal
Development (ZPD) and self-paced learning. Empirically tracks learner success
rates across complexity levels, manages monotonic competency frontier progression,
detects catastrophic forgetting / regression, and biases task sampling.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import logging
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

from symforge.core.ast import ASTNode, AudioNode, SDFNode
from symforge.core.curriculum import CurriculumLevel, CurriculumSampler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZPDConfig:
    """Configuration parameters for Competency Frontier and ZPD sampling.

    Attributes:
        zpd_lower_bound: Success rate lower threshold for ZPD inclusion (0.2).
        zpd_upper_bound: Success rate upper threshold for mastery (0.8).
        window_size: Size of sliding window deque of trials per level (20).
        replay_prob: Cumulative probability allocated for mastered replay (0.1).
        exploration_prob: Cumulative probability for speculative exploration (0.05).
        min_trials_to_advance: Minimum trials required before evaluating mastery (5).
        regression_weight: Proportion of active mass shifted to regressed levels (0.8).
    """

    zpd_lower_bound: float = 0.2
    zpd_upper_bound: float = 0.8
    window_size: int = 20
    replay_prob: float = 0.1
    exploration_prob: float = 0.05
    min_trials_to_advance: int = 5
    regression_weight: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not (0.0 <= self.zpd_lower_bound < self.zpd_upper_bound <= 1.0):
            raise ValueError("Must have 0.0 <= zpd_lower_bound < zpd_upper_bound <= 1.0")
        if not (0.0 <= self.replay_prob <= 1.0):
            raise ValueError("replay_prob must be in [0.0, 1.0]")
        if not (0.0 <= self.exploration_prob <= 1.0):
            raise ValueError("exploration_prob must be in [0.0, 1.0]")
        if self.replay_prob + self.exploration_prob >= 1.0:
            raise ValueError("replay_prob + exploration_prob must be strictly less than 1.0")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.min_trials_to_advance < 1:
            raise ValueError("min_trials_to_advance must be positive")
        if not (0.0 <= self.regression_weight <= 1.0):
            raise ValueError("regression_weight must be in [0.0, 1.0]")


@dataclass
class FrontierState:
    """Snapshot of learner competency frontier and ZPD bands.

    Attributes:
        current_frontier: Active highest competency frontier reached.
        mastered_levels: List of curriculum levels currently considered mastered.
        active_zpd_levels: Levels falling within empirical ZPD band [0.2, 0.8].
        success_rates: Mapping from curriculum level to empirical success rate.
        regressed_levels: Previously mastered levels whose success rate dropped < 0.8.
        total_trials: Cumulative number of trials recorded across all levels.
    """

    current_frontier: CurriculumLevel
    mastered_levels: List[CurriculumLevel]
    active_zpd_levels: List[CurriculumLevel]
    success_rates: Dict[CurriculumLevel, float]
    regressed_levels: List[CurriculumLevel]
    total_trials: int


class CompetencyFrontierTracker:
    """Tracks empirical success rates and calculates ZPD sampling distributions."""

    def __init__(
        self,
        config: Optional[ZPDConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the competency frontier tracker.

        Args:
            config: Optional ZPD configuration parameters.
            seed: Optional random seed for deterministic level sampling.
        """
        self.config = config or ZPDConfig()
        self._rng = random.Random(seed)
        self._current_frontier: CurriculumLevel = CurriculumLevel.L1
        self._total_trials: int = 0
        self._history_mastered: set[CurriculumLevel] = set()
        self._regressed_levels: set[CurriculumLevel] = set()
        self._windows: Dict[CurriculumLevel, collections.deque[bool]] = {
            lvl: collections.deque(maxlen=self.config.window_size)
            for lvl in CurriculumLevel
        }

    @property
    def current_frontier(self) -> CurriculumLevel:
        """Return the current active competency frontier level."""
        return self._current_frontier

    @property
    def total_trials(self) -> int:
        """Return the total number of recorded trials."""
        return self._total_trials

    @property
    def mastered_levels(self) -> List[CurriculumLevel]:
        """Return levels currently mastered and not regressed."""
        return sorted(
            [
                lvl
                for lvl in self._history_mastered
                if lvl not in self._regressed_levels
            ],
            key=lambda x: x.value,
        )

    @property
    def regressed_levels(self) -> List[CurriculumLevel]:
        """Return previously mastered levels currently suffering regression."""
        return sorted(list(self._regressed_levels), key=lambda x: x.value)

    @property
    def active_zpd_levels(self) -> List[CurriculumLevel]:
        """Return active unmastered levels situated in the ZPD interval."""
        return sorted(
            [
                lvl
                for lvl in CurriculumLevel
                if self.config.zpd_lower_bound
                <= self.get_success_rate(lvl)
                <= self.config.zpd_upper_bound
                and lvl not in self._regressed_levels
                and lvl not in self.mastered_levels
            ],
            key=lambda x: x.value,
        )

    def get_success_rate(self, level: Union[CurriculumLevel, int, str]) -> float:
        """Return the current empirical moving-average success rate for a level.

        Args:
            level: Target curriculum complexity level.

        Returns:
            Empirical success rate float in [0.0, 1.0].
        """
        lvl = CurriculumLevel.from_value(level)
        window = self._windows[lvl]
        if not window:
            return 0.0
        return float(sum(window) / len(window))

    def record_trial(
        self, level: Union[CurriculumLevel, int, str], success: bool
    ) -> None:
        """Record trial outcome, update sliding window, and check transitions.

        Args:
            level: Target curriculum complexity level.
            success: Whether the learner successfully synthesized the task.
        """
        lvl = CurriculumLevel.from_value(level)
        self._windows[lvl].append(bool(success))
        self._total_trials += 1

        self._update_mastery_and_regression(lvl)
        self._advance_frontier()

    def _update_mastery_and_regression(self, lvl: CurriculumLevel) -> None:
        """Evaluate mastery and regression state transitions for a level.

        Args:
            lvl: Curriculum level being updated.
        """
        s_rate = self.get_success_rate(lvl)
        trials_count = len(self._windows[lvl])

        if lvl in self._history_mastered:
            if s_rate < self.config.zpd_upper_bound:
                self._regressed_levels.add(lvl)
            else:
                self._regressed_levels.discard(lvl)
            return

        if trials_count < self.config.min_trials_to_advance:
            return

        if s_rate > self.config.zpd_upper_bound:
            self._history_mastered.add(lvl)

    def _advance_frontier(self) -> None:
        """Advance the competency frontier monotonically when frontier is mastered."""
        while (
            self._current_frontier in self.mastered_levels
            and self._current_frontier < CurriculumLevel.L5
        ):
            next_val = self._current_frontier.value + 1
            self._current_frontier = CurriculumLevel(next_val)

    def _compute_replay_weights(
        self, mastered: List[CurriculumLevel]
    ) -> Tuple[Dict[CurriculumLevel, float], float]:
        """Compute replay probability weights for mastered levels."""
        if not mastered:
            return {}, 0.0
        p_each = self.config.replay_prob / len(mastered)
        return {lvl: p_each for lvl in mastered}, self.config.replay_prob

    def _compute_exploration_weights(
        self, exploration: List[CurriculumLevel]
    ) -> Tuple[Dict[CurriculumLevel, float], float]:
        """Compute speculative exploration probability weights."""
        if not exploration:
            return {}, 0.0
        p_each = self.config.exploration_prob / len(exploration)
        return {lvl: p_each for lvl in exploration}, self.config.exploration_prob

    def _compute_active_weights(
        self,
        regressed: List[CurriculumLevel],
        active_targets: List[CurriculumLevel],
        pool: float,
    ) -> Dict[CurriculumLevel, float]:
        """Compute probability weights for regressed and active ZPD levels."""
        if pool <= 0.0:
            return {}
        if regressed and active_targets:
            p_reg = (pool * self.config.regression_weight) / len(regressed)
            p_act = (pool * (1.0 - self.config.regression_weight)) / len(active_targets)
            res = {lvl: p_reg for lvl in regressed}
            res.update({lvl: p_act for lvl in active_targets})
            return res
        if regressed:
            p_reg = pool / len(regressed)
            return {lvl: p_reg for lvl in regressed}
        if active_targets:
            p_act = pool / len(active_targets)
            return {lvl: p_act for lvl in active_targets}
        return {}

    def get_sampling_distribution(self) -> Dict[CurriculumLevel, float]:
        """Compute normalized sampling probability distribution over all levels.

        Returns:
            Dictionary mapping each CurriculumLevel to its sampling probability.
            The sum of all probabilities is guaranteed to equal 1.0 within 1e-5.
        """
        mastered = self.mastered_levels
        regressed = self.regressed_levels
        active_zpd = self.active_zpd_levels

        exploration = [
            lvl
            for lvl in CurriculumLevel
            if lvl > self._current_frontier
            and lvl not in mastered
            and lvl not in regressed
            and lvl not in active_zpd
        ]

        active_targets = list(active_zpd)
        if (
            not active_targets
            and self._current_frontier not in mastered
            and self._current_frontier not in regressed
        ):
            active_targets = [self._current_frontier]

        weights: Dict[CurriculumLevel, float] = {lvl: 0.0 for lvl in CurriculumLevel}
        replay_w, replay_used = self._compute_replay_weights(mastered)
        expl_w, expl_used = self._compute_exploration_weights(exploration)

        pool = max(0.0, 1.0 - replay_used - expl_used)
        act_w = self._compute_active_weights(regressed, active_targets, pool)

        for d in (replay_w, expl_w, act_w):
            for lvl, val in d.items():
                weights[lvl] += val

        total = sum(weights.values())
        if total <= 0.0:
            return {lvl: 1.0 / len(CurriculumLevel) for lvl in CurriculumLevel}
        return {lvl: val / total for lvl, val in weights.items()}

    def sample_next_level(
        self, rng: Optional[random.Random] = None
    ) -> CurriculumLevel:
        """Sample next curriculum level according to active ZPD distribution.

        Args:
            rng: Optional random generator instance.

        Returns:
            CurriculumLevel selected according to ZPD probabilities.
        """
        r = rng if rng is not None else self._rng
        dist = self.get_sampling_distribution()
        levels = list(dist.keys())
        weights = list(dist.values())
        return r.choices(levels, weights=weights, k=1)[0]

    def get_state(self) -> FrontierState:
        """Return current snapshot of frontier state.

        Returns:
            FrontierState containing frontier, mastered, ZPD, and regression data.
        """
        return FrontierState(
            current_frontier=self._current_frontier,
            mastered_levels=self.mastered_levels,
            active_zpd_levels=self.active_zpd_levels,
            success_rates={lvl: self.get_success_rate(lvl) for lvl in CurriculumLevel},
            regressed_levels=self.regressed_levels,
            total_trials=self._total_trials,
        )


class ZPDSampler:
    """Zone of Proximal Development (ZPD) procedural AST task sampler."""

    def __init__(
        self,
        tracker: Optional[CompetencyFrontierTracker] = None,
        audio_sampler: Optional[CurriculumSampler] = None,
        sdf_sampler: Optional[CurriculumSampler] = None,
        default_domain: str = "audio",
        seed: Optional[int] = None,
    ) -> None:
        """Initialize ZPD Sampler across domains.

        Args:
            tracker: Optional tracker instance (default: newly created).
            audio_sampler: Optional audio domain CurriculumSampler.
            sdf_sampler: Optional SDF domain CurriculumSampler.
            default_domain: Fallback domain ('audio' or 'sdf').
            seed: Optional random generator seed.
        """
        self._tracker = tracker or CompetencyFrontierTracker(seed=seed)
        self._audio_sampler = audio_sampler or CurriculumSampler(
            domain="audio", seed=seed
        )
        self._sdf_sampler = sdf_sampler or CurriculumSampler(
            domain="sdf", seed=seed
        )
        self._default_domain = default_domain.lower().strip()
        self._rng = random.Random(seed)

    @property
    def tracker(self) -> CompetencyFrontierTracker:
        """Return associated CompetencyFrontierTracker instance."""
        return self._tracker

    @property
    def state(self) -> FrontierState:
        """Return current frontier state snapshot."""
        return self._tracker.get_state()

    def record_result(
        self, level: Union[CurriculumLevel, int, str], success: bool
    ) -> None:
        """Record trial result in the underlying tracker.

        Args:
            level: Curriculum level synthesized.
            success: Whether synthesis passed quality gate.
        """
        self._tracker.record_trial(level, success)

    def sample_task(
        self,
        domain: Optional[str] = None,
        rng: Optional[random.Random] = None,
        max_attempts: int = 50,
    ) -> Tuple[ASTNode, CurriculumLevel]:
        """Sample a validated procedural AST task from the active ZPD band.

        Args:
            domain: Domain identifier ('audio' or 'sdf').
            rng: Optional custom random generator.
            max_attempts: Maximum generation attempts for triple quality filtering.

        Returns:
            Tuple of (validated ASTNode, sampled CurriculumLevel).

        Raises:
            ValueError: If domain is not 'audio' or 'sdf'.
            RuntimeError: If triple filtering fails within max_attempts.
        """
        dom = (domain or self._default_domain).lower().strip()
        if dom not in ("audio", "sdf"):
            raise ValueError(
                f"Unsupported curriculum domain: '{dom}'. Expected 'audio' or 'sdf'."
            )

        level = self._tracker.sample_next_level(rng=rng)
        sampler = self._audio_sampler if dom == "audio" else self._sdf_sampler
        ast_node = sampler.sample_level(level, max_attempts=max_attempts)
        return ast_node, level
