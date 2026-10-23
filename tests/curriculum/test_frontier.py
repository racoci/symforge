"""Tests for Competency Frontier Tracker and ZPD Sampler."""

import random
import pytest

from symforge.core.ast import AudioNode, SDFNode
from symforge.core.curriculum import CurriculumLevel
from symforge.curriculum.frontier import (
    CompetencyFrontierTracker,
    FrontierState,
    ZPDConfig,
    ZPDSampler,
)


def test_zpd_config_validation() -> None:
    """Validate parameter boundaries and constraints in ZPDConfig."""
    cfg = ZPDConfig()
    assert cfg.zpd_lower_bound == 0.2
    assert cfg.zpd_upper_bound == 0.8
    assert cfg.window_size == 20
    assert cfg.replay_prob == 0.1
    assert cfg.exploration_prob == 0.05

    # Invalid bounds
    with pytest.raises(ValueError):
        ZPDConfig(zpd_lower_bound=0.9, zpd_upper_bound=0.8)

    with pytest.raises(ValueError):
        ZPDConfig(zpd_lower_bound=-0.1)

    # Probabilities sum exceeding 1.0
    with pytest.raises(ValueError):
        ZPDConfig(replay_prob=0.6, exploration_prob=0.5)

    with pytest.raises(ValueError):
        ZPDConfig(window_size=0)

    with pytest.raises(ValueError):
        ZPDConfig(min_trials_to_advance=0)


def test_sliding_window_and_moving_average() -> None:
    """Verify sliding window calculation and moving average accuracy."""
    cfg = ZPDConfig(window_size=5, min_trials_to_advance=3)
    tracker = CompetencyFrontierTracker(config=cfg)

    # Cold start: 0 trials
    assert tracker.get_success_rate(CurriculumLevel.L1) == 0.0

    # 3 successes, 2 failures
    for s in [True, True, True, False, False]:
        tracker.record_trial(CurriculumLevel.L1, s)

    assert tracker.get_success_rate(CurriculumLevel.L1) == pytest.approx(0.6)
    assert tracker.total_trials == 5

    # Push 5 consecutive successes to evict old trials
    for _ in range(5):
        tracker.record_trial(CurriculumLevel.L1, True)

    assert tracker.get_success_rate(CurriculumLevel.L1) == pytest.approx(1.0)
    assert tracker.total_trials == 10


def test_zpd_boundary_identification() -> None:
    """Verify that levels with S(c) in [0.2, 0.8] are identified and receive primary weight."""
    cfg = ZPDConfig(window_size=20, min_trials_to_advance=5)
    tracker = CompetencyFrontierTracker(config=cfg)

    # Record 10 successes and 10 failures on L1 -> S(L1) = 0.50 (in ZPD)
    for _ in range(10):
        tracker.record_trial(CurriculumLevel.L1, True)
        tracker.record_trial(CurriculumLevel.L1, False)

    state = tracker.get_state()
    assert CurriculumLevel.L1 in state.active_zpd_levels
    assert state.success_rates[CurriculumLevel.L1] == pytest.approx(0.50)

    dist = tracker.get_sampling_distribution()
    # L1 should receive dominant sampling probability (active ZPD mass)
    assert dist[CurriculumLevel.L1] > 0.80
    assert abs(sum(dist.values()) - 1.0) < 1e-5


def test_monotonic_frontier_progression() -> None:
    """Pass-to-Complete Gate: Monotonic frontier progression on simulated student."""
    cfg = ZPDConfig(window_size=20, min_trials_to_advance=5)
    tracker = CompetencyFrontierTracker(config=cfg)

    assert tracker.current_frontier == CurriculumLevel.L1

    # Student achieves 95% at L1 (19 successes out of 20)
    for _ in range(19):
        tracker.record_trial(CurriculumLevel.L1, True)
    tracker.record_trial(CurriculumLevel.L1, False)

    assert tracker.get_success_rate(CurriculumLevel.L1) == pytest.approx(0.95)
    assert CurriculumLevel.L1 in tracker.mastered_levels
    # Tracker advances frontier monotonically to L2
    assert tracker.current_frontier == CurriculumLevel.L2

    # Student achieves 85% at L2 (17 successes out of 20)
    for _ in range(17):
        tracker.record_trial(CurriculumLevel.L2, True)
    for _ in range(3):
        tracker.record_trial(CurriculumLevel.L2, False)

    assert tracker.get_success_rate(CurriculumLevel.L2) == pytest.approx(0.85)
    assert CurriculumLevel.L2 in tracker.mastered_levels
    # Tracker advances frontier monotonically to L3
    assert tracker.current_frontier == CurriculumLevel.L3

    # Student achieves 90% at L3
    for _ in range(18):
        tracker.record_trial(CurriculumLevel.L3, True)
    for _ in range(2):
        tracker.record_trial(CurriculumLevel.L3, False)

    assert tracker.current_frontier == CurriculumLevel.L4


def test_regression_recovery() -> None:
    """Verify catastrophic forgetting detection and automatic recovery re-allocation."""
    cfg = ZPDConfig(window_size=20, min_trials_to_advance=5, regression_weight=0.80)
    tracker = CompetencyFrontierTracker(config=cfg)

    # 1. Master L1
    for _ in range(20):
        tracker.record_trial(CurriculumLevel.L1, True)

    assert tracker.current_frontier == CurriculumLevel.L2
    assert CurriculumLevel.L1 in tracker.mastered_levels

    # 2. Simulate regression on L1: S(L1) drops to 0.50 (10 failures)
    for _ in range(10):
        tracker.record_trial(CurriculumLevel.L1, False)

    state = tracker.get_state()
    assert CurriculumLevel.L1 in state.regressed_levels
    assert CurriculumLevel.L1 not in state.mastered_levels
    assert state.success_rates[CurriculumLevel.L1] == pytest.approx(0.50)

    # 3. Verify sampling distribution shifts weight back to L1
    dist = tracker.get_sampling_distribution()
    assert dist[CurriculumLevel.L1] > dist[CurriculumLevel.L2]
    assert dist[CurriculumLevel.L1] >= 0.65
    assert abs(sum(dist.values()) - 1.0) < 1e-5

    # 4. Student practices L1 again, restoring mastery (push 20 successes)
    for _ in range(20):
        tracker.record_trial(CurriculumLevel.L1, True)

    state_restored = tracker.get_state()
    assert CurriculumLevel.L1 not in state_restored.regressed_levels
    assert CurriculumLevel.L1 in state_restored.mastered_levels
    assert tracker.get_success_rate(CurriculumLevel.L1) == pytest.approx(1.0)

    # Sampling weights shift back to active frontier (L2)
    dist_restored = tracker.get_sampling_distribution()
    assert dist_restored[CurriculumLevel.L2] > dist_restored[CurriculumLevel.L1]
    assert dist_restored[CurriculumLevel.L1] == pytest.approx(cfg.replay_prob)


def test_suppression_of_out_of_reach_levels() -> None:
    """Verify that levels with S(c) < 0.2 receive near-zero probability (<= exploration_prob)."""
    cfg = ZPDConfig(window_size=20, exploration_prob=0.05)
    tracker = CompetencyFrontierTracker(config=cfg)

    # Frontier is at L1. L5 has 0 trials (S(L5) = 0.0 < 0.2)
    dist = tracker.get_sampling_distribution()
    assert dist[CurriculumLevel.L5] <= cfg.exploration_prob
    assert dist[CurriculumLevel.L5] > 0.0

    # Record failures on L5
    for _ in range(10):
        tracker.record_trial(CurriculumLevel.L5, False)

    assert tracker.get_success_rate(CurriculumLevel.L5) < 0.20
    dist_after = tracker.get_sampling_distribution()
    assert dist_after[CurriculumLevel.L5] <= cfg.exploration_prob


def test_sampling_distribution_normalization_invariant() -> None:
    """Verify mathematical invariant sum(P(c)) == 1.0 +- 1e-5 across all states."""
    tracker = CompetencyFrontierTracker(seed=42)

    # Check cold start
    dist0 = tracker.get_sampling_distribution()
    assert abs(sum(dist0.values()) - 1.0) < 1e-5

    # Check across various simulated trials
    rng = random.Random(123)
    for _ in range(50):
        lvl = rng.choice(list(CurriculumLevel))
        outcome = rng.random() > 0.3
        tracker.record_trial(lvl, outcome)
        dist = tracker.get_sampling_distribution()
        assert abs(sum(dist.values()) - 1.0) < 1e-5


def test_sample_next_level() -> None:
    """Verify level sampling according to ZPD distribution."""
    tracker = CompetencyFrontierTracker(seed=99)
    rng = random.Random(99)

    counts = {lvl: 0 for lvl in CurriculumLevel}
    for _ in range(200):
        lvl = tracker.sample_next_level(rng=rng)
        counts[lvl] += 1

    # At cold start, L1 has ~95% of weight, so it should have vast majority of samples
    assert counts[CurriculumLevel.L1] > 160
    assert sum(counts.values()) == 200


def test_zpd_sampler_audio_and_sdf_integration() -> None:
    """Integration test: ZPDSampler samples valid AST tasks with triple quality filtering."""
    tracker = CompetencyFrontierTracker(seed=42)
    sampler = ZPDSampler(tracker=tracker, seed=42)

    # 1. Sample Audio task
    audio_node, audio_lvl = sampler.sample_task(domain="audio")
    assert isinstance(audio_node, AudioNode)
    assert isinstance(audio_lvl, CurriculumLevel)
    # At cold start, sampled level is L1
    assert audio_lvl == CurriculumLevel.L1
    assert audio_node.tree_depth == 1

    # 2. Sample SDF task
    sdf_node, sdf_lvl = sampler.sample_task(domain="sdf")
    assert isinstance(sdf_node, SDFNode)
    assert isinstance(sdf_lvl, CurriculumLevel)
    assert sdf_lvl == CurriculumLevel.L1
    assert sdf_node.tree_depth == 1

    # 3. Advance to L2 and sample
    for _ in range(10):
        sampler.record_result(CurriculumLevel.L1, True)

    assert sampler.tracker.current_frontier == CurriculumLevel.L2
    # Force next sample to be L2 by mocking or direct testing
    sample_lvl = sampler.tracker.sample_next_level()
    assert sample_lvl in (CurriculumLevel.L1, CurriculumLevel.L2)

    # 4. Invalid domain raises ValueError
    with pytest.raises(ValueError, match="Unsupported curriculum domain"):
        sampler.sample_task(domain="robotics")
