"""Automated tests for Procedural PCFG Curriculum Complexity Stratification and Triple Quality Filtering."""

import time
import pytest
import torch

from symforge.core.ast import (
    ASTNode,
    AudioNode,
    SineNode,
    SawNode,
    SquareNode,
    GainNode,
    MixNode,
    SDFNode,
    SphereNode,
    BoxNode,
    DifferenceNode,
    UnionNode,
    FreeParam,
)
from symforge.core.curriculum import (
    CurriculumLevel,
    CurriculumSampler,
    FilterResult,
    PhysicalValidityFilter,
    PerceptualEntropyFilter,
    SemanticDeduplicationFilter,
    TripleQualityFilter,
)
from symforge.core.pcfg import PCFGTaskGenerator


def test_curriculum_level_enum_and_depth_invariants():
    """Verify enum members, string/int parsing, descriptions, and depth invariants."""
    assert CurriculumLevel.L1 == 1
    assert CurriculumLevel.L2 == 2
    assert CurriculumLevel.L3 == 3
    assert CurriculumLevel.L4 == 4
    assert CurriculumLevel.L5 == 5

    assert CurriculumLevel.from_value("L1") == CurriculumLevel.L1
    assert CurriculumLevel.from_value("l3") == CurriculumLevel.L3
    assert CurriculumLevel.from_value(4) == CurriculumLevel.L4
    assert CurriculumLevel.from_value(CurriculumLevel.L5) == CurriculumLevel.L5

    with pytest.raises(ValueError, match="Unknown curriculum level specification"):
        CurriculumLevel.from_value("L99")

    for lvl in CurriculumLevel:
        assert lvl.target_depth == lvl.value
        assert len(lvl.description) > 0


def test_audio_curriculum_generation_levels_l1_to_l5():
    """Verify generation of valid audio ASTs matching discrete depths L1 through L5."""
    sampler = CurriculumSampler(domain="audio", seed=101)

    for lvl in CurriculumLevel:
        for _ in range(10):
            ast = sampler.sample_level(lvl)
            assert isinstance(ast, AudioNode)
            if lvl == CurriculumLevel.L5:
                assert ast.tree_depth >= 5
            else:
                assert ast.tree_depth == lvl.target_depth


def test_sdf_curriculum_generation_levels_l1_to_l5():
    """Verify generation of valid SDF ASTs matching discrete depths L1 through L5."""
    sampler = CurriculumSampler(domain="sdf", seed=202)

    for lvl in CurriculumLevel:
        for _ in range(10):
            ast = sampler.sample_level(lvl)
            assert isinstance(ast, SDFNode)
            if lvl == CurriculumLevel.L5:
                assert ast.tree_depth >= 5
            else:
                assert ast.tree_depth == lvl.target_depth


def test_triple_quality_filter_rejects_audio_clipping():
    """Verify Filter 1 rejects signals exceeding digital maximum amplitude 1.0."""
    q_filter = TripleQualityFilter(domain="audio")

    # Construct an audio tree with excessive gain causing clipping > 1.0
    f = FreeParam("f", 50.0, 2000.0, 440.0)
    osc = SineNode(freq=f)
    g = FreeParam("g", 1.0, 10.0, 3.5)
    clipped_node = GainNode(gain=g, input_node=osc)

    res = q_filter.validate(clipped_node)
    assert not res.passed
    assert res.filter_id == "physical_validity"
    assert "Digital clipping" in res.reason


def test_triple_quality_filter_rejects_audio_silence():
    """Verify Filter 1 rejects signals with amplitude below 0.02 threshold."""
    q_filter = TripleQualityFilter(domain="audio")

    # Construct an audio tree scaled down to inaudible silence
    f = FreeParam("f", 50.0, 2000.0, 440.0)
    osc = SineNode(freq=f)
    g = FreeParam("g", 0.0, 0.1, 0.005)
    silent_node = GainNode(gain=g, input_node=osc)

    res = q_filter.validate(silent_node)
    assert not res.passed
    assert res.filter_id == "physical_validity"
    assert "Silence" in res.reason


def test_triple_quality_filter_rejects_audio_flat_dc_entropy():
    """Verify Filter 2 rejects flat DC signals with near-zero spectral entropy."""
    entropy_filter = PerceptualEntropyFilter(domain="audio")

    # Zero-frequency oscillator / static constant DC offset
    f = FreeParam("f", 0.0, 10.0, 0.0)
    phi = FreeParam("phi", 0.0, 6.28, 1.57079)  # sin(pi/2) = 1.0 constant DC
    dc_node = SineNode(freq=f, phase=phi)

    res = entropy_filter.validate(dc_node)
    assert not res.passed
    assert res.filter_id == "perceptual_entropy"
    assert "flat DC" in res.reason or "variance" in res.reason or "Near-zero" in res.reason


def test_triple_quality_filter_rejects_empty_sdf():
    """Verify Filter 1 rejects empty SDF CSG difference (subtracting larger shape)."""
    q_filter = TripleQualityFilter(domain="sdf")

    # Sphere of radius 1.0 minus sphere of radius 2.0 -> empty solid
    r1 = FreeParam("r1", 0.1, 5.0, 1.0)
    r2 = FreeParam("r2", 0.1, 5.0, 2.0)
    diff_empty = DifferenceNode(a=SphereNode(radius=r1), b=SphereNode(radius=r2))

    res = q_filter.validate(diff_empty)
    assert not res.passed
    assert res.filter_id in ("physical_validity", "perceptual_entropy")
    assert "zero-crossing" in res.reason or "empty" in res.reason or "Vanishing" in res.reason


def test_triple_quality_filter_rejects_unbounded_sdf():
    """Verify Filter 2 rejects unbounded shapes that engulf standard bounding box."""
    entropy_filter = PerceptualEntropyFilter(domain="sdf")

    # Giant sphere of radius 20.0 engulfing [-3, 3]^3 bounding box entirely
    r = FreeParam("r", 0.1, 50.0, 20.0)
    giant_sphere = SphereNode(radius=r)

    res = entropy_filter.validate(giant_sphere)
    assert not res.passed
    assert res.filter_id == "perceptual_entropy"
    assert "bounding box" in res.reason


def test_triple_quality_filter_rejects_duplicate_dags():
    """Verify Filter 3 detects and rejects semantically identical DAG signatures."""
    dedup = SemanticDeduplicationFilter()

    # Two identical sphere nodes with different parameter variable IDs
    s1 = SphereNode(radius=FreeParam("r_alpha", 0.1, 5.0, 1.25))
    s2 = SphereNode(radius=FreeParam("r_beta", 0.1, 5.0, 1.25))

    res1 = dedup.validate(s1)
    assert res1.passed
    assert res1.metadata.get("signature") is not None

    res2 = dedup.validate(s2)
    assert not res2.passed
    assert res2.filter_id == "semantic_deduplication"
    assert "Duplicate semantic DAG signature" in res2.reason

    # Resetting filter allows the signature to be accepted again
    dedup.reset()
    res3 = dedup.validate(s2)
    assert res3.passed


def test_triple_quality_filter_rejects_degenerate_difference():
    """Verify Filter 3 rejects binary self-difference nodes: Difference(A, A)."""
    dedup = SemanticDeduplicationFilter()

    s = SphereNode(radius=FreeParam("r", 0.1, 5.0, 1.0))
    degenerate_diff = DifferenceNode(a=s, b=s)

    res = dedup.validate(degenerate_diff)
    assert not res.passed
    assert res.filter_id == "semantic_deduplication"
    assert "Degenerate self-combination" in res.reason


def test_pcfg_curriculum_pass_to_complete_gate_audio():
    """Pass-to-complete gate: Generates 100 non-degenerate audio derivations per level (500 total) within seconds."""
    sampler = CurriculumSampler(domain="audio", seed=42)

    t0 = time.time()
    for lvl in CurriculumLevel:
        batch = sampler.generate_batch(lvl, batch_size=100)
        assert len(batch) == 100
        for ast in batch:
            assert isinstance(ast, AudioNode)
            if lvl == CurriculumLevel.L5:
                assert ast.tree_depth >= 5
            else:
                assert ast.tree_depth == lvl.target_depth
    t1 = time.time()
    elapsed = t1 - t0
    # Must complete within 10 seconds (typically < 1.0s)
    assert elapsed < 10.0, f"Audio curriculum generation took too long: {elapsed:.2f}s"


def test_pcfg_curriculum_pass_to_complete_gate_sdf():
    """Pass-to-complete gate: Generates 100 non-degenerate SDF derivations per level (500 total) within seconds."""
    sampler = CurriculumSampler(domain="sdf", seed=84)

    t0 = time.time()
    for lvl in CurriculumLevel:
        batch = sampler.generate_batch(lvl, batch_size=100)
        assert len(batch) == 100
        for ast in batch:
            assert isinstance(ast, SDFNode)
            if lvl == CurriculumLevel.L5:
                assert ast.tree_depth >= 5
            else:
                assert ast.tree_depth == lvl.target_depth
    t1 = time.time()
    elapsed = t1 - t0
    # Must complete within 10 seconds (typically < 1.5s)
    assert elapsed < 10.0, f"SDF curriculum generation took too long: {elapsed:.2f}s"


def test_pcfg_task_generator_curriculum_integration():
    """Verify PCFGTaskGenerator integrates sample_level smoothly."""
    pcfg_audio = PCFGTaskGenerator(domain="audio", seed=7)
    audio_ast = pcfg_audio.sample_level(CurriculumLevel.L3)
    assert isinstance(audio_ast, AudioNode)
    assert audio_ast.tree_depth == 3

    pcfg_sdf = PCFGTaskGenerator(domain="sdf", seed=9)
    sdf_ast = pcfg_sdf.sample_level(CurriculumLevel.L4)
    assert isinstance(sdf_ast, SDFNode)
    assert sdf_ast.tree_depth == 4
