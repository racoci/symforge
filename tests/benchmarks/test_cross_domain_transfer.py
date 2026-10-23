"""Tests for Cross-Domain Unified Validation and Transfer Benchmark.

Validates that Audio DSP, SDF 3D Geometry, and Hardware Superoptimization share the
exact same synthesis and abstraction discovery engine without domain-specific algorithmic hacks.
"""

import pytest
import torch

from symforge.abstraction.anti_unification import (
    HardwareVariableNode,
    mine_recurrent_patterns,
)
from symforge.abstraction.grammar_expander import DynamicGrammarLibrary
from symforge.abstraction.mdl import HardwareMacroCallNode
from symforge.benchmarks.cross_domain import (
    BenchmarkTask,
    CrossDomainTransferSummary,
    DomainBenchmarkScore,
    format_cross_domain_markdown_report,
    get_all_benchmark_suites,
    get_audio_benchmark_suite,
    get_hardware_benchmark_suite,
    get_sdf_benchmark_suite,
    run_cross_domain_benchmark,
)
from symforge.core.ast import (
    ASTNode,
    AddHardwareNode,
    AndHardwareNode,
    HardwareHoleNode,
    HardwareNode,
    ImmHardwareNode,
    MulHardwareNode,
    OrHardwareNode,
    RegHardwareNode,
    ShlHardwareNode,
    ShrHardwareNode,
    XorHardwareNode,
    deserialize_node,
    serialize_node,
)
from symforge.search.synthesizer import (
    ReferenceASTScorer,
    TypeGuidedBeamSynthesizer,
)


def test_hardware_execution_semantics_and_wrapping():
    """Verify hardware instruction execution semantics, bitwise operations, and 32-bit wrapping."""
    regs = [10, 20, 0xFFFFFFFF]

    # RegHardwareNode
    assert RegHardwareNode(0).evaluate(regs) == 10
    assert RegHardwareNode(1).evaluate(regs) == 20
    assert RegHardwareNode(2).evaluate(regs) == 0xFFFFFFFF
    assert RegHardwareNode(99).evaluate(regs) == 0  # Out of bounds fallback

    # ImmHardwareNode with 32-bit mask
    assert ImmHardwareNode(5).evaluate(regs) == 5
    assert ImmHardwareNode(0x1_FFFF_FFFF).evaluate(regs) == 0xFFFFFFFF

    # 32-bit wrapping addition: 0xFFFFFFFF + 1 = 0
    add_wrap = AddHardwareNode(RegHardwareNode(2), ImmHardwareNode(1))
    assert add_wrap.evaluate(regs) == 0

    # 32-bit wrapping multiplication: 0x10000 * 0x10000 = 0
    mul_wrap = MulHardwareNode(ImmHardwareNode(0x10000), ImmHardwareNode(0x10000))
    assert mul_wrap.evaluate(regs) == 0

    # Bitwise AND, OR, XOR
    bitwise_regs = [0xF0, 0x0F, 0xAA]
    assert AndHardwareNode(RegHardwareNode(0), RegHardwareNode(1)).evaluate(bitwise_regs) == 0x00
    assert OrHardwareNode(RegHardwareNode(0), RegHardwareNode(1)).evaluate(bitwise_regs) == 0xFF
    assert XorHardwareNode(RegHardwareNode(2), ImmHardwareNode(0xFF)).evaluate(bitwise_regs) == 0x55

    # Logical shifts
    assert ShlHardwareNode(ImmHardwareNode(1), 4).evaluate(regs) == 16
    assert ShrHardwareNode(ImmHardwareNode(32), 2).evaluate(regs) == 8
    # Shift count modulo 32
    assert ShlHardwareNode(ImmHardwareNode(1), 32).evaluate(regs) == 1


def test_hardware_symmetrical_serialization():
    """Verify symmetrical identity f^-1(f(x)) = x across all Hardware AST node types."""
    test_nodes = [
        RegHardwareNode(0),
        RegHardwareNode(3),
        ImmHardwareNode(0),
        ImmHardwareNode(42),
        ImmHardwareNode(0xFFFFFFFF),
        AddHardwareNode(RegHardwareNode(0), RegHardwareNode(1)),
        MulHardwareNode(RegHardwareNode(0), ImmHardwareNode(3)),
        AndHardwareNode(RegHardwareNode(0), RegHardwareNode(2)),
        OrHardwareNode(RegHardwareNode(1), ImmHardwareNode(0xFF)),
        XorHardwareNode(RegHardwareNode(0), RegHardwareNode(1)),
        ShlHardwareNode(RegHardwareNode(0), 2),
        ShrHardwareNode(RegHardwareNode(1), 4),
        HardwareHoleNode("h_hw_test"),
        # Nested multiply-accumulate composite
        AddHardwareNode(
            left=MulHardwareNode(RegHardwareNode(0), RegHardwareNode(1)),
            right=RegHardwareNode(2),
        ),
    ]

    for original in test_nodes:
        serialized = serialize_node(original)
        deserialized = deserialize_node(serialized)
        assert deserialized == original, (
            f"Symmetrical serialization mismatch for {type(original).__name__}: "
            f"{original} != {deserialized}"
        )


def test_hardware_synthesis_beam_search():
    """Verify TypeGuidedBeamSynthesizer finds target hardware programs within budget."""
    synthesizer = TypeGuidedBeamSynthesizer(beam_width=15, max_steps=100)

    # Test 1: Register bypass
    t1 = RegHardwareNode(0)
    res1 = synthesizer.synthesize(
        HardwareHoleNode("root"),
        ReferenceASTScorer(t1, domain="hardware"),
    )
    assert res1.success is True
    assert res1.best_score <= 1e-3
    assert isinstance(res1.best_program, RegHardwareNode)
    assert res1.best_program.reg_idx == 0

    # Test 2: Shift Left by 1
    t2 = ShlHardwareNode(RegHardwareNode(0), 1)
    res2 = synthesizer.synthesize(
        HardwareHoleNode("root"),
        ReferenceASTScorer(t2, domain="hardware"),
    )
    assert res2.success is True
    assert res2.best_score <= 1e-3
    assert isinstance(res2.best_program, ShlHardwareNode)
    assert res2.total_expansions <= 50


def test_recurrent_abstraction_discovery_hardware_mac():
    """Verify recurrent pattern discovery discovers Multiply-Accumulate (MAC) macro from solved programs."""
    prog1 = AddHardwareNode(
        left=MulHardwareNode(left=RegHardwareNode(0), right=RegHardwareNode(1)),
        right=RegHardwareNode(2),
    )
    prog2 = AddHardwareNode(
        left=MulHardwareNode(left=RegHardwareNode(0), right=RegHardwareNode(1)),
        right=ImmHardwareNode(1),
    )
    prog3 = AddHardwareNode(
        left=MulHardwareNode(left=RegHardwareNode(2), right=RegHardwareNode(0)),
        right=RegHardwareNode(1),
    )

    corpus = [prog1, prog2, prog3]
    patterns = mine_recurrent_patterns(corpus, min_frequency=2, min_size=2)

    assert len(patterns) > 0
    # Top abstraction must generalize Multiply-Accumulate
    top_pattern = patterns[0]
    assert isinstance(top_pattern.generalized_ast, AddHardwareNode)
    assert isinstance(top_pattern.generalized_ast.left, MulHardwareNode)
    assert top_pattern.frequency >= 2

    # Register in dynamic library and verify macro expansion
    lib = DynamicGrammarLibrary()
    macro_def = lib.register_macro(top_pattern)
    assert macro_def.domain == "hardware"

    expansions = lib.get_macro_expansions("hardware", lambda p: f"hole_{p}")
    assert len(expansions) > 0
    assert any(isinstance(e, HardwareMacroCallNode) for e in expansions)


def test_cross_domain_suites_structure():
    """Verify all 3 domain benchmark suites contain exactly 5 well-formed tasks."""
    suites = get_all_benchmark_suites()
    assert set(suites.keys()) == {"audio", "sdf", "hardware"}

    for domain_name, task_list in suites.items():
        assert len(task_list) == 5, f"Domain {domain_name} has {len(task_list)} tasks instead of 5"
        for t in task_list:
            assert isinstance(t, BenchmarkTask)
            assert t.domain == domain_name
            assert isinstance(t.target_ast, ASTNode)
            assert isinstance(t.initial_hole, ASTNode)


def test_cross_domain_benchmark_pass_to_complete_gate():
    """Pass-to-Complete Gate: Runs unified cross-domain benchmark asserting >= 80% completion in all domains."""
    summary = run_cross_domain_benchmark(budget_steps=100, seed=42)

    # Invariant 1: Quality gate passed
    assert summary.all_domains_passed_gate is True

    # Invariant 2: Audio domain >= 80% (>= 4/5)
    audio_score = summary.get_score("audio")
    assert audio_score is not None
    assert audio_score.success_rate >= 0.8
    assert audio_score.solved_tasks >= 4
    assert audio_score.total_tasks == 5

    # Invariant 3: SDF domain >= 80% (>= 4/5)
    sdf_score = summary.get_score("sdf")
    assert sdf_score is not None
    assert sdf_score.success_rate >= 0.8
    assert sdf_score.solved_tasks >= 4
    assert sdf_score.total_tasks == 5

    # Invariant 4: Hardware domain >= 80% (>= 4/5)
    hw_score = summary.get_score("hardware")
    assert hw_score is not None
    assert hw_score.success_rate >= 0.8
    assert hw_score.solved_tasks >= 4
    assert hw_score.total_tasks == 5

    # Invariant 5: Total abstractions mined across solved suites
    assert summary.total_abstractions_mined >= 1

    # Invariant 6: Report formatting produces rich compilable markdown
    report = format_cross_domain_markdown_report(summary)
    assert "# Cross-Domain Unified Validation & Transfer Benchmark Report" in report
    assert "PASSED (>= 80% Success Rate across all domains)" in report
    assert "```mermaid" in report
    assert "```text" in report
    assert "AUDIO" in report
    assert "SDF" in report
    assert "HARDWARE" in report
