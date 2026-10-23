"""Automated tests for Minimum Description Length (MDL) Compression Evaluator."""

from __future__ import annotations
import pytest

from symforge.abstraction.anti_unification import (
    AbstractionPattern,
    VariableNode,
    anti_unify,
    collect_variables,
    make_variable,
    mine_recurrent_patterns,
)
from symforge.abstraction.mdl import (
    AudioMacroCallNode,
    MacroCallNode,
    MDLConfig,
    MDLScore,
    SDFMacroCallNode,
    compute_ast_size,
    compute_macro_cost,
    evaluate_mdl,
    filter_and_rank_abstractions,
    make_macro_call,
    rewrite_ast_with_pattern,
)
from symforge.core.ast import (
    ASTNode,
    AudioHoleNode,
    BoxNode,
    CylinderNode,
    DifferenceNode,
    FreeParam,
    GainNode,
    HoleNode,
    LowPassNode,
    MixNode,
    NoiseNode,
    RotateNode,
    SawNode,
    ScaleNode,
    SDFHoleNode,
    SineNode,
    SphereNode,
    TranslateNode,
    UnionNode,
    deserialize_node,
    serialize_node,
)


def _p(name: str, val: float) -> FreeParam:
    """Helper to construct FreeParam."""
    return FreeParam(id=name, min_val=0.0, max_val=100.0, default_val=val)


class TestDiscreteSizeCalculation:
    """Tests for discrete AST description length calculation under MDLConfig."""

    def test_leaf_nodes_size(self) -> None:
        """Leaf nodes incur node_cost plus param_cost per free parameter."""
        cfg = MDLConfig(symbol_base_cost=2.0, node_cost=1.0, param_cost=1.0)

        # Noise has 0 params -> 1.0
        noise = NoiseNode()
        assert compute_ast_size(noise, cfg) == 1.0

        # Sphere has 1 param (radius) -> 1.0 + 1.0 = 2.0
        sphere = SphereNode(radius=_p("r_0", 1.0))
        assert compute_ast_size(sphere, cfg) == 2.0

        # Saw has 1 param (freq) -> 1.0 + 1.0 = 2.0
        saw = SawNode(freq=_p("f_0", 220.0))
        assert compute_ast_size(saw, cfg) == 2.0

        # Sine has 2 params (freq, phase) -> 1.0 + 2.0 = 3.0
        sine = SineNode(freq=_p("f_0", 440.0), phase=_p("p_0", 0.0))
        assert compute_ast_size(sine, cfg) == 3.0

        # Box has 3 params (w, h, d) -> 1.0 + 3.0 = 4.0
        box = BoxNode(width=_p("w_0", 1.0), height=_p("h_0", 2.0), depth=_p("d_0", 3.0))
        assert compute_ast_size(box, cfg) == 4.0

        # Cylinder has 2 params (radius, height) -> 1.0 + 2.0 = 3.0
        cyl = CylinderNode(radius=_p("r_0", 1.0), height=_p("h_0", 2.0))
        assert compute_ast_size(cyl, cfg) == 3.0

    def test_transforms_and_operators_size(self) -> None:
        """Unary transforms and binary combinators account for operators, params, and children."""
        cfg = MDLConfig(node_cost=1.0, param_cost=1.0)

        # Translate (node=1, params=3) + Sphere (node=1, param=1) = 6.0
        sphere = SphereNode(radius=_p("r_0", 1.0))
        trans = TranslateNode(dx=_p("dx", 1.0), dy=_p("dy", 0.0), dz=_p("dz", 0.0), child=sphere)
        assert compute_ast_size(trans, cfg) == 6.0

        # Gain (node=1, param=1) + Saw (node=1, param=1) = 4.0
        saw = SawNode(freq=_p("f_0", 220.0))
        gain = GainNode(gain=_p("g_0", 0.8), input_node=saw)
        assert compute_ast_size(gain, cfg) == 4.0

        # Difference (node=1, params=0) + Cyl1 (3) + Cyl2 (3) = 7.0
        c1 = CylinderNode(radius=_p("r_1", 2.0), height=_p("h_1", 5.0))
        c2 = CylinderNode(radius=_p("r_2", 1.5), height=_p("h_2", 5.5))
        diff = DifferenceNode(a=c1, b=c2)
        assert compute_ast_size(diff, cfg) == 7.0

        # Mix (node=1, param=1) + Sine (3) + Saw (2) = 7.0
        sine = SineNode(freq=_p("f_0", 440.0), phase=_p("p_0", 0.0))
        mix = MixNode(ratio=_p("r_0", 0.5), input1=sine, input2=saw)
        assert compute_ast_size(mix, cfg) == 7.0

    def test_variable_and_hole_nodes_size(self) -> None:
        """Holes and variable slots evaluate strictly to param_cost."""
        cfg = MDLConfig(param_cost=1.5)
        var = VariableNode(name="?x")
        hole = HoleNode(hole_id="h0")
        audio_hole = AudioHoleNode(hole_id="ah0")
        sdf_hole = SDFHoleNode(hole_id="sh0")

        assert compute_ast_size(var, cfg) == 1.5
        assert compute_ast_size(hole, cfg) == 1.5
        assert compute_ast_size(audio_hole, cfg) == 1.5
        assert compute_ast_size(sdf_hole, cfg) == 1.5

    def test_macro_call_node_size(self) -> None:
        """MacroCallNode size equals node_cost plus arguments size."""
        cfg = MDLConfig(node_cost=1.0, param_cost=1.0)
        # Macro call with 2 FreeParams: 1 (node) + 2 (params) = 3.0
        call = make_macro_call(
            macro_name="test_macro",
            arguments={"?x": _p("p1", 1.0), "?y": _p("p2", 2.0)},
        )
        assert compute_ast_size(call, cfg) == 3.0

        # Macro call with 1 AST child (Sphere=2.0) and 1 param: 1 + 2.0 + 1.0 = 4.0
        call_nested = make_macro_call(
            macro_name="test_higher_order",
            arguments={"?shape": SphereNode(radius=_p("r", 1.0)), "?param": _p("p", 3.0)},
        )
        assert compute_ast_size(call_nested, cfg) == 4.0

    def test_custom_cost_sensitivity(self) -> None:
        """Calculations scale faithfully with custom node and param costs."""
        cfg = MDLConfig(node_cost=3.0, param_cost=0.5)
        # Box has 1 node (3.0) + 3 params (1.5) = 4.5
        box = BoxNode(width=_p("w", 1.0), height=_p("h", 1.0), depth=_p("d", 1.0))
        assert compute_ast_size(box, cfg) == 4.5


class TestMDLAcceptanceInvariants:
    """Core mathematical invariants of Minimum Description Length selection."""

    def test_trivial_single_use_abstraction_rejected(self) -> None:
        """Single-use abstraction produces delta_mdl >= 0 and is rejected (prevents library bloat)."""
        cfg = MDLConfig(symbol_base_cost=2.0, node_cost=1.0, param_cost=1.0, threshold=0.0)

        # Single program in corpus
        c1 = CylinderNode(radius=_p("r1", 2.0), height=_p("h1", 5.0))
        c2 = CylinderNode(radius=_p("r2", 1.5), height=_p("h2", 5.5))
        prog = DifferenceNode(a=c1, b=c2)
        corpus = [prog]

        # Pattern representing Difference(Cylinder(?r1, ?h1), Cylinder(?r2, ?h2))
        gen_c1 = CylinderNode(radius=VariableNode("?r1"), height=VariableNode("?h1"))
        gen_c2 = CylinderNode(radius=VariableNode("?r2"), height=VariableNode("?h2"))
        gen_ast = DifferenceNode(a=gen_c1, b=gen_c2)
        pattern = AbstractionPattern(
            name="macro_hollow_cyl",
            generalized_ast=gen_ast,
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=1,
            occurrences=[],
            size=gen_ast.size,
        )

        score = evaluate_mdl(pattern, corpus, cfg)

        # Macro cost = symbol_base_cost (2.0) + Difference(Cyl, Cyl) (7.0) = 9.0
        assert score.dsl_expansion_cost == 9.0
        # Savings on 1 occurrence = 7.0 - 5.0 = 2.0
        assert score.corpus_savings == 2.0
        # Delta MDL = 9.0 - 2.0 = +7.0 > 0
        assert score.delta_mdl == 7.0
        assert not score.is_accepted

    def test_trivial_primitive_abstraction_rejected_even_with_high_frequency(self) -> None:
        """An abstraction that wraps a single primitive achieves 0 net savings and is rejected."""
        cfg = MDLConfig(symbol_base_cost=2.0, node_cost=1.0, param_cost=1.0, threshold=0.0)

        # Corpus of 10 spheres
        corpus = [SphereNode(radius=_p(f"r_{i}", float(i + 1))) for i in range(10)]

        # Trivial pattern wrapping Sphere(?r)
        gen_ast = SphereNode(radius=VariableNode("?r"))
        pattern = AbstractionPattern(
            name="macro_trivial_sphere",
            generalized_ast=gen_ast,
            parameters=["?r"],
            frequency=10,
            occurrences=[],
            size=gen_ast.size,
        )

        score = evaluate_mdl(pattern, corpus, cfg)

        # Original sphere size: 2.0. MacroCall size: 2.0. Savings per occurrence: 0.0!
        assert score.corpus_savings == 0.0
        # DSL expansion cost = 2.0 (base) + 2.0 (body) = 4.0
        assert score.dsl_expansion_cost == 4.0
        assert score.delta_mdl == 4.0
        assert not score.is_accepted

    def test_highly_recurrent_pattern_accepted(self) -> None:
        """Highly recurrent pattern achieves positive compression (delta_mdl < 0) and is accepted."""
        cfg = MDLConfig(symbol_base_cost=2.0, node_cost=1.0, param_cost=1.0, threshold=0.0)

        # 6 programs each containing hollow cylinder
        corpus = [
            DifferenceNode(
                a=CylinderNode(radius=_p(f"r1_{i}", 2.0 + i), height=_p(f"h1_{i}", 5.0)),
                b=CylinderNode(radius=_p(f"r2_{i}", 1.5 + i), height=_p(f"h2_{i}", 5.5)),
            )
            for i in range(6)
        ]

        gen_c1 = CylinderNode(radius=VariableNode("?r1"), height=VariableNode("?h1"))
        gen_c2 = CylinderNode(radius=VariableNode("?r2"), height=VariableNode("?h2"))
        gen_ast = DifferenceNode(a=gen_c1, b=gen_c2)
        pattern = AbstractionPattern(
            name="macro_hollow_cyl",
            generalized_ast=gen_ast,
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=6,
            occurrences=[],
            size=gen_ast.size,
        )

        score = evaluate_mdl(pattern, corpus, cfg)

        # Macro cost = 2.0 + 7.0 = 9.0
        # Savings = 6 * 2.0 = 12.0
        # Delta MDL = 9.0 - 12.0 = -3.0 < 0
        assert score.dsl_expansion_cost == 9.0
        assert score.corpus_savings == 12.0
        assert score.delta_mdl == -3.0
        assert score.is_accepted
        assert score.compression_ratio > 1.0


class TestDomainBenchmarks:
    """Domain benchmarks: Audio DSP and SDF 3D CSG compression evaluation."""

    def test_audio_recurring_mix_benchmark(self) -> None:
        """Evaluates recurring Mix(Sine, Saw) across 5 audio programs, proving positive compression."""
        cfg = MDLConfig(symbol_base_cost=2.0, node_cost=1.0, param_cost=1.0)

        # 5 audio programs featuring Mix(Sine, Saw)
        corpus: list[ASTNode] = [
            MixNode(
                ratio=_p(f"r_{i}", 0.5),
                input1=SineNode(freq=_p(f"f_s_{i}", 440.0 + i * 20), phase=_p(f"p_{i}", 0.0)),
                input2=SawNode(freq=_p(f"f_w_{i}", 220.0 + i * 10)),
            )
            for i in range(5)
        ]

        # Extract pattern via anti-unification of first two programs
        lgg, _, _ = anti_unify(corpus[0], corpus[1])
        vars_list = collect_variables(lgg)
        pattern = AbstractionPattern(
            name="macro_audio_harmonic_mix",
            generalized_ast=lgg,
            parameters=[v.name for v in vars_list],
            frequency=5,
            occurrences=[],
            size=lgg.size,
        )

        score = evaluate_mdl(pattern, corpus, cfg)

        # Mix (2) + Sine (3) + Saw (2) = 7.0 original per program
        # Macro definition: 2.0 + 7.0 = 9.0
        # Macro call: 1 (node) + 4 (params) = 5.0
        # Savings per program: 7.0 - 5.0 = 2.0
        # Total savings across 5 programs: 10.0
        # Delta MDL: 9.0 - 10.0 = -1.0 < 0
        assert score.is_accepted
        assert score.delta_mdl < 0.0
        assert score.matched_programs_count == 5
        assert score.total_occurrences == 5
        assert score.compression_ratio > 1.0

    def test_sdf_recurring_hollow_cylinder_benchmark(self) -> None:
        """Evaluates recurring Difference(Cylinder, Cylinder) across 5 SDF programs, proving compression."""
        cfg = MDLConfig(symbol_base_cost=2.0, node_cost=1.0, param_cost=1.0)

        # 5 SDF programs embedded inside Translate nodes
        corpus: list[ASTNode] = [
            TranslateNode(
                dx=_p(f"dx_{i}", float(i)),
                dy=_p(f"dy_{i}", 0.0),
                dz=_p(f"dz_{i}", 0.0),
                child=DifferenceNode(
                    a=CylinderNode(radius=_p(f"r1_{i}", 2.0), height=_p(f"h1_{i}", 5.0)),
                    b=CylinderNode(radius=_p(f"r2_{i}", 1.5), height=_p(f"h2_{i}", 5.5)),
                ),
            )
            for i in range(5)
        ]

        gen_c1 = CylinderNode(radius=VariableNode("?r1"), height=VariableNode("?h1"))
        gen_c2 = CylinderNode(radius=VariableNode("?r2"), height=VariableNode("?h2"))
        gen_ast = DifferenceNode(a=gen_c1, b=gen_c2)
        pattern = AbstractionPattern(
            name="macro_sdf_hollow_pipe",
            generalized_ast=gen_ast,
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=5,
            occurrences=[],
            size=gen_ast.size,
        )

        score = evaluate_mdl(pattern, corpus, cfg)

        # Macro cost = 2.0 + 7.0 = 9.0
        # Savings on 5 occurrences = 5 * 2.0 = 10.0
        # Delta MDL = 9.0 - 10.0 = -1.0 < 0
        assert score.is_accepted
        assert score.delta_mdl == -1.0
        assert score.matched_programs_count == 5
        assert score.total_occurrences == 5
        assert score.compression_ratio > 1.0


class TestPassToCompleteGate:
    """Pass-to-Complete Gate: Evaluates 20 candidates, accepting only the top compressing abstractions."""

    def test_gate_evaluates_20_candidates_filters_and_ranks(self) -> None:
        """Batch evaluates 20 candidates, rejecting all delta_mdl >= 0 and ranking top compressors."""
        cfg = MDLConfig(symbol_base_cost=2.0, node_cost=1.0, param_cost=1.0)

        # Corpus of 10 programs:
        # - 5 contain composite audio Mix(Sine, Saw)
        # - 5 contain composite SDF Difference(Cyl, Cyl)
        corpus: list[ASTNode] = []
        for i in range(5):
            corpus.append(
                MixNode(
                    ratio=_p(f"r_{i}", 0.5),
                    input1=SineNode(freq=_p(f"fs_{i}", 440.0 + i), phase=_p(f"p_{i}", 0.0)),
                    input2=SawNode(freq=_p(f"fw_{i}", 220.0 + i)),
                )
            )
        for i in range(5):
            corpus.append(
                DifferenceNode(
                    a=CylinderNode(radius=_p(f"rc1_{i}", 3.0), height=_p(f"hc1_{i}", 6.0)),
                    b=CylinderNode(radius=_p(f"rc2_{i}", 2.0), height=_p(f"hc2_{i}", 6.0)),
                )
            )

        # Build 20 candidates:
        candidates: list[AbstractionPattern] = []

        # Candidate 0: Recurring Audio Mix (recurs 5 times -> accepted)
        cand_audio = AbstractionPattern(
            name="macro_recurrent_audio",
            generalized_ast=MixNode(
                ratio=VariableNode("?r"),
                input1=SineNode(freq=VariableNode("?fs"), phase=VariableNode("?p")),
                input2=SawNode(freq=VariableNode("?fw")),
            ),
            parameters=["?r", "?fs", "?p", "?fw"],
            frequency=5,
            occurrences=[],
            size=7,
        )
        candidates.append(cand_audio)

        # Candidate 1: Recurring SDF Hollow Cylinder (recurs 5 times -> accepted)
        cand_sdf = AbstractionPattern(
            name="macro_recurrent_sdf",
            generalized_ast=DifferenceNode(
                a=CylinderNode(radius=VariableNode("?r1"), height=VariableNode("?h1")),
                b=CylinderNode(radius=VariableNode("?r2"), height=VariableNode("?h2")),
            ),
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=5,
            occurrences=[],
            size=7,
        )
        candidates.append(cand_sdf)

        # Candidates 2-10: Trivial single-node wrappers (should all be rejected)
        for idx in range(2, 11):
            candidates.append(
                AbstractionPattern(
                    name=f"macro_trivial_single_node_{idx}",
                    generalized_ast=SphereNode(radius=VariableNode("?r")),
                    parameters=["?r"],
                    frequency=1,
                    occurrences=[],
                    size=2,
                )
            )

        # Candidates 11-19: Spurious complex patterns with 0 matches in corpus
        for idx in range(11, 20):
            candidates.append(
                AbstractionPattern(
                    name=f"macro_unmatched_{idx}",
                    generalized_ast=RotateNode(
                        rx=VariableNode("?rx"),
                        ry=VariableNode("?ry"),
                        rz=VariableNode("?rz"),
                        child=BoxNode(
                            width=VariableNode("?w"),
                            height=VariableNode("?h"),
                            depth=VariableNode("?d"),
                        ),
                    ),
                    parameters=["?rx", "?ry", "?rz", "?w", "?h", "?d"],
                    frequency=0,
                    occurrences=[],
                    size=8,
                )
            )

        assert len(candidates) == 20

        # Execute filter and rank
        top_abstractions = filter_and_rank_abstractions(
            candidates=candidates,
            corpus=corpus,
            config=cfg,
            top_k=10,
        )

        # Only the 2 genuinely compressing abstractions are accepted
        assert len(top_abstractions) == 2
        for score in top_abstractions:
            assert score.is_accepted
            assert score.delta_mdl < 0.0

        accepted_names = {s.pattern_name for s in top_abstractions}
        assert "macro_recurrent_audio" in accepted_names
        assert "macro_recurrent_sdf" in accepted_names

        # Monotonic ordering invariant: ranked by compression gain (-delta_mdl) descending
        for i in range(len(top_abstractions) - 1):
            assert top_abstractions[i].delta_mdl <= top_abstractions[i + 1].delta_mdl


class TestSymmetricalCrossValidationAndSerialization:
    """Symmetrical properties and cross-validation of AST rewriting and serialization."""

    def test_symmetrical_size_savings_identity(self) -> None:
        """Identity invariant: compute_ast_size(p) - compute_ast_size(p[alpha]) == size_savings."""
        cfg = MDLConfig()
        c1 = CylinderNode(radius=_p("r1", 2.0), height=_p("h1", 5.0))
        c2 = CylinderNode(radius=_p("r2", 1.5), height=_p("h2", 5.5))
        prog = DifferenceNode(a=c1, b=c2)

        gen_c1 = CylinderNode(radius=VariableNode("?r1"), height=VariableNode("?h1"))
        gen_c2 = CylinderNode(radius=VariableNode("?r2"), height=VariableNode("?h2"))
        pattern = AbstractionPattern(
            name="macro_hollow_cyl",
            generalized_ast=DifferenceNode(a=gen_c1, b=gen_c2),
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=1,
            occurrences=[],
            size=7,
        )

        rewritten, match_count, savings = rewrite_ast_with_pattern(prog, pattern, cfg)

        assert match_count == 1
        orig_size = compute_ast_size(prog, cfg)
        rewritten_size = compute_ast_size(rewritten, cfg)
        assert orig_size - rewritten_size == savings

    def test_macro_call_serialization_roundtrip(self) -> None:
        """Symmetrical round-trip: deserialize_node(serialize_node(call)) == call."""
        macro_sdf = SDFMacroCallNode(
            macro_name="pipe",
            arguments={"?r1": _p("r1", 2.0), "?h1": _p("h1", 5.0)},
            domain="sdf",
        )
        serialized = serialize_node(macro_sdf)
        deserialized = deserialize_node(serialized)

        assert isinstance(deserialized, SDFMacroCallNode)
        assert deserialized.macro_name == "pipe"
        assert deserialized.domain == "sdf"
        assert deserialized.arguments["?r1"] == _p("r1", 2.0)
        assert deserialized.arguments["?h1"] == _p("h1", 5.0)

    def test_domain_typing_invariance(self) -> None:
        """Typed macro calls preserve AudioNode and SDFNode domain hierarchies."""
        audio_call = make_macro_call("fm_pair", {}, domain="audio")
        sdf_call = make_macro_call("pipe", {}, domain="sdf")
        generic_call = make_macro_call("generic_macro", {}, domain="generic")

        from symforge.core.ast import AudioNode, SDFNode
        assert isinstance(audio_call, AudioNode)
        assert isinstance(sdf_call, SDFNode)
        assert not isinstance(generic_call, AudioNode)
        assert not isinstance(generic_call, SDFNode)


class TestRobustnessAndEdgeCases:
    """Robustness against empty inputs, non-matching patterns, and loop/depth caps."""

    def test_empty_corpus(self) -> None:
        """Empty corpus returns zero savings, delta_mdl > 0, and is_accepted == False."""
        cfg = MDLConfig()
        pattern = AbstractionPattern(
            name="macro_any",
            generalized_ast=NoiseNode(),
            parameters=[],
            frequency=0,
            occurrences=[],
            size=1,
        )
        score = evaluate_mdl(pattern, [], cfg)

        assert score.corpus_savings == 0.0
        assert score.delta_mdl > 0.0
        assert not score.is_accepted
        assert score.compression_ratio == 1.0

    def test_empty_candidates_returns_empty(self) -> None:
        """Empty candidates list returns empty list."""
        assert filter_and_rank_abstractions([], []) == []

    def test_recursion_depth_limit_enforced(self) -> None:
        """Deep recursion raises contextual RuntimeError instead of hanging."""
        cfg = MDLConfig(max_depth=5)
        # Construct deeply nested Translate chain
        curr: ASTNode = SphereNode(radius=_p("r", 1.0))
        for i in range(10):
            curr = TranslateNode(dx=_p("dx", 0.0), dy=_p("dy", 0.0), dz=_p("dz", 0.0), child=curr)

        with pytest.raises(RuntimeError) as exc_info:
            compute_ast_size(curr, cfg)
        assert "recursion depth limit exceeded" in str(exc_info.value)
