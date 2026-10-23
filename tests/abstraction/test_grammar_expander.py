"""Automated tests for Dynamic Lean 4 Grammar Expander & Library Injection Engine."""

from __future__ import annotations
import logging
import math
import pytest
import torch

logger = logging.getLogger(__name__)

from symforge.abstraction.anti_unification import (
    AbstractionPattern,
    VariableNode,
)
from symforge.abstraction.grammar_expander import (
    DynamicGrammarLibrary,
    MacroDefinition,
    MacroParam,
    expand_all_macros_in_ast,
    expand_macro_call,
    generate_lean_macro_def,
    get_active_library,
)
from symforge.abstraction.mdl import (
    AudioMacroCallNode,
    SDFMacroCallNode,
    make_macro_call,
)
from symforge.core.ast import (
    AudioHoleNode,
    BoxNode,
    CylinderNode,
    DifferenceNode,
    FreeParam,
    GainNode,
    MixNode,
    SawNode,
    SDFHoleNode,
    SineNode,
    SphereNode,
    TranslateNode,
    UnionNode,
    has_holes,
)
from symforge.core.transpiler import (
    DifferentiableAudioModule,
    DifferentiableSDFModule,
    transpile_ast_to_module,
)
from symforge.search.synthesizer import (
    AStarSynthesizer,
    ReferenceASTScorer,
    SearchConfig,
    TypeGuidedBeamSynthesizer,
    TypedSynthesizer,
)


def _p(name: str, val: float, min_v: float = 0.0, max_v: float = 10.0) -> FreeParam:
    """Helper creating a test FreeParam."""
    return FreeParam(id=name, min_val=min_v, max_val=max_v, default_val=val)


class TestMacroDefinitionCreation:
    """Tests for inverting AbstractionPatterns into first-class MacroDefinitions."""

    def test_sdf_macro_definition_from_pattern(self) -> None:
        """Pattern with typed child slots maps to MacroParams and valid signatures."""
        pat = AbstractionPattern(
            name="macro_hollow_cylinder",
            generalized_ast=DifferenceNode(
                a=VariableNode("?c1", "SDFNode", "sdf"),
                b=VariableNode("?c2", "SDFNode", "sdf"),
            ),
            parameters=["?c1: SDFNode", "?c2: SDFNode"],
            frequency=2,
            occurrences=[],
            size=3,
        )

        lib = DynamicGrammarLibrary()
        mdef = lib.register_macro(pat)

        assert mdef.name == "macro_hollow_cylinder"
        assert mdef.domain == "sdf"
        assert mdef.return_type == "SDFNode"
        assert len(mdef.parameters) == 2

        p1, p2 = mdef.parameters
        assert p1.name == "c1"
        assert p1.expected_type == "SDFNode"
        assert p1.is_free_param is False
        assert p2.name == "c2"
        assert p2.expected_type == "SDFNode"

    def test_audio_macro_definition_from_pattern(self) -> None:
        """Pattern with continuous FreeParam slots maps to continuous MacroParams."""
        pat = AbstractionPattern(
            name="macro_dual_osc",
            generalized_ast=MixNode(
                ratio=VariableNode("?r", "FreeParam", "param"),
                input1=VariableNode("?s1", "AudioNode", "audio"),
                input2=VariableNode("?s2", "AudioNode", "audio"),
            ),
            parameters=["?r: FreeParam", "?s1: AudioNode", "?s2: AudioNode"],
            frequency=3,
            occurrences=[],
            size=4,
        )

        lib = DynamicGrammarLibrary()
        mdef = lib.register_macro(pat)

        assert mdef.name == "macro_dual_osc"
        assert mdef.domain == "audio"
        assert mdef.return_type == "AudioNode"
        assert len(mdef.parameters) == 3

        p_r = mdef.parameters[0]
        assert p_r.name == "r"
        assert p_r.is_free_param is True

    def test_lean4_code_generation_subprograms(self) -> None:
        """Lean 4 inductive declaration code generation for subprogram slots."""
        pat = AbstractionPattern(
            name="macro_hollow_cylinder",
            generalized_ast=DifferenceNode(
                a=VariableNode("?c1", "SDFNode", "sdf"),
                b=VariableNode("?c2", "SDFNode", "sdf"),
            ),
            parameters=["?c1: SDFNode", "?c2: SDFNode"],
            frequency=1,
            occurrences=[],
            size=3,
        )

        lib = DynamicGrammarLibrary()
        mdef = lib.register_macro(pat)
        code = mdef.to_lean_code()

        # Invariant: generates exact Lean 4 declaration specification
        assert code == "def macro_hollow_cylinder (c1 c2 : SDFNode) : SDFNode := SDFNode.difference c1 c2"

    def test_lean4_code_generation_params(self) -> None:
        """Lean 4 inductive declaration code generation for FreeParam slots."""
        gen_c1 = CylinderNode(
            radius=VariableNode("?r1", "FreeParam", "param"),
            height=VariableNode("?h1", "FreeParam", "param"),
        )
        gen_c2 = CylinderNode(
            radius=VariableNode("?r2", "FreeParam", "param"),
            height=VariableNode("?h2", "FreeParam", "param"),
        )
        pat = AbstractionPattern(
            name="macro_hollow_cylinder",
            generalized_ast=DifferenceNode(a=gen_c1, b=gen_c2),
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=1,
            occurrences=[],
            size=7,
        )

        lib = DynamicGrammarLibrary()
        mdef = lib.register_macro(pat)
        code = mdef.to_lean_code()

        assert code == (
            "def macro_hollow_cylinder (r1 h1 r2 h2 : FreeParam) : SDFNode := "
            "SDFNode.difference (SDFNode.cylinder r1 h1) (SDFNode.cylinder r2 h2)"
        )

    def test_instantiate_call_with_args(self) -> None:
        """Instantiating macro with concrete args produces typed MacroCallNode."""
        pat = AbstractionPattern(
            name="macro_diff",
            generalized_ast=DifferenceNode(
                a=VariableNode("?c1", "SDFNode", "sdf"),
                b=VariableNode("?c2", "SDFNode", "sdf"),
            ),
            parameters=["?c1: SDFNode", "?c2: SDFNode"],
            frequency=1,
            occurrences=[],
            size=3,
        )

        lib = DynamicGrammarLibrary()
        mdef = lib.register_macro(pat)

        s1 = SphereNode(radius=_p("r1", 1.0))
        s2 = SphereNode(radius=_p("r2", 0.5))
        call_node = mdef.instantiate_call({"c1": s1, "c2": s2})

        assert isinstance(call_node, SDFMacroCallNode)
        assert call_node.macro_name == "macro_diff"
        assert call_node.arguments["c1"] == s1
        assert call_node.arguments["c2"] == s2

    def test_instantiate_call_generates_typed_holes(self) -> None:
        """Instantiating macro with missing args populates typed child holes."""
        pat = AbstractionPattern(
            name="macro_diff",
            generalized_ast=DifferenceNode(
                a=VariableNode("?c1", "SDFNode", "sdf"),
                b=VariableNode("?c2", "SDFNode", "sdf"),
            ),
            parameters=["?c1: SDFNode", "?c2: SDFNode"],
            frequency=1,
            occurrences=[],
            size=3,
        )

        lib = DynamicGrammarLibrary()
        mdef = lib.register_macro(pat)
        call_node = mdef.instantiate_call()

        assert isinstance(call_node, SDFMacroCallNode)
        assert isinstance(call_node.arguments["c1"], SDFHoleNode)
        assert isinstance(call_node.arguments["c2"], SDFHoleNode)


class TestSnapshotAndRollback:
    """Tests for versioned snapshot creation and transactional rollback."""

    def test_versioned_snapshot_and_rollback(self) -> None:
        """Snapshot captures library state; rollback safely restores previous state."""
        lib = DynamicGrammarLibrary()
        assert len(lib.macros) == 0

        # Snapshot 1: Clean library
        v1 = lib.snapshot()
        assert v1 == 1

        pat1 = AbstractionPattern(
            name="macro_one",
            generalized_ast=SphereNode(radius=VariableNode("?r", "FreeParam", "param")),
            parameters=["?r"],
            frequency=1,
            occurrences=[],
            size=2,
        )
        lib.register_macro(pat1)
        assert "macro_one" in lib.macros

        # Snapshot 2: Library with macro_one
        v2 = lib.snapshot()
        assert v2 == 2

        pat2 = AbstractionPattern(
            name="macro_two",
            generalized_ast=BoxNode(
                width=VariableNode("?w", "FreeParam", "param"),
                height=VariableNode("?h", "FreeParam", "param"),
                depth=VariableNode("?d", "FreeParam", "param"),
            ),
            parameters=["?w", "?h", "?d"],
            frequency=1,
            occurrences=[],
            size=4,
        )
        lib.register_macro(pat2)
        assert len(lib.macros) == 2
        assert "macro_two" in lib.macros

        # Rollback to v2: macro_two must be removed, macro_one preserved
        lib.rollback(v2)
        assert len(lib.macros) == 1
        assert "macro_one" in lib.macros
        assert "macro_two" not in lib.macros

        # Rollback to v1: clean state restored
        lib.rollback(v1)
        assert len(lib.macros) == 0

    def test_invalid_snapshot_rollback_raises(self) -> None:
        """Rollback to non-existent snapshot raises ValueError."""
        lib = DynamicGrammarLibrary()
        with pytest.raises(ValueError, match="Snapshot version 999 not found"):
            lib.rollback(999)


class TestPCFGProbabilityRenormalization:
    """Tests for probability distribution invariant: sum(P) == 1.0 +- 1e-5."""

    def test_pcfg_probabilities_strictly_sum_to_one(self) -> None:
        """Rule distributions must sum to 1.0 initially and after every registration."""
        lib = DynamicGrammarLibrary()

        # Initial distributions
        for dom in ("audio", "sdf"):
            dist = lib.get_rule_distribution(dom)
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-5, f"{dom} initial distribution sum = {total}"

        # Register SDF macro
        pat_sdf = AbstractionPattern(
            name="macro_hollow_cyl",
            generalized_ast=DifferenceNode(
                a=VariableNode("?c1", "SDFNode", "sdf"),
                b=VariableNode("?c2", "SDFNode", "sdf"),
            ),
            parameters=["?c1: SDFNode", "?c2: SDFNode"],
            frequency=1,
            occurrences=[],
            size=3,
        )
        lib.register_macro(pat_sdf, prior_weight=0.15)

        dist_sdf = lib.get_rule_distribution("sdf")
        assert "macro_hollow_cyl" in dist_sdf
        assert abs(sum(dist_sdf.values()) - 1.0) < 1e-5

        # Register Audio macro
        pat_audio = AbstractionPattern(
            name="macro_osc_mix",
            generalized_ast=MixNode(
                ratio=_p("r", 0.5),
                input1=VariableNode("?s1", "AudioNode", "audio"),
                input2=VariableNode("?s2", "AudioNode", "audio"),
            ),
            parameters=["?s1: AudioNode", "?s2: AudioNode"],
            frequency=1,
            occurrences=[],
            size=4,
        )
        lib.register_macro(pat_audio, prior_weight=0.20)

        dist_audio = lib.get_rule_distribution("audio")
        assert "macro_osc_mix" in dist_audio
        assert abs(sum(dist_audio.values()) - 1.0) < 1e-5

    def test_pcfg_rules_conversion(self) -> None:
        """get_pcfg_rules returns valid GrammarRule instances with proper weights."""
        lib = DynamicGrammarLibrary()
        pat = AbstractionPattern(
            name="macro_double_sphere",
            generalized_ast=UnionNode(
                smoothness=_p("k", 0.1),
                a=SphereNode(radius=_p("r1", 1.0)),
                b=SphereNode(radius=_p("r2", 0.5)),
            ),
            parameters=[],
            frequency=1,
            occurrences=[],
            size=5,
        )
        lib.register_macro(pat, prior_weight=0.10)

        rules = lib.get_pcfg_rules("sdf")
        macro_rules = [r for r in rules if r.rhs_tag == "macro_double_sphere"]
        assert len(macro_rules) == 1
        rule = macro_rules[0]
        assert rule.lhs == "SDFExp"
        assert rule.is_terminal is True  # No child nonterminals
        assert len(rule.child_symbols) == 0
        assert abs(sum(r.weight for r in rules) - 1.0) < 1e-5


class TestMacroExpansionSemanticInvariance:
    """Tests proving semantic invariance: Exec(MacroCall) == Exec(InlinedAST)."""

    def test_macro_expansion_sdf_semantic_identity(self) -> None:
        """Inlined SDF AST produces identical distance field values to MacroCall."""
        lib = DynamicGrammarLibrary()
        gen_c1 = CylinderNode(
            radius=VariableNode("?r1", "FreeParam", "param"),
            height=VariableNode("?h1", "FreeParam", "param"),
        )
        gen_c2 = CylinderNode(
            radius=VariableNode("?r2", "FreeParam", "param"),
            height=VariableNode("?h2", "FreeParam", "param"),
        )
        pat = AbstractionPattern(
            name="macro_hollow_cylinder",
            generalized_ast=DifferenceNode(a=gen_c1, b=gen_c2),
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=1,
            occurrences=[],
            size=7,
        )
        mdef = lib.register_macro(pat)

        r1 = _p("r1", 1.2, 0.1, 5.0)
        h1 = _p("h1", 3.0, 0.1, 5.0)
        r2 = _p("r2", 0.8, 0.1, 5.0)
        h2 = _p("h2", 3.2, 0.1, 5.0)

        macro_call = mdef.instantiate_call({"r1": r1, "h1": h1, "r2": r2, "h2": h2})
        inlined_ast = lib.expand_macro_call(macro_call)

        mod_call = transpile_ast_to_module(macro_call, library=lib)
        mod_inline = transpile_ast_to_module(inlined_ast, library=lib)

        torch.manual_seed(42)
        probe_points = torch.randn(200, 3)

        out_call = mod_call(probe_points)
        out_inline = mod_inline(probe_points)

        max_err = (out_call - out_inline).abs().max().item()
        assert max_err < 1e-5, f"SDF semantic invariance violation: error={max_err}"

    def test_macro_expansion_audio_semantic_identity(self) -> None:
        """Inlined Audio AST produces identical waveform values to MacroCall."""
        lib = DynamicGrammarLibrary()
        gen_s1 = SineNode(
            freq=VariableNode("?f1", "FreeParam", "param"),
            phase=VariableNode("?p1", "FreeParam", "param"),
        )
        gen_s2 = SawNode(freq=VariableNode("?f2", "FreeParam", "param"))
        pat = AbstractionPattern(
            name="macro_dual_osc",
            generalized_ast=MixNode(
                ratio=VariableNode("?r", "FreeParam", "param"),
                input1=gen_s1,
                input2=gen_s2,
            ),
            parameters=["?r", "?f1", "?p1", "?f2"],
            frequency=1,
            occurrences=[],
            size=6,
        )
        mdef = lib.register_macro(pat)

        r = _p("r", 0.4, 0.0, 1.0)
        f1 = _p("f1", 440.0, 20.0, 5000.0)
        p1 = _p("p1", 0.5, 0.0, 6.28)
        f2 = _p("f2", 220.0, 20.0, 5000.0)

        macro_call = mdef.instantiate_call({"r": r, "f1": f1, "p1": p1, "f2": f2})
        inlined_ast = lib.expand_macro_call(macro_call)

        mod_call = transpile_ast_to_module(macro_call, library=lib)
        mod_inline = transpile_ast_to_module(inlined_ast, library=lib)

        t = torch.linspace(0.0, 0.05, 500)
        out_call = mod_call(t)
        out_inline = mod_inline(t)

        max_err = (out_call - out_inline).abs().max().item()
        assert max_err < 1e-5, f"Audio semantic invariance violation: error={max_err}"

    def test_nested_macro_expansion(self) -> None:
        """Inlining correctly resolves recursively nested macro invocations."""
        lib = DynamicGrammarLibrary()

        # Macro 1: scaled sphere
        pat1 = AbstractionPattern(
            name="macro_unit_sphere",
            generalized_ast=SphereNode(radius=_p("r", 1.0)),
            parameters=[],
            frequency=1,
            occurrences=[],
            size=2,
        )
        lib.register_macro(pat1)

        # Macro 2: difference using macro 1
        m1_call = make_macro_call("macro_unit_sphere", {}, domain="sdf")
        pat2 = AbstractionPattern(
            name="macro_diff_spheres",
            generalized_ast=DifferenceNode(a=m1_call, b=m1_call),
            parameters=[],
            frequency=1,
            occurrences=[],
            size=3,
        )
        lib.register_macro(pat2)

        outer_call = make_macro_call("macro_diff_spheres", {}, domain="sdf")
        inlined = lib.expand_macro_call(outer_call)

        # Invariant: Fully inlined to basic primitives
        assert isinstance(inlined, DifferenceNode)
        assert isinstance(inlined.a, SphereNode)
        assert isinstance(inlined.b, SphereNode)


class TestTranspilationAndAutograd:
    """Tests verifying autograd gradient flow through MacroCallNode parameter slots."""

    def test_sdf_macro_transpilation_and_autograd(self) -> None:
        """Gradients flow back into MacroCallNode FreeParam arguments."""
        lib = DynamicGrammarLibrary()
        gen_c1 = CylinderNode(
            radius=VariableNode("?r1", "FreeParam", "param"),
            height=VariableNode("?h1", "FreeParam", "param"),
        )
        gen_c2 = CylinderNode(
            radius=VariableNode("?r2", "FreeParam", "param"),
            height=VariableNode("?h2", "FreeParam", "param"),
        )
        pat = AbstractionPattern(
            name="macro_hollow_cylinder",
            generalized_ast=DifferenceNode(a=gen_c1, b=gen_c2),
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=1,
            occurrences=[],
            size=7,
        )
        mdef = lib.register_macro(pat)

        r1 = _p("r1", 1.0, 0.1, 5.0)
        h1 = _p("h1", 2.0, 0.1, 5.0)
        r2 = _p("r2", 0.5, 0.1, 5.0)
        h2 = _p("h2", 2.5, 0.1, 5.0)

        macro_call = mdef.instantiate_call({"r1": r1, "h1": h1, "r2": r2, "h2": h2})
        mod = transpile_ast_to_module(macro_call, library=lib)

        assert isinstance(mod, DifferentiableSDFModule)
        assert mod.theta.shape[0] == 4

        probe = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=False)
        out = mod(probe)
        loss = out.pow(2).sum()
        loss.backward()

        assert mod.theta.grad is not None
        assert (mod.theta.grad.abs() > 0.0).any()

    def test_audio_macro_transpilation_and_autograd(self) -> None:
        """Autograd gradients propagate through Audio MacroCallNode parameters."""
        lib = DynamicGrammarLibrary()
        gen_s1 = SineNode(
            freq=VariableNode("?f1", "FreeParam", "param"),
            phase=VariableNode("?p1", "FreeParam", "param"),
        )
        gen_s2 = SawNode(freq=VariableNode("?f2", "FreeParam", "param"))
        pat = AbstractionPattern(
            name="macro_dual_osc",
            generalized_ast=MixNode(
                ratio=VariableNode("?r", "FreeParam", "param"),
                input1=gen_s1,
                input2=gen_s2,
            ),
            parameters=["?r", "?f1", "?p1", "?f2"],
            frequency=1,
            occurrences=[],
            size=6,
        )
        mdef = lib.register_macro(pat)

        r = _p("r", 0.5, 0.0, 1.0)
        f1 = _p("f1", 440.0, 20.0, 5000.0)
        p1 = _p("p1", 0.0, 0.0, 6.28)
        f2 = _p("f2", 220.0, 20.0, 5000.0)

        macro_call = mdef.instantiate_call({"r": r, "f1": f1, "p1": p1, "f2": f2})
        mod = transpile_ast_to_module(macro_call, library=lib)

        assert isinstance(mod, DifferentiableAudioModule)
        assert mod.theta.shape[0] == 4

        t = torch.linspace(0.0, 0.01, 100, requires_grad=False)
        out = mod(t)
        loss = out.pow(2).sum()
        loss.backward()

        assert mod.theta.grad is not None
        assert (mod.theta.grad.abs() > 0.0).any()


class TestSynthesizerSearchAccelerationGate:
    """Pass-to-Complete Gate: Verified search step reduction > 40% with registered macro."""

    def test_synthesizer_pass_to_complete_gate_sdf_composite(self) -> None:
        """Synthesizer discovers composite Difference(Cylinder, Cylinder) with >40% step reduction."""
        r1 = _p("r1", 1.0, 0.1, 5.0)
        h1 = _p("h1", 2.0, 0.1, 5.0)
        r2 = _p("r2", 0.5, 0.1, 5.0)
        h2 = _p("h2", 2.5, 0.1, 5.0)
        c1 = CylinderNode(radius=r1, height=h1)
        c2 = CylinderNode(radius=r2, height=h2)
        target = DifferenceNode(a=c1, b=c2)

        scorer = ReferenceASTScorer(target=target, domain="sdf")
        cfg = SearchConfig(max_expansions=100, tolerance=1e-4)

        # 1. Base run without macro
        synth_base = AStarSynthesizer()
        res_base = synth_base.synthesize(
            initial_hole=SDFHoleNode("root"),
            scorer=scorer,
            config=cfg,
        )
        n_base = res_base.total_expansions

        # 2. Register macro in dynamic library
        lib = DynamicGrammarLibrary()
        gen_c1 = CylinderNode(
            radius=VariableNode("?r1", "FreeParam", "param"),
            height=VariableNode("?h1", "FreeParam", "param"),
        )
        gen_c2 = CylinderNode(
            radius=VariableNode("?r2", "FreeParam", "param"),
            height=VariableNode("?h2", "FreeParam", "param"),
        )
        pat = AbstractionPattern(
            name="macro_hollow_cylinder",
            generalized_ast=DifferenceNode(a=gen_c1, b=gen_c2),
            parameters=["?r1", "?h1", "?r2", "?h2"],
            frequency=2,
            occurrences=[(0, {"?r1": r1, "?h1": h1, "?r2": r2, "?h2": h2})],
            size=7,
        )
        lib.register_macro(pat, prior_weight=0.15)

        # 3. Macro run with dynamic library
        synth_macro = AStarSynthesizer(library=lib)
        res_macro = synth_macro.synthesize(
            initial_hole=SDFHoleNode("root"),
            scorer=scorer,
            config=cfg,
        )
        n_macro = res_macro.total_expansions

        assert res_macro.success is True
        assert res_macro.best_score <= cfg.tolerance

        step_reduction = (n_base - n_macro) / n_base
        logger.info(
            "Gate metrics: N_base=%d, N_macro=%d, Reduction=%.2f%%",
            n_base, n_macro, step_reduction * 100.0,
        )

        # Gate Invariant: Step reduction strictly exceeds 40%
        assert step_reduction > 0.40, (
            f"Pass-to-Complete Gate failed: reduction {step_reduction:.2%} <= 40% "
            f"(N_base={n_base}, N_macro={n_macro})"
        )

    def test_synthesizer_pass_to_complete_gate_chain_transforms(self) -> None:
        """Synthesizer discovers chained transforms with >40% step reduction."""
        sphere = SphereNode(radius=_p("r", 1.0, 0.1, 5.0))
        dx = _p("dx", 1.0, -5.0, 5.0)
        dy = _p("dy", 0.0, -5.0, 5.0)
        dz = _p("dz", 0.0, -5.0, 5.0)
        t1 = TranslateNode(dx=dx, dy=dy, dz=dz, child=sphere)
        target = TranslateNode(dx=dx, dy=dy, dz=dz, child=t1)

        scorer = ReferenceASTScorer(target=target, domain="sdf")
        cfg = SearchConfig(max_expansions=100, tolerance=1e-4)

        # Base run
        synth_base = AStarSynthesizer()
        res_base = synth_base.synthesize(
            initial_hole=SDFHoleNode("root"),
            scorer=scorer,
            config=cfg,
        )
        n_base = res_base.total_expansions

        # Register chained translate macro
        lib = DynamicGrammarLibrary()
        pat = AbstractionPattern(
            name="macro_double_translate",
            generalized_ast=target,
            parameters=[],
            frequency=2,
            occurrences=[],
            size=5,
        )
        lib.register_macro(pat, prior_weight=0.15)

        synth_macro = AStarSynthesizer(library=lib)
        res_macro = synth_macro.synthesize(
            initial_hole=SDFHoleNode("root"),
            scorer=scorer,
            config=cfg,
        )
        n_macro = res_macro.total_expansions

        assert res_base.success is True
        assert res_macro.success is True
        assert n_base > n_macro

        step_reduction = (n_base - n_macro) / n_base
        assert step_reduction > 0.40, (
            f"Pass-to-Complete Gate failed: reduction {step_reduction:.2%} <= 40% "
            f"(N_base={n_base}, N_macro={n_macro})"
        )
