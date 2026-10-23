"""Tests for LibraryPruner and MacroDependencyGraph."""

import pytest
import torch

from symforge.abstraction.grammar_expander import (
    DynamicGrammarLibrary,
    MacroParam,
    MacroDefinition,
)
from symforge.abstraction.anti_unification import (
    AbstractionPattern,
    VariableNode,
    SDFVariableNode,
    AudioVariableNode,
)
from symforge.core.ast import (
    BoxNode,
    CylinderNode,
    DifferenceNode,
    FreeParam,
    MixNode,
    SawNode,
    SineNode,
    SphereNode,
    TranslateNode,
    UnionNode,
)
from symforge.core.transpiler import transpile_ast_to_module
from symforge.resilience.library_pruner import (
    LibraryPruner,
    MacroDependencyGraph,
    MacroUtilityMetrics,
    extract_called_macros,
    inline_macro_in_ast,
)


class TestMacroDependencyGraph:
    """Test dependency graph construction, topological sort, and cycle detection."""

    def test_dependency_tracking(self):
        graph = MacroDependencyGraph()
        graph.add_dependency(dependency="macro_base", dependent="macro_derived")

        assert graph.get_dependencies("macro_derived") == {"macro_base"}
        assert graph.get_dependents("macro_base") == {"macro_derived"}
        assert graph.is_leaf("macro_derived") is True
        assert graph.is_leaf("macro_base") is False

    def test_topological_sort_linear(self):
        graph = MacroDependencyGraph()
        graph.add_dependency("macro_a", "macro_b")
        graph.add_dependency("macro_b", "macro_c")

        sorted_nodes = graph.topological_sort()
        assert sorted_nodes.index("macro_a") < sorted_nodes.index("macro_b")
        assert sorted_nodes.index("macro_b") < sorted_nodes.index("macro_c")

    def test_cycle_detection_raises_value_error(self):
        graph = MacroDependencyGraph()
        graph.add_dependency("macro_a", "macro_b")
        graph.add_dependency("macro_b", "macro_a")

        with pytest.raises(ValueError, match="Cyclic dependency"):
            graph.topological_sort()


class TestSafeInlining:
    """Test inlining macros preserves semantic evaluation invariance."""

    def test_safe_inlining_sdf_semantics(self):
        library = DynamicGrammarLibrary()

        # Define base macro: hollow cylinder = Difference(Cylinder, Cylinder)
        r1 = FreeParam("r1", 0.1, 10.0, 1.0)
        h1 = FreeParam("h1", 0.1, 10.0, 2.0)
        r2 = FreeParam("r2", 0.1, 10.0, 0.5)
        h2 = FreeParam("h2", 0.1, 10.0, 2.2)

        base_body = DifferenceNode(
            CylinderNode(r1, h1),
            CylinderNode(r2, h2),
        )
        pattern = AbstractionPattern(
            name="macro_hollow_cyl",
            generalized_ast=base_body,
            parameters=["r1:FreeParam", "h1:FreeParam", "r2:FreeParam", "h2:FreeParam"],
            frequency=5,
            occurrences=[],
            size=3,
        )
        library.register_macro(pattern, "macro_hollow_cyl")

        # Dependent macro: translated hollow cylinder
        dx = FreeParam("dx", -5.0, 5.0, 1.0)
        dy = FreeParam("dy", -5.0, 5.0, 0.0)
        dz = FreeParam("dz", -5.0, 5.0, 0.0)
        hollow_def = library.macros["macro_hollow_cyl"]
        hollow_call = hollow_def.instantiate_call()
        derived_body = TranslateNode(dx, dy, dz, hollow_call)

        derived_pattern = AbstractionPattern(
            name="macro_trans_hollow",
            generalized_ast=derived_body,
            parameters=["dx:FreeParam", "dy:FreeParam", "dz:FreeParam"],
            frequency=3,
            occurrences=[],
            size=4,
        )
        library.register_macro(derived_pattern, "macro_trans_hollow")

        # Verify dependency graph
        pruner = LibraryPruner(max_library_size=1)
        graph = pruner.build_dependency_graph(library)
        assert graph.get_dependents("macro_hollow_cyl") == {"macro_trans_hollow"}

        # Perform inlining
        inlined_ast = inline_macro_in_ast(
            derived_body,
            "macro_hollow_cyl",
            library,
        )

        # Semantics check: evaluate original vs inlined
        mod_orig = transpile_ast_to_module(derived_body)
        mod_inlined = transpile_ast_to_module(inlined_ast)

        pts = torch.randn(20, 3)
        d_orig = mod_orig(pts)
        d_inlined = mod_inlined(pts)

        assert torch.allclose(d_orig, d_inlined, atol=1e-5)


class TestLibraryPrunerGate:
    """Test pruning from 60 macros to 40 macros."""

    def test_pruner_reduces_from_60_to_40_macros_safely(self):
        library = DynamicGrammarLibrary()
        pruner = LibraryPruner(max_library_size=40)

        # Register 60 synthetic macros
        for i in range(60):
            p = AbstractionPattern(
                name=f"macro_shape_{i}",
                generalized_ast=SphereNode(FreeParam(f"r_{i}", 0.1, 5.0, 1.0)),
                parameters=[f"r_{i}"],
                frequency=i + 1,
                occurrences=[],
                size=2,
            )
            library.register_macro(p, f"macro_shape_{i}")
            # Record variable utility: macros 0..19 get low utility, 20..59 get high utility
            pruner.record_usage(f"macro_shape_{i}", savings=float(i + 1), cycle=1)

        assert len(library.macros) == 60

        # Execute pruning
        pruned_names = pruner.prune_to_capacity(library, target_capacity=40)

        # Assertions
        assert len(pruned_names) == 20
        assert len(library.macros) == 40

        # Lowest utility macros (0..19) should have been pruned first
        for i in range(20):
            assert f"macro_shape_{i}" in pruned_names
            assert f"macro_shape_{i}" not in library.macros

        # Remaining macros should be 20..59
        for i in range(20, 60):
            assert f"macro_shape_{i}" in library.macros

        # PCFG prior probabilities must remain normalized
        rules = library.get_rule_distribution("sdf")
        total_prob = sum(rules.values())
        assert pytest.approx(total_prob, rel=1e-4) == 1.0
