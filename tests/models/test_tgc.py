"""Automated test suite for Model 2 (TGC) - Topology & Grammar Connector."""

from __future__ import annotations
import random
import time
from typing import List, Tuple
import pytest
import torch

from symforge.core.ast import (
    ASTNode,
    AudioNode,
    LowPassNode,
    MixNode,
    SDFNode,
    SineNode,
    SphereNode,
    UnionNode,
    has_holes,
)
from symforge.core.transpiler import transpile_ast_to_module
from symforge.models.tgc import (
    AUDIO_VOCABULARY,
    DOMAIN_VOCABULARIES,
    SDF_VOCABULARY,
    TGC,
    TopologyGrammarConnector,
    assemble,
    batch_assemble,
    normalize_primitive_name,
)


def extract_ast_primitives_with_depth(
    node: ASTNode, current_depth: int = 0
) -> List[Tuple[str, int]]:
    """Traverse AST and collect (primitive_name, depth_from_root) tuples."""
    results: List[Tuple[str, int]] = []
    tag = type(node).__name__
    for suffix in ("Node", "Filter", "Oscillator", "Envelope", "SDF", "CSG"):
        if tag.endswith(suffix) and len(tag) > len(suffix):
            tag = tag[: -len(suffix)]
    results.append((tag, current_depth))
    for child in node.children:
        results.extend(extract_ast_primitives_with_depth(child, current_depth + 1))
    return results


class TestTGCArchitecture:
    """Validate TGC initialization, vocabularies, and normalization."""

    def test_vocabulary_consistency(self):
        """Invariant: TGC vocabularies must cover all required primitives."""
        assert "Sine" in AUDIO_VOCABULARY
        assert "LowPass" in AUDIO_VOCABULARY
        assert "Mix" in AUDIO_VOCABULARY
        assert "ADSR" in AUDIO_VOCABULARY
        assert "ExpDecay" in AUDIO_VOCABULARY
        assert "FM" in AUDIO_VOCABULARY
        assert "LFO" in AUDIO_VOCABULARY

        assert "Sphere" in SDF_VOCABULARY
        assert "Box" in SDF_VOCABULARY
        assert "Cylinder" in SDF_VOCABULARY
        assert "Torus" in SDF_VOCABULARY
        assert "Plane" in SDF_VOCABULARY
        assert "Translate" in SDF_VOCABULARY
        assert "Rotate" in SDF_VOCABULARY
        assert "Scale" in SDF_VOCABULARY
        assert "Union" in SDF_VOCABULARY
        assert "SmoothUnion" in SDF_VOCABULARY
        assert "Difference" in SDF_VOCABULARY
        assert "Intersection" in SDF_VOCABULARY

    def test_normalize_primitive_name_variants(self):
        """Invariant: Case-insensitive and suffix stripping matches canonical names."""
        assert normalize_primitive_name("sine", "audio") == "Sine"
        assert normalize_primitive_name("LowPassNode", "audio") == "LowPass"
        assert normalize_primitive_name("exp_decay_envelope", "audio") == "ExpDecay"
        assert normalize_primitive_name("SPHERE_SDF", "sdf") == "Sphere"
        assert normalize_primitive_name("SmoothUnion", "sdf") == "SmoothUnion"
        assert normalize_primitive_name("NonExistent", "audio") is None

    def test_invalid_domain_raises_value_error(self):
        """Invariant: Passing an unknown domain raises ValueError."""
        tgc = TGC()
        with pytest.raises(ValueError, match="Unknown domain"):
            tgc.assemble([("Sine", 0)], domain="quantum")

    def test_empty_or_unrecognized_input_returns_none(self):
        """Invariant: Empty list or unrecognized primitives return None."""
        assert assemble([], domain="audio") is None
        assert assemble([("AlienPrimitive", 1)], domain="sdf") is None


class TestTGCStructuralAssemblyAudio:
    """Validate Audio domain assembly across Curriculum Levels L1 to L4."""

    def test_audio_level_1_isolated_primitives(self):
        """Curriculum Level 1: Isolated leaf oscillators and noise."""
        for prim in ("Sine", "Saw", "Square", "Noise", "FM", "LFO"):
            pairs = [(prim, 0)]
            ast = assemble(pairs, domain="audio")
            assert ast is not None
            assert isinstance(ast, AudioNode)
            assert not has_holes(ast)

            # Invariant: single node tree
            assert ast.tree_depth == 1
            extracted = extract_ast_primitives_with_depth(ast)
            assert extracted[0][0] == prim
            assert extracted[0][1] == 0

            # Symmetrical validation: transpilation to PyTorch module
            module = transpile_ast_to_module(ast)
            t = torch.linspace(0.0, 0.01, 100)
            y = module(t)
            assert y.shape == (100,)
            assert not torch.isnan(y).any()
            assert not torch.isinf(y).any()

    def test_audio_level_2_single_modifiers_shuffled(self):
        """Curriculum Level 2: Modifier with leaf, deliberately shuffled."""
        cases = [
            [("Sine", 1), ("LowPass", 0)],
            [("Saw", 1), ("Gain", 0)],
            [("Square", 1), ("ADSR", 0)],
            [("Noise", 1), ("ExpDecay", 0)],
        ]
        for shuffled in cases:
            # Randomize order
            shuffled_copy = list(shuffled)
            random.Random(42).shuffle(shuffled_copy)

            ast = assemble(shuffled_copy, domain="audio")
            assert ast is not None
            assert isinstance(ast, AudioNode)
            assert not has_holes(ast)
            assert ast.tree_depth == 2

            # Invariant: modifier at root, child at depth 1
            extracted = dict(extract_ast_primitives_with_depth(ast))
            for p, d in shuffled:
                assert p in extracted
                assert abs(extracted[p] - d) <= 1

            # Symmetrical validation: transpilation
            module = transpile_ast_to_module(ast)
            t = torch.linspace(0.0, 0.01, 100)
            y = module(t)
            assert y.shape == (100,)
            assert not torch.isnan(y).any()

    def test_audio_level_3_binary_mix_shuffled(self):
        """Curriculum Level 3: Binary composition (Mix) with nested modifier."""
        # Mix(LowPass(Sine), Saw)
        pairs = [
            ("Mix", 0),
            ("LowPass", 1),
            ("Sine", 2),
            ("Saw", 1),
        ]
        shuffled = list(pairs)
        random.Random(123).shuffle(shuffled)

        ast = assemble(shuffled, domain="audio")
        assert ast is not None
        assert isinstance(ast, MixNode)
        assert not has_holes(ast)
        assert ast.tree_depth == 3

        extracted = [p for p, _ in extract_ast_primitives_with_depth(ast)]
        for p, _ in pairs:
            assert p in extracted

        # Symmetrical validation
        module = transpile_ast_to_module(ast)
        t = torch.linspace(0.0, 0.01, 100)
        y = module(t)
        assert y.shape == (100,)
        assert not torch.isnan(y).any()

    def test_audio_level_4_nested_graph_shuffled(self):
        """Curriculum Level 4: Deep composition with ADSR, Mix, LowPass, Oscillators."""
        pairs = [
            ("ADSR", 0),
            ("Mix", 1),
            ("LowPass", 2),
            ("Sine", 3),
            ("Saw", 2),
        ]
        shuffled = list(pairs)
        random.Random(999).shuffle(shuffled)

        ast = assemble(shuffled, domain="audio")
        assert ast is not None
        assert not has_holes(ast)
        assert ast.tree_depth >= 3

        extracted = [p for p, _ in extract_ast_primitives_with_depth(ast)]
        for p, _ in pairs:
            assert p in extracted

        # Symmetrical validation
        module = transpile_ast_to_module(ast)
        t = torch.linspace(0.0, 0.01, 100)
        y = module(t)
        assert y.shape == (100,)
        assert not torch.isnan(y).any()


class TestTGCStructuralAssemblySDF:
    """Validate SDF domain assembly across Curriculum Levels L1 to L4."""

    def test_sdf_level_1_isolated_primitives(self):
        """Curriculum Level 1: Isolated 3D CSG geometric primitives."""
        for prim in ("Sphere", "Box", "Cylinder", "Torus", "Plane"):
            pairs = [(prim, 0)]
            ast = assemble(pairs, domain="sdf")
            assert ast is not None
            assert isinstance(ast, SDFNode)
            assert not has_holes(ast)
            assert ast.tree_depth == 1

            extracted = extract_ast_primitives_with_depth(ast)
            assert extracted[0][0] == prim
            assert extracted[0][1] == 0

            # Symmetrical validation: transpilation to PyTorch module
            module = transpile_ast_to_module(ast)
            p = torch.randn(50, 3)
            d = module(p)
            assert d.shape == (50,)
            assert not torch.isnan(d).any()

    def test_sdf_level_2_spatial_transforms_shuffled(self):
        """Curriculum Level 2: Spatial transforms (Translate, Rotate, Scale)."""
        cases = [
            [("Sphere", 1), ("Translate", 0)],
            [("Box", 1), ("Rotate", 0)],
            [("Cylinder", 1), ("Scale", 0)],
        ]
        for shuffled in cases:
            ast = assemble(shuffled, domain="sdf")
            assert ast is not None
            assert isinstance(ast, SDFNode)
            assert not has_holes(ast)
            assert ast.tree_depth == 2

            extracted = dict(extract_ast_primitives_with_depth(ast))
            for p, d in shuffled:
                assert p in extracted
                assert abs(extracted[p] - d) <= 1

            # Symmetrical validation
            module = transpile_ast_to_module(ast)
            p = torch.randn(50, 3)
            d = module(p)
            assert d.shape == (50,)
            assert not torch.isnan(d).any()

    def test_sdf_level_3_csg_booleans_shuffled(self):
        """Curriculum Level 3: Binary CSG booleans (Union, SmoothUnion, Difference)."""
        cases = [
            [("Union", 0), ("Sphere", 1), ("Box", 1)],
            [("SmoothUnion", 0), ("Sphere", 1), ("Box", 1)],
            [("Difference", 0), ("Box", 1), ("Cylinder", 1)],
            [("Intersection", 0), ("Sphere", 1), ("Torus", 1)],
        ]
        for pairs in cases:
            shuffled = list(pairs)
            random.Random(77).shuffle(shuffled)

            ast = assemble(shuffled, domain="sdf")
            assert ast is not None
            assert not has_holes(ast)
            assert ast.tree_depth == 2

            extracted = [p for p, _ in extract_ast_primitives_with_depth(ast)]
            for p, _ in pairs:
                assert p in extracted

            # Symmetrical validation
            module = transpile_ast_to_module(ast)
            p = torch.randn(50, 3)
            d = module(p)
            assert d.shape == (50,)
            assert not torch.isnan(d).any()

    def test_sdf_level_4_nested_csg_tree_shuffled(self):
        """Curriculum Level 4: Nested CSG composition with transforms."""
        # Difference(Union(Translate(Sphere), Box), Cylinder)
        pairs = [
            ("Difference", 0),
            ("Union", 1),
            ("Translate", 2),
            ("Sphere", 3),
            ("Box", 2),
            ("Cylinder", 1),
        ]
        shuffled = list(pairs)
        random.Random(555).shuffle(shuffled)

        ast = assemble(shuffled, domain="sdf")
        assert ast is not None
        assert not has_holes(ast)
        assert ast.tree_depth >= 3

        extracted = [p for p, _ in extract_ast_primitives_with_depth(ast)]
        for p, _ in pairs:
            assert p in extracted

        # Symmetrical validation
        module = transpile_ast_to_module(ast)
        p = torch.randn(50, 3)
        d = module(p)
        assert d.shape == (50,)
        assert not torch.isnan(d).any()


class TestTGCRobustnessAndNoise:
    """Verify graceful handling of incomplete, noisy, or distorted multiset predictions."""

    def test_incomplete_audio_multiset_closes_holes(self):
        """Robustness: Single operator without child leaf synthesizes fallback leaf."""
        pairs = [("LowPass", 0)]
        ast = assemble(pairs, domain="audio")
        assert ast is not None
        assert isinstance(ast, LowPassNode)
        assert not has_holes(ast)
        assert isinstance(ast.input_node, SineNode)

        # Symmetrical validation
        module = transpile_ast_to_module(ast)
        t = torch.linspace(0.0, 0.01, 50)
        y = module(t)
        assert y.shape == (50,)

    def test_incomplete_sdf_multiset_closes_holes(self):
        """Robustness: Binary boolean with only one child synthesizes second child."""
        pairs = [("Union", 0), ("Sphere", 1)]
        ast = assemble(pairs, domain="sdf")
        assert ast is not None
        assert isinstance(ast, UnionNode)
        assert not has_holes(ast)
        assert isinstance(ast.a, SphereNode) or isinstance(ast.b, SphereNode)

        # Symmetrical validation
        module = transpile_ast_to_module(ast)
        p = torch.randn(20, 3)
        d = module(p)
        assert d.shape == (20,)

    def test_noisy_multiset_with_unknown_tokens(self):
        """Robustness: Ignores unknown noisy tokens and builds valid tree."""
        pairs = [("UnknownModulator", 2), ("Sine", 1), ("LowPass", 0), ("GarbageNoiseToken", 3)]
        ast = assemble(pairs, domain="audio")
        assert ast is not None
        assert isinstance(ast, LowPassNode)
        assert not has_holes(ast)

    def test_extra_primitives_in_multiset(self):
        """Robustness: Extra primitives that cannot connect are handled gracefully."""
        # 3 leaves for a single root
        pairs = [("Sphere", 0), ("Box", 0), ("Cylinder", 0)]
        ast = assemble(pairs, domain="sdf")
        assert ast is not None
        assert isinstance(ast, SDFNode)
        assert not has_holes(ast)


class TestTGCBatchAssemble:
    """Validate batch assembly utility."""

    def test_batch_assemble_audio_and_sdf(self):
        """Batch assemble multiple program specifications."""
        batch_audio = [
            [("Sine", 0)],
            [("LowPass", 0), ("Saw", 1)],
            [("Mix", 0), ("Sine", 1), ("Noise", 1)],
        ]
        results_audio = batch_assemble(batch_audio, domain="audio")
        assert len(results_audio) == 3
        for ast in results_audio:
            assert ast is not None
            assert not has_holes(ast)
            assert isinstance(ast, AudioNode)

        batch_sdf = [
            [("Sphere", 0)],
            [("Translate", 0), ("Box", 1)],
            [("Union", 0), ("Sphere", 1), ("Cylinder", 1)],
        ]
        results_sdf = batch_assemble(batch_sdf, domain="sdf")
        assert len(results_sdf) == 3
        for ast in results_sdf:
            assert ast is not None
            assert not has_holes(ast)
            assert isinstance(ast, SDFNode)


class TestTGCLatencyBenchmark:
    """Pass-to-Complete Gate: Assembly latency benchmark across 20 programs."""

    def test_latency_under_50ms_per_program(self):
        """Invariant: Average assembly latency across 20 programs must be < 50 ms."""
        programs_audio = [
            [("Sine", 0)],
            [("Saw", 0)],
            [("LowPass", 0), ("Sine", 1)],
            [("Gain", 0), ("Saw", 1)],
            [("ADSR", 0), ("Square", 1)],
            [("Mix", 0), ("Sine", 1), ("Saw", 1)],
            [("Mix", 0), ("LowPass", 1), ("Sine", 2), ("Noise", 1)],
            [("ADSR", 0), ("Mix", 1), ("Sine", 2), ("Saw", 2)],
            [("LowPass", 0), ("Mix", 1), ("Gain", 2), ("Sine", 3), ("Saw", 2)],
            [("ExpDecay", 0), ("LowPass", 1), ("Noise", 2)],
        ]

        programs_sdf = [
            [("Sphere", 0)],
            [("Box", 0)],
            [("Translate", 0), ("Sphere", 1)],
            [("Rotate", 0), ("Box", 1)],
            [("Scale", 0), ("Cylinder", 1)],
            [("Union", 0), ("Sphere", 1), ("Box", 1)],
            [("Difference", 0), ("Box", 1), ("Cylinder", 1)],
            [("Intersection", 0), ("Sphere", 1), ("Torus", 1)],
            [("SmoothUnion", 0), ("Translate", 1), ("Sphere", 2), ("Box", 1)],
            [("Difference", 0), ("Union", 1), ("Sphere", 2), ("Box", 2), ("Cylinder", 1)],
        ]

        all_20_programs = [("audio", p) for p in programs_audio] + [("sdf", p) for p in programs_sdf]
        assert len(all_20_programs) == 20

        tgc = TGC()

        # Warmup pass
        for dom, pairs in all_20_programs:
            _ = tgc.assemble(pairs, domain=dom)

        # Benchmark pass across 20 programs
        t0 = time.perf_counter()
        for dom, pairs in all_20_programs:
            assembled = tgc.assemble(pairs, domain=dom)
            assert assembled is not None
            assert not has_holes(assembled)
        total_time = time.perf_counter() - t0

        avg_latency_ms = (total_time / len(all_20_programs)) * 1000.0

        # Pass-to-complete gate: average latency < 50 ms
        assert avg_latency_ms < 50.0, f"Assembly average latency {avg_latency_ms:.2f} ms exceeded 50 ms gate"
