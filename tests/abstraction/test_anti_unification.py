"""Automated tests for Subgraph Anti-Unification and Structural Pattern Extraction."""

from __future__ import annotations
import time
import pytest

from symforge.abstraction.anti_unification import (
    AbstractionPattern,
    AudioVariableNode,
    MacroParamStr,
    SDFVariableNode,
    VariableNode,
    anti_unify,
    apply_substitution,
    collect_variables,
    extract_subtrees,
    make_variable,
    match_pattern,
    mine_recurrent_patterns,
)
from symforge.core.ast import (
    ASTNode,
    BoxNode,
    CylinderNode,
    DifferenceNode,
    FreeParam,
    GainNode,
    LowPassNode,
    MixNode,
    SawNode,
    SineNode,
    SphereNode,
    SquareNode,
    TranslateNode,
    UnionNode,
    deserialize_node,
    serialize_node,
)


def _p(name: str, val: float) -> FreeParam:
    """Helper to construct FreeParam."""
    return FreeParam(id=name, min_val=0.0, max_val=100.0, default_val=val)


class TestAntiUnificationProperties:
    """Core mathematical properties of Plotkin / Reynolds Anti-Unification."""

    def test_identity_property_sdf(self) -> None:
        """Identity invariant: LGG(t, t) == t with empty substitutions for SDF."""
        sphere = SphereNode(radius=_p("r_0", 2.5))
        box = BoxNode(width=_p("w_0", 1.0), height=_p("h_0", 2.0), depth=_p("d_0", 3.0))
        csg = DifferenceNode(a=sphere, b=box)

        lgg, s1, s2 = anti_unify(csg, csg)

        assert lgg == csg
        assert len(s1) == 0
        assert len(s2) == 0
        assert apply_substitution(lgg, s1) == csg
        assert apply_substitution(lgg, s2) == csg

    def test_identity_property_audio(self) -> None:
        """Identity invariant: LGG(t, t) == t for Audio nodes."""
        sine = SineNode(freq=_p("f_0", 440.0))
        saw = SawNode(freq=_p("f_1", 220.0))
        mix = MixNode(ratio=_p("mix_0", 0.5), input1=sine, input2=saw)

        lgg, s1, s2 = anti_unify(mix, mix)

        assert lgg == mix
        assert len(s1) == 0
        assert len(s2) == 0
        assert apply_substitution(lgg, s1) == mix
        assert apply_substitution(lgg, s2) == mix

    def test_parameter_abstraction(self) -> None:
        """Parameter abstraction: LGG(Sphere(r1), Sphere(r2)) abstracts parameters into variable."""
        r1 = _p("r_1", 1.5)
        r2 = _p("r_2", 3.0)
        t1 = SphereNode(radius=r1)
        t2 = SphereNode(radius=r2)

        lgg, s1, s2 = anti_unify(t1, t2)

        assert isinstance(lgg, SphereNode)
        assert isinstance(lgg.radius, VariableNode)
        assert lgg.radius.expected_type == "FreeParam"
        assert lgg.radius.domain == "param"

        var_name = lgg.radius.name
        assert s1[var_name] == r1
        assert s2[var_name] == r2

        # Symmetrical round-trip identity
        assert apply_substitution(lgg, s1) == t1
        assert apply_substitution(lgg, s2) == t2

    def test_operator_divergence(self) -> None:
        """Operator divergence: LGG(Box, Cylinder) abstracts to a typed variable."""
        box = BoxNode(width=_p("w", 2.0), height=_p("h", 2.0), depth=_p("d", 2.0))
        cyl = CylinderNode(radius=_p("r", 1.0), height=_p("h", 4.0))

        lgg, s1, s2 = anti_unify(box, cyl)

        assert isinstance(lgg, VariableNode)
        assert isinstance(lgg, SDFVariableNode)
        assert lgg.domain == "sdf"
        assert lgg.expected_type == "SDFNode"
        assert s1[lgg.name] == box
        assert s2[lgg.name] == cyl

        assert apply_substitution(lgg, s1) == box
        assert apply_substitution(lgg, s2) == cyl

    def test_exact_composite_csg_discovery(self) -> None:
        """Identifies shared structure: Difference(Cylinder, Cylinder) and Difference(Box, Box) -> Difference(?x, ?y)."""
        cyl_outer = CylinderNode(radius=_p("r1", 3.0), height=_p("h1", 5.0))
        cyl_inner = CylinderNode(radius=_p("r2", 1.0), height=_p("h2", 6.0))
        t1 = DifferenceNode(a=cyl_outer, b=cyl_inner)

        box_outer = BoxNode(width=_p("w1", 4.0), height=_p("h1", 4.0), depth=_p("d1", 4.0))
        box_inner = BoxNode(width=_p("w2", 2.0), height=_p("h2", 2.0), depth=_p("d2", 5.0))
        t2 = DifferenceNode(a=box_outer, b=box_inner)

        lgg, s1, s2 = anti_unify(t1, t2)

        assert isinstance(lgg, DifferenceNode)
        assert isinstance(lgg.a, VariableNode)
        assert isinstance(lgg.b, VariableNode)
        assert lgg.a.name != lgg.b.name
        assert lgg.a.name == "?x"
        assert lgg.b.name == "?y"

        assert s1["?x"] == cyl_outer
        assert s1["?y"] == cyl_inner
        assert s2["?x"] == box_outer
        assert s2["?y"] == box_inner

        assert apply_substitution(lgg, s1) == t1
        assert apply_substitution(lgg, s2) == t2

    def test_exact_audio_discovery(self) -> None:
        """Identifies shared structure: Mix(Sine, Saw) and Mix(Sine, Square) -> Mix(Sine, ?x)."""
        shared_ratio = _p("ratio", 0.5)
        shared_sine = SineNode(freq=_p("freq_carrier", 440.0))
        saw = SawNode(freq=_p("freq_saw", 220.0))
        square = SquareNode(freq=_p("freq_sq", 880.0))

        t1 = MixNode(ratio=shared_ratio, input1=shared_sine, input2=saw)
        t2 = MixNode(ratio=shared_ratio, input1=shared_sine, input2=square)

        lgg, s1, s2 = anti_unify(t1, t2)

        assert isinstance(lgg, MixNode)
        assert lgg.ratio == shared_ratio
        assert lgg.input1 == shared_sine
        assert isinstance(lgg.input2, VariableNode)
        assert isinstance(lgg.input2, AudioVariableNode)
        assert lgg.input2.name == "?x"

        assert s1["?x"] == saw
        assert s2["?x"] == square

        assert apply_substitution(lgg, s1) == t1
        assert apply_substitution(lgg, s2) == t2

    def test_repeated_variable_binding(self) -> None:
        """Repeated variable binding: Difference(A, A) and Difference(B, B) -> Difference(?x, ?x)."""
        shape_a = SphereNode(radius=_p("r_a", 2.0))
        shape_b = BoxNode(width=_p("w_b", 1.0), height=_p("h_b", 1.0), depth=_p("d_b", 1.0))

        t1 = DifferenceNode(a=shape_a, b=shape_a)
        t2 = DifferenceNode(a=shape_b, b=shape_b)

        lgg, s1, s2 = anti_unify(t1, t2)

        assert isinstance(lgg, DifferenceNode)
        assert isinstance(lgg.a, VariableNode)
        assert isinstance(lgg.b, VariableNode)
        # Identical differences must map to the same variable
        assert lgg.a.name == lgg.b.name
        assert lgg.a == lgg.b
        assert lgg.a.name == "?x"

        assert len(s1) == 1
        assert len(s2) == 1
        assert s1["?x"] == shape_a
        assert s2["?x"] == shape_b

        assert apply_substitution(lgg, s1) == t1
        assert apply_substitution(lgg, s2) == t2


class TestSubtreeExtractionAndMatching:
    """Tests for subtree candidate harvesting and pattern matching."""

    def test_extract_subtrees_min_size(self) -> None:
        """Verify extract_subtrees filters out single leaf primitives when min_size=2."""
        sphere = SphereNode(radius=_p("r", 1.0))
        box = BoxNode(width=_p("w", 2.0), height=_p("h", 2.0), depth=_p("d", 2.0))
        diff = DifferenceNode(a=sphere, b=box)
        trans = TranslateNode(dx=_p("x", 0.0), dy=_p("y", 1.0), dz=_p("z", 0.0), child=diff)

        # min_size=2 should include trans (size=4) and diff (size=3), but exclude sphere/box (size=1)
        subtrees = extract_subtrees(trans, min_size=2)
        assert len(subtrees) == 2
        assert trans in subtrees
        assert diff in subtrees
        assert sphere not in subtrees
        assert box not in subtrees

    def test_match_pattern_success_and_conflict(self) -> None:
        """Verify pattern matching detects matching instances and rejects conflicting bindings."""
        var_x = SDFVariableNode(name="?x", expected_type="SDFNode", domain="sdf")
        var_y = SDFVariableNode(name="?y", expected_type="SDFNode", domain="sdf")
        pattern = DifferenceNode(a=var_x, b=var_y)

        cyl = CylinderNode(radius=_p("r", 1.0), height=_p("h", 2.0))
        box = BoxNode(width=_p("w", 3.0), height=_p("h", 3.0), depth=_p("d", 3.0))
        target = DifferenceNode(a=cyl, b=box)

        sub = match_pattern(pattern, target)
        assert sub is not None
        assert sub["?x"] == cyl
        assert sub["?y"] == box

        # Repeated variable conflict test: pattern Difference(?x, ?x)
        rep_pattern = DifferenceNode(a=var_x, b=var_x)
        assert match_pattern(rep_pattern, target) is None

        # Repeated variable success: target Difference(cyl, cyl)
        target_rep = DifferenceNode(a=cyl, b=cyl)
        sub_rep = match_pattern(rep_pattern, target_rep)
        assert sub_rep is not None
        assert sub_rep["?x"] == cyl


class TestCorpusPatternMining:
    """Tests for recurrent macro discovery over program corpora."""

    def test_pass_to_complete_gate_pairs_discovery(self) -> None:
        """Pass-to-Complete Gate: Correctly discovers candidate macro-primitives from pairs."""
        # 1. SDF Pair
        cyl1 = CylinderNode(radius=_p("r1", 2.0), height=_p("h1", 5.0))
        cyl2 = CylinderNode(radius=_p("r2", 1.0), height=_p("h2", 6.0))
        sdf_p1 = DifferenceNode(a=cyl1, b=cyl2)

        box1 = BoxNode(width=_p("w1", 4.0), height=_p("h1", 4.0), depth=_p("d1", 4.0))
        box2 = BoxNode(width=_p("w2", 2.0), height=_p("h2", 2.0), depth=_p("d2", 5.0))
        sdf_p2 = DifferenceNode(a=box1, b=box2)

        sdf_patterns = mine_recurrent_patterns([sdf_p1, sdf_p2], min_frequency=2, min_size=2)
        assert len(sdf_patterns) >= 1
        best_sdf = sdf_patterns[0]
        assert isinstance(best_sdf.generalized_ast, DifferenceNode)
        assert best_sdf.frequency == 2
        assert len(best_sdf.occurrences) == 2
        assert len(best_sdf.parameters) == 2

        # 2. Audio Pair
        ratio = _p("mix_ratio", 0.5)
        sine = SineNode(freq=_p("fc", 440.0))
        saw = SawNode(freq=_p("fs", 220.0))
        sq = SquareNode(freq=_p("fq", 880.0))

        audio_p1 = MixNode(ratio=ratio, input1=sine, input2=saw)
        audio_p2 = MixNode(ratio=ratio, input1=sine, input2=sq)

        audio_patterns = mine_recurrent_patterns([audio_p1, audio_p2], min_frequency=2, min_size=2)
        assert len(audio_patterns) >= 1
        best_audio = audio_patterns[0]
        assert isinstance(best_audio.generalized_ast, MixNode)
        assert best_audio.frequency == 2
        assert len(best_audio.occurrences) == 2

    def test_corpus_mining_20_programs_under_2_seconds(self) -> None:
        """Corpus mining: Mines patterns from 20 programs (Audio and SDF) in < 2.0 seconds."""
        corpus: List[ASTNode] = []

        # 10 SDF programs sharing hollow difference or union structures
        for i in range(10):
            r_out = _p(f"rout_{i}", 2.0 + i * 0.1)
            r_in = _p(f"rin_{i}", 1.0 + i * 0.1)
            if i % 2 == 0:
                outer = CylinderNode(radius=r_out, height=_p(f"h_{i}", 5.0))
                inner = CylinderNode(radius=r_in, height=_p(f"h2_{i}", 6.0))
            else:
                outer = BoxNode(width=r_out, height=r_out, depth=r_out)
                inner = SphereNode(radius=r_in)
            prog = DifferenceNode(a=outer, b=inner)
            corpus.append(prog)

        # 10 Audio programs sharing Mix(Sine, ?x) and LowPass filtering
        shared_sine = SineNode(freq=_p("base_sine", 440.0))
        ratio = _p("mix_r", 0.5)
        for j in range(10):
            freq = _p(f"osc_f_{j}", 100.0 + j * 50.0)
            if j % 2 == 0:
                osc = SawNode(freq=freq)
            else:
                osc = SquareNode(freq=freq)
            mix = MixNode(ratio=ratio, input1=shared_sine, input2=osc)
            lp = LowPassNode(cutoff=_p(f"cut_{j}", 1000.0), resonance=_p(f"res_{j}", 0.7), input_node=mix)
            corpus.append(lp)

        assert len(corpus) == 20

        # Measure mining runtime
        t0 = time.perf_counter()
        patterns = mine_recurrent_patterns(
            corpus,
            min_frequency=2,
            min_size=2,
            max_patterns=50,
            max_duration_seconds=2.0,
        )
        elapsed = time.perf_counter() - t0

        # Hard performance invariant
        assert elapsed < 2.0, f"Mining took {elapsed:.3f}s, exceeding 2.0s limit"
        assert len(patterns) >= 2

        # Validate extracted pattern invariants
        for pat in patterns:
            assert pat.frequency >= 2
            assert pat.size >= 2
            assert len(pat.occurrences) >= 2
            # Symmetrically verify that each occurrence matches the referenced program
            for prog_idx, sub in pat.occurrences:
                instantiated = apply_substitution(pat.generalized_ast, sub)
                # Instantiated must be a valid subtree of the target program
                subtrees = extract_subtrees(corpus[prog_idx], min_size=pat.size)
                assert instantiated in subtrees or instantiated == corpus[prog_idx]


class TestSymmetricalCrossValidationAndSerialization:
    """Serialization and symmetric inversion guarantees."""

    def test_variable_node_serialization_roundtrip(self) -> None:
        """Verify serialize_node and deserialize_node round-trip on VariableNode."""
        var = make_variable("?x", "AudioNode", "audio")
        data = serialize_node(var)
        restored = deserialize_node(data)
        assert restored == var
        assert isinstance(restored, AudioVariableNode)
        assert restored.domain == "audio"

    def test_macro_param_str_symmetrical_equality(self) -> None:
        """MacroParamStr matches both untyped '?x' and typed '?x: SDFNode'."""
        p = MacroParamStr("?x", "SDFNode")
        assert p == "?x"
        assert p == "?x: SDFNode"
        assert p.name == "?x"
        assert p.expected_type == "SDFNode"
        assert "?x" in [p]
        assert "?x: SDFNode" in [p]

    def test_invalid_substitution_type_error(self) -> None:
        """apply_substitution raises TypeError if non-ASTNode passed into ASTNode hole."""
        var = VariableNode("?x")
        with pytest.raises(TypeError, match="Cannot substitute non-ASTNode value"):
            apply_substitution(var, {"?x": 42})
