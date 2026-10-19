import pytest
from symforge.core.ast import (
    FreeParam,
    SineNode,
    SawNode,
    SquareNode,
    NoiseNode,
    ADSRNode,
    LowPassNode,
    GainNode,
    MixNode,
    FMNode,
    SphereNode,
    BoxNode,
    CylinderNode,
    TorusNode,
    TranslateNode,
    RotateNode,
    ScaleNode,
    UnionNode,
    IntersectionNode,
    DifferenceNode,
    serialize_node,
    deserialize_node,
)


def test_free_param_properties():
    p = FreeParam(id="p0", min_val=0.0, max_val=10.0, default_val=1.0)
    assert p.id == "p0"
    assert p.min_val == 0.0
    assert p.max_val == 10.0
    assert p.default_val == 1.0


def test_audio_ast_construction_and_invariants():
    f0 = FreeParam("f0", 20.0, 20000.0, 440.0)
    saw = SawNode(freq=f0)
    assert saw.size == 1
    assert saw.depth == 1
    assert len(saw.collect_params()) == 1

    fc = FreeParam("fc", 100.0, 10000.0, 1000.0)
    q = FreeParam("q", 0.1, 10.0, 0.707)
    lp = LowPassNode(cutoff=fc, resonance=q, input_node=saw)
    assert lp.size == 2
    assert lp.depth == 2
    params = lp.collect_params()
    assert len(params) == 3
    assert {p.id for p in params} == {"f0", "fc", "q"}


def test_sdf_ast_construction_and_invariants():
    r = FreeParam("r", 0.1, 5.0, 1.0)
    sphere = SphereNode(radius=r)
    w = FreeParam("w", 0.1, 5.0, 0.5)
    h = FreeParam("h", 0.1, 5.0, 0.5)
    d = FreeParam("d", 0.1, 5.0, 0.5)
    box = BoxNode(width=w, height=h, depth=d)

    sm = FreeParam("k", 0.0, 1.0, 0.1)
    union_node = UnionNode(smoothness=sm, a=sphere, b=box)
    assert union_node.size == 3
    assert union_node.depth == 2
    assert len(union_node.collect_params()) == 5


def test_ast_serialization_roundtrip_audio():
    f0 = FreeParam("f0", 20.0, 20000.0, 440.0)
    saw = SawNode(freq=f0)
    fc = FreeParam("fc", 100.0, 10000.0, 1000.0)
    q = FreeParam("q", 0.1, 10.0, 0.707)
    lp = LowPassNode(cutoff=fc, resonance=q, input_node=saw)

    raw_dict = serialize_node(lp)
    recovered = deserialize_node(raw_dict)
    assert recovered == lp
    assert serialize_node(recovered) == raw_dict


def test_ast_serialization_roundtrip_sdf():
    r = FreeParam("r", 0.1, 5.0, 1.0)
    sphere = SphereNode(radius=r)
    dx = FreeParam("dx", -5.0, 5.0, 1.0)
    dy = FreeParam("dy", -5.0, 5.0, 0.0)
    dz = FreeParam("dz", -5.0, 5.0, 0.0)
    translated = TranslateNode(dx=dx, dy=dy, dz=dz, child=sphere)

    raw_dict = serialize_node(translated)
    recovered = deserialize_node(raw_dict)
    assert recovered == translated
    assert serialize_node(recovered) == raw_dict


def test_hole_nodes_and_substitution():
    from symforge.core.ast import (
        AudioHoleNode,
        SDFHoleNode,
        find_holes,
        has_holes,
        substitute_hole,
        update_ast_params,
    )

    h1 = AudioHoleNode("h1")
    assert has_holes(h1)
    assert len(find_holes(h1)) == 1

    fc = FreeParam("fc", 100.0, 10000.0, 1000.0)
    res = FreeParam("res", 0.1, 10.0, 0.707)
    partial_lp = LowPassNode(cutoff=fc, resonance=res, input_node=h1)

    assert has_holes(partial_lp)
    holes = find_holes(partial_lp)
    assert len(holes) == 1
    assert holes[0].hole_id == "h1"

    f0 = FreeParam("f0", 20.0, 20000.0, 440.0)
    saw = SawNode(freq=f0)

    complete_lp = substitute_hole(partial_lp, "h1", saw)
    assert not has_holes(complete_lp)
    assert isinstance(complete_lp, LowPassNode)
    assert complete_lp.input_node == saw

    # Test update_ast_params
    updated = update_ast_params(complete_lp, {"f0": 880.0, "fc": 2000.0})
    assert isinstance(updated, LowPassNode)
    assert updated.cutoff.default_val == 2000.0
    assert updated.input_node.freq.default_val == 880.0

