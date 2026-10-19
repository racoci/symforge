import pytest
from symforge.core.ast import (
    FreeParam,
    SawNode,
    LowPassNode,
    SphereNode,
    BoxNode,
    UnionNode,
    serialize_node,
    deserialize_node,
)
from symforge.core.bridge import LeanBridge


def test_lean_bridge_available():
    bridge = LeanBridge()
    assert bridge.is_available()


def test_lean_bridge_example_fetching():
    bridge = LeanBridge()
    audio_ast = bridge.get_example_audio()
    assert isinstance(audio_ast, LowPassNode)
    assert audio_ast.size >= 2
    assert len(audio_ast.collect_params()) >= 2

    sdf_ast = bridge.get_example_sdf()
    assert isinstance(sdf_ast, UnionNode)
    assert sdf_ast.size >= 2


def test_lean_bridge_ast_validation():
    bridge = LeanBridge()
    f0 = FreeParam("f0", 20.0, 20000.0, 440.0)
    saw = SawNode(freq=f0)
    fc = FreeParam("fc", 100.0, 10000.0, 1000.0)
    q = FreeParam("q", 0.1, 10.0, 0.707)
    lp = LowPassNode(cutoff=fc, resonance=q, input_node=saw)

    val_res = bridge.validate_audio(lp)
    assert val_res["status"] == "ok"
    assert val_res["size"] == 2
    assert val_res["paramCount"] == 3


def test_lean_bridge_1000_roundtrip_identity():
    # Pass-to-complete gate: 100% round-trip serialization pass across 1,000 programs
    bridge = LeanBridge()
    for i in range(1000):
        # Generate varied configurations
        freq_val = 100.0 + (i % 800)
        cutoff_val = 500.0 + (i % 2000)
        saw = SawNode(freq=FreeParam(f"f_{i}", 20.0, 20000.0, freq_val))
        lp = LowPassNode(
            cutoff=FreeParam(f"fc_{i}", 100.0, 10000.0, cutoff_val),
            resonance=FreeParam(f"q_{i}", 0.1, 10.0, 1.0),
            input_node=saw,
        )
        serialized = serialize_node(lp)
        deserialized = deserialize_node(serialized)
        assert deserialized == lp
