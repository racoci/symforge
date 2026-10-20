"""Tests for Supervised Production Tuple Harvester."""

from __future__ import annotations
import io
import json
import pytest

from symforge.core.ast import (
    ASTNode,
    FreeParam,
    SphereNode,
    BoxNode,
    CylinderNode,
    TranslateNode,
    RotateNode,
    DifferenceNode,
    UnionNode,
    IntersectionNode,
    SineNode,
    FMNode,
    ADSRNode,
    MixNode,
    LowPassNode,
    GainNode,
)
from symforge.core.pcfg import PCFGTaskGenerator
from symforge.data.harvester import ProductionTuple, ProductionTupleHarvester


# ============================================================================
# Harvester Invariant & Tree Traversal Tests
# ============================================================================


def test_harvester_single_leaf_sdf():
    harvester = ProductionTupleHarvester()
    r = FreeParam("r1", 0.1, 10.0, 2.5)
    sphere = SphereNode(radius=r)

    tuples = harvester.harvest(sphere)
    assert len(tuples) == 1
    t0 = tuples[0]

    assert t0.nonterminal == "SDF"
    assert t0.production_id == "sdf_sphere"
    assert "Sphere(radius: FreeParam)" in t0.production
    assert t0.depth_from_root == 0
    assert t0.depth_to_leaves == 0
    assert t0.parent_context == ["Root"]
    assert t0.grandparent_context is None
    assert t0.left_sibling_context is None
    assert t0.ancestor_chain == ["Root"]
    assert t0.expected_type == "SDF"
    assert t0.num_free_params == 1
    assert len(t0.free_params) == 1
    assert t0.free_params[0]["id"] == "r1"
    assert t0.free_params[0]["defaultVal"] == 2.5


def test_harvester_composite_sdf_invariants():
    harvester = ProductionTupleHarvester()

    # Tree structure:
    # Difference (root, depth 0, depth_to_leaves 2)
    #   -> child 0 (a): Translate (depth 1, depth_to_leaves 1)
    #        -> child 0: Box (depth 2, depth_to_leaves 0)
    #   -> child 1 (b): Sphere (depth 1, depth_to_leaves 0)
    dx = FreeParam("dx", -10, 10, 1.0)
    dy = FreeParam("dy", -10, 10, 2.0)
    dz = FreeParam("dz", -10, 10, 3.0)
    w = FreeParam("w", 0.1, 20, 5.0)
    h = FreeParam("h", 0.1, 20, 5.0)
    d = FreeParam("d", 0.1, 20, 5.0)
    box = BoxNode(width=w, height=h, depth=d)
    trans = TranslateNode(dx=dx, dy=dy, dz=dz, child=box)
    r = FreeParam("r", 0.1, 10, 2.0)
    sphere = SphereNode(radius=r)
    diff = DifferenceNode(a=trans, b=sphere)

    tuples = harvester.harvest(diff)
    # Total nodes in AST: Difference(1) + Translate(1) + Box(1) + Sphere(1) = 4
    assert len(tuples) == diff.size == 4

    # Node 0: Difference (Root)
    t_diff = tuples[0]
    assert t_diff.production_id == "sdf_difference"
    assert t_diff.depth_from_root == 0
    assert t_diff.depth_to_leaves == 2
    assert t_diff.parent_context == ["Root"]
    assert t_diff.grandparent_context is None
    assert t_diff.left_sibling_context is None
    assert t_diff.ancestor_chain == ["Root"]
    assert t_diff.num_free_params == 0

    # Node 1: Translate (Child 0 of Difference)
    t_trans = tuples[1]
    assert t_trans.production_id == "sdf_translate"
    assert t_trans.depth_from_root == 1
    assert t_trans.depth_to_leaves == 1
    assert t_trans.parent_context == ["Difference"]
    assert t_trans.grandparent_context == "Root"
    assert t_trans.left_sibling_context is None
    assert t_trans.ancestor_chain == ["Root", "Difference"]
    assert t_trans.num_free_params == 3

    # Node 2: Box (Child 0 of Translate)
    t_box = tuples[2]
    assert t_box.production_id == "sdf_box"
    assert t_box.depth_from_root == 2
    assert t_box.depth_to_leaves == 0
    assert t_box.parent_context == ["Translate"]
    assert t_box.grandparent_context == "Difference"
    assert t_box.left_sibling_context is None
    assert t_box.ancestor_chain == ["Root", "Difference", "Translate"]
    assert t_box.num_free_params == 3

    # Node 3: Sphere (Child 1 of Difference, sibling to Translate)
    t_sphere = tuples[3]
    assert t_sphere.production_id == "sdf_sphere"
    assert t_sphere.depth_from_root == 1
    assert t_sphere.depth_to_leaves == 0
    assert t_sphere.parent_context == ["Difference"]
    assert t_sphere.grandparent_context == "Root"
    assert t_sphere.left_sibling_context == "Translate"
    assert t_sphere.ancestor_chain == ["Root", "Difference"]
    assert t_sphere.num_free_params == 1


def test_harvester_composite_audio_invariants():
    harvester = ProductionTupleHarvester()

    # Audio tree:
    # Mix (root, depth 0)
    #   -> child 0: ADSR (depth 1)
    #        -> gate: Sine (depth 2)
    #   -> child 1: FM (depth 1)
    f_sine = FreeParam("f_sine", 20, 2000, 440)
    p_sine = FreeParam("p_sine", 0, 6.28, 0)
    sine = SineNode(freq=f_sine, phase=p_sine)

    att = FreeParam("att", 0.001, 1.0, 0.01)
    dec = FreeParam("dec", 0.001, 1.0, 0.2)
    sus = FreeParam("sus", 0.0, 1.0, 0.7)
    rel = FreeParam("rel", 0.001, 1.0, 0.3)
    adsr = ADSRNode(attack=att, decay=dec, sustain=sus, release=rel, gate=sine)

    fc = FreeParam("fc", 100, 1000, 300)
    fm = FreeParam("fm", 10, 500, 150)
    mi = FreeParam("mi", 0, 10, 2)
    fm_node = FMNode(carrier=fc, mod_freq=fm, mod_index=mi)

    ratio = FreeParam("ratio", 0, 1, 0.5)
    mix = MixNode(ratio=ratio, input1=adsr, input2=fm_node)

    tuples = harvester.harvest(mix)
    assert len(tuples) == mix.size == 4

    # Mix
    t_mix = tuples[0]
    assert t_mix.nonterminal == "Audio"
    assert t_mix.production_id == "audio_mix"
    assert t_mix.depth_from_root == 0
    assert t_mix.depth_to_leaves == 2

    # ADSR
    t_adsr = tuples[1]
    assert t_adsr.production_id == "audio_adsr"
    assert t_adsr.depth_from_root == 1
    assert t_adsr.depth_to_leaves == 1
    assert t_adsr.parent_context == ["Mix"]
    assert t_adsr.grandparent_context == "Root"
    assert t_adsr.left_sibling_context is None

    # Sine (gate of ADSR)
    t_sine = tuples[2]
    assert t_sine.production_id == "audio_sine"
    assert t_sine.depth_from_root == 2
    assert t_sine.depth_to_leaves == 0
    assert t_sine.parent_context == ["ADSR"]
    assert t_sine.grandparent_context == "Mix"

    # FM (input2 of Mix, sibling of ADSR)
    t_fm = tuples[3]
    assert t_fm.production_id == "audio_fm"
    assert t_fm.depth_from_root == 1
    assert t_fm.depth_to_leaves == 0
    assert t_fm.parent_context == ["Mix"]
    assert t_fm.grandparent_context == "Root"
    assert t_fm.left_sibling_context == "ADSR"


def test_harvester_batch_and_jsonl_export():
    harvester = ProductionTupleHarvester()

    gen_sdf = PCFGTaskGenerator(domain="sdf", seed=42)
    gen_audio = PCFGTaskGenerator(domain="audio", seed=42)

    trees = [gen_sdf.sample(max_depth=3) for _ in range(5)] + [
        gen_audio.sample(max_depth=3) for _ in range(5)
    ]

    tuples = harvester.harvest_batch(trees)
    total_expected = sum(t.size for t in trees)
    assert len(tuples) == total_expected

    # Dict conversion
    dict_list = harvester.to_dict_list(tuples)
    assert len(dict_list) == len(tuples)
    assert all(isinstance(d, dict) for d in dict_list)
    assert all("production_id" in d for d in dict_list)

    # JSONL export to buffer
    buf = io.StringIO()
    harvester.export_jsonl(tuples, buf)
    buf.seek(0)
    lines = buf.getvalue().strip().split("\n")
    assert len(lines) == len(tuples)

    # Validate JSON syntax per line
    for line in lines:
        parsed = json.loads(line)
        assert "nonterminal" in parsed
        assert "production" in parsed
        assert "production_id" in parsed
        assert "depth_from_root" in parsed
        assert "depth_to_leaves" in parsed
        assert "ancestor_chain" in parsed


def test_harvester_frequency_tables_and_cooccurrence():
    harvester = ProductionTupleHarvester()

    gen_sdf = PCFGTaskGenerator(domain="sdf", seed=100)
    trees = [gen_sdf.sample(max_depth=4) for _ in range(15)]
    tuples = harvester.harvest_batch(trees)

    freqs = harvester.compute_production_frequencies(tuples)
    assert "SDF" in freqs
    sdf_freq = freqs["SDF"]
    assert "counts" in sdf_freq
    assert "probabilities" in sdf_freq
    assert "total" in sdf_freq

    # Probabilities should sum to 1.0
    prob_sum = sum(sdf_freq["probabilities"].values())
    assert prob_sum == pytest.approx(1.0, rel=1e-4)

    # Co-occurrence
    cooc = harvester.compute_context_cooccurrence(tuples)
    assert isinstance(cooc, dict)
    # Check that parent contexts exist
    assert any(ctx.startswith("parent:") for ctx in cooc.keys())
    for ctx, prod_counts in cooc.items():
        assert sum(prod_counts.values()) > 0
