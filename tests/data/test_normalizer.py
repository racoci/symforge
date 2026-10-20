"""Tests for OpenSCAD and Dexed domain code normalizers."""

from __future__ import annotations
import json
import math
import pytest
import torch

from symforge.core.ast import (
    ASTNode,
    AudioNode,
    SDFNode,
    SphereNode,
    BoxNode,
    CylinderNode,
    TranslateNode,
    RotateNode,
    ScaleNode,
    UnionNode,
    DifferenceNode,
    IntersectionNode,
    ADSRNode,
    FMNode,
    MixNode,
    SineNode,
    serialize_node,
    deserialize_node,
)
from symforge.core.transpiler import transpile_ast_to_module
from symforge.data.normalizer import OpenSCADNormalizer, DexedPatchNormalizer


# ============================================================================
# OpenSCADNormalizer Tests
# ============================================================================


def test_openscad_primitives_parsing():
    normalizer = OpenSCADNormalizer()

    # Cube scalar
    ast1 = normalizer.normalize("cube(10);")
    assert isinstance(ast1, BoxNode)
    assert ast1.width.default_val == pytest.approx(10.0)
    assert ast1.height.default_val == pytest.approx(10.0)
    assert ast1.depth.default_val == pytest.approx(10.0)

    # Cube vector
    ast2 = normalizer.normalize("cube([10, 20, 30]);")
    assert isinstance(ast2, BoxNode)
    assert ast2.width.default_val == pytest.approx(10.0)
    assert ast2.height.default_val == pytest.approx(20.0)
    assert ast2.depth.default_val == pytest.approx(30.0)

    # Cube named parameters
    ast3 = normalizer.normalize("cube(size=[5.5, 6.5, 7.5], center=true);")
    assert isinstance(ast3, BoxNode)
    assert ast3.width.default_val == pytest.approx(5.5)
    assert ast3.height.default_val == pytest.approx(6.5)
    assert ast3.depth.default_val == pytest.approx(7.5)

    # Sphere scalar
    ast4 = normalizer.normalize("sphere(7);")
    assert isinstance(ast4, SphereNode)
    assert ast4.radius.default_val == pytest.approx(7.0)

    # Sphere named r
    ast5 = normalizer.normalize("sphere(r=8.5);")
    assert isinstance(ast5, SphereNode)
    assert ast5.radius.default_val == pytest.approx(8.5)

    # Sphere named d (diameter)
    ast6 = normalizer.normalize("sphere(d=16.0);")
    assert isinstance(ast6, SphereNode)
    assert ast6.radius.default_val == pytest.approx(8.0)

    # Cylinder named r, h
    ast7 = normalizer.normalize("cylinder(h=25, r=5);")
    assert isinstance(ast7, CylinderNode)
    assert ast7.height.default_val == pytest.approx(25.0)
    assert ast7.radius.default_val == pytest.approx(5.0)

    # Cylinder positional h, r1, r2
    ast8 = normalizer.normalize("cylinder(30, 4, 4);")
    assert isinstance(ast8, CylinderNode)
    assert ast8.height.default_val == pytest.approx(30.0)
    assert ast8.radius.default_val == pytest.approx(4.0)

    # Cylinder with diameter d
    ast9 = normalizer.normalize("cylinder(h=15, d=10);")
    assert isinstance(ast9, CylinderNode)
    assert ast9.height.default_val == pytest.approx(15.0)
    assert ast9.radius.default_val == pytest.approx(5.0)


def test_openscad_transforms_parsing():
    normalizer = OpenSCADNormalizer()

    # Translate
    scad_trans = "translate([10, -5, 2.5]) cube(4);"
    ast = normalizer.normalize(scad_trans)
    assert isinstance(ast, TranslateNode)
    assert ast.dx.default_val == pytest.approx(10.0)
    assert ast.dy.default_val == pytest.approx(-5.0)
    assert ast.dz.default_val == pytest.approx(2.5)
    assert isinstance(ast.child, BoxNode)

    # Rotate (degrees converted to radians)
    scad_rot = "rotate([0, 90, 180]) sphere(5);"
    ast_rot = normalizer.normalize(scad_rot)
    assert isinstance(ast_rot, RotateNode)
    assert ast_rot.rx.default_val == pytest.approx(0.0)
    assert ast_rot.ry.default_val == pytest.approx(math.radians(90))
    assert ast_rot.rz.default_val == pytest.approx(math.radians(180))
    assert isinstance(ast_rot.child, SphereNode)

    # Scale scalar and vector
    scad_scale = "scale([2, 2, 2]) cylinder(h=10, r=3);"
    ast_scale = normalizer.normalize(scad_scale)
    assert isinstance(ast_scale, ScaleNode)
    assert ast_scale.s.default_val == pytest.approx(2.0)

    # Chained transforms
    scad_chain = "translate([1, 2, 3]) rotate([0, 45, 0]) scale(1.5) sphere(4);"
    ast_chain = normalizer.normalize(scad_chain)
    assert isinstance(ast_chain, TranslateNode)
    assert isinstance(ast_chain.child, RotateNode)
    assert isinstance(ast_chain.child.child, ScaleNode)
    assert isinstance(ast_chain.child.child.child, SphereNode)


def test_openscad_csg_combinators_and_nesting():
    normalizer = OpenSCADNormalizer()

    # Union
    scad_union = "union() { cube(10); sphere(5); }"
    ast_union = normalizer.normalize(scad_union)
    assert isinstance(ast_union, UnionNode)
    assert isinstance(ast_union.a, BoxNode)
    assert isinstance(ast_union.b, SphereNode)

    # Difference
    scad_diff = "difference() { cube(20); sphere(8); }"
    ast_diff = normalizer.normalize(scad_diff)
    assert isinstance(ast_diff, DifferenceNode)
    assert isinstance(ast_diff.a, BoxNode)
    assert isinstance(ast_diff.b, SphereNode)

    # Intersection
    scad_inter = "intersection() { cube(15); cylinder(h=20, r=6); }"
    ast_inter = normalizer.normalize(scad_inter)
    assert isinstance(ast_inter, IntersectionNode)
    assert isinstance(ast_inter.a, BoxNode)
    assert isinstance(ast_inter.b, CylinderNode)

    # Multi-child fold (N > 2)
    scad_multi = "difference() { cube(30); sphere(10); cylinder(h=40, r=4); }"
    ast_multi = normalizer.normalize(scad_multi)
    assert isinstance(ast_multi, DifferenceNode)
    # Binary folding: Difference(Difference(cube, sphere), cylinder)
    assert isinstance(ast_multi.a, DifferenceNode)
    assert isinstance(ast_multi.a.a, BoxNode)
    assert isinstance(ast_multi.a.b, SphereNode)
    assert isinstance(ast_multi.b, CylinderNode)


def test_openscad_robustness_comments_and_variables():
    normalizer = OpenSCADNormalizer()

    scad_code = """
    // Global parameters
    $fn = 64;
    side = 20;
    hole_radius = side / 4; // 5.0
    height = 2 * side;      // 40.0

    /* Multi-line comment block
       describing CSG operation */
    difference() {
        translate([0, 0, 10]) {
            cube([side, side, height]);
        }
        // Subtracted core
        cylinder(r=hole_radius, h=height + 10);
    }
    """
    ast = normalizer.normalize(scad_code)
    assert isinstance(ast, DifferenceNode)
    assert isinstance(ast.a, TranslateNode)
    assert isinstance(ast.a.child, BoxNode)
    assert ast.a.child.width.default_val == pytest.approx(20.0)
    assert ast.a.child.height.default_val == pytest.approx(20.0)
    assert ast.a.child.depth.default_val == pytest.approx(40.0)

    assert isinstance(ast.b, CylinderNode)
    assert ast.b.radius.default_val == pytest.approx(5.0)
    assert ast.b.height.default_val == pytest.approx(50.0)


def test_openscad_symmetrical_transpilation_and_eval():
    normalizer = OpenSCADNormalizer()
    scad_code = """
    difference() {
        cube([10, 10, 10]);
        translate([5, 5, 5]) sphere(r=4);
    }
    """
    ast = normalizer.normalize(scad_code)
    assert isinstance(ast, SDFNode)

    # Symmetrical validation: transpile to differentiable PyTorch module
    module = transpile_ast_to_module(ast)
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
        [15.0, 15.0, 15.0],
    ], dtype=torch.float32)

    distances = module(points)
    assert distances.shape == (3,)
    assert not torch.isnan(distances).any()

    # Differentiable backward gradient flow
    loss = distances.sum()
    loss.backward()
    assert module.theta.grad is not None
    assert torch.isfinite(module.theta.grad).all()


# ============================================================================
# DexedPatchNormalizer Tests
# ============================================================================


def test_dexed_patch_algorithm_1_and_envelopes():
    normalizer = DexedPatchNormalizer()

    # Alg 1: Carriers are Op 1 and Op 3 (or Op 1 and Op 4 depending on numbering)
    # Mixed output with ADSR envelopes
    patch_data = {
        "name": "E.PIANO 1",
        "algorithm": 1,
        "operators": [
            {
                "operator": 1,
                "coarse": 1,
                "fine": 0,
                "output_level": 99,
                "attack": 0.01,
                "decay": 0.3,
                "sustain": 0.7,
                "release": 0.5,
            },
            {
                "operator": 2,
                "coarse": 2,
                "fine": 5,
                "output_level": 80,
            },
            {
                "operator": 3,
                "coarse": 1,
                "fine": 0,
                "output_level": 90,
                "attack": 0.02,
                "decay": 0.4,
                "sustain": 0.5,
                "release": 0.8,
            },
            {
                "operator": 4,
                "coarse": 3,
                "fine": 0,
                "output_level": 70,
            },
            {
                "operator": 5,
                "coarse": 1,
                "fine": 0,
                "output_level": 60,
            },
            {
                "operator": 6,
                "coarse": 4,
                "fine": 0,
                "output_level": 50,
            },
        ],
    }

    ast = normalizer.normalize(patch_data)
    assert isinstance(ast, AudioNode)

    # In Alg 1, multiple carriers are mixed via MixNode
    assert isinstance(ast, MixNode)
    assert isinstance(ast.input1, ADSRNode)
    assert isinstance(ast.input2, ADSRNode)

    # Carrier 1 is modulated by Op 2 -> FMNode
    assert isinstance(ast.input1.gate, FMNode)
    assert ast.input1.gate.carrier.default_val > 0.0
    assert ast.input1.gate.mod_freq.default_val > 0.0
    assert ast.input1.gate.mod_index.default_val > 0.0


def test_dexed_patch_dx7_eg_rate_level_conversion():
    normalizer = DexedPatchNormalizer()

    patch_data = {
        "name": "BRASS 1",
        "algorithm": 5,
        "operators": [
            {
                "operator": 1,
                "coarse": 1,
                "fine": 0,
                "output_level": 99,
                "eg_rate": [95, 20, 20, 50],
                "eg_level": [99, 95, 80, 0],
            },
            {
                "operator": 2,
                "coarse": 1,
                "fine": 0,
                "output_level": 75,
            },
            {
                "operator": 3,
                "coarse": 2,
                "fine": 0,
                "output_level": 85,
                "eg_rate": [90, 30, 10, 40],
                "eg_level": [99, 80, 50, 0],
            },
            {
                "operator": 4,
                "coarse": 1,
                "fine": 0,
                "output_level": 70,
            },
            {
                "operator": 5,
                "coarse": 1,
                "fine": 0,
                "output_level": 90,
                "eg_rate": [99, 50, 50, 60],
                "eg_level": [99, 90, 70, 0],
            },
            {
                "operator": 6,
                "coarse": 3,
                "fine": 0,
                "output_level": 60,
            },
        ],
    }

    ast = normalizer.normalize(patch_data)
    assert isinstance(ast, AudioNode)
    # Check that ADSR envelope parameters were converted into valid seconds/levels
    params = ast.collect_params()
    param_names = [p.id for p in params]
    assert any("attack" in name for name in param_names)
    assert any("decay" in name for name in param_names)
    assert any("sustain" in name for name in param_names)
    assert any("release" in name for name in param_names)

    # Check bounds
    for p in params:
        assert p.min_val <= p.default_val <= p.max_val


def test_dexed_patch_algorithm_32_additive():
    normalizer = DexedPatchNormalizer()

    # Alg 32: All 6 operators are carriers mixed together
    patch_data = {
        "algorithm": 32,
        "operators": [
            {"operator": i, "coarse": i, "fine": 0, "output_level": 90}
            for i in range(1, 7)
        ],
    }

    ast = normalizer.normalize(patch_data)
    assert isinstance(ast, AudioNode)
    assert isinstance(ast, MixNode)

    # All unmodulated carriers become SineNode (or ADSR wrapped SineNode)
    params = ast.collect_params()
    assert len(params) >= 6


def test_dexed_patch_simplified_preset():
    normalizer = DexedPatchNormalizer()

    preset = {
        "carrier": {
            "freq": 440.0,
            "attack": 0.05,
            "decay": 0.2,
            "sustain": 0.6,
            "release": 0.4,
        },
        "modulator": {
            "freq": 220.0,
            "index": 3.0,
        },
    }

    ast = normalizer.normalize(preset)
    assert isinstance(ast, ADSRNode)
    assert isinstance(ast.gate, FMNode)
    assert ast.gate.carrier.default_val == pytest.approx(440.0)
    assert ast.gate.mod_freq.default_val == pytest.approx(220.0)
    assert ast.gate.mod_index.default_val == pytest.approx(3.0)


def test_dexed_symmetrical_transpilation_and_lean_roundtrip():
    normalizer = DexedPatchNormalizer()

    preset = {
        "carrier": {"freq": 330.0, "attack": 0.02, "decay": 0.1, "sustain": 0.8, "release": 0.3},
        "modulator": {"freq": 165.0, "index": 2.0},
    }
    ast = normalizer.normalize(preset)

    # Symmetrical validation 1: Transpile and evaluate audio module
    module = transpile_ast_to_module(ast)
    t = torch.linspace(0.0, 0.1, 500, dtype=torch.float32)
    y = module(t)
    assert y.shape == (500,)
    assert not torch.isnan(y).any()

    loss = y.sum()
    loss.backward()
    assert module.theta.grad is not None
    assert torch.isfinite(module.theta.grad).all()

    # Symmetrical validation 2: Lean 4 serialization and deserialization
    lean_dict = serialize_node(ast)
    reconstructed = deserialize_node(lean_dict)
    assert reconstructed.size == ast.size
    assert reconstructed.tree_depth == ast.tree_depth
