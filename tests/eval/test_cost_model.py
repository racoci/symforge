import pytest
from symforge.core.ast import (
    FreeParam,
    SineNode,
    SawNode,
    LowPassNode,
    MixNode,
    SphereNode,
    UnionNode,
)
from symforge.eval.cost import HardwareCostModel, compute_program_cost


def test_hardware_cost_monotonicity_audio():
    cost_model = HardwareCostModel()

    f0 = FreeParam("f0", 20.0, 1000.0, 440.0)
    saw = SawNode(freq=f0)
    cost_base = cost_model.compute_cost(saw)

    fc = FreeParam("fc", 100.0, 5000.0, 1000.0)
    q = FreeParam("q", 0.1, 5.0, 0.707)
    lp = LowPassNode(cutoff=fc, resonance=q, input_node=saw)
    cost_composite = cost_model.compute_cost(lp)

    # Monotonicity invariant: adding operations increases hardware cost
    assert cost_composite > cost_base


def test_hardware_cost_monotonicity_sdf():
    cost_model = HardwareCostModel()

    r1 = FreeParam("r1", 0.1, 5.0, 1.0)
    sphere1 = SphereNode(radius=r1)
    cost_sphere = cost_model.compute_cost(sphere1)

    r2 = FreeParam("r2", 0.1, 5.0, 0.5)
    sphere2 = SphereNode(radius=r2)
    sm = FreeParam("k", 0.0, 1.0, 0.1)
    union_node = UnionNode(smoothness=sm, a=sphere1, b=sphere2)
    cost_union = cost_model.compute_cost(union_node)

    assert cost_union > cost_sphere


def test_hardware_cost_dag_deduplication():
    cost_model = HardwareCostModel()

    f0 = FreeParam("f0", 20.0, 1000.0, 440.0)
    saw = SawNode(freq=f0)
    ratio = FreeParam("r", 0.0, 1.0, 0.5)
    # Both branches of MixNode are the exact same saw subexpression
    mix = MixNode(ratio=ratio, input1=saw, input2=saw)

    tree_cost = cost_model.compute_cost(mix, use_dag_dedup=False)
    dag_cost = cost_model.compute_cost(mix, use_dag_dedup=True)

    # DAG deduplication must reduce hardware execution cost for repeated subexpressions
    assert dag_cost < tree_cost
    # The difference should exactly equal the cost of the deduplicated saw node
    saw_cost = cost_model.compute_cost(saw)
    assert abs((tree_cost - dag_cost) - saw_cost) < 1e-5


def test_compute_program_cost_convenience():
    f0 = FreeParam("f0", 20.0, 1000.0, 440.0)
    saw = SawNode(freq=f0)

    cost1 = compute_program_cost(saw)
    cost2 = HardwareCostModel().compute_cost(saw)
    assert cost1 == cost2
