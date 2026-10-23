"""Hardware execution and computational complexity cost model C_H(G) for SymForge AST/DAG programs."""

from __future__ import annotations
from typing import Dict, Optional, Set, Type
from symforge.core.ast import (
    ASTNode,
    AudioNode,
    SineNode,
    SawNode,
    SquareNode,
    NoiseNode,
    ADSRNode,
    LowPassNode,
    GainNode,
    MixNode,
    FMNode,
    SDFNode,
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
    FreeParam,
    HardwareNode,
    ImmHardwareNode,
    RegHardwareNode,
    AddHardwareNode,
    MulHardwareNode,
    AndHardwareNode,
    OrHardwareNode,
    XorHardwareNode,
    ShlHardwareNode,
    ShrHardwareNode,
    HardwareHoleNode,
)
from symforge.core.dag import ast_to_dag, DAGNode


class HardwareCostModel:
    """Estimates the compiled hardware execution cost C_H(G) of a program graph.

    Considers:
    - Base operation cost (FLOPs, transcendental functions like sin/cos/exp, branches)
    - Memory access and register overhead from continuous free parameters
    - DAG common subexpression deduplication
    """

    DEFAULT_NODE_COSTS: Dict[Type[ASTNode], float] = {
        # Audio domain
        SineNode: 5.0,
        SawNode: 3.0,
        SquareNode: 4.0,
        NoiseNode: 2.5,
        ADSRNode: 6.0,
        LowPassNode: 8.0,
        GainNode: 1.5,
        MixNode: 3.0,
        FMNode: 9.0,
        # SDF domain
        SphereNode: 3.0,
        BoxNode: 5.0,
        CylinderNode: 5.0,
        TorusNode: 7.0,
        TranslateNode: 2.0,
        RotateNode: 8.0,
        ScaleNode: 3.0,
        UnionNode: 5.0,
        IntersectionNode: 2.0,
        DifferenceNode: 2.5,
        # Hardware domain
        ImmHardwareNode: 0.5,
        RegHardwareNode: 0.5,
        AddHardwareNode: 1.0,
        MulHardwareNode: 3.0,
        AndHardwareNode: 1.0,
        OrHardwareNode: 1.0,
        XorHardwareNode: 1.0,
        ShlHardwareNode: 1.0,
        ShrHardwareNode: 1.0,
        HardwareHoleNode: 1.0,
    }

    def __init__(
        self,
        node_costs: Optional[Dict[Type[ASTNode], float]] = None,
        param_cost_weight: float = 0.5,
    ):
        self.node_costs = dict(self.DEFAULT_NODE_COSTS)
        if node_costs:
            self.node_costs.update(node_costs)
        self.param_cost_weight = float(param_cost_weight)

    def get_node_base_cost(self, node: ASTNode) -> float:
        """Return base operation cost for an AST node type."""
        for cls, cost in self.node_costs.items():
            if isinstance(node, cls):
                return cost
        return 2.0  # Generic default for unspecified nodes

    def compute_cost(self, node: ASTNode, use_dag_dedup: bool = False) -> float:
        """Compute the total hardware execution cost for an AST.

        Args:
            node: Root AST node.
            use_dag_dedup: If True, deduplicates common subexpressions via DAG canonicalization,
                           reflecting shared hardware execution buffers.

        Returns:
            Scalar hardware cost C_H(G) >= 0.0.
        """
        if use_dag_dedup:
            dag_root = ast_to_dag(node)
            return self._compute_dag_cost(dag_root)
        return self._compute_ast_cost(node)

    def _compute_ast_cost(self, node: ASTNode) -> float:
        base_cost = self.get_node_base_cost(node)
        param_cost = self.param_cost_weight * len(node._free_param_fields())
        children_cost = sum(self._compute_ast_cost(child) for child in node.children)
        return base_cost + param_cost + children_cost

    def _compute_dag_cost(self, root: DAGNode) -> float:
        visited: Set[DAGNode] = set()
        total_cost = 0.0

        def traverse(curr: DAGNode) -> None:
            nonlocal total_cost
            if curr in visited:
                return
            visited.add(curr)

            base_cost = self.get_node_base_cost(curr.ast_template)
            param_cost = self.param_cost_weight * len(curr.ast_template._free_param_fields())
            total_cost += base_cost + param_cost

            for child in curr.children:
                traverse(child)

        traverse(root)
        return total_cost


def compute_program_cost(
    node: ASTNode,
    cost_model: Optional[HardwareCostModel] = None,
    use_dag_dedup: bool = False,
) -> float:
    """Convenience helper to compute the hardware cost C_H(G) of a program."""
    if cost_model is None:
        cost_model = HardwareCostModel()
    return cost_model.compute_cost(node, use_dag_dedup=use_dag_dedup)
