"""Canonical Directed Acyclic Graph (DAG) representation and Common Subexpression Elimination."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple
from symforge.core.ast import ASTNode, FreeParam, deserialize_node, serialize_node


@dataclass
class DAGNode:
    """Canonical node in a directed acyclic graph."""

    ast_template: ASTNode
    children: Tuple[DAGNode, ...]
    signature: str

    def __hash__(self) -> int:
        return hash(self.signature)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DAGNode):
            return False
        return self.signature == other.signature


def ast_to_dag(node: ASTNode, memo: Dict[str, DAGNode] | None = None) -> DAGNode:
    """Convert an AST into a canonical deduplicated DAG."""
    if memo is None:
        memo = {}

    child_dags: List[DAGNode] = []
    # Hard iteration cap for children loop
    loop_cap = 1000
    iterations = 0
    for child in node.children:
        iterations += 1
        if iterations > loop_cap:
            raise RuntimeError(f"Safety iteration cap exceeded during AST->DAG conversion (iterations={iterations})")
        child_dags.append(ast_to_dag(child, memo))

    # Compute deterministic structural signature
    node_dict = serialize_node(node)
    tag = next(iter(node_dict.keys()))

    # Build signature including children signatures
    child_sigs = ",".join(c.signature for c in child_dags)
    params_str = ",".join(f"{p.id}:{p.default_val}" for p in node._free_param_fields())
    sig = f"{tag}(params=[{params_str}],children=[{child_sigs}])"

    if sig in memo:
        return memo[sig]

    dag_node = DAGNode(
        ast_template=node,
        children=tuple(child_dags),
        signature=sig,
    )
    memo[sig] = dag_node
    return dag_node


def dag_to_ast(dag: DAGNode) -> ASTNode:
    """Reconstruct an AST tree from a DAG node."""
    # We can serialize the template node's fields and re-attach child ASTs
    raw_dict = serialize_node(dag.ast_template)
    tag, fields = next(iter(raw_dict.items()))

    # If leaf node
    if not dag.children:
        return deserialize_node(raw_dict)

    reconstructed_children = [dag_to_ast(c) for c in dag.children]

    # Map reconstructed children into fields
    new_fields = dict(fields)
    if tag in ("translate", "rotate", "scale"):
        new_fields["child"] = serialize_node(reconstructed_children[0])
    elif tag in ("union", "smoothUnion", "intersection", "difference"):
        new_fields["a"] = serialize_node(reconstructed_children[0])
        new_fields["b"] = serialize_node(reconstructed_children[1])
    elif tag in ("adsr", "lowpass", "gain", "expDecay"):
        child_key = "gate" if tag == "adsr" else "input"
        new_fields[child_key] = serialize_node(reconstructed_children[0])
    elif tag == "mix":
        new_fields["input1"] = serialize_node(reconstructed_children[0])
        new_fields["input2"] = serialize_node(reconstructed_children[1])

    return deserialize_node({tag: new_fields})


def common_subexpression_elimination(root: DAGNode) -> List[DAGNode]:
    """Find all subgraphs with multiple incoming references (CSE candidates)."""
    counts: Dict[str, int] = {}
    node_map: Dict[str, DAGNode] = {}

    def traverse(node: DAGNode, visited: Set[str]) -> None:
        counts[node.signature] = counts.get(node.signature, 0) + 1
        node_map[node.signature] = node
        if node.signature in visited:
            return
        visited.add(node.signature)
        for child in node.children:
            traverse(child, visited)

    traverse(root, set())

    # Filter to non-trivial subgraphs (having children) referenced more than once
    candidates: List[DAGNode] = []
    for sig, count in counts.items():
        if count > 1 and len(node_map[sig].children) > 0:
            candidates.append(node_map[sig])

    return candidates
