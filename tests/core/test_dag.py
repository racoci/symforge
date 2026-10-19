import pytest
from symforge.core.ast import FreeParam, SphereNode, BoxNode, UnionNode
from symforge.core.dag import ast_to_dag, dag_to_ast, common_subexpression_elimination


def test_ast_to_dag_deduplication():
    # If two branches use identical subtrees, DAG representation should share identity
    r = FreeParam("r", 0.1, 5.0, 1.0)
    sphere1 = SphereNode(radius=r)
    sphere2 = SphereNode(radius=r)
    sm = FreeParam("k", 0.0, 1.0, 0.1)
    tree = UnionNode(smoothness=sm, a=sphere1, b=sphere2)

    dag = ast_to_dag(tree)
    # Both children of union in DAG should point to the exact same object reference
    assert dag.children[0] is dag.children[1]

    # Converting back to AST preserves semantic equality
    recovered_ast = dag_to_ast(dag)
    assert recovered_ast == tree


def test_cse_subgraph_detection():
    r = FreeParam("r", 0.1, 5.0, 1.0)
    sphere = SphereNode(radius=r)
    w = FreeParam("w", 0.1, 5.0, 0.5)
    h = FreeParam("h", 0.1, 5.0, 0.5)
    d = FreeParam("d", 0.1, 5.0, 0.5)
    box = BoxNode(width=w, height=h, depth=d)

    sm = FreeParam("k", 0.0, 1.0, 0.1)
    sub = UnionNode(smoothness=sm, a=sphere, b=box)

    # Composite tree with repeating subgraphs
    root = UnionNode(smoothness=sm, a=sub, b=sub)
    dag = ast_to_dag(root)
    assert dag.children[0] is dag.children[1]
    cses = common_subexpression_elimination(dag)
    assert len(cses) >= 1
