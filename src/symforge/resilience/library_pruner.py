"""Library Bloat Pruning & Dependency Tracking Engine for SymForge.

Prevents combinatorial search bloat by capping the dynamic grammar library size,
tracking macro utility metrics (EMA of usage * savings), constructing a topological
macro dependency graph, and executing safe leaf inlining before eviction.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from symforge.abstraction.grammar_expander import (
    DynamicGrammarLibrary,
    MacroDefinition,
    _clean_param_name,
    generate_lean_macro_def,
)
from symforge.abstraction.anti_unification import VariableNode
from symforge.abstraction.mdl import (
    MacroCallNode,
    make_macro_call,
)
from symforge.core.ast import (
    ASTNode,
    AudioHoleNode,
    FreeParam,
    SDFHoleNode,
)

logger = logging.getLogger(__name__)


def _normalize_macro_name(raw_name: str) -> str:
    """Normalize macro identifier into canonical lowercase snake_case format."""
    clean = raw_name.replace("-", "_").lower()
    if not clean.startswith("macro_"):
        clean = f"macro_{clean}"
    return clean


@dataclasses.dataclass
class MacroUtilityMetrics:
    """Utility metrics and historical usage statistics for a library macro.

    Attributes:
        macro_name: Canonical identifier of the registered macro.
        usage_count: Total cumulative invocations in accepted synthesis solutions.
        recent_savings: AST description length savings achieved in latest cycle.
        last_used_cycle: Meta-controller cycle index when macro was most recently used.
        utility_score: Exponential moving average (EMA) of (usage_count * savings).
    """

    macro_name: str
    usage_count: int = 0
    recent_savings: float = 0.0
    last_used_cycle: int = 0
    utility_score: float = 0.0

    def record_usage(
        self,
        savings: float = 1.0,
        cycle: int = 0,
        alpha: float = 0.2,
    ) -> None:
        """Update metrics and compute the new EMA utility score.

        Args:
            savings: MDL description length savings in this cycle.
            cycle: Current wake-sleep cycle counter.
            alpha: EMA smoothing coefficient (default 0.2).
        """
        self.usage_count += 1
        self.recent_savings = max(0.0, savings)
        self.last_used_cycle = max(self.last_used_cycle, cycle)

        instant_signal = float(self.usage_count) * self.recent_savings
        if self.utility_score == 0.0:
            self.utility_score = instant_signal
        else:
            self.utility_score = (1.0 - alpha) * self.utility_score + alpha * instant_signal


def extract_called_macros(node: Any) -> Set[str]:
    """Recursively collect all macro names invoked within an AST node."""
    called: Set[str] = set()
    if isinstance(node, MacroCallNode):
        called.add(_normalize_macro_name(node.macro_name))
        for v in node.arguments.values():
            called.update(extract_called_macros(v))
        return called

    if not isinstance(node, ASTNode):
        return called

    for fld in dataclasses.fields(node):
        val = getattr(node, fld.name)
        if isinstance(val, ASTNode):
            called.update(extract_called_macros(val))
        elif isinstance(val, (list, tuple)):
            for item in val:
                called.update(extract_called_macros(item))
        elif isinstance(val, dict):
            for item in val.values():
                called.update(extract_called_macros(item))
    return called


class MacroDependencyGraph:
    """Directed dependency graph tracking macro-to-macro invocations.

    Edges:
        A -> B represents 'A is a dependency of B' (i.e. Macro B calls Macro A).
        - dependencies[B] contains A (B depends on A).
        - dependents[A] contains B (A is called by B).
    """

    def __init__(self) -> None:
        self.nodes: Set[str] = set()
        self.dependencies: Dict[str, Set[str]] = {}
        self.dependents: Dict[str, Set[str]] = {}

    def add_macro(self, name: str) -> None:
        """Register a macro vertex in the dependency graph."""
        clean = _normalize_macro_name(name)
        self.nodes.add(clean)
        self.dependencies.setdefault(clean, set())
        self.dependents.setdefault(clean, set())

    def add_dependency(self, dependency: str, dependent: str) -> None:
        """Record that 'dependent' invokes 'dependency' (dependency -> dependent)."""
        dep = _normalize_macro_name(dependency)
        dpt = _normalize_macro_name(dependent)
        self.add_macro(dep)
        self.add_macro(dpt)
        self.dependencies[dpt].add(dep)
        self.dependents[dep].add(dpt)

    def remove_dependency(self, dependency: str, dependent: str) -> None:
        """Remove a dependency edge between dependency and dependent."""
        dep = _normalize_macro_name(dependency)
        dpt = _normalize_macro_name(dependent)
        if dpt in self.dependencies:
            self.dependencies[dpt].discard(dep)
        if dep in self.dependents:
            self.dependents[dep].discard(dpt)

    def get_dependencies(self, name: str) -> Set[str]:
        """Return the set of macros called by 'name'."""
        return set(self.dependencies.get(_normalize_macro_name(name), set()))

    def get_dependents(self, name: str) -> Set[str]:
        """Return the set of macros that call 'name'."""
        return set(self.dependents.get(_normalize_macro_name(name), set()))

    def is_leaf(self, name: str) -> bool:
        """True iff no other macro depends on 'name'."""
        return len(self.get_dependents(name)) == 0

    def topological_sort(self) -> List[str]:
        """Return topological ordering where dependencies precede dependents.

        Raises:
            ValueError: If a cyclic dependency is detected.
        """
        in_degree: Dict[str, int] = {
            node: len(self.dependencies.get(node, set())) for node in self.nodes
        }
        queue: collections.deque[str] = collections.deque(
            sorted([node for node, deg in in_degree.items() if deg == 0])
        )
        sorted_nodes: List[str] = []

        visited_count = 0
        safety_cap = len(self.nodes) + 10
        while queue:
            visited_count += 1
            if visited_count > safety_cap:
                raise RuntimeError("Safety iteration cap exceeded during topological sort")
            node = queue.popleft()
            sorted_nodes.append(node)
            for dpt in sorted(self.dependents.get(node, set())):
                in_degree[dpt] -= 1
                if in_degree[dpt] == 0:
                    queue.append(dpt)

        if len(sorted_nodes) != len(self.nodes):
            remaining = set(self.nodes) - set(sorted_nodes)
            raise ValueError(
                f"Cyclic dependency detected among macros: {sorted(remaining)}"
            )

        return sorted_nodes


def _substitute_node(node: Any, sub: Dict[str, Any]) -> Any:
    """Recursively replace VariableNodes and parameters with provided arguments."""
    if isinstance(node, VariableNode):
        clean_name = _clean_param_name(node.name)
        val = sub.get(clean_name, sub.get(node.name, sub.get(f"?{clean_name}", node)))
        if isinstance(val, (int, float)) and node.expected_type == "FreeParam":
            return FreeParam(
                id=clean_name,
                min_val=0.0,
                max_val=100.0,
                default_val=float(val),
            )
        return val

    if isinstance(node, FreeParam):
        clean_id = _clean_param_name(node.id)
        val = sub.get(clean_id, sub.get(node.id, node))
        if isinstance(val, (int, float)):
            return FreeParam(
                id=node.id,
                min_val=node.min_val,
                max_val=node.max_val,
                default_val=float(val),
            )
        return val

    if isinstance(node, MacroCallNode):
        new_args: Dict[str, Any] = {}
        for k, v in node.arguments.items():
            new_args[k] = _substitute_node(v, sub)
        return make_macro_call(node.macro_name, new_args, domain=node.domain)

    if not isinstance(node, ASTNode):
        return node

    new_fields: Dict[str, Any] = {}
    for fld in dataclasses.fields(node):
        val = getattr(node, fld.name)
        if isinstance(val, ASTNode):
            new_fields[fld.name] = _substitute_node(val, sub)
        elif isinstance(val, FreeParam):
            new_fields[fld.name] = _substitute_node(val, sub)
        elif isinstance(val, (list, tuple)):
            new_fields[fld.name] = type(val)(_substitute_node(item, sub) for item in val)
        elif isinstance(val, dict):
            new_fields[fld.name] = {k: _substitute_node(v, sub) for k, v in val.items()}
        else:
            new_fields[fld.name] = val
    return type(node)(**new_fields)


def inline_macro_in_ast(
    node: ASTNode,
    target_macro_name: str,
    library: DynamicGrammarLibrary,
    max_depth: int = 10,
) -> ASTNode:
    """Replace all calls to target_macro_name in node with its expanded body.

    Args:
        node: The root AST node to transform.
        target_macro_name: The name of the macro to inline.
        library: DynamicGrammarLibrary providing macro definitions.
        max_depth: Maximum recursion depth for nested expansion.

    Returns:
        New ASTNode with target macro invocations safely inlined.
    """
    if max_depth <= 0:
        return node

    target_clean = _normalize_macro_name(target_macro_name)

    if isinstance(node, MacroCallNode):
        # First inline target macro inside argument subtrees
        inlined_args: Dict[str, Any] = {}
        for k, v in node.arguments.items():
            if isinstance(v, ASTNode):
                inlined_args[k] = inline_macro_in_ast(
                    v, target_macro_name, library, max_depth - 1
                )
            else:
                inlined_args[k] = v

        node_macro_clean = _normalize_macro_name(node.macro_name)
        if node_macro_clean == target_clean:
            target_def = (
                library.macros.get(target_clean)
                or DynamicGrammarLibrary._GLOBAL_REGISTRY.get(target_clean)
            )
            if target_def is None:
                raise KeyError(
                    f"Target macro '{target_macro_name}' not found in library for inlining."
                )

            # Build parameter substitution dictionary
            sub: Dict[str, Any] = {}
            for k, v in inlined_args.items():
                clean_k = _clean_param_name(k)
                sub[clean_k] = v
                sub[f"?{clean_k}"] = v
                sub[k] = v

            # Supply defaults for missing parameters
            for param in target_def.parameters:
                if param.name not in sub and f"?{param.name}" not in sub:
                    if param.is_free_param:
                        p_id = f"{target_def.name}_{param.name}"
                        sub[param.name] = FreeParam(
                            id=p_id,
                            min_val=param.min_val,
                            max_val=param.max_val,
                            default_val=param.default_val,
                        )
                        sub[f"?{param.name}"] = sub[param.name]
                    elif param.expected_type == "AudioNode":
                        sub[param.name] = AudioHoleNode(f"hole_{param.name}")
                        sub[f"?{param.name}"] = sub[param.name]
                    elif param.expected_type == "SDFNode":
                        sub[param.name] = SDFHoleNode(f"hole_{param.name}")
                        sub[f"?{param.name}"] = sub[param.name]

            expanded_body = _substitute_node(target_def.body, sub)
            return inline_macro_in_ast(
                expanded_body, target_macro_name, library, max_depth - 1
            )
        else:
            return make_macro_call(
                node.macro_name, inlined_args, domain=node.domain
            )

    if not isinstance(node, ASTNode):
        return node

    new_fields: Dict[str, Any] = {}
    for fld in dataclasses.fields(node):
        val = getattr(node, fld.name)
        if isinstance(val, ASTNode):
            new_fields[fld.name] = inline_macro_in_ast(
                val, target_macro_name, library, max_depth - 1
            )
        elif isinstance(val, (list, tuple)):
            new_fields[fld.name] = type(val)(
                inline_macro_in_ast(item, target_macro_name, library, max_depth - 1)
                if isinstance(item, ASTNode)
                else item
                for item in val
            )
        elif isinstance(val, dict):
            new_fields[fld.name] = {
                k: inline_macro_in_ast(v, target_macro_name, library, max_depth - 1)
                if isinstance(v, ASTNode)
                else v
                for k, v in val.items()
            }
        else:
            new_fields[fld.name] = val

    return type(node)(**new_fields)


class LibraryPruner:
    """Capacity-bounded grammar pruner with dependency tracking and safe inlining."""

    def __init__(
        self,
        max_library_size: int = 40,
        alpha: float = 0.2,
    ) -> None:
        """Initialize LibraryPruner.

        Args:
            max_library_size: Hard cap on registered macros in the library (default 40).
            alpha: EMA smoothing coefficient for utility tracking.
        """
        self.max_library_size = max(1, max_library_size)
        self.alpha = alpha
        self.metrics: Dict[str, MacroUtilityMetrics] = {}

    def record_usage(
        self,
        macro_name: str,
        savings: float = 1.0,
        cycle: int = 0,
    ) -> None:
        """Record macro invocation and update EMA utility score."""
        clean = _normalize_macro_name(macro_name)
        if clean not in self.metrics:
            self.metrics[clean] = MacroUtilityMetrics(macro_name=clean)
        self.metrics[clean].record_usage(
            savings=savings, cycle=cycle, alpha=self.alpha
        )

    def get_metrics(self, macro_name: str) -> MacroUtilityMetrics:
        """Retrieve metrics for a macro, initializing if not present."""
        clean = _normalize_macro_name(macro_name)
        if clean not in self.metrics:
            self.metrics[clean] = MacroUtilityMetrics(macro_name=clean)
        return self.metrics[clean]

    def get_utility_score(self, macro_name: str) -> float:
        """Return the current EMA utility score of a macro."""
        clean = _normalize_macro_name(macro_name)
        if clean in self.metrics:
            return self.metrics[clean].utility_score
        return 0.0

    def build_dependency_graph(
        self,
        library: DynamicGrammarLibrary,
    ) -> MacroDependencyGraph:
        """Construct the macro dependency graph from current library definitions."""
        graph = MacroDependencyGraph()
        for name in library.macros:
            graph.add_macro(name)

        for dependent_name, mdef in library.macros.items():
            called_macros = extract_called_macros(mdef.body)
            for dep_name in called_macros:
                if dep_name in library.macros:
                    graph.add_dependency(
                        dependency=dep_name, dependent=dependent_name
                    )

        return graph

    def prune_to_capacity(
        self,
        library: DynamicGrammarLibrary,
        target_capacity: Optional[int] = None,
    ) -> List[str]:
        """Prune low-utility macros until library size is <= target_capacity.

        If a macro targeted for removal has dependent macros, it is safely
        inlined into all dependent macros prior to unregistration, preserving
        runtime execution invariance Exec(B_inlined) == Exec(B_original).

        Args:
            library: DynamicGrammarLibrary to prune.
            target_capacity: Target size limit (defaults to self.max_library_size).

        Returns:
            List of pruned macro names in eviction order.
        """
        cap = self.max_library_size if target_capacity is None else max(0, target_capacity)
        if len(library.macros) <= cap:
            return []

        pruned_names: List[str] = []
        safety_counter = 0
        max_iterations = len(library.macros) * 3 + 10

        while len(library.macros) > cap:
            safety_counter += 1
            if safety_counter > max_iterations:
                logger.error(
                    "Safety iteration cap exceeded during prune_to_capacity "
                    "(current=%d, target=%d)",
                    len(library.macros),
                    cap,
                )
                break

            graph = self.build_dependency_graph(library)
            current_names = list(library.macros.keys())

            # Sort candidate macros by utility ascending
            # Tie breakers: last_used_cycle ascending, usage_count ascending, name
            current_names.sort(
                key=lambda m: (
                    self.get_utility_score(m),
                    self.get_metrics(m).last_used_cycle,
                    self.get_metrics(m).usage_count,
                    m,
                )
            )

            cand = current_names[0]
            dependents = graph.get_dependents(cand)

            if dependents:
                logger.info(
                    "LibraryPruner: Executing safe inlining for macro '%s' into %d dependents: %s",
                    cand,
                    len(dependents),
                    sorted(dependents),
                )
                for dep_name in sorted(dependents):
                    if dep_name in library.macros:
                        dep_macro = library.macros[dep_name]
                        inlined_body = inline_macro_in_ast(
                            dep_macro.body,
                            cand,
                            library,
                        )
                        dep_macro.body = inlined_body
                        try:
                            dep_macro.lean_code = generate_lean_macro_def(
                                macro_name=dep_macro.name,
                                parameters=dep_macro.parameters,
                                body=dep_macro.body,
                                return_type=dep_macro.return_type,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Could not regenerate lean_code for dependent macro dep_name=%s cand=%s: %s",
                                dep_name,
                                cand,
                                exc,
                            )

            # Unregister cand from library and renormalize PCFG priors
            removed = library.unregister_macro(cand)
            if removed is not None:
                pruned_names.append(cand)
            else:
                # Direct deletion fallback
                library.macros.pop(cand, None)
                pruned_names.append(cand)

            logger.info(
                "LibraryPruner: Evicted macro '%s'. Current library size: %d / %d",
                cand,
                len(library.macros),
                cap,
            )

        return pruned_names
