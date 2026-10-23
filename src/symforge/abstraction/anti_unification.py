"""Subgraph Anti-Unification and Structural Pattern Extraction (Plotkin / Reynolds).

This module implements first-order term anti-unification over SymForge ASTs
to discover Least General Generalizations (LGG) and extract recurrent macro
patterns from solved program corpora.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from symforge.core.ast import (
    ASTNode,
    AudioNode,
    FreeParam,
    HardwareNode,
    SDFNode,
)

# Type alias for variable substitutions
Substitution = Dict[str, Any]


@dataclass(frozen=True)
class VariableNode(ASTNode):
    """Represents an abstracted variable slot in a generalized pattern."""

    name: str
    expected_type: str = "ASTNode"
    domain: str = "generic"

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return ()

    def _free_param_fields(self) -> List[FreeParam]:
        return []

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "variable": {
                "name": self.name,
                "expectedType": self.expected_type,
                "domain": self.domain,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Support serialization if used inside parameter fields."""
        return {
            "id": self.name,
            "variable": True,
            "name": self.name,
            "expectedType": self.expected_type,
            "domain": self.domain,
        }


@dataclass(frozen=True)
class AudioVariableNode(AudioNode, VariableNode):
    """Typed variable node in the audio domain."""
    pass


@dataclass(frozen=True)
class SDFVariableNode(SDFNode, VariableNode):
    """Typed variable node in the SDF domain."""
    pass


@dataclass(frozen=True)
class HardwareVariableNode(HardwareNode, VariableNode):
    """Typed variable node in the hardware domain."""

    def evaluate(self, registers: List[int]) -> int:
        raise RuntimeError(f"Cannot evaluate unbound variable node: {self.name}")


def make_variable(
    name: str,
    expected_type: str = "ASTNode",
    domain: str = "generic",
) -> VariableNode:
    """Factory creating typed variable node based on domain."""
    if domain == "audio" or expected_type == "AudioNode":
        return AudioVariableNode(name=name, expected_type=expected_type, domain="audio")
    if domain == "sdf" or expected_type == "SDFNode":
        return SDFVariableNode(name=name, expected_type=expected_type, domain="sdf")
    if domain == "hardware" or expected_type == "HardwareNode":
        return HardwareVariableNode(name=name, expected_type=expected_type, domain="hardware")
    return VariableNode(name=name, expected_type=expected_type, domain=domain)


class MacroParamStr(str):
    """Parameter descriptor string that behaves symmetrically as a typed or untyped name."""

    name: str
    expected_type: str

    def __new__(cls, name: str, expected_type: str = "ASTNode") -> MacroParamStr:
        val = f"{name}: {expected_type}"
        instance = super().__new__(cls, val)
        instance.name = name
        instance.expected_type = expected_type
        return instance

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return str(self) == other or self.name == other
        return False

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class AbstractionPattern:
    """Represents an invented macro pattern candidate."""

    name: str
    generalized_ast: ASTNode
    parameters: List[str]
    frequency: int
    occurrences: List[Tuple[int, Substitution]]
    size: int

    @property
    def var_names(self) -> List[str]:
        """Extract plain variable names without type annotations."""
        return [p.split(":")[0].strip() for p in self.parameters]


class VariableNameGenerator:
    """Generates unique deterministic variable names (?x, ?y, ?z, ?w, ?u, ?v, ...)."""

    DEFAULT_NAMES = ["?x", "?y", "?z", "?w", "?u", "?v", "?a", "?b", "?c", "?d", "?e", "?f"]

    def __init__(self, prefix: str = "?") -> None:
        self.prefix = prefix
        self.index = 0

    def next_name(self) -> str:
        if self.index < len(self.DEFAULT_NAMES):
            name = self.DEFAULT_NAMES[self.index]
        else:
            name = f"{self.prefix}v{self.index}"
        self.index += 1
        return name


def _infer_domain_and_type(u1: Any, u2: Any) -> Tuple[str, str]:
    """Infer domain and expected type for an abstracted variable slot."""
    if isinstance(u1, AudioNode) and isinstance(u2, AudioNode):
        return "audio", "AudioNode"
    if isinstance(u1, SDFNode) and isinstance(u2, SDFNode):
        return "sdf", "SDFNode"
    if isinstance(u1, HardwareNode) and isinstance(u2, HardwareNode):
        return "hardware", "HardwareNode"
    if isinstance(u1, FreeParam) or isinstance(u2, FreeParam):
        return "param", "FreeParam"
    if isinstance(u1, (int, float)) and isinstance(u2, (int, float)):
        return "param", type(u1).__name__
    return "generic", "ASTNode"


def anti_unify(
    t1: ASTNode,
    t2: ASTNode,
    var_prefix: str = "?",
    max_depth: int = 100,
) -> Tuple[ASTNode, Dict[str, Any], Dict[str, Any]]:
    """Compute the Least General Generalization (LGG) of two AST terms.

    Args:
        t1: First AST term.
        t2: Second AST term.
        var_prefix: Variable name prefix (default '?').
        max_depth: Maximum recursion depth to prevent stack overflow.

    Returns:
        Tuple (generalized_ast, sub1, sub2) where sub1(generalized) == t1
        and sub2(generalized) == t2.
    """
    bindings: Dict[Tuple[Any, Any], VariableNode] = {}
    sub1: Dict[str, Any] = {}
    sub2: Dict[str, Any] = {}
    var_gen = VariableNameGenerator(prefix=var_prefix)

    def _rec(u1: Any, u2: Any, depth: int) -> Any:
        if depth > max_depth:
            raise RuntimeError(
                f"Anti-unification recursion limit exceeded (depth={depth}, u1={type(u1)}, u2={type(u2)})"
            )

        # Invariant 1: Identity property
        if u1 == u2:
            return u1

        # Invariant 2: Reuse variable for identical differences
        pair_key = (u1, u2)
        if pair_key in bindings:
            return bindings[pair_key]

        # Case 3: Both are ASTNodes with matching constructor
        if isinstance(u1, ASTNode) and isinstance(u2, ASTNode) and type(u1) is type(u2):
            field_dict: Dict[str, Any] = {}
            for f in dataclasses.fields(u1):
                v1 = getattr(u1, f.name)
                v2 = getattr(u2, f.name)
                field_dict[f.name] = _rec(v1, v2, depth + 1)
            return type(u1)(**field_dict)

        # Case 4: Operator divergence or parameter mismatch -> abstract into variable
        domain, expected_type = _infer_domain_and_type(u1, u2)
        name = var_gen.next_name()
        var = make_variable(name=name, expected_type=expected_type, domain=domain)

        bindings[pair_key] = var
        sub1[name] = u1
        sub2[name] = u2
        return var

    lgg = _rec(t1, t2, 0)
    return lgg, sub1, sub2


def apply_substitution(node: ASTNode, sub: Substitution) -> ASTNode:
    """Apply a substitution map to replace variables in an AST."""
    if isinstance(node, VariableNode):
        val = sub.get(node.name, node)
        if isinstance(val, ASTNode):
            return val
        raise TypeError(
            f"Cannot substitute non-ASTNode value {val!r} into ASTNode position for variable {node.name}"
        )

    if not isinstance(node, ASTNode):
        return node

    new_fields: Dict[str, Any] = {}
    for fld in dataclasses.fields(node):
        val = getattr(node, fld.name)
        if isinstance(val, VariableNode):
            new_fields[fld.name] = sub.get(val.name, val)
        elif isinstance(val, ASTNode):
            new_fields[fld.name] = apply_substitution(val, sub)
        else:
            new_fields[fld.name] = val
    return type(node)(**new_fields)


def extract_subtrees(
    ast: ASTNode,
    min_size: int = 2,
    max_subtrees: int = 500,
    max_depth: int = 100,
) -> List[ASTNode]:
    """Extract candidate non-trivial subtrees from an AST."""
    subtrees: List[ASTNode] = []
    seen_sigs: Set[str] = set()

    def _collect(curr: ASTNode, depth: int) -> None:
        if depth > max_depth or len(subtrees) >= max_subtrees:
            return

        sig = repr(curr)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            if curr.size >= min_size:
                subtrees.append(curr)

        loop_cap = 100
        iterations = 0
        for child in curr.children:
            iterations += 1
            if iterations > loop_cap:
                raise RuntimeError(
                    f"Child loop safety cap exceeded in extract_subtrees "
                    f"(iterations={iterations}, node={type(curr)})"
                )
            _collect(child, depth + 1)

    _collect(ast, 0)
    return subtrees


def collect_variables(node: ASTNode, max_depth: int = 100) -> List[VariableNode]:
    """Collect all unique VariableNodes in an AST in depth-first order."""
    vars_list: List[VariableNode] = []
    seen: Set[str] = set()

    def _walk(curr: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(curr, VariableNode):
            if curr.name not in seen:
                seen.add(curr.name)
                vars_list.append(curr)
            return

        if not isinstance(curr, ASTNode):
            return

        for fld in dataclasses.fields(curr):
            val = getattr(curr, fld.name)
            if isinstance(val, VariableNode):
                if val.name not in seen:
                    seen.add(val.name)
                    vars_list.append(val)
            elif isinstance(val, ASTNode):
                _walk(val, depth + 1)

    _walk(node, 0)
    return vars_list


def match_pattern(
    pattern: ASTNode,
    target: ASTNode,
    max_depth: int = 100,
) -> Optional[Substitution]:
    """Match a generalized pattern against a target AST subtree."""
    sub: Dict[str, Any] = {}
    if not _match_rec(pattern, target, sub, depth=0, max_depth=max_depth):
        return None
    return sub


def _match_rec(
    pat: Any,
    tgt: Any,
    sub: Dict[str, Any],
    depth: int = 0,
    max_depth: int = 100,
) -> bool:
    if depth > max_depth:
        return False

    if isinstance(pat, VariableNode):
        if pat.domain == "audio" and not isinstance(tgt, AudioNode):
            return False
        if pat.domain == "sdf" and not isinstance(tgt, SDFNode):
            return False
        if pat.domain == "hardware" and not isinstance(tgt, HardwareNode):
            return False
        if pat.domain == "param" and not isinstance(tgt, (FreeParam, float, int)):
            return False

        if pat.name in sub:
            return bool(sub[pat.name] == tgt)
        sub[pat.name] = tgt
        return True

    if type(pat) is not type(tgt):
        return False

    if isinstance(pat, ASTNode):
        for fld in dataclasses.fields(pat):
            p_val = getattr(pat, fld.name)
            t_val = getattr(tgt, fld.name)
            if not _match_rec(p_val, t_val, sub, depth + 1, max_depth):
                return False
        return True

    return bool(pat == tgt)


def _generate_pattern_name(ast: ASTNode, index: int) -> str:
    """Generate an idiomatic macro name for an abstraction pattern."""
    domain = "generic"
    if isinstance(ast, AudioNode):
        domain = "audio"
    elif isinstance(ast, SDFNode):
        domain = "sdf"
    elif isinstance(ast, HardwareNode):
        domain = "hardware"

    op_name = type(ast).__name__.replace("Node", "").lower()
    return f"macro_{domain}_{op_name}_{index}"


def mine_recurrent_patterns(
    programs: List[ASTNode],
    min_frequency: int = 2,
    min_size: int = 2,
    max_patterns: int = 50,
    max_duration_seconds: float = 2.0,
) -> List[AbstractionPattern]:
    """Mine recurrent abstraction patterns across a corpus of solved programs.

    Args:
        programs: List of AST programs from the solved corpus.
        min_frequency: Minimum number of distinct programs instantiating the pattern.
        min_size: Minimum structural node count for candidate patterns.
        max_patterns: Maximum number of patterns to return.
        max_duration_seconds: Hard execution budget in seconds (< 2.0s).

    Returns:
        List of ranked AbstractionPattern candidates.
    """
    if len(programs) < min_frequency:
        return []

    start_time = time.perf_counter()

    # Step 1: Extract candidate subtrees for each program
    prog_subtrees: List[List[ASTNode]] = []
    for p in programs:
        subtrees = extract_subtrees(p, min_size=min_size)
        if not subtrees and p.size >= min_size:
            subtrees = [p]
        prog_subtrees.append(subtrees)

    # Step 2: Pairwise anti-unification over candidate subtrees
    candidates: Dict[str, ASTNode] = {}
    max_pairs = 1000
    pair_count = 0
    n_progs = len(programs)

    for i in range(n_progs):
        for j in range(i + 1, n_progs):
            pair_count += 1
            if pair_count > max_pairs:
                break
            if (time.perf_counter() - start_time) > (max_duration_seconds * 0.5):
                break

            for s1 in prog_subtrees[i]:
                for s2 in prog_subtrees[j]:
                    # Domain optimization: skip cross-domain pairings
                    if (isinstance(s1, AudioNode) and not isinstance(s2, AudioNode)) or \
                       (isinstance(s1, SDFNode) and not isinstance(s2, SDFNode)) or \
                       (isinstance(s1, HardwareNode) and not isinstance(s2, HardwareNode)):
                        continue

                    lgg, _, _ = anti_unify(s1, s2)
                    if lgg.size < min_size or isinstance(lgg, VariableNode):
                        continue

                    sig = repr(lgg)
                    if sig not in candidates:
                        candidates[sig] = lgg
                        if len(candidates) >= max_patterns * 2:
                            break

    # Step 3: Corpus matching and occurrence evaluation
    patterns: List[AbstractionPattern] = []

    for cand in candidates.values():
        if (time.perf_counter() - start_time) > (max_duration_seconds * 0.9):
            break

        occurrences: List[Tuple[int, Substitution]] = []
        matched_progs: Set[int] = set()

        for prog_idx, subtrees in enumerate(prog_subtrees):
            for st in subtrees:
                sub = match_pattern(cand, st)
                if sub is not None:
                    occurrences.append((prog_idx, sub))
                    matched_progs.add(prog_idx)
                    break

        freq = len(matched_progs)
        if freq >= min_frequency:
            cand_vars = collect_variables(cand)
            params = [MacroParamStr(v.name, v.expected_type) for v in cand_vars]
            pat_name = _generate_pattern_name(cand, len(patterns))
            pattern = AbstractionPattern(
                name=pat_name,
                generalized_ast=cand,
                parameters=params,
                frequency=freq,
                occurrences=occurrences,
                size=cand.size,
            )
            patterns.append(pattern)

    # Step 4: Rank by DreamCoder compression heuristic
    patterns.sort(
        key=lambda p: (p.frequency * p.size, p.frequency, p.size),
        reverse=True,
    )
    return patterns[:max_patterns]
