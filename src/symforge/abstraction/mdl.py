"""Minimum Description Length (MDL) Compression Evaluator (DreamCoder).

This module implements the Minimum Description Length (MDL) compression metric
for program synthesis library learning. It evaluates whether candidate
abstractions discovered via anti-unification provide net description length
compression over a solved program corpus, strictly pruning unhelpful or
single-use macros to prevent library bloat.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from symforge.abstraction.anti_unification import (
    AbstractionPattern,
    VariableNode,
    match_pattern,
)
from symforge.core.ast import (
    ASTNode,
    AudioNode,
    FreeParam,
    HardwareNode,
    HoleNode,
    SDFNode,
)


@dataclass(frozen=True)
class MDLConfig:
    """Configuration parameters for Minimum Description Length (MDL) evaluation.

    Attributes:
        symbol_base_cost: Base cost in bits/tokens for introducing a new DSL production name.
        node_cost: Cost per operator or primitive constructor in an AST.
        param_cost: Cost per variable argument or continuous parameter slot.
        threshold: Acceptance cutoff. Candidate is accepted iff delta_mdl < threshold.
        max_depth: Maximum recursion depth for AST traversals.
        max_iterations: Safety cap for iterative corpus evaluations.
    """

    symbol_base_cost: float = 2.0
    node_cost: float = 1.0
    param_cost: float = 1.0
    threshold: float = 0.0
    max_depth: int = 100
    max_iterations: int = 1000


@dataclass
class MDLScore:
    """Evaluation score and compression metrics for an abstraction pattern candidate.

    Attributes:
        pattern_name: Identifier of the candidate abstraction.
        dsl_expansion_cost: Size of the macro definition added to the library (Size(alpha)).
        corpus_savings: Total description length saved across all matching programs.
        delta_mdl: Net description length change (Size(alpha) - corpus_savings).
        compression_ratio: Relative corpus compression (L(D) / (L(D) + delta_mdl)).
        is_accepted: True iff delta_mdl < threshold (strictly compresses corpus).
        matched_programs_count: Number of distinct programs containing the pattern.
        total_occurrences: Total times the pattern appears across the corpus.
    """

    pattern_name: str
    dsl_expansion_cost: float
    corpus_savings: float
    delta_mdl: float
    compression_ratio: float
    is_accepted: bool
    matched_programs_count: int
    total_occurrences: int


@dataclass(frozen=True)
class MacroCallNode(ASTNode):
    """Represents an instantiated invocation of a library macro."""

    macro_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    domain: str = "generic"

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return tuple(v for v in self.arguments.values() if isinstance(v, ASTNode))

    def _free_param_fields(self) -> List[FreeParam]:
        return [v for v in self.arguments.values() if isinstance(v, FreeParam)]

    def to_lean_dict(self) -> Dict[str, Any]:
        serialized_args: Dict[str, Any] = {}
        for k, v in self.arguments.items():
            if isinstance(v, ASTNode):
                serialized_args[k] = v.to_lean_dict()
            elif isinstance(v, FreeParam):
                serialized_args[k] = v.to_dict()
            else:
                serialized_args[k] = v

        return {
            "macroCall": {
                "name": self.macro_name,
                "arguments": serialized_args,
                "domain": self.domain,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.to_lean_dict()


@dataclass(frozen=True)
class AudioMacroCallNode(AudioNode, MacroCallNode):
    """Typed macro call node residing within the audio domain."""
    pass


@dataclass(frozen=True)
class SDFMacroCallNode(SDFNode, MacroCallNode):
    """Typed macro call node residing within the SDF domain."""
    pass


@dataclass(frozen=True)
class HardwareMacroCallNode(HardwareNode, MacroCallNode):
    """Typed macro call node residing within the hardware domain."""

    def evaluate(self, registers: List[int]) -> int:
        raise RuntimeError(f"Cannot directly evaluate unexpanded HardwareMacroCallNode: {self.macro_name}")


def make_macro_call(
    macro_name: str,
    arguments: Dict[str, Any],
    domain: str = "generic",
) -> MacroCallNode:
    """Factory creating a domain-typed MacroCallNode."""
    if domain == "audio":
        return AudioMacroCallNode(macro_name=macro_name, arguments=arguments, domain="audio")
    if domain == "sdf":
        return SDFMacroCallNode(macro_name=macro_name, arguments=arguments, domain="sdf")
    if domain == "hardware":
        return HardwareMacroCallNode(macro_name=macro_name, arguments=arguments, domain="hardware")
    return MacroCallNode(macro_name=macro_name, arguments=arguments, domain=domain)


def _infer_pattern_domain(ast: ASTNode) -> str:
    """Infers the domain ('audio', 'sdf', 'hardware', or 'generic') of a pattern AST."""
    if isinstance(ast, AudioNode):
        return "audio"
    if isinstance(ast, SDFNode):
        return "sdf"
    if isinstance(ast, HardwareNode):
        return "hardware"
    return "generic"


def compute_ast_size(ast: ASTNode, config: MDLConfig = MDLConfig()) -> float:
    """Calculates discrete description length of an AST under the MDL cost model.

    Args:
        ast: Root node of the AST.
        config: MDLConfig specifying symbol, node, and param costs.

    Returns:
        Discrete description length (float).
    """
    def _calc(curr: Any, depth: int) -> float:
        if depth > config.max_depth:
            raise RuntimeError(
                f"AST recursion depth limit exceeded in compute_ast_size "
                f"(depth={depth}, node_type={type(curr).__name__})"
            )

        if isinstance(curr, (VariableNode, HoleNode)):
            return config.param_cost

        if isinstance(curr, MacroCallNode):
            cost = config.node_cost
            for arg_val in curr.arguments.values():
                if isinstance(arg_val, ASTNode):
                    cost += _calc(arg_val, depth + 1)
                else:
                    cost += config.param_cost
            return cost

        if not isinstance(curr, ASTNode):
            return config.param_cost if isinstance(curr, (FreeParam, int, float)) else 0.0

        cost = config.node_cost
        for param in curr._free_param_fields():
            if isinstance(param, ASTNode):
                cost += _calc(param, depth + 1)
            else:
                cost += config.param_cost

        for child in curr.children:
            cost += _calc(child, depth + 1)

        return cost

    return _calc(ast, 0)


def compute_macro_cost(pattern: AbstractionPattern, config: MDLConfig = MDLConfig()) -> float:
    """Calculates the cost of registering a macro in the DSL library.

    Args:
        pattern: Candidate AbstractionPattern.
        config: MDLConfig with base symbol and node costs.

    Returns:
        Registration cost in description length units.
    """
    body_size = compute_ast_size(pattern.generalized_ast, config)
    return config.symbol_base_cost + body_size


def rewrite_ast_with_pattern(
    ast: ASTNode,
    pattern: AbstractionPattern,
    config: Optional[MDLConfig] = None,
) -> Tuple[ASTNode, int, float]:
    """Rewrites matching subtrees in `ast` with `MacroCallNode`.

    Performs outermost-first deterministic matching to prevent overlapping or
    conflicting replacements, returning the rewritten tree, match count,
    and net discrete size savings.

    Args:
        ast: Root node of the target AST.
        pattern: Candidate AbstractionPattern.
        config: Optional MDLConfig.

    Returns:
        Tuple (rewritten_ast, match_count, size_savings).
    """
    cfg = config if config is not None else MDLConfig()
    domain = _infer_pattern_domain(pattern.generalized_ast)
    match_counter = [0]

    def _rec_rewrite(curr: ASTNode, depth: int) -> ASTNode:
        if depth > cfg.max_depth:
            raise RuntimeError(
                f"Recursion limit exceeded in rewrite_ast_with_pattern "
                f"(depth={depth}, node_type={type(curr).__name__})"
            )

        sub = match_pattern(pattern.generalized_ast, curr, max_depth=cfg.max_depth)
        if sub is not None:
            match_counter[0] += 1
            rewritten_args: Dict[str, Any] = {}
            for k, v in sub.items():
                if isinstance(v, ASTNode):
                    rewritten_args[k] = _rec_rewrite(v, depth + 1)
                else:
                    rewritten_args[k] = v
            return make_macro_call(pattern.name, rewritten_args, domain=domain)

        if isinstance(curr, MacroCallNode):
            new_args: Dict[str, Any] = {}
            for k, v in curr.arguments.items():
                if isinstance(v, ASTNode):
                    new_args[k] = _rec_rewrite(v, depth + 1)
                else:
                    new_args[k] = v
            return make_macro_call(curr.macro_name, new_args, domain=curr.domain)

        if not isinstance(curr, ASTNode):
            return curr

        new_fields: Dict[str, Any] = {}
        for fld in dataclasses.fields(curr):
            val = getattr(curr, fld.name)
            if isinstance(val, ASTNode):
                new_fields[fld.name] = _rec_rewrite(val, depth + 1)
            elif isinstance(val, (list, tuple)):
                rw_items = []
                for item in val:
                    if isinstance(item, ASTNode):
                        rw_items.append(_rec_rewrite(item, depth + 1))
                    else:
                        rw_items.append(item)
                new_fields[fld.name] = type(val)(rw_items)
            else:
                new_fields[fld.name] = val

        return type(curr)(**new_fields)

    rewritten_ast = _rec_rewrite(ast, 0)
    match_count = match_counter[0]

    orig_size = compute_ast_size(ast, cfg)
    rewritten_size = compute_ast_size(rewritten_ast, cfg)
    size_savings = orig_size - rewritten_size if match_count > 0 else 0.0

    return rewritten_ast, match_count, size_savings


def evaluate_mdl(
    pattern: AbstractionPattern,
    corpus: List[ASTNode],
    config: MDLConfig = MDLConfig(),
) -> MDLScore:
    """Evaluates a single pattern candidate against a corpus of programs.

    Args:
        pattern: Candidate AbstractionPattern.
        corpus: List of AST programs in the solved dataset.
        config: MDLConfig specifying cost parameters and threshold.

    Returns:
        MDLScore containing compression metrics and acceptance status.
    """
    corpus_size_before = sum(compute_ast_size(p, config) for p in corpus)
    dsl_expansion_cost = compute_macro_cost(pattern, config)

    corpus_savings = 0.0
    matched_programs_count = 0
    total_occurrences = 0

    max_corpus_items = config.max_iterations
    iterations = 0

    for prog in corpus:
        iterations += 1
        if iterations > max_corpus_items:
            raise RuntimeError(
                f"Corpus iteration safety cap exceeded in evaluate_mdl "
                f"(iterations={iterations}, max={max_corpus_items})"
            )

        _, matches, savings = rewrite_ast_with_pattern(prog, pattern, config)
        if matches > 0:
            matched_programs_count += 1
            total_occurrences += matches
            corpus_savings += savings

    delta_mdl = dsl_expansion_cost - corpus_savings

    if corpus_size_before <= 0.0:
        compression_ratio = 1.0
    else:
        new_corpus_size = corpus_size_before + delta_mdl
        compression_ratio = (
            corpus_size_before / max(1e-9, new_corpus_size)
            if new_corpus_size > 0.0
            else 1.0
        )

    is_accepted = bool(delta_mdl < config.threshold)

    return MDLScore(
        pattern_name=pattern.name,
        dsl_expansion_cost=dsl_expansion_cost,
        corpus_savings=corpus_savings,
        delta_mdl=delta_mdl,
        compression_ratio=compression_ratio,
        is_accepted=is_accepted,
        matched_programs_count=matched_programs_count,
        total_occurrences=total_occurrences,
    )


def filter_and_rank_abstractions(
    candidates: List[AbstractionPattern],
    corpus: List[ASTNode],
    config: MDLConfig = MDLConfig(),
    top_k: int = 10,
) -> List[MDLScore]:
    """Ranks candidates in descending order of compression gain (-Delta MDL),
    filtering out rejected candidates (Delta MDL >= threshold).

    Args:
        candidates: List of candidate AbstractionPatterns.
        corpus: List of AST programs in the solved dataset.
        config: MDLConfig specifying cost parameters and threshold.
        top_k: Maximum number of accepted abstractions to return.

    Returns:
        List of ranked accepted MDLScore objects.
    """
    scores: List[MDLScore] = []
    max_candidates = config.max_iterations
    iterations = 0

    for cand in candidates:
        iterations += 1
        if iterations > max_candidates:
            raise RuntimeError(
                f"Candidate iteration safety cap exceeded in filter_and_rank_abstractions "
                f"(iterations={iterations}, max={max_candidates})"
            )

        score = evaluate_mdl(cand, corpus, config)
        if score.is_accepted:
            scores.append(score)

    scores.sort(key=lambda s: (-s.delta_mdl, s.corpus_savings), reverse=True)
    return scores[:top_k]
