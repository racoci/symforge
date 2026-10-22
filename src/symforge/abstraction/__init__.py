"""SymForge Abstraction Discovery: Anti-unification, MDL, and grammar expansion."""

from symforge.abstraction.anti_unification import (
    AbstractionPattern,
    AudioVariableNode,
    MacroParamStr,
    SDFVariableNode,
    Substitution,
    VariableNameGenerator,
    VariableNode,
    anti_unify,
    apply_substitution,
    collect_variables,
    extract_subtrees,
    make_variable,
    match_pattern,
    mine_recurrent_patterns,
)

__all__ = [
    "AbstractionPattern",
    "AudioVariableNode",
    "MacroParamStr",
    "SDFVariableNode",
    "Substitution",
    "VariableNameGenerator",
    "VariableNode",
    "anti_unify",
    "apply_substitution",
    "collect_variables",
    "extract_subtrees",
    "make_variable",
    "match_pattern",
    "mine_recurrent_patterns",
]
