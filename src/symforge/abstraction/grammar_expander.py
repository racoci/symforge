"""Dynamic Lean 4 Grammar Expander & Library Injection Engine (DreamCoder / SymForge).

This module inverts discovered abstraction patterns into first-class DSL productions,
maintains versioned grammar libraries with transactional rollback, renormalizes
probabilistic CFG priors, and enables single-step macro instantiation during synthesis.
"""

from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from symforge.abstraction.anti_unification import (
    AbstractionPattern,
    AudioVariableNode,
    SDFVariableNode,
    VariableNode,
    collect_variables,
)
from symforge.abstraction.mdl import (
    AudioMacroCallNode,
    HardwareMacroCallNode,
    MacroCallNode,
    SDFMacroCallNode,
    make_macro_call,
)
from symforge.core.ast import (
    ADSRNode,
    ASTNode,
    AudioHoleNode,
    AudioNode,
    BoxNode,
    CylinderNode,
    DifferenceNode,
    ExpDecayNode,
    FMNode,
    FreeParam,
    GainNode,
    HardwareHoleNode,
    HardwareNode,
    HoleNode,
    IntersectionNode,
    LFONode,
    LowPassNode,
    MixNode,
    NoiseNode,
    PlaneNode,
    RotateNode,
    SawNode,
    ScaleNode,
    SDFHoleNode,
    SDFNode,
    SineNode,
    SmoothUnionNode,
    SphereNode,
    SquareNode,
    TorusNode,
    TranslateNode,
    UnionNode,
)
from symforge.core.pcfg import GrammarRule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MacroParam:
    """Typed parameter slot for an invented macro production.

    Attributes:
        name: Parameter name (clean identifier, e.g. 'c1', 'r1').
        expected_type: Type constraint (e.g. 'SDFNode', 'AudioNode', 'FreeParam').
        domain: Domain of the parameter ('sdf', 'audio', 'param', 'generic').
        is_free_param: Whether the parameter is a continuous parameter.
        default_val: Default numerical value for parameter initialization.
        min_val: Lower bound for parameter optimization.
        max_val: Upper bound for parameter optimization.
    """

    name: str
    expected_type: str
    domain: str
    is_free_param: bool = False
    default_val: float = 0.0
    min_val: float = 0.0
    max_val: float = 1.0


@dataclass
class MacroDefinition:
    """First-class DSL macro production registered in the inductive environment."""

    name: str
    parameters: List[MacroParam]
    body: ASTNode
    domain: str
    return_type: str
    lean_code: str
    prior_weight: float = 0.15
    pattern: Optional[AbstractionPattern] = None

    def to_lean_code(self) -> str:
        """Return generated Lean 4 inductive declaration code."""
        return self.lean_code

    def instantiate_call(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        hole_id_fn: Optional[Callable[[str], str]] = None,
    ) -> MacroCallNode:
        """Instantiate a MacroCallNode with provided or generated arguments."""
        args: Dict[str, Any] = {}
        provided = arguments or {}

        # Build normalized lookup for provided arguments
        lookup: Dict[str, Any] = {}
        for k, v in provided.items():
            lookup[k] = v
            clean_k = k.lstrip("?")
            lookup[clean_k] = v
            lookup[f"?{clean_k}"] = v

        counter = 0
        for param in self.parameters:
            matched_val = lookup.get(param.name) or lookup.get(f"?{param.name}")
            if matched_val is not None:
                args[param.name] = matched_val
                continue

            counter += 1
            if param.expected_type == "AudioNode":
                hole_id = hole_id_fn("audio") if hole_id_fn else f"h_{self.name}_{counter}"
                args[param.name] = AudioHoleNode(hole_id=hole_id)
            elif param.expected_type == "SDFNode":
                hole_id = hole_id_fn("sdf") if hole_id_fn else f"h_{self.name}_{counter}"
                args[param.name] = SDFHoleNode(hole_id=hole_id)
            elif param.expected_type == "HardwareNode":
                hole_id = hole_id_fn("hardware") if hole_id_fn else f"h_{self.name}_{counter}"
                args[param.name] = HardwareHoleNode(name=hole_id, hole_id=hole_id)
            elif param.is_free_param or param.expected_type == "FreeParam":
                param_id = f"{self.name}_{param.name}_{counter}"
                args[param.name] = FreeParam(
                    id=param_id,
                    min_val=param.min_val,
                    max_val=param.max_val,
                    default_val=param.default_val,
                )
            else:
                args[param.name] = param.default_val

        return make_macro_call(self.name, args, domain=self.domain)


def _clean_param_name(raw: str) -> str:
    """Normalize variable slot string to valid Lean/Python identifier."""
    name = raw.split(":")[0].strip().lstrip("?")
    if not name or not name[0].isalpha():
        name = f"p_{name}"
    return name.replace("-", "_")


def _ast_to_lean_expr(node: ASTNode, var_map: Dict[str, str]) -> str:
    """Recursively convert an AST into valid Lean 4 syntax."""
    if isinstance(node, VariableNode):
        return var_map.get(node.name, _clean_param_name(node.name))

    if isinstance(node, FreeParam):
        clean_id = _clean_param_name(node.id)
        if clean_id in var_map.values():
            return clean_id
        return f'FreeParam.mk "{node.id}" {node.min_val} {node.max_val} {node.default_val}'

    # SDF Primitives
    if isinstance(node, SphereNode):
        r_str = _ast_to_lean_expr(node.radius, var_map)
        return f"SDFNode.sphere {r_str}"
    if isinstance(node, BoxNode):
        w = _ast_to_lean_expr(node.width, var_map)
        h = _ast_to_lean_expr(node.height, var_map)
        d = _ast_to_lean_expr(node.depth, var_map)
        return f"SDFNode.box {w} {h} {d}"
    if isinstance(node, CylinderNode):
        r = _ast_to_lean_expr(node.radius, var_map)
        h = _ast_to_lean_expr(node.height, var_map)
        return f"SDFNode.cylinder {r} {h}"
    if isinstance(node, TorusNode):
        r1 = _ast_to_lean_expr(node.major_radius, var_map)
        r2 = _ast_to_lean_expr(node.minor_radius, var_map)
        return f"SDFNode.torus {r1} {r2}"
    if isinstance(node, PlaneNode):
        nx = _ast_to_lean_expr(node.nx, var_map)
        ny = _ast_to_lean_expr(node.ny, var_map)
        nz = _ast_to_lean_expr(node.nz, var_map)
        off = _ast_to_lean_expr(node.offset, var_map)
        return f"SDFNode.plane {nx} {ny} {nz} {off}"

    # SDF CSG and Transforms
    if isinstance(node, TranslateNode):
        dx = _ast_to_lean_expr(node.dx, var_map)
        dy = _ast_to_lean_expr(node.dy, var_map)
        dz = _ast_to_lean_expr(node.dz, var_map)
        c = _ast_to_lean_expr(node.child, var_map)
        return f"SDFNode.translate {dx} {dy} {dz} ({c})"
    if isinstance(node, RotateNode):
        rx = _ast_to_lean_expr(node.rx, var_map)
        ry = _ast_to_lean_expr(node.ry, var_map)
        rz = _ast_to_lean_expr(node.rz, var_map)
        c = _ast_to_lean_expr(node.child, var_map)
        return f"SDFNode.rotate {rx} {ry} {rz} ({c})"
    if isinstance(node, ScaleNode):
        s = _ast_to_lean_expr(node.s, var_map)
        c = _ast_to_lean_expr(node.child, var_map)
        return f"SDFNode.scale {s} ({c})"
    if isinstance(node, UnionNode):
        sm = _ast_to_lean_expr(node.smoothness, var_map)
        a = _ast_to_lean_expr(node.a, var_map)
        b = _ast_to_lean_expr(node.b, var_map)
        return f"SDFNode.union {sm} ({a}) ({b})"
    if isinstance(node, IntersectionNode):
        a = _ast_to_lean_expr(node.a, var_map)
        b = _ast_to_lean_expr(node.b, var_map)
        a_wrap = f"({a})" if " " in a else a
        b_wrap = f"({b})" if " " in b else b
        return f"SDFNode.intersection {a_wrap} {b_wrap}"
    if isinstance(node, DifferenceNode):
        a = _ast_to_lean_expr(node.a, var_map)
        b = _ast_to_lean_expr(node.b, var_map)
        a_wrap = f"({a})" if " " in a else a
        b_wrap = f"({b})" if " " in b else b
        return f"SDFNode.difference {a_wrap} {b_wrap}"

    # Audio Primitives
    if isinstance(node, SineNode):
        f = _ast_to_lean_expr(node.freq, var_map)
        p = _ast_to_lean_expr(node.phase, var_map)
        return f"AudioNode.sine {f} {p}"
    if isinstance(node, SawNode):
        f = _ast_to_lean_expr(node.freq, var_map)
        return f"AudioNode.saw {f}"
    if isinstance(node, SquareNode):
        f = _ast_to_lean_expr(node.freq, var_map)
        d = _ast_to_lean_expr(node.duty, var_map)
        return f"AudioNode.square {f} {d}"
    if isinstance(node, NoiseNode):
        return f"AudioNode.noise {node.seed}"
    if isinstance(node, MixNode):
        r = _ast_to_lean_expr(node.ratio, var_map)
        in1 = _ast_to_lean_expr(node.input1, var_map)
        in2 = _ast_to_lean_expr(node.input2, var_map)
        in1_wrap = f"({in1})" if " " in in1 else in1
        in2_wrap = f"({in2})" if " " in in2 else in2
        return f"AudioNode.mix {r} {in1_wrap} {in2_wrap}"
    if isinstance(node, LowPassNode):
        c = _ast_to_lean_expr(node.cutoff, var_map)
        res = _ast_to_lean_expr(node.resonance, var_map)
        inp = _ast_to_lean_expr(node.input_node, var_map)
        inp_wrap = f"({inp})" if " " in inp else inp
        return f"AudioNode.lowpass {c} {res} {inp_wrap}"
    if isinstance(node, GainNode):
        g = _ast_to_lean_expr(node.gain, var_map)
        inp = _ast_to_lean_expr(node.input_node, var_map)
        inp_wrap = f"({inp})" if " " in inp else inp
        return f"AudioNode.gain {g} {inp_wrap}"
    if isinstance(node, FMNode):
        c = _ast_to_lean_expr(node.carrier, var_map)
        mf = _ast_to_lean_expr(node.mod_freq, var_map)
        mi = _ast_to_lean_expr(node.mod_index, var_map)
        return f"AudioNode.fm {c} {mf} {mi}"

    # Generic or unknown fallback
    return f"{type(node).__name__}"


def generate_lean_macro_def(
    macro_name: str,
    parameters: List[MacroParam],
    body: ASTNode,
    return_type: str,
) -> str:
    """Generate Lean 4 inductive declaration code for a macro."""
    clean_name = macro_name.replace("-", "_").lower()
    if not clean_name.startswith("macro_"):
        clean_name = f"macro_{clean_name}"

    # Group adjacent parameters with identical expected_type
    var_map: Dict[str, str] = {}
    grouped_clauses: List[str] = []
    current_group: List[str] = []
    current_type: Optional[str] = None

    for param in parameters:
        c_name = _clean_param_name(param.name)
        var_map[param.name] = c_name
        var_map[f"?{param.name.lstrip('?')}"] = c_name

        p_type = param.expected_type
        if p_type in ("float", "int"):
            p_type = "Float" if p_type == "float" else "Int"

        if current_type is None:
            current_type = p_type
            current_group.append(c_name)
        elif current_type == p_type:
            current_group.append(c_name)
        else:
            grouped_clauses.append(f"({' '.join(current_group)} : {current_type})")
            current_group = [c_name]
            current_type = p_type

    if current_group and current_type:
        grouped_clauses.append(f"({' '.join(current_group)} : {current_type})")

    params_sig = f" {' '.join(grouped_clauses)}" if grouped_clauses else ""
    body_expr = _ast_to_lean_expr(body, var_map)

    return f"def {clean_name}{params_sig} : {return_type} := {body_expr}"


def _infer_param_defaults(
    clean_name: str,
    domain: str,
    sub_val: Any,
) -> Tuple[float, float, float]:
    """Infer parameter bounds and default value."""
    if isinstance(sub_val, FreeParam):
        return sub_val.default_val, sub_val.min_val, sub_val.max_val
    if isinstance(sub_val, (int, float)):
        val = float(sub_val)
        return val, min(0.0, val), max(10.0, val * 2.0)

    name_lower = clean_name.lower()
    if "freq" in name_lower:
        return 440.0, 20.0, 5000.0
    if "phase" in name_lower:
        return 0.0, 0.0, 6.28
    if "cutoff" in name_lower:
        return 1000.0, 100.0, 10000.0
    if "radius" in name_lower or name_lower.startswith("r"):
        return 1.0, 0.1, 5.0
    if "height" in name_lower or name_lower.startswith("h"):
        return 1.0, 0.1, 5.0
    if "width" in name_lower or name_lower.startswith("w"):
        return 1.0, 0.1, 5.0
    if "depth" in name_lower or name_lower.startswith("d"):
        return 1.0, 0.1, 5.0
    if domain == "audio":
        return 1.0, 0.0, 2.0
    return 1.0, 0.1, 5.0


def pattern_to_macro_definition(
    pattern: AbstractionPattern,
    name: Optional[str] = None,
    prior_weight: float = 0.15,
) -> MacroDefinition:
    """Convert an AbstractionPattern into a MacroDefinition."""
    macro_name = name or pattern.name
    clean_macro_name = macro_name.replace("-", "_").lower()
    if not clean_macro_name.startswith("macro_"):
        clean_macro_name = f"macro_{clean_macro_name}"

    # Determine domain and return type
    if isinstance(pattern.generalized_ast, AudioNode):
        domain = "audio"
        return_type = "AudioNode"
    elif isinstance(pattern.generalized_ast, SDFNode):
        domain = "sdf"
        return_type = "SDFNode"
    elif isinstance(pattern.generalized_ast, HardwareNode):
        domain = "hardware"
        return_type = "HardwareNode"
    else:
        domain = "generic"
        return_type = "ASTNode"

    variables = collect_variables(pattern.generalized_ast)
    var_dict = {v.name: v for v in variables}

    # Extract sample substitution values if available
    first_sub: Dict[str, Any] = {}
    if pattern.occurrences:
        first_sub = pattern.occurrences[0][1]

    params: List[MacroParam] = []
    processed_names: Set[str] = set()

    # Process explicit pattern parameters
    for p_str in pattern.parameters:
        p_name = p_str.split(":")[0].strip()
        c_name = _clean_param_name(p_name)
        if c_name in processed_names:
            continue
        processed_names.add(c_name)

        declared_type = p_str.split(":")[1].strip() if ":" in p_str else None
        v_node = var_dict.get(p_name) or var_dict.get(f"?{c_name}")

        exp_type = declared_type or (v_node.expected_type if v_node else "ASTNode")
        p_domain = v_node.domain if v_node else domain

        is_free = exp_type in ("FreeParam", "float", "int") or p_domain == "param"
        sub_val = first_sub.get(p_name) or first_sub.get(f"?{c_name}")
        def_val, min_v, max_v = _infer_param_defaults(c_name, domain, sub_val)

        params.append(
            MacroParam(
                name=c_name,
                expected_type=exp_type,
                domain=p_domain,
                is_free_param=is_free,
                default_val=def_val,
                min_val=min_v,
                max_val=max_v,
            )
        )

    # Catch any remaining VariableNodes in the body AST
    for v in variables:
        c_name = _clean_param_name(v.name)
        if c_name in processed_names:
            continue
        processed_names.add(c_name)

        is_free = v.expected_type in ("FreeParam", "float", "int") or v.domain == "param"
        sub_val = first_sub.get(v.name) or first_sub.get(f"?{c_name}")
        def_val, min_v, max_v = _infer_param_defaults(c_name, domain, sub_val)

        params.append(
            MacroParam(
                name=c_name,
                expected_type=v.expected_type,
                domain=v.domain,
                is_free_param=is_free,
                default_val=def_val,
                min_val=min_v,
                max_val=max_v,
            )
        )

    lean_code = generate_lean_macro_def(
        macro_name=clean_macro_name,
        parameters=params,
        body=pattern.generalized_ast,
        return_type=return_type,
    )

    return MacroDefinition(
        name=clean_macro_name,
        parameters=params,
        body=pattern.generalized_ast,
        domain=domain,
        return_type=return_type,
        lean_code=lean_code,
        prior_weight=prior_weight,
        pattern=pattern,
    )


class DynamicGrammarLibrary:
    """Versioned library of base productions and injected macro rules."""

    _GLOBAL_REGISTRY: Dict[str, MacroDefinition] = {}
    _ACTIVE_INSTANCE: Optional[DynamicGrammarLibrary] = None

    def __init__(self) -> None:
        self.macros: Dict[str, MacroDefinition] = {}
        self._snapshots: Dict[int, Dict[str, Any]] = {}
        self._version_counter: int = 0

        # Base rule distributions normalized to 1.0
        self.rule_weights: Dict[str, Dict[str, float]] = {
            "audio": {
                "LowPass": 0.20,
                "Gain": 0.1333,
                "Mix": 0.1333,
                "FM": 0.10,
                "Sine": 0.1667,
                "Saw": 0.1667,
                "Square": 0.10,
            },
            "sdf": {
                "Union": 0.1667,
                "Intersection": 0.0833,
                "Difference": 0.0833,
                "Translate": 0.1111,
                "Sphere": 0.1944,
                "Box": 0.1667,
                "Cylinder": 0.1111,
                "Torus": 0.0834,
            },
        }
        self._renormalize_distributions()
        DynamicGrammarLibrary._ACTIVE_INSTANCE = self

    def _renormalize_distributions(self) -> None:
        """Strictly normalize probability weights to sum to 1.0 +- 1e-5."""
        for dom, weights in self.rule_weights.items():
            tot = sum(weights.values())
            if tot > 0:
                self.rule_weights[dom] = {k: v / tot for k, v in weights.items()}

    def snapshot(self) -> int:
        """Capture transactional snapshot of grammar state."""
        self._version_counter += 1
        ver = self._version_counter
        self._snapshots[ver] = {
            "macros": {k: dataclasses.replace(v) for k, v in self.macros.items()},
            "rule_weights": {d: dict(weights) for d, weights in self.rule_weights.items()},
        }
        return ver

    def rollback(self, version: int) -> None:
        """Revert grammar back to snapshot state safely."""
        if version not in self._snapshots:
            raise ValueError(f"Snapshot version {version} not found in library history")

        snap = self._snapshots[version]
        self.macros = {k: dataclasses.replace(v) for k, v in snap["macros"].items()}
        self.rule_weights = {d: dict(weights) for d, weights in snap["rule_weights"].items()}

        # Prune newer snapshots
        prune_keys = [k for k in self._snapshots if k > version]
        for k in prune_keys:
            del self._snapshots[k]

        self._sync_global_registry()

    def _sync_global_registry(self) -> None:
        """Synchronize class-level registry with local library state."""
        DynamicGrammarLibrary._GLOBAL_REGISTRY.clear()
        for k, v in self.macros.items():
            DynamicGrammarLibrary._GLOBAL_REGISTRY[k] = v

    def register_macro(
        self,
        pattern: AbstractionPattern,
        name: Optional[str] = None,
        prior_weight: float = 0.15,
    ) -> MacroDefinition:
        """Register macro pattern, validate types, and update PCFG priors."""
        macro_def = pattern_to_macro_definition(pattern, name=name, prior_weight=prior_weight)
        key = macro_def.name

        self.macros[key] = macro_def
        DynamicGrammarLibrary._GLOBAL_REGISTRY[key] = macro_def

        # Update PCFG rule distribution
        dom = macro_def.domain if macro_def.domain in self.rule_weights else "sdf"
        weights = self.rule_weights[dom]

        # Rescale existing weights to make room for prior_weight
        scale = max(0.01, 1.0 - prior_weight)
        new_weights = {r: w * scale for r, w in weights.items() if r != key}
        new_weights[key] = prior_weight
        self.rule_weights[dom] = new_weights
        self._renormalize_distributions()

        return macro_def

    def get_rule_distribution(self, domain: str) -> Dict[str, float]:
        """Return normalized rule probability distribution for the domain."""
        return dict(self.rule_weights.get(domain, {}))

    def get_pcfg_rules(self, domain: str) -> List[GrammarRule]:
        """Return GrammarRules for PCFG generator."""
        rules: List[GrammarRule] = []
        if domain == "audio":
            lhs = "AudioExp"
        elif domain == "sdf":
            lhs = "SDFExp"
        else:
            lhs = "HardwareExp"
        weights = self.rule_weights.get(domain, {})

        for tag, w in weights.items():
            if tag in self.macros:
                m = self.macros[tag]
                children_symbols = tuple(
                    lhs for p in m.parameters if p.expected_type in ("AudioNode", "SDFNode", "HardwareNode")
                )
                is_terminal = len(children_symbols) == 0
                rules.append(
                    GrammarRule(
                        lhs=lhs,
                        rhs_tag=tag,
                        child_symbols=children_symbols,
                        weight=w,
                        is_terminal=is_terminal,
                    )
                )
            else:
                rules.append(
                    GrammarRule(
                        lhs=lhs,
                        rhs_tag=tag,
                        child_symbols=(lhs,) if tag in ("LowPass", "Gain", "Translate", "Shl", "Shr") else ((lhs, lhs) if tag in ("Mix", "Union", "Intersection", "Difference", "Add", "Mul", "And", "Or", "Xor") else ()),
                        weight=w,
                        is_terminal=tag in ("Sine", "Saw", "Square", "Noise", "FM", "Sphere", "Box", "Cylinder", "Torus", "Imm", "Reg"),
                    )
                )
        return rules

    def get_macro_expansions(
        self,
        domain: str,
        hole_id_fn: Callable[[str], str],
    ) -> List[ASTNode]:
        """Instantiate single-step MacroCallNode expansions for typed holes."""
        expansions: List[ASTNode] = []
        for m in self.macros.values():
            if m.domain == domain or domain == "generic":
                call_node = m.instantiate_call(hole_id_fn=hole_id_fn)
                expansions.append(call_node)
        return expansions

    def expand_macro_call(self, macro_call: MacroCallNode) -> ASTNode:
        """Inline macro body with provided argument substitutions."""
        m_name = macro_call.macro_name.replace("-", "_").lower()
        if not m_name.startswith("macro_") and f"macro_{m_name}" in self.macros:
            m_name = f"macro_{m_name}"

        macro_def = self.macros.get(m_name) or DynamicGrammarLibrary._GLOBAL_REGISTRY.get(m_name)
        if not macro_def:
            raise KeyError(
                f"Macro '{macro_call.macro_name}' is not registered in DynamicGrammarLibrary."
            )

        # Build comprehensive substitution map
        sub: Dict[str, Any] = {}
        for k, v in macro_call.arguments.items():
            clean_k = _clean_param_name(k)
            sub[clean_k] = v
            sub[f"?{clean_k}"] = v
            sub[k] = v

        # Fill missing parameters from MacroParam defaults
        for param in macro_def.parameters:
            if param.name not in sub and f"?{param.name}" not in sub:
                if param.is_free_param:
                    p_id = f"{macro_def.name}_{param.name}"
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

        # Apply substitutions to macro body
        inlined = self._substitute_body(macro_def.body, sub)
        # Recursively expand any nested macro calls
        return self.inline_all_macros(inlined)

    def _substitute_body(self, node: Any, sub: Dict[str, Any]) -> Any:
        """Recursively replace VariableNodes with argument values."""
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
            return sub.get(clean_id, sub.get(node.id, node))

        if not isinstance(node, ASTNode):
            return node

        new_fields: Dict[str, Any] = {}
        for fld in dataclasses.fields(node):
            val = getattr(node, fld.name)
            if isinstance(val, ASTNode):
                new_fields[fld.name] = self._substitute_body(val, sub)
            elif isinstance(val, (list, tuple)):
                new_fields[fld.name] = type(val)(self._substitute_body(item, sub) for item in val)
            else:
                new_fields[fld.name] = self._substitute_body(val, sub)

        return type(node)(**new_fields)

    def inline_all_macros(self, ast: ASTNode) -> ASTNode:
        """Recursively inline all MacroCallNode instances in an AST."""
        if isinstance(ast, MacroCallNode):
            # First inline argument subtrees
            inlined_args: Dict[str, Any] = {}
            for k, v in ast.arguments.items():
                if isinstance(v, ASTNode):
                    inlined_args[k] = self.inline_all_macros(v)
                else:
                    inlined_args[k] = v
            prepared_call = make_macro_call(ast.macro_name, inlined_args, domain=ast.domain)
            return self.expand_macro_call(prepared_call)

        if not isinstance(ast, ASTNode):
            return ast

        new_fields: Dict[str, Any] = {}
        for fld in dataclasses.fields(ast):
            val = getattr(ast, fld.name)
            if isinstance(val, ASTNode):
                new_fields[fld.name] = self.inline_all_macros(val)
            elif isinstance(val, (list, tuple)):
                new_fields[fld.name] = type(val)(
                    self.inline_all_macros(item) if isinstance(item, ASTNode) else item
                    for item in val
                )
            else:
                new_fields[fld.name] = val

        return type(ast)(**new_fields)


def get_active_library() -> DynamicGrammarLibrary:
    """Retrieve or create active singleton DynamicGrammarLibrary."""
    if DynamicGrammarLibrary._ACTIVE_INSTANCE is None:
        DynamicGrammarLibrary._ACTIVE_INSTANCE = DynamicGrammarLibrary()
    return DynamicGrammarLibrary._ACTIVE_INSTANCE


def expand_macro_call(
    macro_call: MacroCallNode,
    library: Optional[DynamicGrammarLibrary] = None,
) -> ASTNode:
    """Inline macro call with registered substitutions."""
    lib = library or get_active_library()
    return lib.expand_macro_call(macro_call)


def expand_all_macros_in_ast(
    ast: ASTNode,
    library: Optional[DynamicGrammarLibrary] = None,
) -> ASTNode:
    """Recursively expand all MacroCallNodes across an entire AST."""
    lib = library or get_active_library()
    return lib.inline_all_macros(ast)
