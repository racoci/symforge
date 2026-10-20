"""Domain Code Normalizers converting external DSLs (OpenSCAD, Dexed) to SymForge ASTs."""

from __future__ import annotations
import ast
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

from symforge.core.ast import (
    ADSRNode,
    ASTNode,
    AudioNode,
    BoxNode,
    CylinderNode,
    DifferenceNode,
    FMNode,
    FreeParam,
    IntersectionNode,
    MixNode,
    RotateNode,
    ScaleNode,
    SDFNode,
    SineNode,
    SphereNode,
    TranslateNode,
    UnionNode,
)


# ============================================================================
# Safe OpenSCAD Expression Evaluator
# ============================================================================


class _SafeMathEvaluator(ast.NodeVisitor):
    """Safely evaluates arithmetic expressions, variables, and vectors without eval()."""

    def __init__(self, env: Dict[str, Any]):
        self.env = env

    def evaluate(self, node: ast.AST) -> Any:
        return self.visit(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Num(self, node: ast.Num) -> Any:  # Python <= 3.7 compatibility
        return node.n

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.env:
            return self.env[node.id]
        if node.id.lower() == "pi":
            return math.pi
        if node.id.lower() == "true":
            return True
        if node.id.lower() == "false":
            return False
        raise ValueError(f"Undefined variable or identifier in OpenSCAD expression: '{node.id}'")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        val = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            denom = right if abs(right) > 1e-8 else (1e-8 if right >= 0 else -1e-8)
            return left / denom
        if isinstance(node.op, (ast.Pow,)):
            return left ** right
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    def visit_List(self, node: ast.List) -> List[Any]:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> List[Any]:
        return [self.visit(elt) for elt in node.elts]

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are supported.")
        func_name = node.func.id.lower()
        args = [self.visit(arg) for arg in node.args]
        math_funcs = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sqrt": math.sqrt,
            "abs": abs,
            "min": min,
            "max": max,
            "radians": math.radians,
            "degrees": math.degrees,
        }
        if func_name in math_funcs:
            return math_funcs[func_name](*args)
        raise ValueError(f"Unsupported function call in OpenSCAD expression: '{node.func.id}'")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ============================================================================
# OpenSCAD Normalizer
# ============================================================================


class OpenSCADNormalizer:
    """Parses OpenSCAD CST/code into canonical SymForge SDFNode ASTs."""

    def __init__(self) -> None:
        self.param_counter = 0

    def _next_param(
        self, name: str, min_val: float, max_val: float, default_val: float
    ) -> FreeParam:
        self.param_counter += 1
        return FreeParam(
            id=f"{name}_{self.param_counter}",
            min_val=min_val,
            max_val=max_val,
            default_val=default_val,
        )

    def _strip_comments(self, code: str) -> str:
        """Strip single-line and multi-line comments from OpenSCAD source code."""
        # Multi-line /* ... */
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        # Single-line // ...
        code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
        return code

    def normalize(self, code: str) -> SDFNode:
        """Parse OpenSCAD code and return a valid SymForge SDFNode tree."""
        clean_code = self._strip_comments(code)
        tokens = self._tokenize(clean_code)
        parser = _OpenSCADParser(tokens, self._next_param)
        return parser.parse_program()

    def _tokenize(self, code: str) -> List[str]:
        """Tokenize OpenSCAD syntax into discrete semantic symbols."""
        token_spec = [
            ("STRING", r'"[^"\\]*(?:\\.[^"\\]*)*"'),
            ("NUMBER", r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"),
            ("ID", r"[$a-zA-Z_][a-zA-Z0-9_$]*"),
            ("PUNCT", r"[()\[\]{};,=+\-*/^]"),
            ("WS", r"\s+"),
        ]
        tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_spec)
        tokens: List[str] = []
        for match in re.finditer(tok_regex, code):
            kind = match.lastgroup
            value = match.group()
            if kind != "WS":
                tokens.append(value)
        return tokens


class _OpenSCADParser:
    """Recursive descent parser for OpenSCAD statements and CSG combinators."""

    def __init__(self, tokens: List[str], param_factory: Any):
        self.tokens = tokens
        self.pos = 0
        self.env: Dict[str, Any] = {
            "pi": math.pi,
            "PI": math.pi,
            "true": True,
            "false": False,
        }
        self.param_factory = param_factory

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError(f"Unexpected end of input. Expected: '{expected}'")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected '{expected}', found '{tok}' at token {self.pos}")
        self.pos += 1
        return tok

    def parse_program(self) -> SDFNode:
        """Parse top-level OpenSCAD statements and fold multiple objects into a Union."""
        nodes: List[SDFNode] = []
        max_steps = 50000
        step = 0
        while self.peek() is not None:
            step += 1
            if step > max_steps:
                raise RuntimeError("OpenSCAD parser iteration limit exceeded.")

            tok = self.peek()
            if tok == ";":
                self.consume(";")
                continue

            # Variable assignment: var = expr ;
            if (
                self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1] == "="
                and tok not in ("cube", "sphere", "cylinder", "translate", "rotate", "scale", "union", "difference", "intersection")
            ):
                var_name = self.consume()
                self.consume("=")
                expr_str = self._collect_until_semicolon()
                self.consume(";")
                val = self._evaluate_expr_str(expr_str)
                self.env[var_name] = val
                continue

            # Parse geometric statement
            node = self.parse_statement()
            if node is not None:
                nodes.append(node)

        if not nodes:
            # Default empty fallback: small unit box
            w = self.param_factory("w", 0.01, 100.0, 1.0)
            h = self.param_factory("h", 0.01, 100.0, 1.0)
            d = self.param_factory("d", 0.01, 100.0, 1.0)
            return BoxNode(width=w, height=h, depth=d)

        if len(nodes) == 1:
            return nodes[0]

        # Fold multiple top-level nodes into UnionNode
        current = nodes[0]
        for next_node in nodes[1:]:
            sm = self.param_factory("sm", 0.0, 1.0, 0.0)
            current = UnionNode(smoothness=sm, a=current, b=next_node)
        return current

    def _collect_until_semicolon(self) -> str:
        parts: List[str] = []
        bracket_level = 0
        paren_level = 0
        while self.peek() is not None:
            tok = self.peek()
            if tok == ";" and bracket_level == 0 and paren_level == 0:
                break
            tok = self.consume()
            if tok in ("(", "{"):
                paren_level += 1
            elif tok in (")", "}"):
                paren_level -= 1
            elif tok == "[":
                bracket_level += 1
            elif tok == "]":
                bracket_level -= 1
            parts.append(tok)
        return " ".join(parts)

    def _evaluate_expr_str(self, expr_str: str) -> Any:
        try:
            parsed = ast.parse(expr_str, mode="eval")
            evaluator = _SafeMathEvaluator(self.env)
            return evaluator.evaluate(parsed)
        except Exception as e:
            logger.error(
                "OpenSCAD expression evaluation failed for '%s' with environment keys=%s: %s",
                expr_str,
                list(self.env.keys()),
                e,
            )
            raise ValueError(f"Failed to evaluate expression '{expr_str}': {e}") from e

    def parse_statement(self) -> SDFNode:
        """Parse a primitive, transform, or CSG combinator statement."""
        name = self.consume()
        args, kwargs = self.parse_call_args()

        # Primitives
        if name == "cube":
            self.consume_optional(";")
            return self._build_cube(args, kwargs)

        if name == "sphere":
            self.consume_optional(";")
            return self._build_sphere(args, kwargs)

        if name == "cylinder":
            self.consume_optional(";")
            return self._build_cylinder(args, kwargs)

        # Spatial transforms
        if name in ("translate", "rotate", "scale"):
            child = self.parse_child_or_block()
            if name == "translate":
                return self._build_translate(args, kwargs, child)
            if name == "rotate":
                return self._build_rotate(args, kwargs, child)
            if name == "scale":
                return self._build_scale(args, kwargs, child)

        # CSG Boolean Combinators
        if name in ("union", "difference", "intersection"):
            children = self.parse_children_block()
            if not children:
                # Return empty default box
                w = self.param_factory("w", 0.01, 100.0, 1.0)
                h = self.param_factory("h", 0.01, 100.0, 1.0)
                d = self.param_factory("d", 0.01, 100.0, 1.0)
                return BoxNode(width=w, height=h, depth=d)

            if len(children) == 1:
                return children[0]

            if name == "union":
                res = children[0]
                for c in children[1:]:
                    sm = self.param_factory("sm", 0.0, 1.0, 0.0)
                    res = UnionNode(smoothness=sm, a=res, b=c)
                return res

            if name == "difference":
                res = children[0]
                for c in children[1:]:
                    res = DifferenceNode(a=res, b=c)
                return res

            if name == "intersection":
                res = children[0]
                for c in children[1:]:
                    res = IntersectionNode(a=res, b=c)
                return res

        raise ValueError(f"Unknown OpenSCAD module or primitive: '{name}'")

    def consume_optional(self, expected: str) -> bool:
        if self.peek() == expected:
            self.consume(expected)
            return True
        return False

    def parse_call_args(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Parse positional and keyword arguments inside ( ... )."""
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if self.peek() != "(":
            return args, kwargs

        self.consume("(")
        if self.peek() == ")":
            self.consume(")")
            return args, kwargs

        while self.peek() is not None:
            # Check for keyword argument: id = expr
            if (
                self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1] == "="
                and self.peek() not in ("==", "!=", "<=", ">=")
            ):
                key = self.consume()
                self.consume("=")
                val = self.parse_single_expr()
                kwargs[key] = val
            else:
                val = self.parse_single_expr()
                args.append(val)

            if self.peek() == ",":
                self.consume(",")
                continue
            if self.peek() == ")":
                self.consume(")")
                break

        return args, kwargs

    def parse_single_expr(self) -> Any:
        """Parse an argument expression until a comma or closing parenthesis."""
        parts: List[str] = []
        bracket_level = 0
        paren_level = 0
        while self.peek() is not None:
            tok = self.peek()
            if (tok == "," or tok == ")") and bracket_level == 0 and paren_level == 0:
                break
            tok = self.consume()
            if tok in ("(", "{"):
                paren_level += 1
            elif tok in (")", "}"):
                paren_level -= 1
            elif tok == "[":
                bracket_level += 1
            elif tok == "]":
                bracket_level -= 1
            parts.append(tok)
        expr_str = " ".join(parts)
        return self._evaluate_expr_str(expr_str)

    def parse_child_or_block(self) -> SDFNode:
        """Parse the target child of a transformation (either { ... } or a single statement)."""
        if self.peek() == "{":
            children = self.parse_children_block()
            if len(children) == 1:
                return children[0]
            # Fold multiple into UnionNode
            res = children[0]
            for c in children[1:]:
                sm = self.param_factory("sm", 0.0, 1.0, 0.0)
                res = UnionNode(smoothness=sm, a=res, b=c)
            return res

        return self.parse_statement()

    def parse_children_block(self) -> List[SDFNode]:
        """Parse a block of statements enclosed in { ... }."""
        self.consume("{")
        children: List[SDFNode] = []
        max_steps = 10000
        step = 0
        while self.peek() is not None and self.peek() != "}":
            step += 1
            if step > max_steps:
                raise RuntimeError("Block parsing loop cap exceeded.")

            if self.peek() == ";":
                self.consume(";")
                continue

            # Variable assignment inside block
            if (
                self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1] == "="
                and self.peek() not in ("cube", "sphere", "cylinder", "translate", "rotate", "scale", "union", "difference", "intersection")
            ):
                var_name = self.consume()
                self.consume("=")
                expr_str = self._collect_until_semicolon()
                self.consume(";")
                val = self._evaluate_expr_str(expr_str)
                self.env[var_name] = val
                continue

            child = self.parse_statement()
            if child is not None:
                children.append(child)

        self.consume("}")
        return children

    # ------------------------------------------------------------------------
    # Primitive Construction Helpers
    # ------------------------------------------------------------------------

    def _build_cube(self, args: List[Any], kwargs: Dict[str, Any]) -> BoxNode:
        size_val = kwargs.get("size", args[0] if args else 1.0)
        if isinstance(size_val, (list, tuple)):
            w_val = float(size_val[0])
            h_val = float(size_val[1]) if len(size_val) > 1 else w_val
            d_val = float(size_val[2]) if len(size_val) > 2 else w_val
        else:
            w_val = float(size_val)
            h_val = float(size_val)
            d_val = float(size_val)

        w = self.param_factory("w", 0.01, 100.0, w_val)
        h = self.param_factory("h", 0.01, 100.0, h_val)
        d = self.param_factory("d", 0.01, 100.0, d_val)
        return BoxNode(width=w, height=h, depth=d)

    def _build_sphere(self, args: List[Any], kwargs: Dict[str, Any]) -> SphereNode:
        if "r" in kwargs:
            r_val = float(kwargs["r"])
        elif "d" in kwargs:
            r_val = float(kwargs["d"]) / 2.0
        elif args:
            r_val = float(args[0])
        else:
            r_val = 1.0

        r = self.param_factory("r", 0.01, 100.0, r_val)
        return SphereNode(radius=r)

    def _build_cylinder(self, args: List[Any], kwargs: Dict[str, Any]) -> CylinderNode:
        # OpenSCAD: cylinder(h, r1, r2) or cylinder(h=..., r=...) or cylinder(h=..., d=...)
        h_val = float(kwargs.get("h", args[0] if args else 1.0))
        if "r" in kwargs:
            r_val = float(kwargs["r"])
        elif "r1" in kwargs:
            r1 = float(kwargs["r1"])
            r2 = float(kwargs.get("r2", r1))
            r_val = (r1 + r2) / 2.0
        elif "d" in kwargs:
            r_val = float(kwargs["d"]) / 2.0
        elif len(args) >= 3:
            r_val = (float(args[1]) + float(args[2])) / 2.0
        elif len(args) >= 2:
            r_val = float(args[1])
        else:
            r_val = 1.0

        r = self.param_factory("r", 0.01, 100.0, r_val)
        h = self.param_factory("h", 0.01, 100.0, h_val)
        return CylinderNode(radius=r, height=h)

    def _build_translate(
        self, args: List[Any], kwargs: Dict[str, Any], child: SDFNode
    ) -> TranslateNode:
        v = kwargs.get("v", args[0] if args else [0.0, 0.0, 0.0])
        if isinstance(v, (list, tuple)):
            dx_val = float(v[0]) if len(v) > 0 else 0.0
            dy_val = float(v[1]) if len(v) > 1 else 0.0
            dz_val = float(v[2]) if len(v) > 2 else 0.0
        else:
            dx_val = dy_val = dz_val = float(v)

        dx = self.param_factory("dx", -100.0, 100.0, dx_val)
        dy = self.param_factory("dy", -100.0, 100.0, dy_val)
        dz = self.param_factory("dz", -100.0, 100.0, dz_val)
        return TranslateNode(dx=dx, dy=dy, dz=dz, child=child)

    def _build_rotate(
        self, args: List[Any], kwargs: Dict[str, Any], child: SDFNode
    ) -> RotateNode:
        a = kwargs.get("a", args[0] if args else [0.0, 0.0, 0.0])
        if isinstance(a, (list, tuple)):
            rx_deg = float(a[0]) if len(a) > 0 else 0.0
            ry_deg = float(a[1]) if len(a) > 1 else 0.0
            rz_deg = float(a[2]) if len(a) > 2 else 0.0
        else:
            rx_deg = ry_deg = 0.0
            rz_deg = float(a)

        # Convert OpenSCAD degrees to radians for PyTorch rotation matrices
        rx = self.param_factory("rx", -6.28318, 6.28318, math.radians(rx_deg))
        ry = self.param_factory("ry", -6.28318, 6.28318, math.radians(ry_deg))
        rz = self.param_factory("rz", -6.28318, 6.28318, math.radians(rz_deg))
        return RotateNode(rx=rx, ry=ry, rz=rz, child=child)

    def _build_scale(
        self, args: List[Any], kwargs: Dict[str, Any], child: SDFNode
    ) -> ScaleNode:
        v = kwargs.get("v", args[0] if args else 1.0)
        if isinstance(v, (list, tuple)):
            # Uniform approximation of 3D scaling vector
            s_val = (float(v[0]) + float(v[1]) + float(v[2])) / 3.0
        else:
            s_val = float(v)

        s = self.param_factory("s", 0.01, 10.0, s_val)
        return ScaleNode(s=s, child=child)


# ============================================================================
# DexedPatchNormalizer
# ============================================================================


class DexedPatchNormalizer:
    """Parses Dexed / Yamaha DX7 voice patch definitions into SymForge AudioNode ASTs."""

    # Yamaha DX7 Standard 32 Algorithms
    _DX7_ALGORITHMS: Dict[int, Dict[str, Any]] = {
        1: {"carriers": [1, 3], "modulators": {1: [2], 3: [4], 4: [5], 5: [6]}},
        2: {"carriers": [1, 3], "modulators": {1: [2], 3: [4], 4: [5], 5: [6]}},
        3: {"carriers": [1, 4], "modulators": {1: [2], 2: [3], 4: [5], 5: [6]}},
        4: {"carriers": [1, 4], "modulators": {1: [2], 2: [3], 4: [5], 5: [6]}},
        5: {"carriers": [1, 3, 5], "modulators": {1: [2], 3: [4], 5: [6]}},
        6: {"carriers": [1, 3, 5], "modulators": {1: [2], 3: [4], 5: [6]}},
        7: {"carriers": [1, 3], "modulators": {1: [2], 3: [4], 4: [5, 6]}},
        8: {"carriers": [1, 3], "modulators": {1: [2], 3: [4], 4: [5, 6]}},
        9: {"carriers": [1, 3], "modulators": {1: [2], 3: [4, 5, 6]}},
        10: {"carriers": [1, 3], "modulators": {1: [2], 3: [4, 5, 6]}},
        11: {"carriers": [1, 4], "modulators": {1: [2], 2: [3], 4: [5, 6]}},
        12: {"carriers": [1, 3], "modulators": {1: [2], 3: [4, 5], 5: [6]}},
        13: {"carriers": [1, 3], "modulators": {1: [2], 3: [4, 5], 5: [6]}},
        14: {"carriers": [1, 3], "modulators": {1: [2], 3: [4], 4: [5], 5: [6]}},
        15: {"carriers": [1, 3], "modulators": {1: [2], 3: [4], 4: [5], 5: [6]}},
        16: {"carriers": [1], "modulators": {1: [2], 2: [3], 3: [4], 4: [5], 5: [6]}},
        17: {"carriers": [1], "modulators": {1: [2], 2: [3], 3: [4], 4: [5], 5: [6]}},
        18: {"carriers": [1], "modulators": {1: [2], 2: [3], 3: [4, 5], 5: [6]}},
        19: {"carriers": [1, 4, 5], "modulators": {1: [2], 2: [3], 5: [6]}},
        20: {"carriers": [1, 2, 4], "modulators": {2: [3], 4: [5], 5: [6]}},
        21: {"carriers": [1, 2, 4], "modulators": {2: [3], 4: [5], 5: [6]}},
        22: {"carriers": [1, 2, 3, 5], "modulators": {3: [4], 5: [6]}},
        23: {"carriers": [1, 2, 3, 5], "modulators": {3: [4], 5: [6]}},
        24: {"carriers": [1, 2, 3, 4, 5], "modulators": {5: [6]}},
        25: {"carriers": [1, 2, 3, 4, 5], "modulators": {5: [6]}},
        26: {"carriers": [1, 2, 3], "modulators": {3: [4], 4: [5], 5: [6]}},
        27: {"carriers": [1, 2, 3], "modulators": {3: [4, 5], 5: [6]}},
        28: {"carriers": [1, 3, 5], "modulators": {1: [2], 3: [4], 5: [6]}},
        29: {"carriers": [1, 2, 3, 4], "modulators": {4: [5], 5: [6]}},
        30: {"carriers": [1, 2, 3, 4], "modulators": {4: [5], 5: [6]}},
        31: {"carriers": [1, 2, 3, 4, 5], "modulators": {5: [6]}},
        32: {"carriers": [1, 2, 3, 4, 5, 6], "modulators": {}},
    }

    def __init__(self, base_freq: float = 440.0) -> None:
        self.base_freq = base_freq
        self.param_counter = 0

    def _next_param(
        self, name: str, min_val: float, max_val: float, default_val: float
    ) -> FreeParam:
        self.param_counter += 1
        return FreeParam(
            id=f"{name}_{self.param_counter}",
            min_val=min_val,
            max_val=max_val,
            default_val=default_val,
        )

    def normalize(self, data: Union[str, Dict[str, Any]]) -> AudioNode:
        """Normalize a Dexed / DX7 voice patch into an executable AudioNode tree."""
        if isinstance(data, str):
            patch = json.loads(data)
        elif isinstance(data, dict):
            patch = data
        else:
            raise TypeError(f"Expected JSON string or dictionary, got {type(data)}")

        # Case 1: Direct / Simplified carrier-modulator preset
        if "carrier" in patch:
            return self._normalize_simplified_preset(patch)

        # Case 2: Standard DX7 patch with algorithm
        algorithm_idx = int(patch.get("algorithm", patch.get("algo", 1)))
        operators_dict = self._parse_operators(patch)

        alg_info = self._DX7_ALGORITHMS.get(
            algorithm_idx,
            {"carriers": [1], "modulators": {1: [2]}},
        )

        carrier_indices = alg_info["carriers"]
        modulators_map = alg_info["modulators"]

        carrier_nodes: List[AudioNode] = []
        for c_idx in carrier_indices:
            c_op = operators_dict.get(c_idx, {})
            c_freq = self._extract_frequency(c_op)
            c_adsr = self._extract_adsr(c_op)

            # Check if this carrier is modulated
            mod_indices = modulators_map.get(c_idx, [])
            if mod_indices:
                primary_mod_idx = mod_indices[0]
                m_op = operators_dict.get(primary_mod_idx, {})
                m_freq = self._extract_frequency(m_op)
                m_index = self._extract_mod_index(m_op)

                carrier_param = self._next_param(
                    f"c{c_idx}_freq", 20.0, 5000.0, c_freq
                )
                mod_freq_param = self._next_param(
                    f"m{primary_mod_idx}_freq", 1.0, 5000.0, m_freq
                )
                mod_idx_param = self._next_param(
                    f"m{primary_mod_idx}_index", 0.0, 20.0, m_index
                )

                gate_node = FMNode(
                    carrier=carrier_param,
                    mod_freq=mod_freq_param,
                    mod_index=mod_idx_param,
                )
            else:
                # Unmodulated carrier -> SineNode
                carrier_param = self._next_param(
                    f"c{c_idx}_freq", 20.0, 5000.0, c_freq
                )
                phase_param = self._next_param(
                    f"c{c_idx}_phase", 0.0, 6.28318, 0.0
                )
                gate_node = SineNode(freq=carrier_param, phase=phase_param)

            # Wrap in ADSR envelope
            adsr_node = ADSRNode(
                attack=self._next_param(f"op{c_idx}_attack", 0.001, 5.0, c_adsr[0]),
                decay=self._next_param(f"op{c_idx}_decay", 0.001, 5.0, c_adsr[1]),
                sustain=self._next_param(f"op{c_idx}_sustain", 0.0, 1.0, c_adsr[2]),
                release=self._next_param(f"op{c_idx}_release", 0.001, 5.0, c_adsr[3]),
                gate=gate_node,
            )
            carrier_nodes.append(adsr_node)

        # Mix multiple carriers
        if not carrier_nodes:
            # Fallback sine
            f0 = self._next_param("f0", 20.0, 5000.0, self.base_freq)
            return SineNode(freq=f0)

        if len(carrier_nodes) == 1:
            return carrier_nodes[0]

        # Fold multiple carriers with MixNode
        current_mix = carrier_nodes[0]
        for idx, next_carrier in enumerate(carrier_nodes[1:], start=1):
            ratio_param = self._next_param(f"mix_ratio_{idx}", 0.0, 1.0, 0.5)
            current_mix = MixNode(
                ratio=ratio_param,
                input1=current_mix,
                input2=next_carrier,
            )
        return current_mix

    def _normalize_simplified_preset(self, patch: Dict[str, Any]) -> AudioNode:
        carrier_info = patch["carrier"]
        c_freq = float(carrier_info.get("freq", self.base_freq))

        if "modulator" in patch:
            mod_info = patch["modulator"]
            m_freq = float(mod_info.get("freq", self.base_freq / 2.0))
            m_idx = float(mod_info.get("index", mod_info.get("mod_index", 2.0)))

            c_param = self._next_param("c_freq", 20.0, 5000.0, c_freq)
            mf_param = self._next_param("m_freq", 1.0, 5000.0, m_freq)
            mi_param = self._next_param("m_index", 0.0, 20.0, m_idx)
            gate = FMNode(carrier=c_param, mod_freq=mf_param, mod_index=mi_param)
        else:
            c_param = self._next_param("c_freq", 20.0, 5000.0, c_freq)
            p_param = self._next_param("c_phase", 0.0, 6.28318, 0.0)
            gate = SineNode(freq=c_param, phase=p_param)

        att = float(carrier_info.get("attack", 0.01))
        dec = float(carrier_info.get("decay", 0.2))
        sus = float(carrier_info.get("sustain", 0.7))
        rel = float(carrier_info.get("release", 0.5))

        return ADSRNode(
            attack=self._next_param("attack", 0.001, 5.0, att),
            decay=self._next_param("decay", 0.001, 5.0, dec),
            sustain=self._next_param("sustain", 0.0, 1.0, sus),
            release=self._next_param("release", 0.001, 5.0, rel),
            gate=gate,
        )

    def _parse_operators(self, patch: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Normalize various operator representations into a unified {op_idx: dict} mapping."""
        op_dict: Dict[int, Dict[str, Any]] = {}
        raw_ops = patch.get("operators", patch.get("oscillators", patch.get("op", [])))

        if isinstance(raw_ops, list):
            for i, op in enumerate(raw_ops, start=1):
                idx = int(op.get("operator", op.get("op", i)))
                op_dict[idx] = op
        elif isinstance(raw_ops, dict):
            for k, op in raw_ops.items():
                idx = int(k.replace("op", "").replace("operator", ""))
                op_dict[idx] = op

        # Check for flat op{i}_ keys
        for i in range(1, 7):
            if i not in op_dict:
                flat_op: Dict[str, Any] = {}
                prefix = f"op{i}_"
                for k, v in patch.items():
                    if k.startswith(prefix):
                        flat_op[k[len(prefix) :]] = v
                if flat_op:
                    op_dict[i] = flat_op

        return op_dict

    def _extract_frequency(self, op: Dict[str, Any]) -> float:
        if "freq" in op:
            return float(op["freq"])
        coarse = int(op.get("coarse", op.get("freq_coarse", 1)))
        fine = float(op.get("fine", op.get("freq_fine", 0.0)))
        mult = 0.5 if coarse == 0 else float(coarse)
        fine_mult = 1.0 + (fine / 100.0)
        return self.base_freq * mult * fine_mult

    def _extract_mod_index(self, op: Dict[str, Any]) -> float:
        if "index" in op:
            return float(op["index"])
        if "mod_index" in op:
            return float(op["mod_index"])
        output_level = float(op.get("output_level", op.get("level", 99.0)))
        return (output_level / 99.0) * 10.0

    def _extract_adsr(self, op: Dict[str, Any]) -> Tuple[float, float, float, float]:
        if "attack" in op:
            att = float(op["attack"])
            dec = float(op.get("decay", 0.2))
            sus = float(op.get("sustain", 0.7))
            rel = float(op.get("release", 0.5))
            return att, dec, sus, rel

        # Check EG rate & level format
        eg_rate = op.get("eg_rate", op.get("rates", None))
        eg_level = op.get("eg_level", op.get("levels", None))

        if eg_rate is not None and eg_level is not None:
            r1 = float(eg_rate[0]) if len(eg_rate) > 0 else 95.0
            r2 = float(eg_rate[1]) if len(eg_rate) > 1 else 20.0
            r4 = float(eg_rate[3]) if len(eg_rate) > 3 else 50.0
            l3 = float(eg_level[2]) if len(eg_level) > 2 else 80.0

            att = max(0.001, min(5.0, (99.0 - r1) / 99.0 * 2.0))
            dec = max(0.001, min(5.0, (99.0 - r2) / 99.0 * 2.0))
            sus = max(0.0, min(1.0, l3 / 99.0))
            rel = max(0.001, min(5.0, (99.0 - r4) / 99.0 * 3.0))
            return att, dec, sus, rel

        # Default envelope
        return 0.01, 0.2, 0.7, 0.5
