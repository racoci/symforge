"""Probabilistic Context-Free Grammar (PCFG) synthetic task generator for procedural synthesis."""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
)


@dataclass(frozen=True)
class GrammarRule:
    """Production rule in the probabilistic grammar."""

    lhs: str
    rhs_tag: str
    child_symbols: Tuple[str, ...]
    weight: float
    is_terminal: bool


class PCFGTaskGenerator:
    """Stochastic grammar engine generating procedural AST target programs."""

    def __init__(self, domain: str = "audio", seed: Optional[int] = None):
        if domain not in ("audio", "sdf"):
            raise ValueError(f"Unsupported domain: '{domain}'. Expected 'audio' or 'sdf'.")

        self.domain = domain
        self.rng = random.Random(seed)
        self.param_counter = 0
        self.rules: List[GrammarRule] = self._build_rules(domain)

    def _build_rules(self, domain: str) -> List[GrammarRule]:
        if domain == "audio":
            return [
                # Non-terminal audio rules
                GrammarRule("AudioExp", "LowPass", ("AudioExp",), weight=0.3, is_terminal=False),
                GrammarRule("AudioExp", "Gain", ("AudioExp",), weight=0.2, is_terminal=False),
                GrammarRule("AudioExp", "Mix", ("AudioExp", "AudioExp"), weight=0.2, is_terminal=False),
                GrammarRule("AudioExp", "FM", (), weight=0.15, is_terminal=True),
                GrammarRule("AudioExp", "Sine", (), weight=0.25, is_terminal=True),
                GrammarRule("AudioExp", "Saw", (), weight=0.25, is_terminal=True),
                GrammarRule("AudioExp", "Square", (), weight=0.15, is_terminal=True),
            ]
        else:  # sdf domain
            return [
                # Non-terminal SDF rules
                GrammarRule("SDFExp", "Union", ("SDFExp", "SDFExp"), weight=0.3, is_terminal=False),
                GrammarRule("SDFExp", "Intersection", ("SDFExp", "SDFExp"), weight=0.15, is_terminal=False),
                GrammarRule("SDFExp", "Difference", ("SDFExp", "SDFExp"), weight=0.15, is_terminal=False),
                GrammarRule("SDFExp", "Translate", ("SDFExp",), weight=0.2, is_terminal=False),
                GrammarRule("SDFExp", "Sphere", (), weight=0.35, is_terminal=True),
                GrammarRule("SDFExp", "Box", (), weight=0.3, is_terminal=True),
                GrammarRule("SDFExp", "Cylinder", (), weight=0.2, is_terminal=True),
                GrammarRule("SDFExp", "Torus", (), weight=0.15, is_terminal=True),
            ]

    def _next_param(self, name: str, min_val: float, max_val: float, default_val: float) -> FreeParam:
        self.param_counter += 1
        return FreeParam(
            id=f"{name}_{self.param_counter}",
            min_val=min_val,
            max_val=max_val,
            default_val=default_val,
        )

    def sample(self, max_depth: int = 4) -> ASTNode:
        """Sample a valid AST derivation tree from the grammar within depth bounds."""
        start_symbol = "AudioExp" if self.domain == "audio" else "SDFExp"
        return self._expand(start_symbol, current_depth=1, max_depth=max_depth)

    def _expand(self, symbol: str, current_depth: int, max_depth: int) -> ASTNode:
        matching_rules = [r for r in self.rules if r.lhs == symbol]
        if not matching_rules:
            raise RuntimeError(f"No grammar rules defined for symbol '{symbol}'")

        # Depth bounding constraint: force terminal if depth limit reached
        if current_depth >= max_depth:
            terminal_rules = [r for r in matching_rules if r.is_terminal]
            if terminal_rules:
                matching_rules = terminal_rules

        # Sample rule according to weights
        weights = [r.weight for r in matching_rules]
        chosen_rule: GrammarRule = self.rng.choices(matching_rules, weights=weights, k=1)[0]

        # Expand child subtrees
        children = [
            self._expand(c_sym, current_depth + 1, max_depth)
            for c_sym in chosen_rule.child_symbols
        ]

        return self._instantiate_node(chosen_rule.rhs_tag, children)

    def _instantiate_node(self, tag: str, children: List[ASTNode]) -> ASTNode:
        # Audio domain nodes
        if tag == "Sine":
            freq = self._next_param("freq", 20.0, 5000.0, self.rng.uniform(100.0, 1000.0))
            phase = self._next_param("phase", 0.0, 6.28, self.rng.uniform(0.0, 3.14))
            return SineNode(freq=freq, phase=phase)

        if tag == "Saw":
            freq = self._next_param("freq", 20.0, 5000.0, self.rng.uniform(100.0, 800.0))
            return SawNode(freq=freq)

        if tag == "Square":
            freq = self._next_param("freq", 20.0, 5000.0, self.rng.uniform(100.0, 800.0))
            duty = self._next_param("duty", 0.1, 0.9, 0.5)
            return SquareNode(freq=freq, duty=duty)

        if tag == "FM":
            carrier = self._next_param("fc", 50.0, 2000.0, self.rng.uniform(150.0, 600.0))
            mod_freq = self._next_param("fm", 10.0, 1000.0, self.rng.uniform(50.0, 300.0))
            mod_index = self._next_param("mi", 0.1, 10.0, self.rng.uniform(0.5, 4.0))
            return FMNode(carrier=carrier, mod_freq=mod_freq, mod_index=mod_index)

        if tag == "LowPass":
            cutoff = self._next_param("cutoff", 100.0, 10000.0, self.rng.uniform(500.0, 4000.0))
            res = self._next_param("res", 0.1, 10.0, self.rng.uniform(0.5, 2.0))
            return LowPassNode(cutoff=cutoff, resonance=res, input_node=children[0])  # type: ignore

        if tag == "Gain":
            gain = self._next_param("gain", 0.0, 2.0, self.rng.uniform(0.2, 1.0))
            return GainNode(gain=gain, input_node=children[0])  # type: ignore

        if tag == "Mix":
            ratio = self._next_param("ratio", 0.0, 1.0, 0.5)
            return MixNode(ratio=ratio, input1=children[0], input2=children[1])  # type: ignore

        # SDF domain nodes
        if tag == "Sphere":
            r = self._next_param("r", 0.1, 3.0, self.rng.uniform(0.5, 1.5))
            return SphereNode(radius=r)

        if tag == "Box":
            w = self._next_param("w", 0.1, 3.0, self.rng.uniform(0.4, 1.2))
            h = self._next_param("h", 0.1, 3.0, self.rng.uniform(0.4, 1.2))
            d = self._next_param("d", 0.1, 3.0, self.rng.uniform(0.4, 1.2))
            return BoxNode(width=w, height=h, depth=d)

        if tag == "Cylinder":
            r = self._next_param("r", 0.1, 3.0, self.rng.uniform(0.3, 1.0))
            h = self._next_param("h", 0.1, 3.0, self.rng.uniform(0.5, 1.5))
            return CylinderNode(radius=r, height=h)

        if tag == "Torus":
            r1 = self._next_param("r1", 0.5, 3.0, self.rng.uniform(1.0, 2.0))
            r2 = self._next_param("r2", 0.05, 0.8, self.rng.uniform(0.1, 0.4))
            return TorusNode(major_radius=r1, minor_radius=r2)

        if tag == "Translate":
            dx = self._next_param("dx", -2.0, 2.0, self.rng.uniform(-0.5, 0.5))
            dy = self._next_param("dy", -2.0, 2.0, self.rng.uniform(-0.5, 0.5))
            dz = self._next_param("dz", -2.0, 2.0, self.rng.uniform(-0.5, 0.5))
            return TranslateNode(dx=dx, dy=dy, dz=dz, child=children[0])  # type: ignore

        if tag == "Union":
            sm = self._next_param("k", 0.0, 0.5, self.rng.uniform(0.05, 0.25))
            return UnionNode(smoothness=sm, a=children[0], b=children[1])  # type: ignore

        if tag == "Intersection":
            return IntersectionNode(a=children[0], b=children[1])  # type: ignore

        if tag == "Difference":
            return DifferenceNode(a=children[0], b=children[1])  # type: ignore

        raise ValueError(f"Unknown production tag: {tag}")
