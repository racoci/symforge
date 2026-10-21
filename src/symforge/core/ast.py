"""Typed Abstract Syntax Tree (AST) representations mirroring Lean 4 inductive definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type


@dataclass(frozen=True)
class FreeParam:
    """Continuous free parameter subject to optimization."""

    id: str
    min_val: float
    max_val: float
    default_val: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "minVal": float(self.min_val),
            "maxVal": float(self.max_val),
            "defaultVal": float(self.default_val),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FreeParam:
        return cls(
            id=str(data["id"]),
            min_val=float(data["minVal"]),
            max_val=float(data["maxVal"]),
            default_val=float(data["defaultVal"]),
        )


@dataclass(frozen=True)
class ASTNode:
    """Base class for all immutable AST nodes in SymForge."""

    @property
    def size(self) -> int:
        count = 1
        for child in self.children:
            count += child.size
        return count

    @property
    def tree_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.tree_depth for child in self.children)

    def __getattr__(self, name: str) -> Any:
        if name == "depth":
            return self.tree_depth
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return ()

    def collect_params(self) -> List[FreeParam]:
        params: List[FreeParam] = []
        for fld in self._free_param_fields():
            params.append(fld)
        for child in self.children:
            params.extend(child.collect_params())
        return params

    def _free_param_fields(self) -> List[FreeParam]:
        return []

    def to_lean_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_lean_dict(cls, data: Dict[str, Any]) -> ASTNode:
        raise NotImplementedError


# ==========================================
# Audio Domain Nodes
# ==========================================


@dataclass(frozen=True)
class AudioNode(ASTNode):
    pass


@dataclass(frozen=True)
class SineNode(AudioNode):
    freq: FreeParam
    phase: FreeParam = field(default_factory=lambda: FreeParam("phase_0", 0.0, 6.28318, 0.0))

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.freq, self.phase]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {"sine": {"freq": self.freq.to_dict(), "phase": self.phase.to_dict()}}


@dataclass(frozen=True)
class SawNode(AudioNode):
    freq: FreeParam

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.freq]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {"saw": {"freq": self.freq.to_dict()}}


@dataclass(frozen=True)
class SquareNode(AudioNode):
    freq: FreeParam
    duty: FreeParam = field(default_factory=lambda: FreeParam("duty_0", 0.01, 0.99, 0.5))

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.freq, self.duty]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {"square": {"freq": self.freq.to_dict(), "duty": self.duty.to_dict()}}


@dataclass(frozen=True)
class NoiseNode(AudioNode):
    seed: int = 42

    def to_lean_dict(self) -> Dict[str, Any]:
        return {"noise": {"seed": self.seed}}


@dataclass(frozen=True)
class ADSRNode(AudioNode):
    attack: FreeParam
    decay: FreeParam
    sustain: FreeParam
    release: FreeParam
    gate: AudioNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.gate,)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.attack, self.decay, self.sustain, self.release]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "adsr": {
                "attack": self.attack.to_dict(),
                "decay": self.decay.to_dict(),
                "sustain": self.sustain.to_dict(),
                "release": self.release.to_dict(),
                "gate": self.gate.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class LowPassNode(AudioNode):
    cutoff: FreeParam
    resonance: FreeParam
    input_node: AudioNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.input_node,)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.cutoff, self.resonance]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "lowpass": {
                "cutoff": self.cutoff.to_dict(),
                "resonance": self.resonance.to_dict(),
                "input": self.input_node.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class GainNode(AudioNode):
    gain: FreeParam
    input_node: AudioNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.input_node,)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.gain]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "gain": {
                "gain": self.gain.to_dict(),
                "input": self.input_node.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class MixNode(AudioNode):
    ratio: FreeParam
    input1: AudioNode
    input2: AudioNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.input1, self.input2)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.ratio]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "mix": {
                "ratio": self.ratio.to_dict(),
                "input1": self.input1.to_lean_dict(),
                "input2": self.input2.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class FMNode(AudioNode):
    carrier: FreeParam
    mod_freq: FreeParam
    mod_index: FreeParam

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.carrier, self.mod_freq, self.mod_index]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "fm": {
                "carrier": self.carrier.to_dict(),
                "modFreq": self.mod_freq.to_dict(),
                "modIndex": self.mod_index.to_dict(),
            }
        }


@dataclass(frozen=True)
class ExpDecayNode(AudioNode):
    decay_time: FreeParam
    input_node: AudioNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.input_node,)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.decay_time]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "expDecay": {
                "decayTime": self.decay_time.to_dict(),
                "input": self.input_node.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class LFONode(AudioNode):
    freq: FreeParam
    amount: FreeParam = field(default_factory=lambda: FreeParam("amount_0", 0.0, 1.0, 0.5))

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.freq, self.amount]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "lfo": {
                "freq": self.freq.to_dict(),
                "amount": self.amount.to_dict(),
            }
        }


# ==========================================
# 3D SDF Domain Nodes
# ==========================================


@dataclass(frozen=True)
class SDFNode(ASTNode):
    pass


@dataclass(frozen=True)
class SphereNode(SDFNode):
    radius: FreeParam

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.radius]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {"sphere": {"radius": self.radius.to_dict()}}


@dataclass(frozen=True)
class BoxNode(SDFNode):
    width: FreeParam
    height: FreeParam
    depth: FreeParam

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.width, self.height, self.depth]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "box": {
                "width": self.width.to_dict(),
                "height": self.height.to_dict(),
                "depth": self.depth.to_dict(),
            }
        }


@dataclass(frozen=True)
class CylinderNode(SDFNode):
    radius: FreeParam
    height: FreeParam

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.radius, self.height]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "cylinder": {
                "radius": self.radius.to_dict(),
                "height": self.height.to_dict(),
            }
        }


@dataclass(frozen=True)
class TorusNode(SDFNode):
    major_radius: FreeParam
    minor_radius: FreeParam

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.major_radius, self.minor_radius]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "torus": {
                "majorRadius": self.major_radius.to_dict(),
                "minorRadius": self.minor_radius.to_dict(),
            }
        }


@dataclass(frozen=True)
class PlaneNode(SDFNode):
    nx: FreeParam = field(default_factory=lambda: FreeParam("nx_0", -1.0, 1.0, 0.0))
    ny: FreeParam = field(default_factory=lambda: FreeParam("ny_0", -1.0, 1.0, 1.0))
    nz: FreeParam = field(default_factory=lambda: FreeParam("nz_0", -1.0, 1.0, 0.0))
    offset: FreeParam = field(default_factory=lambda: FreeParam("offset_0", -5.0, 5.0, 0.0))

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.nx, self.ny, self.nz, self.offset]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "plane": {
                "nx": self.nx.to_dict(),
                "ny": self.ny.to_dict(),
                "nz": self.nz.to_dict(),
                "offset": self.offset.to_dict(),
            }
        }


@dataclass(frozen=True)
class TranslateNode(SDFNode):
    dx: FreeParam
    dy: FreeParam
    dz: FreeParam
    child: SDFNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.child,)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.dx, self.dy, self.dz]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "translate": {
                "dx": self.dx.to_dict(),
                "dy": self.dy.to_dict(),
                "dz": self.dz.to_dict(),
                "child": self.child.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class RotateNode(SDFNode):
    rx: FreeParam
    ry: FreeParam
    rz: FreeParam
    child: SDFNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.child,)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.rx, self.ry, self.rz]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "rotate": {
                "rx": self.rx.to_dict(),
                "ry": self.ry.to_dict(),
                "rz": self.rz.to_dict(),
                "child": self.child.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class ScaleNode(SDFNode):
    s: FreeParam
    child: SDFNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.child,)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.s]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "scale": {
                "s": self.s.to_dict(),
                "child": self.child.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class UnionNode(SDFNode):
    smoothness: FreeParam
    a: SDFNode
    b: SDFNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.a, self.b)

    def _free_param_fields(self) -> List[FreeParam]:
        return [self.smoothness]

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "union": {
                "smoothness": self.smoothness.to_dict(),
                "a": self.a.to_lean_dict(),
                "b": self.b.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class SmoothUnionNode(UnionNode):
    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "smoothUnion": {
                "smoothness": self.smoothness.to_dict(),
                "a": self.a.to_lean_dict(),
                "b": self.b.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class IntersectionNode(SDFNode):
    a: SDFNode
    b: SDFNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.a, self.b)

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "intersection": {
                "a": self.a.to_lean_dict(),
                "b": self.b.to_lean_dict(),
            }
        }


@dataclass(frozen=True)
class DifferenceNode(SDFNode):
    a: SDFNode
    b: SDFNode

    @property
    def children(self) -> Tuple[ASTNode, ...]:
        return (self.a, self.b)

    def to_lean_dict(self) -> Dict[str, Any]:
        return {
            "difference": {
                "a": self.a.to_lean_dict(),
                "b": self.b.to_lean_dict(),
            }
        }


# ==========================================
# Placeholder Holes & Partial Programs
# ==========================================


@dataclass(frozen=True)
class HoleNode(ASTNode):
    """Base class for typed placeholder holes in partial AST expressions."""

    hole_id: str

    def to_lean_dict(self) -> Dict[str, Any]:
        return {"hole": {"id": self.hole_id}}


@dataclass(frozen=True)
class AudioHoleNode(AudioNode, HoleNode):
    """Typed hole representing an unexpanded AudioNode."""

    hole_id: str

    def to_lean_dict(self) -> Dict[str, Any]:
        return {"audioHole": {"id": self.hole_id}}


@dataclass(frozen=True)
class SDFHoleNode(SDFNode, HoleNode):
    """Typed hole representing an unexpanded SDFNode."""

    hole_id: str

    def to_lean_dict(self) -> Dict[str, Any]:
        return {"sdfHole": {"id": self.hole_id}}


def find_holes(node: ASTNode) -> List[HoleNode]:
    """Find all hole nodes in depth-first order within an AST."""
    holes: List[HoleNode] = []
    if isinstance(node, HoleNode):
        holes.append(node)
    for child in node.children:
        holes.extend(find_holes(child))
    return holes


def has_holes(node: ASTNode) -> bool:
    """Check if an AST contains any unexpanded holes."""
    if isinstance(node, HoleNode):
        return True
    return any(has_holes(child) for child in node.children)


def substitute_hole(node: ASTNode, hole_id: str, replacement: ASTNode) -> ASTNode:
    """Substitute a specific hole identified by hole_id with a replacement ASTNode."""
    if isinstance(node, HoleNode) and node.hole_id == hole_id:
        return replacement

    # Audio composite nodes
    if isinstance(node, ADSRNode):
        return ADSRNode(
            attack=node.attack,
            decay=node.decay,
            sustain=node.sustain,
            release=node.release,
            gate=substitute_hole(node.gate, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, ExpDecayNode):
        return ExpDecayNode(
            decay_time=node.decay_time,
            input_node=substitute_hole(node.input_node, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, LowPassNode):
        return LowPassNode(
            cutoff=node.cutoff,
            resonance=node.resonance,
            input_node=substitute_hole(node.input_node, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, GainNode):
        return GainNode(
            gain=node.gain,
            input_node=substitute_hole(node.input_node, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, MixNode):
        return MixNode(
            ratio=node.ratio,
            input1=substitute_hole(node.input1, hole_id, replacement),  # type: ignore
            input2=substitute_hole(node.input2, hole_id, replacement),  # type: ignore
        )

    # SDF composite nodes
    if isinstance(node, TranslateNode):
        return TranslateNode(
            dx=node.dx,
            dy=node.dy,
            dz=node.dz,
            child=substitute_hole(node.child, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, RotateNode):
        return RotateNode(
            rx=node.rx,
            ry=node.ry,
            rz=node.rz,
            child=substitute_hole(node.child, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, ScaleNode):
        return ScaleNode(
            s=node.s,
            child=substitute_hole(node.child, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, SmoothUnionNode):
        return SmoothUnionNode(
            smoothness=node.smoothness,
            a=substitute_hole(node.a, hole_id, replacement),  # type: ignore
            b=substitute_hole(node.b, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, UnionNode):
        return UnionNode(
            smoothness=node.smoothness,
            a=substitute_hole(node.a, hole_id, replacement),  # type: ignore
            b=substitute_hole(node.b, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, IntersectionNode):
        return IntersectionNode(
            a=substitute_hole(node.a, hole_id, replacement),  # type: ignore
            b=substitute_hole(node.b, hole_id, replacement),  # type: ignore
        )
    if isinstance(node, DifferenceNode):
        return DifferenceNode(
            a=substitute_hole(node.a, hole_id, replacement),  # type: ignore
            b=substitute_hole(node.b, hole_id, replacement),  # type: ignore
        )

    return node


def _update_param(p: FreeParam, param_values: Dict[str, float]) -> FreeParam:
    if p.id in param_values:
        return FreeParam(
            id=p.id,
            min_val=p.min_val,
            max_val=p.max_val,
            default_val=float(param_values[p.id]),
        )
    return p


def update_ast_params(node: ASTNode, param_values: Dict[str, float]) -> ASTNode:
    """Return a new AST with updated default values for continuous FreeParams."""
    if isinstance(node, SineNode):
        return SineNode(
            freq=_update_param(node.freq, param_values),
            phase=_update_param(node.phase, param_values),
        )
    if isinstance(node, SawNode):
        return SawNode(freq=_update_param(node.freq, param_values))
    if isinstance(node, SquareNode):
        return SquareNode(
            freq=_update_param(node.freq, param_values),
            duty=_update_param(node.duty, param_values),
        )
    if isinstance(node, ADSRNode):
        return ADSRNode(
            attack=_update_param(node.attack, param_values),
            decay=_update_param(node.decay, param_values),
            sustain=_update_param(node.sustain, param_values),
            release=_update_param(node.release, param_values),
            gate=update_ast_params(node.gate, param_values),  # type: ignore
        )
    if isinstance(node, LowPassNode):
        return LowPassNode(
            cutoff=_update_param(node.cutoff, param_values),
            resonance=_update_param(node.resonance, param_values),
            input_node=update_ast_params(node.input_node, param_values),  # type: ignore
        )
    if isinstance(node, GainNode):
        return GainNode(
            gain=_update_param(node.gain, param_values),
            input_node=update_ast_params(node.input_node, param_values),  # type: ignore
        )
    if isinstance(node, MixNode):
        return MixNode(
            ratio=_update_param(node.ratio, param_values),
            input1=update_ast_params(node.input1, param_values),  # type: ignore
            input2=update_ast_params(node.input2, param_values),  # type: ignore
        )
    if isinstance(node, FMNode):
        return FMNode(
            carrier=_update_param(node.carrier, param_values),
            mod_freq=_update_param(node.mod_freq, param_values),
            mod_index=_update_param(node.mod_index, param_values),
        )
    if isinstance(node, ExpDecayNode):
        return ExpDecayNode(
            decay_time=_update_param(node.decay_time, param_values),
            input_node=update_ast_params(node.input_node, param_values),  # type: ignore
        )
    if isinstance(node, LFONode):
        return LFONode(
            freq=_update_param(node.freq, param_values),
            amount=_update_param(node.amount, param_values),
        )
    if isinstance(node, SphereNode):
        return SphereNode(radius=_update_param(node.radius, param_values))
    if isinstance(node, BoxNode):
        return BoxNode(
            width=_update_param(node.width, param_values),
            height=_update_param(node.height, param_values),
            depth=_update_param(node.depth, param_values),
        )
    if isinstance(node, CylinderNode):
        return CylinderNode(
            radius=_update_param(node.radius, param_values),
            height=_update_param(node.height, param_values),
        )
    if isinstance(node, TorusNode):
        return TorusNode(
            major_radius=_update_param(node.major_radius, param_values),
            minor_radius=_update_param(node.minor_radius, param_values),
        )
    if isinstance(node, PlaneNode):
        return PlaneNode(
            nx=_update_param(node.nx, param_values),
            ny=_update_param(node.ny, param_values),
            nz=_update_param(node.nz, param_values),
            offset=_update_param(node.offset, param_values),
        )
    if isinstance(node, TranslateNode):
        return TranslateNode(
            dx=_update_param(node.dx, param_values),
            dy=_update_param(node.dy, param_values),
            dz=_update_param(node.dz, param_values),
            child=update_ast_params(node.child, param_values),  # type: ignore
        )
    if isinstance(node, RotateNode):
        return RotateNode(
            rx=_update_param(node.rx, param_values),
            ry=_update_param(node.ry, param_values),
            rz=_update_param(node.rz, param_values),
            child=update_ast_params(node.child, param_values),  # type: ignore
        )
    if isinstance(node, ScaleNode):
        return ScaleNode(
            s=_update_param(node.s, param_values),
            child=update_ast_params(node.child, param_values),  # type: ignore
        )
    if isinstance(node, SmoothUnionNode):
        return SmoothUnionNode(
            smoothness=_update_param(node.smoothness, param_values),
            a=update_ast_params(node.a, param_values),  # type: ignore
            b=update_ast_params(node.b, param_values),  # type: ignore
        )
    if isinstance(node, UnionNode):
        return UnionNode(
            smoothness=_update_param(node.smoothness, param_values),
            a=update_ast_params(node.a, param_values),  # type: ignore
            b=update_ast_params(node.b, param_values),  # type: ignore
        )
    if isinstance(node, IntersectionNode):
        return IntersectionNode(
            a=update_ast_params(node.a, param_values),  # type: ignore
            b=update_ast_params(node.b, param_values),  # type: ignore
        )
    if isinstance(node, DifferenceNode):
        return DifferenceNode(
            a=update_ast_params(node.a, param_values),  # type: ignore
            b=update_ast_params(node.b, param_values),  # type: ignore
        )

    return node


# ==========================================
# Serialization & Deserialization
# ==========================================


def serialize_node(node: ASTNode) -> Dict[str, Any]:
    """Serialize an AST node to Lean 4 JSON schema representation."""
    return node.to_lean_dict()


def deserialize_node(data: Dict[str, Any]) -> ASTNode:
    """Deserialize an AST node from Lean 4 JSON schema representation."""
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Invalid Lean AST JSON encoding: {data}")

    tag, fields = next(iter(data.items()))

    # Audio nodes
    if tag == "sine":
        return SineNode(
            freq=FreeParam.from_dict(fields["freq"]),
            phase=FreeParam.from_dict(fields["phase"]),
        )
    if tag == "saw":
        return SawNode(freq=FreeParam.from_dict(fields["freq"]))
    if tag == "square":
        return SquareNode(
            freq=FreeParam.from_dict(fields["freq"]),
            duty=FreeParam.from_dict(fields["duty"]),
        )
    if tag == "noise":
        return NoiseNode(seed=int(fields.get("seed", 42)))
    if tag == "adsr":
        return ADSRNode(
            attack=FreeParam.from_dict(fields["attack"]),
            decay=FreeParam.from_dict(fields["decay"]),
            sustain=FreeParam.from_dict(fields["sustain"]),
            release=FreeParam.from_dict(fields["release"]),
            gate=deserialize_node(fields["gate"]),  # type: ignore
        )
    if tag == "lowpass":
        return LowPassNode(
            cutoff=FreeParam.from_dict(fields["cutoff"]),
            resonance=FreeParam.from_dict(fields["resonance"]),
            input_node=deserialize_node(fields["input"]),  # type: ignore
        )
    if tag == "gain":
        return GainNode(
            gain=FreeParam.from_dict(fields["gain"]),
            input_node=deserialize_node(fields["input"]),  # type: ignore
        )
    if tag == "mix":
        return MixNode(
            ratio=FreeParam.from_dict(fields["ratio"]),
            input1=deserialize_node(fields["input1"]),  # type: ignore
            input2=deserialize_node(fields["input2"]),  # type: ignore
        )
    if tag == "fm":
        return FMNode(
            carrier=FreeParam.from_dict(fields["carrier"]),
            mod_freq=FreeParam.from_dict(fields["modFreq"]),
            mod_index=FreeParam.from_dict(fields["modIndex"]),
        )
    if tag == "expDecay":
        return ExpDecayNode(
            decay_time=FreeParam.from_dict(fields["decayTime"]),
            input_node=deserialize_node(fields["input"]),  # type: ignore
        )
    if tag == "lfo":
        return LFONode(
            freq=FreeParam.from_dict(fields["freq"]),
            amount=FreeParam.from_dict(fields["amount"]),
        )

    # SDF nodes
    if tag == "sphere":
        return SphereNode(radius=FreeParam.from_dict(fields["radius"]))
    if tag == "box":
        return BoxNode(
            width=FreeParam.from_dict(fields["width"]),
            height=FreeParam.from_dict(fields["height"]),
            depth=FreeParam.from_dict(fields["depth"]),
        )
    if tag == "cylinder":
        return CylinderNode(
            radius=FreeParam.from_dict(fields["radius"]),
            height=FreeParam.from_dict(fields["height"]),
        )
    if tag == "torus":
        return TorusNode(
            major_radius=FreeParam.from_dict(fields["majorRadius"]),
            minor_radius=FreeParam.from_dict(fields["minorRadius"]),
        )
    if tag == "plane":
        return PlaneNode(
            nx=FreeParam.from_dict(fields["nx"]),
            ny=FreeParam.from_dict(fields["ny"]),
            nz=FreeParam.from_dict(fields["nz"]),
            offset=FreeParam.from_dict(fields["offset"]),
        )
    if tag == "translate":
        return TranslateNode(
            dx=FreeParam.from_dict(fields["dx"]),
            dy=FreeParam.from_dict(fields["dy"]),
            dz=FreeParam.from_dict(fields["dz"]),
            child=deserialize_node(fields["child"]),  # type: ignore
        )
    if tag == "rotate":
        return RotateNode(
            rx=FreeParam.from_dict(fields["rx"]),
            ry=FreeParam.from_dict(fields["ry"]),
            rz=FreeParam.from_dict(fields["rz"]),
            child=deserialize_node(fields["child"]),  # type: ignore
        )
    if tag == "scale":
        return ScaleNode(
            s=FreeParam.from_dict(fields["s"]),
            child=deserialize_node(fields["child"]),  # type: ignore
        )
    if tag == "union":
        return UnionNode(
            smoothness=FreeParam.from_dict(fields["smoothness"]),
            a=deserialize_node(fields["a"]),  # type: ignore
            b=deserialize_node(fields["b"]),  # type: ignore
        )
    if tag == "smoothUnion":
        return SmoothUnionNode(
            smoothness=FreeParam.from_dict(fields["smoothness"]),
            a=deserialize_node(fields["a"]),  # type: ignore
            b=deserialize_node(fields["b"]),  # type: ignore
        )
    if tag == "intersection":
        return IntersectionNode(
            a=deserialize_node(fields["a"]),  # type: ignore
            b=deserialize_node(fields["b"]),  # type: ignore
        )
    if tag == "difference":
        return DifferenceNode(
            a=deserialize_node(fields["a"]),  # type: ignore
            b=deserialize_node(fields["b"]),  # type: ignore
        )

    # Hole nodes
    if tag == "audioHole":
        return AudioHoleNode(hole_id=str(fields["id"]))
    if tag == "sdfHole":
        return SDFHoleNode(hole_id=str(fields["id"]))
    if tag == "hole":
        return HoleNode(hole_id=str(fields["id"]))

    raise ValueError(f"Unknown node constructor tag: {tag}")
