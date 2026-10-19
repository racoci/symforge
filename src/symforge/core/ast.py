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

    raise ValueError(f"Unknown node constructor tag: {tag}")
