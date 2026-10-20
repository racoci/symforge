"""Procedural PCFG Curriculum Complexity Stratification and Triple Quality Filtering."""

from __future__ import annotations
import enum
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch

logger = logging.getLogger(__name__)

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
    serialize_node,
    deserialize_node,
)
from symforge.core.dag import DAGNode, ast_to_dag
from symforge.core.transpiler import transpile_ast_to_module


class CurriculumLevel(enum.IntEnum):
    """Discrete curriculum complexity levels based on AST tree depth and structural composition."""

    L1 = 1  # Depth 1: Isolated primitive
    L2 = 2  # Depth 2: Single modifier / transform
    L3 = 3  # Depth 3: Binary combination or nested modifier
    L4 = 4  # Depth 4: Nested modulation / blend
    L5 = 5  # Depth 5+: Multi-stage graph

    @classmethod
    def from_value(cls, val: Union[CurriculumLevel, int, str]) -> CurriculumLevel:
        """Parse curriculum level from enum, int, or string representation."""
        if isinstance(val, CurriculumLevel):
            return val
        if isinstance(val, int):
            return cls(val)
        if isinstance(val, str):
            clean = val.strip().upper()
            if clean in cls.__members__:
                return cls[clean]
            if clean.isdigit():
                return cls(int(clean))
        raise ValueError(f"Unknown curriculum level specification: {val}")

    @property
    def target_depth(self) -> int:
        """Expected tree depth (or minimum depth for L5)."""
        return int(self.value)

    @property
    def description(self) -> str:
        """Human-readable structural description."""
        descriptions = {
            CurriculumLevel.L1: "Depth 1: Isolated primitive",
            CurriculumLevel.L2: "Depth 2: Single modifier / transform",
            CurriculumLevel.L3: "Depth 3: Binary combination / composition",
            CurriculumLevel.L4: "Depth 4: Nested modulation / blend",
            CurriculumLevel.L5: "Depth 5+: Multi-stage graph",
        }
        return descriptions[self]


@dataclass(frozen=True)
class FilterResult:
    """Outcome of evaluating an AST candidate through a quality filter."""

    passed: bool
    reason: str = "ok"
    filter_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def canonicalize_ast_for_dedup(node: ASTNode, precision: int = 4) -> ASTNode:
    """Canonicalize continuous parameter IDs and round values for deterministic DAG deduplication."""
    raw = serialize_node(node)
    counter = [0]

    def _rec_canon(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "id" in obj and "minVal" in obj and "maxVal" in obj and "defaultVal" in obj:
                new_p = dict(obj)
                new_p["id"] = f"p{counter[0]}"
                counter[0] += 1
                new_p["defaultVal"] = round(float(obj["defaultVal"]), precision)
                return new_p
            return {k: _rec_canon(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_rec_canon(x) for x in obj]
        return obj

    canon_dict = _rec_canon(raw)
    return deserialize_node(canon_dict)


class PhysicalValidityFilter:
    """Filter 1: Validates numerical stability, clipping, silence, and geometric zero-crossings."""

    def __init__(
        self,
        domain: str = "audio",
        probe_res: int = 15,
        audio_duration: float = 0.5,
        sample_rate: int = 8000,
    ) -> None:
        self.domain = domain
        self.probe_res = probe_res
        self.audio_t = torch.linspace(0.0, audio_duration, int(audio_duration * sample_rate))
        self.sdf_probes, self.sdf_boundary = self._init_sdf_grid(probe_res)

    def _init_sdf_grid(self, res: int) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.linspace(-3.0, 3.0, res)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        probes = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)

        b_mask = (
            (torch.abs(gx) >= 2.95)
            | (torch.abs(gy) >= 2.95)
            | (torch.abs(gz) >= 2.95)
        )
        boundary_probes = torch.stack(
            [gx[b_mask].flatten(), gy[b_mask].flatten(), gz[b_mask].flatten()], dim=-1
        )
        return probes, boundary_probes

    def validate(self, node: ASTNode) -> FilterResult:
        """Validate physical correctness of the candidate AST."""
        try:
            mod = transpile_ast_to_module(node)
            with torch.no_grad():
                if isinstance(node, AudioNode):
                    return self._validate_audio(mod(self.audio_t))
                if isinstance(node, SDFNode):
                    return self._validate_sdf(mod(self.sdf_probes))
                return FilterResult(False, f"Unsupported AST node type: {type(node)}", "physical_validity")
        except Exception as exc:
            logger.warning("Physical validity check failed: %s (node=%s)", exc, type(node).__name__)
            return FilterResult(False, f"Runtime evaluation error: {exc}", "physical_validity")

    def _validate_audio(self, sig: torch.Tensor) -> FilterResult:
        if torch.isnan(sig).any() or torch.isinf(sig).any():
            return FilterResult(False, "Numerical divergence: signal contains NaN/Inf", "physical_validity")

        max_val = torch.max(torch.abs(sig)).item()
        if max_val > 1.001:
            return FilterResult(False, f"Digital clipping: max absolute amplitude {max_val:.3f} > 1.0", "physical_validity")

        if max_val < 0.02:
            return FilterResult(False, f"Silence: max absolute amplitude {max_val:.4f} < 0.02", "physical_validity")

        return FilterResult(True, "ok", "physical_validity", {"max_abs": max_val})

    def _validate_sdf(self, d: torch.Tensor) -> FilterResult:
        if torch.isnan(d).any() or torch.isinf(d).any():
            return FilterResult(False, "Numerical divergence: SDF contains NaN/Inf", "physical_validity")

        has_interior = (d < 0).any().item()
        has_exterior = (d > 0).any().item()
        if not (has_interior and has_exterior):
            return FilterResult(False, "Missing zero-crossing: shape is completely empty or completely filled", "physical_validity")

        return FilterResult(True, "ok", "physical_validity", {"interior_points": int((d < 0).sum().item())})


class PerceptualEntropyFilter:
    """Filter 2: Rejects perceptual triviality (flat DC offsets, near-zero spectral entropy, vanishing shapes)."""

    def __init__(
        self,
        domain: str = "audio",
        probe_res: int = 15,
        audio_duration: float = 0.5,
        sample_rate: int = 8000,
        min_spectral_entropy: float = 0.005,
        min_volume: float = 1e-4,
    ) -> None:
        self.domain = domain
        self.probe_res = probe_res
        self.min_spectral_entropy = min_spectral_entropy
        self.min_volume = min_volume
        self.audio_t = torch.linspace(0.0, audio_duration, int(audio_duration * sample_rate))
        self.sdf_probes, self.sdf_boundary = self._init_sdf_grid(probe_res)

    def _init_sdf_grid(self, res: int) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.linspace(-3.0, 3.0, res)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        probes = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)

        b_mask = (
            (torch.abs(gx) >= 2.95)
            | (torch.abs(gy) >= 2.95)
            | (torch.abs(gz) >= 2.95)
        )
        boundary_probes = torch.stack(
            [gx[b_mask].flatten(), gy[b_mask].flatten(), gz[b_mask].flatten()], dim=-1
        )
        return probes, boundary_probes

    def validate(self, node: ASTNode) -> FilterResult:
        """Validate perceptual non-triviality."""
        try:
            mod = transpile_ast_to_module(node)
            with torch.no_grad():
                if isinstance(node, AudioNode):
                    return self._validate_audio_entropy(mod(self.audio_t))
                if isinstance(node, SDFNode):
                    d_all = mod(self.sdf_probes)
                    d_bnd = mod(self.sdf_boundary)
                    return self._validate_sdf_volume(d_all, d_bnd)
                return FilterResult(False, f"Unsupported AST domain: {type(node)}", "perceptual_entropy")
        except Exception as exc:
            logger.warning("Perceptual entropy check failed: %s (node=%s)", exc, type(node).__name__)
            return FilterResult(False, f"Perceptual entropy check error: {exc}", "perceptual_entropy")

    def _validate_audio_entropy(self, sig: torch.Tensor) -> FilterResult:
        var = torch.var(sig).item()
        if var < 1e-5:
            return FilterResult(False, f"Trivial signal: flat DC offset with variance {var:.6e} < 1e-5", "perceptual_entropy")

        fft = torch.fft.rfft(sig)
        psd = torch.abs(fft) ** 2
        psd_sum = torch.sum(psd).item()
        if psd_sum < 1e-8:
            return FilterResult(False, "Negligible spectral energy", "perceptual_entropy")

        p = psd / psd_sum
        p_safe = torch.clamp(p, min=1e-12)
        entropy = -torch.sum(p_safe * torch.log2(p_safe)).item()
        max_entropy = math.log2(float(p.numel()))
        norm_entropy = entropy / max_entropy

        if norm_entropy < self.min_spectral_entropy:
            return FilterResult(False, f"Near-zero spectral entropy: {norm_entropy:.5f} < {self.min_spectral_entropy}", "perceptual_entropy")

        return FilterResult(True, "ok", "perceptual_entropy", {"spectral_entropy": norm_entropy, "variance": var})

    def _validate_sdf_volume(self, d_grid: torch.Tensor, d_boundary: torch.Tensor) -> FilterResult:
        interior_count = (d_grid < 0).sum().item()
        total_count = d_grid.numel()
        vol = (interior_count / total_count) * (6.0 ** 3)

        if vol < self.min_volume:
            return FilterResult(False, f"Vanishing shape volume: {vol:.6e} < {self.min_volume}", "perceptual_entropy")

        if (d_boundary < 0).all().item():
            return FilterResult(False, "Shape boundary outside standard bounding box [-3, 3]^3", "perceptual_entropy")

        return FilterResult(True, "ok", "perceptual_entropy", {"volume": vol})


class SemanticDeduplicationFilter:
    """Filter 3: Rejects structurally/semantically duplicate graphs and redundant subtrees via canonical DAG."""

    def __init__(self, precision: int = 4) -> None:
        self.precision = precision
        self.seen_signatures: Set[str] = set()

    def reset(self) -> None:
        """Reset the seen signatures registry for a new batch."""
        self.seen_signatures.clear()

    def validate(self, node: ASTNode) -> FilterResult:
        """Validate uniqueness against canonical DAG representation."""
        try:
            # Check for degenerate binary self-combination (e.g. Difference(A, A) = empty)
            if self._is_degenerate_self_op(node):
                return FilterResult(False, "Degenerate self-combination in binary node", "semantic_deduplication")

            canon_node = canonicalize_ast_for_dedup(node, precision=self.precision)
            dag = ast_to_dag(canon_node)
            sig = dag.signature

            if sig in self.seen_signatures:
                return FilterResult(False, f"Duplicate semantic DAG signature: {sig}", "semantic_deduplication")

            self.seen_signatures.add(sig)
            return FilterResult(True, "ok", "semantic_deduplication", {"signature": sig})
        except Exception as exc:
            logger.warning("Deduplication check failed: %s (node=%s)", exc, type(node).__name__)
            return FilterResult(False, f"Deduplication processing error: {exc}", "semantic_deduplication")

    def _is_degenerate_self_op(self, node: ASTNode) -> bool:
        if isinstance(node, (DifferenceNode, IntersectionNode)):
            sig_a = ast_to_dag(canonicalize_ast_for_dedup(node.a, precision=self.precision)).signature
            sig_b = ast_to_dag(canonicalize_ast_for_dedup(node.b, precision=self.precision)).signature
            return sig_a == sig_b
        return False


class TripleQualityFilter:
    """Unified pipeline executing Physical Validity, Perceptual Entropy, and Semantic Deduplication."""

    def __init__(
        self,
        domain: str = "audio",
        probe_res: int = 15,
        min_spectral_entropy: float = 0.005,
        min_volume: float = 1e-4,
    ) -> None:
        self.domain = domain
        self.physical = PhysicalValidityFilter(domain, probe_res=probe_res)
        self.entropy = PerceptualEntropyFilter(
            domain,
            probe_res=probe_res,
            min_spectral_entropy=min_spectral_entropy,
            min_volume=min_volume,
        )
        self.dedup = SemanticDeduplicationFilter()

    def reset(self) -> None:
        """Reset deduplication state for fresh batch generation."""
        self.dedup.reset()

    def validate(self, node: ASTNode) -> FilterResult:
        """Run the complete 3-stage validation pipeline on an AST node."""
        res_phys = self.physical.validate(node)
        if not res_phys.passed:
            return res_phys

        res_ent = self.entropy.validate(node)
        if not res_ent.passed:
            return res_ent

        res_dedup = self.dedup.validate(node)
        if not res_dedup.passed:
            return res_dedup

        return FilterResult(True, "ok", None, {"signature": res_dedup.metadata.get("signature")})


class CurriculumSampler:
    """Procedural PCFG sampler with depth stratification and quality filtering."""

    def __init__(
        self,
        domain: str = "audio",
        seed: Optional[int] = None,
        enable_filtering: bool = True,
        probe_res: int = 15,
    ) -> None:
        if domain not in ("audio", "sdf"):
            raise ValueError(f"Unsupported domain: '{domain}'. Expected 'audio' or 'sdf'.")

        self.domain = domain
        self.rng = random.Random(seed)
        self.param_counter = 0
        self.enable_filtering = enable_filtering
        self.filter = TripleQualityFilter(domain=domain, probe_res=probe_res)

    def _param(self, name: str, min_v: float, max_v: float, def_v: float) -> FreeParam:
        self.param_counter += 1
        return FreeParam(
            id=f"{name}_{self.param_counter}",
            min_val=min_v,
            max_val=max_v,
            default_val=round(def_v, 4),
        )

    # -------------------------------------------------------------
    # Audio Domain Stratified Generators (L1 to L5)
    # -------------------------------------------------------------

    def _sample_audio_l1(self) -> AudioNode:
        choice = self.rng.choice(["sine", "saw", "square", "noise"])
        if choice == "sine":
            f = self._param("f", 50.0, 2000.0, self.rng.uniform(100.0, 1000.0))
            phi = self._param("phi", 0.0, 6.28, self.rng.uniform(0.0, 3.14))
            return SineNode(freq=f, phase=phi)
        if choice == "saw":
            f = self._param("f", 50.0, 1500.0, self.rng.uniform(100.0, 800.0))
            return SawNode(freq=f)
        if choice == "square":
            f = self._param("f", 50.0, 1500.0, self.rng.uniform(100.0, 800.0))
            d = self._param("duty", 0.1, 0.9, self.rng.uniform(0.3, 0.7))
            return SquareNode(freq=f, duty=d)
        return NoiseNode(seed=self.rng.randint(1, 10000))

    def _sample_audio_l2(self) -> AudioNode:
        child = self._sample_audio_l1()
        choice = self.rng.choice(["lowpass", "gain", "adsr"])
        if choice == "lowpass":
            fc = self._param("fc", 200.0, 4000.0, self.rng.uniform(400.0, 2500.0))
            q = self._param("q", 0.5, 2.0, self.rng.uniform(0.7, 1.4))
            return LowPassNode(cutoff=fc, resonance=q, input_node=child)
        if choice == "gain":
            g = self._param("gain", 0.2, 0.95, self.rng.uniform(0.3, 0.9))
            return GainNode(gain=g, input_node=child)
        att = self._param("att", 0.01, 0.1, 0.02)
        dec = self._param("dec", 0.02, 0.2, 0.05)
        sus = self._param("sus", 0.4, 0.9, 0.7)
        rel = self._param("rel", 0.02, 0.2, 0.05)
        return ADSRNode(attack=att, decay=dec, sustain=sus, release=rel, gate=child)

    def _sample_audio_l3(self) -> AudioNode:
        choice = self.rng.choice(["mix", "lowpass", "gain", "adsr"])
        if choice == "mix":
            c1 = self._sample_audio_l2()
            c2 = self._sample_audio_l1()
            r = self._param("ratio", 0.2, 0.8, self.rng.uniform(0.3, 0.7))
            return MixNode(ratio=r, input1=c1, input2=c2)
        c = self._sample_audio_l2()
        if choice == "lowpass":
            fc = self._param("fc", 300.0, 4000.0, self.rng.uniform(500.0, 2500.0))
            q = self._param("q", 0.5, 2.0, 1.0)
            return LowPassNode(cutoff=fc, resonance=q, input_node=c)
        if choice == "gain":
            g = self._param("gain", 0.3, 0.95, 0.8)
            return GainNode(gain=g, input_node=c)
        att = self._param("att", 0.01, 0.1, 0.02)
        dec = self._param("dec", 0.02, 0.2, 0.05)
        sus = self._param("sus", 0.4, 0.9, 0.7)
        rel = self._param("rel", 0.02, 0.2, 0.05)
        return ADSRNode(attack=att, decay=dec, sustain=sus, release=rel, gate=c)

    def _sample_audio_l4(self) -> AudioNode:
        choice = self.rng.choice(["lowpass", "gain", "adsr", "mix"])
        if choice == "mix":
            c1 = self._sample_audio_l3()
            c2 = self._sample_audio_l2()
            r = self._param("ratio", 0.2, 0.8, self.rng.uniform(0.3, 0.7))
            return MixNode(ratio=r, input1=c1, input2=c2)
        c = self._sample_audio_l3()
        if choice == "lowpass":
            fc = self._param("fc", 300.0, 4000.0, self.rng.uniform(500.0, 2500.0))
            q = self._param("q", 0.5, 2.0, 1.0)
            return LowPassNode(cutoff=fc, resonance=q, input_node=c)
        if choice == "gain":
            g = self._param("gain", 0.3, 0.95, 0.8)
            return GainNode(gain=g, input_node=c)
        att = self._param("att", 0.01, 0.1, 0.02)
        dec = self._param("dec", 0.02, 0.2, 0.05)
        sus = self._param("sus", 0.4, 0.9, 0.7)
        rel = self._param("rel", 0.02, 0.2, 0.05)
        return ADSRNode(attack=att, decay=dec, sustain=sus, release=rel, gate=c)

    def _sample_audio_l5(self) -> AudioNode:
        choice = self.rng.choice(["mix", "lowpass", "adsr", "gain"])
        if choice == "mix":
            c1 = self._sample_audio_l4()
            c2 = self._sample_audio_l3()
            r = self._param("ratio", 0.2, 0.8, self.rng.uniform(0.3, 0.7))
            return MixNode(ratio=r, input1=c1, input2=c2)
        c = self._sample_audio_l4()
        if choice == "lowpass":
            fc = self._param("fc", 300.0, 4000.0, self.rng.uniform(500.0, 2500.0))
            q = self._param("q", 0.5, 2.0, 1.0)
            return LowPassNode(cutoff=fc, resonance=q, input_node=c)
        if choice == "gain":
            g = self._param("gain", 0.3, 0.95, 0.8)
            return GainNode(gain=g, input_node=c)
        att = self._param("att", 0.01, 0.1, 0.02)
        dec = self._param("dec", 0.02, 0.2, 0.05)
        sus = self._param("sus", 0.4, 0.9, 0.7)
        rel = self._param("rel", 0.02, 0.2, 0.05)
        return ADSRNode(attack=att, decay=dec, sustain=sus, release=rel, gate=c)

    # -------------------------------------------------------------
    # SDF Domain Stratified Generators (L1 to L5)
    # -------------------------------------------------------------

    def _sample_sdf_l1(self) -> SDFNode:
        choice = self.rng.choice(["sphere", "box", "cylinder", "torus"])
        if choice == "sphere":
            r = self._param("r", 0.3, 2.0, self.rng.uniform(0.5, 1.5))
            return SphereNode(radius=r)
        if choice == "box":
            w = self._param("w", 0.3, 2.0, self.rng.uniform(0.4, 1.2))
            h = self._param("h", 0.3, 2.0, self.rng.uniform(0.4, 1.2))
            d = self._param("d", 0.3, 2.0, self.rng.uniform(0.4, 1.2))
            return BoxNode(width=w, height=h, depth=d)
        if choice == "cylinder":
            r = self._param("r", 0.3, 1.5, self.rng.uniform(0.3, 1.0))
            h = self._param("h", 0.3, 2.0, self.rng.uniform(0.5, 1.5))
            return CylinderNode(radius=r, height=h)
        r1 = self._param("r1", 0.8, 1.6, self.rng.uniform(1.0, 1.4))
        r2 = self._param("r2", 0.15, 0.4, self.rng.uniform(0.2, 0.35))
        return TorusNode(major_radius=r1, minor_radius=r2)

    def _sample_sdf_l2(self) -> SDFNode:
        child = self._sample_sdf_l1()
        choice = self.rng.choice(["translate", "rotate", "scale"])
        if choice == "translate":
            dx = self._param("dx", -1.0, 1.0, self.rng.uniform(-0.5, 0.5))
            dy = self._param("dy", -1.0, 1.0, self.rng.uniform(-0.5, 0.5))
            dz = self._param("dz", -1.0, 1.0, self.rng.uniform(-0.5, 0.5))
            return TranslateNode(dx=dx, dy=dy, dz=dz, child=child)
        if choice == "rotate":
            rx = self._param("rx", -3.14, 3.14, self.rng.uniform(-1.57, 1.57))
            ry = self._param("ry", -3.14, 3.14, self.rng.uniform(-1.57, 1.57))
            rz = self._param("rz", -3.14, 3.14, self.rng.uniform(-1.57, 1.57))
            return RotateNode(rx=rx, ry=ry, rz=rz, child=child)
        s = self._param("s", 0.5, 1.5, self.rng.uniform(0.7, 1.3))
        return ScaleNode(s=s, child=child)

    def _sample_sdf_l3(self) -> SDFNode:
        choice = self.rng.choice(["union", "difference", "intersection", "translate", "rotate", "scale"])
        if choice in ("union", "difference", "intersection"):
            c1 = self._sample_sdf_l2()
            c2 = self._sample_sdf_l1()
            if choice == "union":
                sm = self._param("k", 0.02, 0.3, self.rng.uniform(0.05, 0.2))
                return UnionNode(smoothness=sm, a=c1, b=c2)
            if choice == "difference":
                return DifferenceNode(a=c1, b=c2)
            return IntersectionNode(a=c1, b=c2)
        c = self._sample_sdf_l2()
        if choice == "translate":
            dx = self._param("dx", -1.0, 1.0, self.rng.uniform(-0.4, 0.4))
            dy = self._param("dy", -1.0, 1.0, self.rng.uniform(-0.4, 0.4))
            dz = self._param("dz", -1.0, 1.0, self.rng.uniform(-0.4, 0.4))
            return TranslateNode(dx=dx, dy=dy, dz=dz, child=c)
        if choice == "rotate":
            rx = self._param("rx", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
            ry = self._param("ry", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
            rz = self._param("rz", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
            return RotateNode(rx=rx, ry=ry, rz=rz, child=c)
        s = self._param("s", 0.6, 1.4, self.rng.uniform(0.8, 1.2))
        return ScaleNode(s=s, child=c)

    def _sample_sdf_l4(self) -> SDFNode:
        choice = self.rng.choice(["union", "difference", "intersection", "translate", "rotate", "scale"])
        if choice in ("union", "difference", "intersection"):
            c1 = self._sample_sdf_l3()
            c2 = self._sample_sdf_l2()
            if choice == "union":
                sm = self._param("k", 0.02, 0.3, self.rng.uniform(0.05, 0.2))
                return UnionNode(smoothness=sm, a=c1, b=c2)
            if choice == "difference":
                return DifferenceNode(a=c1, b=c2)
            return IntersectionNode(a=c1, b=c2)
        c = self._sample_sdf_l3()
        if choice == "translate":
            dx = self._param("dx", -0.8, 0.8, self.rng.uniform(-0.3, 0.3))
            dy = self._param("dy", -0.8, 0.8, self.rng.uniform(-0.3, 0.3))
            dz = self._param("dz", -0.8, 0.8, self.rng.uniform(-0.3, 0.3))
            return TranslateNode(dx=dx, dy=dy, dz=dz, child=c)
        if choice == "rotate":
            rx = self._param("rx", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
            ry = self._param("ry", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
            rz = self._param("rz", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
            return RotateNode(rx=rx, ry=ry, rz=rz, child=c)
        s = self._param("s", 0.7, 1.3, self.rng.uniform(0.85, 1.15))
        return ScaleNode(s=s, child=c)

    def _sample_sdf_l5(self) -> SDFNode:
        choice = self.rng.choice(["union", "difference", "translate", "rotate"])
        if choice in ("union", "difference"):
            c1 = self._sample_sdf_l4()
            c2 = self._sample_sdf_l3()
            if choice == "union":
                sm = self._param("k", 0.02, 0.25, self.rng.uniform(0.05, 0.15))
                return UnionNode(smoothness=sm, a=c1, b=c2)
            return DifferenceNode(a=c1, b=c2)
        c = self._sample_sdf_l4()
        if choice == "translate":
            dx = self._param("dx", -0.6, 0.6, self.rng.uniform(-0.3, 0.3))
            dy = self._param("dy", -0.6, 0.6, self.rng.uniform(-0.3, 0.3))
            dz = self._param("dz", -0.6, 0.6, self.rng.uniform(-0.3, 0.3))
            return TranslateNode(dx=dx, dy=dy, dz=dz, child=c)
        rx = self._param("rx", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
        ry = self._param("ry", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
        rz = self._param("rz", -3.14, 3.14, self.rng.uniform(-1.0, 1.0))
        return RotateNode(rx=rx, ry=ry, rz=rz, child=c)

    # -------------------------------------------------------------
    # Unified Stratified Dispatcher
    # -------------------------------------------------------------

    def sample_raw(self, level: Union[CurriculumLevel, int, str]) -> ASTNode:
        """Sample an unvalidated raw AST candidate matching the target curriculum level."""
        lvl = CurriculumLevel.from_value(level)
        generators = {
            ("audio", CurriculumLevel.L1): self._sample_audio_l1,
            ("audio", CurriculumLevel.L2): self._sample_audio_l2,
            ("audio", CurriculumLevel.L3): self._sample_audio_l3,
            ("audio", CurriculumLevel.L4): self._sample_audio_l4,
            ("audio", CurriculumLevel.L5): self._sample_audio_l5,
            ("sdf", CurriculumLevel.L1): self._sample_sdf_l1,
            ("sdf", CurriculumLevel.L2): self._sample_sdf_l2,
            ("sdf", CurriculumLevel.L3): self._sample_sdf_l3,
            ("sdf", CurriculumLevel.L4): self._sample_sdf_l4,
            ("sdf", CurriculumLevel.L5): self._sample_sdf_l5,
        }
        gen = generators.get((self.domain, lvl))
        if gen is None:
            raise ValueError(f"No generator registered for domain={self.domain}, level={lvl.name}")
        return gen()

    def sample_level(
        self,
        level: Union[CurriculumLevel, int, str],
        max_attempts: int = 100,
    ) -> ASTNode:
        """Sample a validated AST candidate satisfying all quality filters for the specified level."""
        lvl = CurriculumLevel.from_value(level)
        attempts = 0
        last_reason = "No attempts made"
        last_filter = "none"

        while attempts < max_attempts:
            attempts += 1
            candidate = self.sample_raw(lvl)

            # Enforce depth invariant
            if lvl == CurriculumLevel.L5:
                if candidate.tree_depth < 5:
                    continue
            elif candidate.tree_depth != lvl.target_depth:
                continue

            if not self.enable_filtering:
                return candidate

            res = self.filter.validate(candidate)
            if res.passed:
                return candidate

            last_reason = res.reason
            last_filter = res.filter_id or "unknown"

        raise RuntimeError(
            f"Failed to generate valid candidate for level={lvl.name} in domain={self.domain} "
            f"within {max_attempts} attempts. Last rejection by filter='{last_filter}': {last_reason}"
        )

    def generate_batch(
        self,
        level: Union[CurriculumLevel, int, str],
        batch_size: int = 100,
        max_attempts: int = 2000,
        reset_dedup: bool = True,
    ) -> List[ASTNode]:
        """Generate a batch of unique, validated derivations for the given curriculum level."""
        lvl = CurriculumLevel.from_value(level)
        if reset_dedup:
            self.filter.reset()

        batch: List[ASTNode] = []
        attempts = 0

        while len(batch) < batch_size and attempts < max_attempts:
            attempts += 1
            try:
                candidate = self.sample_level(lvl, max_attempts=50)
                batch.append(candidate)
            except RuntimeError as err:
                logger.warning("Sampling attempt %d for level %s failed: %s", attempts, lvl.name, err)
                continue

        if len(batch) < batch_size:
            raise RuntimeError(
                f"Generated only {len(batch)}/{batch_size} candidates for {lvl.name} in {attempts} attempts."
            )
        return batch

    def generate_curriculum(
        self,
        samples_per_level: int = 100,
    ) -> Dict[CurriculumLevel, List[ASTNode]]:
        """Generate a full curriculum dataset across all 5 discrete complexity levels."""
        dataset: Dict[CurriculumLevel, List[ASTNode]] = {}
        for lvl in CurriculumLevel:
            dataset[lvl] = self.generate_batch(lvl, batch_size=samples_per_level)
        return dataset
