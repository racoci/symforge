"""Differentiable PyTorch compilation engine for SymForge AST representations."""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

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


class DifferentiableProgram(nn.Module):
    """Base class for AST graphs compiled into differentiable PyTorch modules."""

    def __init__(self, root: ASTNode):
        super().__init__()
        self.root = root
        # Deduplicate parameters while preserving stable order
        params = root.collect_params()
        seen = set()
        unique_params: List[FreeParam] = []
        for p in params:
            if p.id not in seen:
                seen.add(p.id)
                unique_params.append(p)

        self.param_specs = unique_params
        self.param_map = {p.id: i for i, p in enumerate(unique_params)}

        # Continuous parameter vector Theta initialized to default values
        init_vals = [p.default_val for p in unique_params]
        self.theta = nn.Parameter(torch.tensor(init_vals, dtype=torch.float32))

    def get_param(self, p: FreeParam) -> torch.Tensor:
        """Fetch parameter value from Theta tensor."""
        idx = self.param_map[p.id]
        return self.theta[idx]


class DifferentiableAudioModule(DifferentiableProgram):
    """Compiles an AudioNode AST into a differentiable 1D waveform synthesizer."""

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute synthesized audio output over time tensor t in seconds."""
        return self._eval(self.root, t)

    def _eval(self, node: ASTNode, t: torch.Tensor) -> torch.Tensor:
        if isinstance(node, SineNode):
            f = self.get_param(node.freq)
            phi = self.get_param(node.phase)
            return torch.sin(2.0 * math.pi * f * t + phi)

        if isinstance(node, SawNode):
            f = self.get_param(node.freq)
            # Differentiable sawtooth approximation
            phase = f * t
            return 2.0 * (phase - torch.floor(phase + 0.5))

        if isinstance(node, SquareNode):
            f = self.get_param(node.freq)
            return torch.tanh(15.0 * torch.sin(2.0 * math.pi * f * t))

        if isinstance(node, NoiseNode):
            g = torch.Generator().manual_seed(node.seed)
            noise = torch.rand(t.shape, generator=g, device=t.device, dtype=t.dtype) * 2.0 - 1.0
            return torch.clamp(noise, -1.0, 1.0)

        if isinstance(node, ADSRNode):
            att = self.get_param(node.attack)
            dec = self.get_param(node.decay)
            sus = self.get_param(node.sustain)
            rel = self.get_param(node.release)
            env = self._adsr_envelope(t, att, dec, sus, rel)
            gate_sig = self._eval(node.gate, t)
            return env * gate_sig

        if isinstance(node, LowPassNode):
            fc = self.get_param(node.cutoff)
            q = self.get_param(node.resonance)
            in_sig = self._eval(node.input_node, t)
            return self._lowpass_filter(in_sig, fc, q, t)

        if isinstance(node, GainNode):
            g = self.get_param(node.gain)
            return g * self._eval(node.input_node, t)

        if isinstance(node, MixNode):
            r = self.get_param(node.ratio)
            s1 = self._eval(node.input1, t)
            s2 = self._eval(node.input2, t)
            return torch.lerp(s1, s2, r)

        if isinstance(node, FMNode):
            c = self.get_param(node.carrier)
            mf = self.get_param(node.mod_freq)
            mi = self.get_param(node.mod_index)
            modulator = mi * torch.sin(2.0 * math.pi * mf * t)
            return torch.sin(2.0 * math.pi * c * t + modulator)

        raise TypeError(f"Unsupported audio node type: {type(node)}")

    def _adsr_envelope(
        self,
        t: torch.Tensor,
        att: torch.Tensor,
        dec: torch.Tensor,
        sus: torch.Tensor,
        rel: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable piecewise envelope."""
        att = torch.clamp(att, min=0.001)
        dec = torch.clamp(dec, min=0.001)
        t_att = att
        t_dec = t_att + dec
        # Smooth interpolation
        attack_phase = torch.clamp(t / t_att, 0.0, 1.0)
        decay_phase = torch.clamp(1.0 - (t - t_att) / dec, 0.0, 1.0) * (1.0 - sus) + sus
        return torch.where(t < t_att, attack_phase, decay_phase)

    def _lowpass_filter(
        self,
        in_sig: torch.Tensor,
        fc: torch.Tensor,
        q: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable single-pole smoothing lowpass filter."""
        dt = (t[1] - t[0]) if len(t) > 1 else torch.tensor(1.0 / 16000.0)
        rc = 1.0 / (2.0 * math.pi * torch.clamp(fc, min=10.0))
        alpha = dt / (rc + dt)
        # Apply smoothing
        return alpha * in_sig + (1.0 - alpha) * in_sig.roll(1)


class DifferentiableSDFModule(DifferentiableProgram):
    """Compiles an SDFNode AST into a differentiable 3D distance field."""

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute signed distance at 3D point coordinates (N, 3)."""
        return self._eval(self.root, p)

    def _eval(self, node: ASTNode, p: torch.Tensor) -> torch.Tensor:
        if isinstance(node, SphereNode):
            r = self.get_param(node.radius)
            return torch.norm(p, dim=-1) - r

        if isinstance(node, BoxNode):
            w = self.get_param(node.width)
            h = self.get_param(node.height)
            d = self.get_param(node.depth)
            bounds = torch.stack([w, h, d])
            q = torch.abs(p) - bounds
            outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
            inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
            return outside + inside

        if isinstance(node, CylinderNode):
            r = self.get_param(node.radius)
            h = self.get_param(node.height)
            d_xy = torch.norm(p[..., :2], dim=-1) - r
            d_z = torch.abs(p[..., 2]) - h
            outside = torch.norm(
                torch.clamp(torch.stack([d_xy, d_z], dim=-1), min=0.0), dim=-1
            )
            inside = torch.clamp(torch.maximum(d_xy, d_z), max=0.0)
            return outside + inside

        if isinstance(node, TorusNode):
            r1 = self.get_param(node.major_radius)
            r2 = self.get_param(node.minor_radius)
            q_x = torch.norm(p[..., :2], dim=-1) - r1
            q_y = p[..., 2]
            return torch.norm(torch.stack([q_x, q_y], dim=-1), dim=-1) - r2

        if isinstance(node, TranslateNode):
            dx = self.get_param(node.dx)
            dy = self.get_param(node.dy)
            dz = self.get_param(node.dz)
            offset = torch.stack([dx, dy, dz])
            return self._eval(node.child, p - offset)

        if isinstance(node, RotateNode):
            rx = self.get_param(node.rx)
            ry = self.get_param(node.ry)
            rz = self.get_param(node.rz)
            rot_matrix = self._rotation_matrix(rx, ry, rz)
            # p @ R
            p_rot = torch.matmul(p, rot_matrix)
            return self._eval(node.child, p_rot)

        if isinstance(node, ScaleNode):
            s = torch.clamp(self.get_param(node.s), min=0.001)
            return s * self._eval(node.child, p / s)

        if isinstance(node, UnionNode):
            k = self.get_param(node.smoothness)
            d1 = self._eval(node.a, p)
            d2 = self._eval(node.b, p)
            return self._smooth_min(d1, d2, k)

        if isinstance(node, IntersectionNode):
            d1 = self._eval(node.a, p)
            d2 = self._eval(node.b, p)
            return torch.maximum(d1, d2)

        if isinstance(node, DifferenceNode):
            d1 = self._eval(node.a, p)
            d2 = self._eval(node.b, p)
            return torch.maximum(d1, -d2)

        raise TypeError(f"Unsupported SDF node type: {type(node)}")

    def _smooth_min(
        self, d1: torch.Tensor, d2: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """Polynomial smooth minimum for CSG boolean union."""
        k = torch.clamp(k, min=1e-5)
        h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return torch.lerp(d2, d1, h) - k * h * (1.0 - h)

    def _rotation_matrix(
        self, rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor
    ) -> torch.Tensor:
        """3D Euler angles rotation matrix."""
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        rx_mat = torch.stack([
            torch.tensor(1.0, device=rx.device), torch.tensor(0.0, device=rx.device), torch.tensor(0.0, device=rx.device),
            torch.tensor(0.0, device=rx.device), cx, -sx,
            torch.tensor(0.0, device=rx.device), sx, cx,
        ]).reshape(3, 3)

        ry_mat = torch.stack([
            cy, torch.tensor(0.0, device=ry.device), sy,
            torch.tensor(0.0, device=ry.device), torch.tensor(1.0, device=ry.device), torch.tensor(0.0, device=ry.device),
            -sy, torch.tensor(0.0, device=ry.device), cy,
        ]).reshape(3, 3)

        rz_mat = torch.stack([
            cz, -sz, torch.tensor(0.0, device=rz.device),
            sz, cz, torch.tensor(0.0, device=rz.device),
            torch.tensor(0.0, device=rz.device), torch.tensor(0.0, device=rz.device), torch.tensor(1.0, device=rz.device),
        ]).reshape(3, 3)

        return torch.matmul(rz_mat, torch.matmul(ry_mat, rx_mat))


def transpile_ast_to_module(node: ASTNode) -> DifferentiableProgram:
    """Factory compiling an AST node to its corresponding differentiable module."""
    if isinstance(node, AudioNode):
        return DifferentiableAudioModule(node)
    if isinstance(node, SDFNode):
        return DifferentiableSDFModule(node)
    raise TypeError(f"Cannot transpile AST of type: {type(node)}")
