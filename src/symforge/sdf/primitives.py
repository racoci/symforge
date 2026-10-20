"""Functional and differentiable PyTorch 3D SDF primitives and CSG combinators."""

from __future__ import annotations
from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn

from symforge.core.ast import FreeParam


def _ensure_scalar_param(
    val: Union[float, int, torch.Tensor, nn.Parameter, FreeParam],
    name: str = "param",
) -> nn.Parameter:
    """Convert scalar input into an nn.Parameter with autograd tracking."""
    if isinstance(val, FreeParam):
        p = nn.Parameter(torch.tensor(float(val.default_val), dtype=torch.float32))
        setattr(p, "_free_param", val)
        return p
    if isinstance(val, nn.Parameter):
        return val
    if isinstance(val, torch.Tensor):
        if val.requires_grad:
            return nn.Parameter(val)
        return nn.Parameter(val.clone().detach().to(torch.float32))
    if isinstance(val, (int, float)):
        return nn.Parameter(torch.tensor(float(val), dtype=torch.float32))
    raise TypeError(f"Cannot initialize scalar parameter '{name}' from type {type(val)}")


def _extract_element(elem: Union[float, int, torch.Tensor, FreeParam]) -> float:
    """Extract float value from scalar or FreeParam."""
    if isinstance(elem, FreeParam):
        return float(elem.default_val)
    if isinstance(elem, torch.Tensor):
        return float(elem.item())
    return float(elem)


def _ensure_vector3_param(
    val: Union[float, int, Sequence[Union[float, int, torch.Tensor, FreeParam]], torch.Tensor, nn.Parameter, FreeParam],
    name: str = "vector_param",
) -> nn.Parameter:
    """Convert 3D vector or uniform scalar into an nn.Parameter of shape (3,)."""
    if isinstance(val, FreeParam):
        v = float(val.default_val)
        p = nn.Parameter(torch.tensor([v, v, v], dtype=torch.float32))
        setattr(p, "_free_param", val)
        return p
    if isinstance(val, (int, float)):
        v = float(val)
        return nn.Parameter(torch.tensor([v, v, v], dtype=torch.float32))
    if isinstance(val, (list, tuple)):
        if len(val) != 3:
            raise ValueError(f"Parameter '{name}' must have 3 components, got {len(val)}")
        vals = [_extract_element(x) for x in val]
        p = nn.Parameter(torch.tensor(vals, dtype=torch.float32))
        setattr(p, "_free_params", [x for x in val if isinstance(x, FreeParam)])
        return p
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            v = float(val.item())
            return nn.Parameter(torch.tensor([v, v, v], dtype=torch.float32))
        if val.numel() == 3:
            t = val.reshape(3).to(torch.float32)
            return nn.Parameter(t) if val.requires_grad else nn.Parameter(t.clone().detach())
        raise ValueError(f"Expected 1 or 3 elements for '{name}', got {val.numel()}")
    raise TypeError(f"Cannot initialize vector parameter '{name}' from type {type(val)}")


class SphereSDF(nn.Module):
    """Analytical signed distance field for a sphere centered at the origin: d(p) = ||p|| - r."""

    def __init__(
        self,
        radius: Union[float, int, torch.Tensor, nn.Parameter, FreeParam] = 1.0,
    ) -> None:
        super().__init__()
        self.radius = _ensure_scalar_param(radius, "radius")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute signed distance field at query points p of shape (..., 3)."""
        return torch.linalg.vector_norm(p, dim=-1) - self.radius


class BoxSDF(nn.Module):
    """Analytical signed distance field for an axis-aligned box with half extents."""

    def __init__(
        self,
        half_extents: Union[float, int, Sequence[Union[float, int, torch.Tensor, FreeParam]], torch.Tensor, nn.Parameter, FreeParam] = 1.0,
    ) -> None:
        super().__init__()
        self.half_extents = _ensure_vector3_param(half_extents, "half_extents")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute signed distance field at query points p of shape (..., 3)."""
        q = torch.abs(p) - self.half_extents
        q_out = torch.clamp(q, min=0.0)
        d_out = torch.linalg.vector_norm(q_out, dim=-1)
        q_max = torch.max(q, dim=-1).values
        d_in = torch.clamp(q_max, max=0.0)
        return d_out + d_in


class CylinderSDF(nn.Module):
    """Analytical signed distance field for a capped cylinder along the z-axis."""

    def __init__(
        self,
        radius: Union[float, int, torch.Tensor, nn.Parameter, FreeParam] = 1.0,
        height: Union[float, int, torch.Tensor, nn.Parameter, FreeParam] = 1.0,
    ) -> None:
        super().__init__()
        self.radius = _ensure_scalar_param(radius, "radius")
        self.height = _ensure_scalar_param(height, "height")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute signed distance field at query points p of shape (..., 3)."""
        d_xy = torch.linalg.vector_norm(p[..., :2], dim=-1) - self.radius
        d_z = torch.abs(p[..., 2]) - self.height
        d_out_vec = torch.clamp(torch.stack([d_xy, d_z], dim=-1), min=0.0)
        d_out = torch.linalg.vector_norm(d_out_vec, dim=-1)
        d_in = torch.clamp(torch.maximum(d_xy, d_z), max=0.0)
        return d_out + d_in


class TorusSDF(nn.Module):
    """Analytical signed distance field for a torus aligned in the xy-plane."""

    def __init__(
        self,
        major_radius: Union[float, int, torch.Tensor, nn.Parameter, FreeParam] = 2.0,
        minor_radius: Union[float, int, torch.Tensor, nn.Parameter, FreeParam] = 0.5,
    ) -> None:
        super().__init__()
        self.major_radius = _ensure_scalar_param(major_radius, "major_radius")
        self.minor_radius = _ensure_scalar_param(minor_radius, "minor_radius")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute signed distance field at query points p of shape (..., 3)."""
        rho = torch.linalg.vector_norm(p[..., :2], dim=-1) - self.major_radius
        q = torch.stack([rho, p[..., 2]], dim=-1)
        return torch.linalg.vector_norm(q, dim=-1) - self.minor_radius


class PlaneSDF(nn.Module):
    """Analytical signed distance field for an infinite plane: d(p) = p . n + offset."""

    def __init__(
        self,
        normal: Union[Sequence[Union[float, int, torch.Tensor, FreeParam]], torch.Tensor, nn.Parameter] = (0.0, 0.0, 1.0),
        offset: Union[float, int, torch.Tensor, nn.Parameter, FreeParam] = 0.0,
    ) -> None:
        super().__init__()
        self.normal = _ensure_vector3_param(normal, "normal")
        self.offset = _ensure_scalar_param(offset, "offset")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute signed distance field at query points p of shape (..., 3)."""
        norm_n = torch.clamp(torch.linalg.vector_norm(self.normal), min=1e-8)
        n_hat = self.normal / norm_n
        return torch.sum(p * n_hat, dim=-1) + self.offset


class TranslateSDF(nn.Module):
    """Spatial translation transform: d_trans(p) = d(p - offset)."""

    def __init__(
        self,
        sdf: nn.Module,
        offset: Union[float, int, Sequence[Union[float, int, torch.Tensor, FreeParam]], torch.Tensor, nn.Parameter, FreeParam] = 0.0,
    ) -> None:
        super().__init__()
        self.sdf = sdf
        self.offset = _ensure_vector3_param(offset, "offset")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute translated signed distance field."""
        return self.sdf(p - self.offset)


class RotateSDF(nn.Module):
    """Spatial rotation transform via Euler angles (rx, ry, rz) in radians."""

    def __init__(
        self,
        sdf: nn.Module,
        angles: Union[Sequence[Union[float, int, torch.Tensor, FreeParam]], torch.Tensor, nn.Parameter] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        self.sdf = sdf
        self.angles = _ensure_vector3_param(angles, "angles")

    def _build_rotation_matrix(self) -> torch.Tensor:
        """Construct 3D rotation matrix matching Euler angle rotation."""
        rx, ry, rz = self.angles[0], self.angles[1], self.angles[2]
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        rx_mat = torch.stack([
            torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx),
            torch.zeros_like(rx), cx, -sx,
            torch.zeros_like(rx), sx, cx,
        ]).reshape(3, 3)

        ry_mat = torch.stack([
            cy, torch.zeros_like(ry), sy,
            torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry),
            -sy, torch.zeros_like(ry), cy,
        ]).reshape(3, 3)

        rz_mat = torch.stack([
            cz, -sz, torch.zeros_like(rz),
            sz, cz, torch.zeros_like(rz),
            torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz),
        ]).reshape(3, 3)

        return torch.matmul(rz_mat, torch.matmul(ry_mat, rx_mat))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute rotated signed distance field."""
        rot_matrix = self._build_rotation_matrix()
        p_local = torch.matmul(p, rot_matrix)
        return self.sdf(p_local)


class ScaleSDF(nn.Module):
    """Spatial isotropic scaling transform: d_scaled(p) = s * d(p / s)."""

    def __init__(
        self,
        sdf: nn.Module,
        factor: Union[float, int, torch.Tensor, nn.Parameter, FreeParam] = 1.0,
    ) -> None:
        super().__init__()
        self.sdf = sdf
        self.factor = _ensure_scalar_param(factor, "factor")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute scaled signed distance field."""
        s = torch.clamp(self.factor, min=1e-6)
        return s * self.sdf(p / s)


class UnionSDF(nn.Module):
    """CSG Boolean Union: d(p) = min(d1(p), d2(p))."""

    def __init__(self, sdf1: nn.Module, sdf2: nn.Module) -> None:
        super().__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute union of two SDF fields."""
        return torch.minimum(self.sdf1(p), self.sdf2(p))


class IntersectionSDF(nn.Module):
    """CSG Boolean Intersection: d(p) = max(d1(p), d2(p))."""

    def __init__(self, sdf1: nn.Module, sdf2: nn.Module) -> None:
        super().__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute intersection of two SDF fields."""
        return torch.maximum(self.sdf1(p), self.sdf2(p))


class DifferenceSDF(nn.Module):
    """CSG Boolean Difference: d(p) = max(d1(p), -d2(p))."""

    def __init__(self, sdf1: nn.Module, sdf2: nn.Module) -> None:
        super().__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute difference of two SDF fields (sdf1 minus sdf2)."""
        return torch.maximum(self.sdf1(p), -self.sdf2(p))


class SmoothUnionSDF(nn.Module):
    """CSG Smooth Union using polynomial smooth minimum."""

    def __init__(
        self,
        sdf1: nn.Module,
        sdf2: nn.Module,
        k: Union[float, int, torch.Tensor, nn.Parameter, FreeParam] = 0.1,
    ) -> None:
        super().__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.k = _ensure_scalar_param(k, "k")

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute smooth union blend of two SDF fields."""
        d1 = self.sdf1(p)
        d2 = self.sdf2(p)
        k_val = torch.clamp(self.k, min=1e-6)
        h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k_val, 0.0, 1.0)
        return torch.lerp(d2, d1, h) - k_val * h * (1.0 - h)
