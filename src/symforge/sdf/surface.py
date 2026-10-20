"""Surface raymarching feature extractor and contrastive geometric perceptual loss for 3D SDFs."""

from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from symforge.eval.loss import compute_infonce_loss


def _generate_canonical_camera_rays(
    grid_size: int = 6,
    cam_dist: float = 2.5,
    extent: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate orthographic camera ray origins and directions from 6 axial viewpoints."""
    cams = [
        torch.tensor([cam_dist, 0.0, 0.0]),
        torch.tensor([-cam_dist, 0.0, 0.0]),
        torch.tensor([0.0, cam_dist, 0.0]),
        torch.tensor([0.0, -cam_dist, 0.0]),
        torch.tensor([0.0, 0.0, cam_dist]),
        torch.tensor([0.0, 0.0, -cam_dist]),
    ]
    rays_o_list: List[torch.Tensor] = []
    rays_d_list: List[torch.Tensor] = []
    coords = torch.linspace(-extent, extent, grid_size)

    for c in cams:
        d_center = -c / torch.linalg.vector_norm(c)
        up = torch.tensor([0.0, 1.0, 0.0]) if abs(d_center[1].item()) < 0.9 else torch.tensor([1.0, 0.0, 0.0])
        v_axis = torch.linalg.cross(d_center, up)
        v_axis = v_axis / torch.linalg.vector_norm(v_axis)
        u_axis = torch.linalg.cross(v_axis, d_center)

        for u in coords:
            for v in coords:
                ro = c + u * u_axis + v * v_axis
                rays_o_list.append(ro)
                rays_d_list.append(d_center)

    return torch.stack(rays_o_list), torch.stack(rays_d_list)


class SurfaceRaymarchFeatureExtractor(nn.Module):
    """Extracts normalized geometric feature embeddings z_sdf from 3D SDF models."""

    def __init__(
        self,
        grid_resolution: int = 9,
        bounds: float = 1.5,
        num_ray_views: int = 6,
        fourier_freqs: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.grid_resolution = int(grid_resolution)
        self.bounds = float(bounds)
        self.eps = float(eps)
        self.fourier_freqs = tuple(float(f) for f in fourier_freqs)

        # 3D spatial probe grid in [-bounds, bounds]^3
        coords = torch.linspace(-self.bounds, self.bounds, self.grid_resolution)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        probes = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
        self.register_buffer("probes", probes, persistent=False)

        # Canonical camera ray origins and directions for raymarching
        rays_o, rays_d = _generate_canonical_camera_rays(grid_size=num_ray_views)
        self.register_buffer("canonical_rays_o", rays_o, persistent=False)
        self.register_buffer("canonical_rays_d", rays_d, persistent=False)

    def extract_surface_points(
        self,
        sdf: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        rays_o: Optional[torch.Tensor] = None,
        rays_d: Optional[torch.Tensor] = None,
        max_steps: int = 64,
        eps: Optional[float] = None,
        t_max: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sphere trace along rays to find surface points on the zero-level set: |d(p)| < eps."""
        threshold = self.eps if eps is None else float(eps)
        ro = self.canonical_rays_o if rays_o is None else rays_o
        rd = self.canonical_rays_d if rays_d is None else rays_d

        t = torch.zeros(ro.shape[0], device=ro.device, dtype=ro.dtype)
        hit_mask = torch.zeros(ro.shape[0], device=ro.device, dtype=torch.bool)
        step_limit = min(max_steps, 256)

        for _ in range(step_limit):
            p_curr = ro + t.unsqueeze(-1) * rd
            d_curr = sdf(p_curr)
            is_hit = torch.abs(d_curr) < threshold
            hit_mask = hit_mask | is_hit
            step_size = torch.clamp(d_curr, min=threshold * 0.5)
            t = torch.where(hit_mask | (t > t_max), t, t + step_size)
            if torch.all(hit_mask | (t > t_max)):
                break

        surface_pts = ro + t.unsqueeze(-1) * rd
        return surface_pts, hit_mask

    def compute_normals(
        self,
        sdf: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        points: torch.Tensor,
        delta: float = 1e-3,
    ) -> torch.Tensor:
        """Compute surface normals via central finite differences: n = grad(d) / ||grad(d)||_2."""
        dev = points.device
        dtype = points.dtype
        e_x = torch.tensor([delta, 0.0, 0.0], device=dev, dtype=dtype)
        e_y = torch.tensor([0.0, delta, 0.0], device=dev, dtype=dtype)
        e_z = torch.tensor([0.0, 0.0, delta], device=dev, dtype=dtype)

        dx = (sdf(points + e_x) - sdf(points - e_x)) / (2.0 * delta)
        dy = (sdf(points + e_y) - sdf(points - e_y)) / (2.0 * delta)
        dz = (sdf(points + e_z) - sdf(points - e_z)) / (2.0 * delta)

        grad = torch.stack([dx, dy, dz], dim=-1)
        norm = torch.clamp(torch.linalg.vector_norm(grad, dim=-1, keepdim=True), min=1e-8)
        return grad / norm

    def sample_epsilon_boundary(
        self,
        sdf: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        num_samples: int = 256,
        eps_band: float = 0.05,
    ) -> torch.Tensor:
        """Stratified sampling near the zero-level set isosurface."""
        surf_pts, mask = self.extract_surface_points(sdf)
        valid = surf_pts[mask]
        if valid.shape[0] == 0:
            return torch.zeros((num_samples, 3), device=surf_pts.device)
        indices = torch.randint(0, valid.shape[0], (num_samples,), device=valid.device)
        noise = torch.randn((num_samples, 3), device=valid.device) * (eps_band * 0.5)
        return valid[indices] + noise

    def _extract_single(
        self,
        sdf: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        """Compute 1D unnormalized feature vector Phi(d) for a single SDF model."""
        probes = self.probes
        d = sdf(probes)

        # 1. Global spatial distance profile (mean-centered to isolate geometry from translation)
        d_centered = d - d.mean()

        # 2. Surface-weighted normal field
        surf_weight = torch.exp(-d**2 / 0.1)
        normals = self.compute_normals(sdf, probes, delta=0.01)
        surf_normals = (normals * surf_weight.unsqueeze(-1)).flatten()

        # 3. Multi-frequency Fourier features of distance near isosurface
        fourier_list: List[torch.Tensor] = []
        for f in self.fourier_freqs:
            fourier_list.append(torch.sin(f * 3.14159265 * d))
            fourier_list.append(torch.cos(f * 3.14159265 * d))
        fourier_feats = torch.cat(fourier_list, dim=-1)

        phi = torch.cat([d_centered, surf_normals, fourier_feats], dim=-1)
        return phi

    def forward(
        self,
        sdf_input: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor], Sequence[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]],
    ) -> torch.Tensor:
        """Extract L2-normalized geometric embedding vector z_sdf = Phi(d) / ||Phi(d)||_2."""
        if isinstance(sdf_input, (list, tuple)):
            embs = [self._extract_single(item) for item in sdf_input]
            batch_phi = torch.stack(embs, dim=0)
            return F.normalize(batch_phi, p=2, dim=-1, eps=1e-8)

        phi = self._extract_single(sdf_input).unsqueeze(0)
        return F.normalize(phi, p=2, dim=-1, eps=1e-8)

    def compute_loss(
        self,
        query: Union[nn.Module, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
        positive: Union[nn.Module, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
        negatives: Union[Sequence[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]], torch.Tensor],
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive perceptual loss directly from SDF models."""
        z_q = query if isinstance(query, torch.Tensor) else self.forward(query)
        z_pos = positive if isinstance(positive, torch.Tensor) else self.forward(positive)

        if isinstance(negatives, torch.Tensor):
            z_negs = negatives
        else:
            neg_embs = [self.forward(n) if not isinstance(n, torch.Tensor) else n for n in negatives]
            z_negs = torch.cat(neg_embs, dim=0)

        return compute_infonce_loss(
            query=z_q,
            positive=z_pos,
            negatives=z_negs,
            temperature=temperature,
            normalize=True,
        )
