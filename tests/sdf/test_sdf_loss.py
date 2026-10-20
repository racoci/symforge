"""Tests for Surface Raymarching Feature Extraction and Contrastive Geometric Loss."""

import pytest
import torch

from symforge.sdf.primitives import (
    SphereSDF,
    BoxSDF,
    CylinderSDF,
    TorusSDF,
    TranslateSDF,
)
from symforge.sdf.surface import SurfaceRaymarchFeatureExtractor
from symforge.eval.loss import compute_infonce_loss


class TestSurfaceRaymarchFeatureExtractor:
    """Test suite for Surface Raymarching feature extraction and geometric embedding."""

    def test_extractor_output_shape_and_normalization(self):
        """Verify feature embedding has shape (1, D) and exact unit L2 norm."""
        sphere = SphereSDF(radius=1.0)
        extractor = SurfaceRaymarchFeatureExtractor()
        emb = extractor(sphere)

        assert emb.ndim == 2
        assert emb.shape[0] == 1
        # L2-normalization invariant: ||z||_2 == 1.0
        norm = torch.norm(emb, p=2, dim=-1).item()
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_batch_extraction(self):
        """Verify extractor handles multiple SDF instances and guarantees unit norms."""
        shapes = [
            SphereSDF(radius=1.0),
            BoxSDF(half_extents=1.0),
            CylinderSDF(radius=0.8, height=1.2),
            TorusSDF(major_radius=1.5, minor_radius=0.4),
        ]
        extractor = SurfaceRaymarchFeatureExtractor()
        embs = extractor(shapes)

        assert embs.shape[0] == len(shapes)
        norms = torch.norm(embs, p=2, dim=-1)
        for n in norms:
            assert n.item() == pytest.approx(1.0, abs=1e-5)

    def test_surface_points_extraction_and_normals(self):
        """Verify sphere tracing converges to zero-level set and central finite differences yield accurate normals."""
        sphere = SphereSDF(radius=1.2)
        extractor = SurfaceRaymarchFeatureExtractor()

        # Raymarch surface points
        surface_pts, mask = extractor.extract_surface_points(sphere)
        assert mask.sum().item() > 0

        valid_pts = surface_pts[mask]
        # Distances at surface points must be near 0 (zero-level set: |d| < 1e-3)
        d_surf = sphere(valid_pts)
        assert torch.all(torch.abs(d_surf) < 1e-3)

        # Normals computed via central finite differences
        normals = extractor.compute_normals(sphere, valid_pts)
        # For a sphere centered at origin, analytical normal is p / ||p||
        expected_normals = valid_pts / torch.norm(valid_pts, dim=-1, keepdim=True)
        torch.testing.assert_close(normals, expected_normals, atol=1e-3, rtol=1e-3)

    def test_contrastive_separation_sphere_vs_box(self):
        """Invariant: InfoNCE contrastive separation between Sphere and Box achieves separation margin > 2.0."""
        sphere = SphereSDF(radius=1.0)
        box = BoxSDF(half_extents=1.0)
        cylinder = CylinderSDF(radius=1.0, height=1.0)

        extractor = SurfaceRaymarchFeatureExtractor()
        z_sphere = extractor(sphere)
        z_box = extractor(box)
        z_cyl = extractor(cylinder)

        tau = 0.07
        sim_pos = torch.sum(z_sphere * z_sphere, dim=-1).item()
        sim_neg_box = torch.sum(z_sphere * z_box, dim=-1).item()
        sim_neg_cyl = torch.sum(z_sphere * z_cyl, dim=-1).item()

        # Margin: (sim_pos - sim_neg) / tau
        margin_box = (sim_pos - sim_neg_box) / tau
        margin_cyl = (sim_pos - sim_neg_cyl) / tau

        assert margin_box > 2.0, f"Expected margin > 2.0, got {margin_box}"
        assert margin_cyl > 2.0, f"Expected margin > 2.0, got {margin_cyl}"

        # Evaluate InfoNCE loss with Sphere query, Sphere positive, and Box/Cylinder negatives
        negatives = torch.cat([z_box, z_cyl], dim=0)
        loss = compute_infonce_loss(
            query=z_sphere,
            positive=z_sphere,
            negatives=negatives,
            temperature=tau,
        )
        assert loss.item() < 0.05

        # Query Box when positive is Sphere: should have high loss
        loss_mismatch = compute_infonce_loss(
            query=z_box,
            positive=z_sphere,
            negatives=negatives,
            temperature=tau,
        )
        assert loss_mismatch.item() > 1.0

    def test_contrastive_loss_direct_integration(self):
        """Verify extractor compute_loss method directly computes InfoNCE over SDF instances."""
        sphere = SphereSDF(radius=1.0)
        box = BoxSDF(half_extents=1.0)
        cylinder = CylinderSDF(radius=1.0, height=1.0)

        extractor = SurfaceRaymarchFeatureExtractor()
        loss = extractor.compute_loss(
            query=sphere,
            positive=sphere,
            negatives=[box, cylinder],
            temperature=0.07,
        )
        assert loss.item() < 0.05

    def test_contrastive_loss_gradient_flow(self):
        """Verify autograd backpropagates gradients through feature extraction to shape parameters."""
        # Query sphere initialized with wrong radius (r = 1.5 instead of target 1.0)
        query_sphere = SphereSDF(radius=1.5)
        target_sphere = SphereSDF(radius=1.0)
        distractor_box = BoxSDF(half_extents=1.0)

        extractor = SurfaceRaymarchFeatureExtractor()
        loss = extractor.compute_loss(
            query=query_sphere,
            positive=target_sphere,
            negatives=[distractor_box],
            temperature=0.07,
        )

        loss.backward()
        assert query_sphere.radius.grad is not None
        assert query_sphere.radius.grad.item() != 0.0
