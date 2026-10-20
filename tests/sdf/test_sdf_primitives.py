"""Tests for 3D SDF primitives, spatial transforms, and CSG combinators."""

import math
import pytest
import torch

from symforge.core.ast import FreeParam
from symforge.sdf.primitives import (
    SphereSDF,
    BoxSDF,
    CylinderSDF,
    TorusSDF,
    PlaneSDF,
    TranslateSDF,
    RotateSDF,
    ScaleSDF,
    UnionSDF,
    IntersectionSDF,
    DifferenceSDF,
    SmoothUnionSDF,
)


class TestAnalyticalPrimitives:
    """Test suite for analytical 3D SDF primitives."""

    def test_sphere_sdf_exact_distance(self):
        """Verify exact mathematical distance for Sphere: d(p) = ||p|| - r."""
        radius = 1.5
        sphere = SphereSDF(radius=radius)

        # Center point
        p_center = torch.tensor([[0.0, 0.0, 0.0]])
        assert sphere(p_center).item() == pytest.approx(-radius, abs=1e-5)

        # Surface points
        p_surface = torch.tensor([
            [1.5, 0.0, 0.0],
            [0.0, -1.5, 0.0],
            [0.0, 0.0, 1.5],
        ])
        d_surf = sphere(p_surface)
        for val in d_surf:
            assert val.item() == pytest.approx(0.0, abs=1e-5)

        # Outside points
        p_outside = torch.tensor([
            [2.5, 0.0, 0.0],
            [0.0, 3.5, 0.0],
            [1.5, 1.5, 1.5],
        ])
        expected = torch.norm(p_outside, dim=-1) - radius
        d_out = sphere(p_outside)
        torch.testing.assert_close(d_out, expected, atol=1e-5, rtol=1e-5)

        # Batch shape handling (B, N, 3) -> (B, N)
        p_batch = torch.randn(4, 10, 3)
        d_batch = sphere(p_batch)
        assert d_batch.shape == (4, 10)
        torch.testing.assert_close(d_batch, torch.norm(p_batch, dim=-1) - radius, atol=1e-5, rtol=1e-5)

    def test_box_sdf_exact_distance(self):
        """Verify exact mathematical distance for Box: d(p) with half extents."""
        bx, by, bz = 1.0, 2.0, 3.0
        box = BoxSDF(half_extents=[bx, by, bz])

        # Center point: inside, closest face is at distance bx=1.0 -> d = -1.0
        p_center = torch.tensor([[0.0, 0.0, 0.0]])
        assert box(p_center).item() == pytest.approx(-1.0, abs=1e-5)

        # Points on face centers
        p_faces = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        for val in box(p_faces):
            assert val.item() == pytest.approx(0.0, abs=1e-5)

        # Outside facing face x
        p_face_x = torch.tensor([[2.5, 0.0, 0.0]])
        assert box(p_face_x).item() == pytest.approx(1.5, abs=1e-5)

        # Outside facing edge (x, y)
        p_edge = torch.tensor([[2.0, 3.0, 0.0]])  # dx = 1.0, dy = 1.0, dz = 0 -> dist = sqrt(2)
        assert box(p_edge).item() == pytest.approx(math.sqrt(2.0), abs=1e-5)

        # Outside facing corner (x, y, z)
        p_corner = torch.tensor([[2.0, 3.0, 4.0]])  # dx=1, dy=1, dz=1 -> dist = sqrt(3)
        assert box(p_corner).item() == pytest.approx(math.sqrt(3.0), abs=1e-5)

        # Scalar half_extents (cube)
        cube = BoxSDF(half_extents=1.0)
        assert cube(torch.tensor([[2.0, 0.0, 0.0]])).item() == pytest.approx(1.0, abs=1e-5)

    def test_cylinder_sdf_exact_distance(self):
        """Verify exact mathematical distance for Capped Cylinder along z-axis."""
        radius = 1.0
        height = 2.0  # half-height along z
        cyl = CylinderSDF(radius=radius, height=height)

        # Center point
        p_center = torch.tensor([[0.0, 0.0, 0.0]])
        assert cyl(p_center).item() == pytest.approx(-1.0, abs=1e-5)

        # Points on lateral surface
        p_side = torch.tensor([[1.0, 0.0, 0.5]])
        assert cyl(p_side).item() == pytest.approx(0.0, abs=1e-5)

        # Points on top cap surface
        p_cap = torch.tensor([[0.5, 0.0, 2.0]])
        assert cyl(p_cap).item() == pytest.approx(0.0, abs=1e-5)

        # Outside radially: dist = ||p_xy|| - r
        p_out_rad = torch.tensor([[2.5, 0.0, 0.0]])
        assert cyl(p_out_rad).item() == pytest.approx(1.5, abs=1e-5)

        # Outside axially: dist = |z| - h
        p_out_ax = torch.tensor([[0.0, 0.0, 3.5]])
        assert cyl(p_out_ax).item() == pytest.approx(1.5, abs=1e-5)

        # Outside corner rim: dist = sqrt(dx^2 + dz^2)
        p_out_rim = torch.tensor([[2.0, 0.0, 3.0]])  # dr=1.0, dz=1.0 -> sqrt(2)
        assert cyl(p_out_rim).item() == pytest.approx(math.sqrt(2.0), abs=1e-5)

    def test_torus_sdf_exact_distance(self):
        """Verify exact mathematical distance for Torus in xy-plane."""
        r_major = 2.0
        r_minor = 0.5
        torus = TorusSDF(major_radius=r_major, minor_radius=r_minor)

        # Tube core circle in xy-plane
        p_core = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        for val in torus(p_core):
            assert val.item() == pytest.approx(-r_minor, abs=1e-5)

        # Tube surface
        p_surf = torch.tensor([
            [2.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 0.0, 0.5],
            [2.0, 0.0, -0.5],
        ])
        for val in torus(p_surf):
            assert val.item() == pytest.approx(0.0, abs=1e-5)

        # Origin: dist = r_major - r_minor
        p_orig = torch.tensor([[0.0, 0.0, 0.0]])
        assert torus(p_orig).item() == pytest.approx(r_major - r_minor, abs=1e-5)

    def test_plane_sdf_exact_distance(self):
        """Verify exact mathematical distance for Plane: d(p) = p . n + offset."""
        normal = [0.0, 0.0, 1.0]
        offset = -1.0
        plane = PlaneSDF(normal=normal, offset=offset)

        # Point on plane z = 1.0
        p_on = torch.tensor([[5.0, -3.0, 1.0]])
        assert plane(p_on).item() == pytest.approx(0.0, abs=1e-5)

        # Point above plane z = 3.0 -> dist = 2.0
        p_above = torch.tensor([[0.0, 0.0, 3.0]])
        assert plane(p_above).item() == pytest.approx(2.0, abs=1e-5)

        # Point below plane z = 0.0 -> dist = -1.0
        p_below = torch.tensor([[0.0, 0.0, 0.0]])
        assert plane(p_below).item() == pytest.approx(-1.0, abs=1e-5)

        # Unnormalized normal auto-normalization
        plane_unnorm = PlaneSDF(normal=[0.0, 3.0, 4.0], offset=0.0)
        p_test = torch.tensor([[0.0, 3.0, 4.0]])  # ||n|| = 5, p . (n/5) = 25/5 = 5.0
        assert plane_unnorm(p_test).item() == pytest.approx(5.0, abs=1e-5)


class TestSpatialTransforms:
    """Test suite for spatial transforms."""

    def test_translate_sdf(self):
        """Verify translation transforms query points into local coordinates."""
        sphere = SphereSDF(radius=1.0)
        offset = [1.0, 2.0, 3.0]
        translated = TranslateSDF(sphere, offset=offset)

        # Translated center point (1, 2, 3) has distance -1.0
        p_center = torch.tensor([[1.0, 2.0, 3.0]])
        assert translated(p_center).item() == pytest.approx(-1.0, abs=1e-5)

        # Surface point at (2, 2, 3)
        p_surf = torch.tensor([[2.0, 2.0, 3.0]])
        assert translated(p_surf).item() == pytest.approx(0.0, abs=1e-5)

        # Inversion symmetry: Translate + offset then -offset recovers original field
        inv_translated = TranslateSDF(translated, offset=[-1.0, -2.0, -3.0])
        p_samples = torch.randn(10, 3)
        torch.testing.assert_close(sphere(p_samples), inv_translated(p_samples), atol=1e-5, rtol=1e-5)

    def test_rotate_sdf(self):
        """Verify rotation transforms query points into local coordinates."""
        box = BoxSDF(half_extents=[1.0, 2.0, 0.5])
        # Rotate 90 degrees around z-axis: swaps x and y extents
        angles = [0.0, 0.0, math.pi / 2.0]
        rotated = RotateSDF(box, angles=angles)

        # After 90 deg z-rotation, face that was at y=2 is now at x=-2 or x=2
        p_face = torch.tensor([[2.0, 0.0, 0.0]])
        assert abs(rotated(p_face).item()) < 1e-4

        # Inversion symmetry: Rotate by theta then -theta recovers original field
        inv_rotated = RotateSDF(rotated, angles=[0.0, 0.0, -math.pi / 2.0])
        p_samples = torch.randn(10, 3)
        torch.testing.assert_close(box(p_samples), inv_rotated(p_samples), atol=1e-4, rtol=1e-4)

    def test_scale_sdf(self):
        """Verify scaling preserves exact signed distance: d'(p) = s * d(p / s)."""
        sphere = SphereSDF(radius=1.0)
        factor = 2.5
        scaled = ScaleSDF(sphere, factor=factor)

        # Scaled sphere has radius 2.5
        p_surf = torch.tensor([[2.5, 0.0, 0.0]])
        assert scaled(p_surf).item() == pytest.approx(0.0, abs=1e-5)

        # Distance to point (5, 0, 0) is 5.0 - 2.5 = 2.5
        p_out = torch.tensor([[5.0, 0.0, 0.0]])
        assert scaled(p_out).item() == pytest.approx(2.5, abs=1e-5)

        # Inversion symmetry
        inv_scaled = ScaleSDF(scaled, factor=1.0 / factor)
        p_samples = torch.randn(10, 3)
        torch.testing.assert_close(sphere(p_samples), inv_scaled(p_samples), atol=1e-5, rtol=1e-5)


class TestCSGCombinators:
    """Test suite for CSG Boolean combinators."""

    def test_union_sdf(self):
        """Verify Union is exact minimum: d = min(d1, d2)."""
        s1 = SphereSDF(radius=1.0)
        s2 = SphereSDF(radius=1.0)
        t_s2 = TranslateSDF(s2, offset=[3.0, 0.0, 0.0])
        union = UnionSDF(s1, t_s2)

        p = torch.tensor([
            [0.0, 0.0, 0.0],  # inside s1
            [3.0, 0.0, 0.0],  # inside s2
            [1.5, 0.0, 0.0],  # halfway between s1 and s2: dist to both is 0.5
        ])
        expected = torch.minimum(s1(p), t_s2(p))
        torch.testing.assert_close(union(p), expected, atol=1e-5, rtol=1e-5)

    def test_intersection_sdf(self):
        """Verify Intersection is exact maximum: d = max(d1, d2)."""
        s1 = SphereSDF(radius=1.5)
        s2 = SphereSDF(radius=1.5)
        t_s2 = TranslateSDF(s2, offset=[1.0, 0.0, 0.0])
        inter = IntersectionSDF(s1, t_s2)

        p = torch.randn(20, 3)
        expected = torch.maximum(s1(p), t_s2(p))
        torch.testing.assert_close(inter(p), expected, atol=1e-5, rtol=1e-5)

    def test_difference_sdf(self):
        """Verify Difference is exact maximum with inverted subtrahend: d = max(d1, -d2)."""
        box = BoxSDF(half_extents=2.0)
        sphere = SphereSDF(radius=1.0)
        diff = DifferenceSDF(box, sphere)

        # Center point was inside box (-2.0) and inside sphere (-1.0).
        # Difference gives max(-2.0, -(-1.0)) = max(-2.0, 1.0) = 1.0 (outside cavity).
        p_center = torch.tensor([[0.0, 0.0, 0.0]])
        assert diff(p_center).item() == pytest.approx(1.0, abs=1e-5)

        p = torch.randn(20, 3)
        expected = torch.maximum(box(p), -sphere(p))
        torch.testing.assert_close(diff(p), expected, atol=1e-5, rtol=1e-5)

    def test_smooth_union_sdf(self):
        """Verify SmoothUnion performs polynomial smooth minimum and converges to union as k -> 0."""
        s1 = SphereSDF(radius=1.0)
        s2 = SphereSDF(radius=1.0)
        t_s2 = TranslateSDF(s2, offset=[1.5, 0.0, 0.0])

        k = 0.2
        smooth_union = SmoothUnionSDF(s1, t_s2, k=k)

        # Far from intersection, should match standard union closely
        p_far = torch.tensor([[-5.0, 0.0, 0.0]])
        assert abs(smooth_union(p_far).item() - torch.minimum(s1(p_far), t_s2(p_far)).item()) < 1e-4

        # Near boundary where d1 == d2, smooth minimum reduces distance by k/4
        p_mid = torch.tensor([[0.75, 0.0, 0.0]])
        d1 = s1(p_mid).item()
        d2 = t_s2(p_mid).item()
        assert d1 == pytest.approx(d2, abs=1e-5)
        expected_mid = d1 - k / 4.0
        assert smooth_union(p_mid).item() == pytest.approx(expected_mid, abs=1e-4)

        # Convergence as k -> 0
        smooth_union_tiny_k = SmoothUnionSDF(s1, t_s2, k=1e-5)
        p_samples = torch.randn(20, 3)
        std_union = torch.minimum(s1(p_samples), t_s2(p_samples))
        torch.testing.assert_close(smooth_union_tiny_k(p_samples), std_union, atol=1e-4, rtol=1e-4)


class TestAutogradAndFreeParam:
    """Test suite for gradient backpropagation and FreeParam integration."""

    def test_autograd_sphere_radius(self):
        """Verify non-zero autograd gradients backpropagate through FreeParam radius."""
        param_r = FreeParam("sphere_radius", 0.1, 5.0, 1.0)
        sphere = SphereSDF(radius=param_r)

        p = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=False)
        d = sphere(p)
        loss = d.sum()
        loss.backward()

        assert sphere.radius.grad is not None
        # d/dr (||p|| - r) = -1.0
        assert sphere.radius.grad.item() == pytest.approx(-1.0, abs=1e-5)

    def test_autograd_box_extents(self):
        """Verify non-zero autograd gradients backpropagate through box half_extents."""
        box = BoxSDF(half_extents=[1.0, 2.0, 3.0])
        p = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
        d = box(p)
        loss = d.sum()
        loss.backward()

        assert box.half_extents.grad is not None
        assert torch.all(box.half_extents.grad != 0.0)

    def test_autograd_translate_offset(self):
        """Verify non-zero gradients backpropagate through translation offset."""
        sphere = SphereSDF(radius=1.0)
        offset_params = [
            FreeParam("dx", -5.0, 5.0, 1.0),
            FreeParam("dy", -5.0, 5.0, 0.0),
            FreeParam("dz", -5.0, 5.0, 0.0),
        ]
        trans = TranslateSDF(sphere, offset=offset_params)

        p = torch.tensor([[3.0, 0.0, 0.0]])
        d = trans(p)
        loss = d.sum()
        loss.backward()

        assert trans.offset.grad is not None
        assert trans.offset.grad[0].item() != 0.0

    def test_autograd_smooth_union_k(self):
        """Verify non-zero gradient backpropagates through smooth union parameter k."""
        s1 = SphereSDF(radius=1.0)
        s2 = SphereSDF(radius=1.0)
        t_s2 = TranslateSDF(s2, offset=[1.5, 0.0, 0.0])
        param_k = FreeParam("k_smooth", 0.01, 1.0, 0.2)
        su = SmoothUnionSDF(s1, t_s2, k=param_k)

        p = torch.tensor([[0.75, 0.0, 0.0]])
        d = su(p)
        loss = d.sum()
        loss.backward()

        assert su.k.grad is not None
        # At mid-point, d = d0 - k/4 -> d(loss)/dk = -0.25
        assert su.k.grad.item() == pytest.approx(-0.25, abs=1e-3)
