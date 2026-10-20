"""3D Signed Distance Field (SDF) CSG primitives and geometric perceptual loss engine."""

from symforge.sdf.primitives import (
    BoxSDF,
    CylinderSDF,
    DifferenceSDF,
    IntersectionSDF,
    PlaneSDF,
    RotateSDF,
    ScaleSDF,
    SmoothUnionSDF,
    SphereSDF,
    TorusSDF,
    TranslateSDF,
    UnionSDF,
)
from symforge.sdf.surface import (
    SurfaceRaymarchFeatureExtractor,
)

__all__ = [
    "SphereSDF",
    "BoxSDF",
    "CylinderSDF",
    "TorusSDF",
    "PlaneSDF",
    "TranslateSDF",
    "RotateSDF",
    "ScaleSDF",
    "UnionSDF",
    "IntersectionSDF",
    "DifferenceSDF",
    "SmoothUnionSDF",
    "SurfaceRaymarchFeatureExtractor",
]
