"""ShapeTools: shape type constants and geometry helpers."""

from __future__ import annotations

from typing import Literal

ShapeType = Literal["rectangle", "trapezoid", "ellipse", "polygon"]

SHAPE_TYPES: list[ShapeType] = ["rectangle", "trapezoid", "ellipse", "polygon"]


def default_geometry(shape_type: ShapeType) -> dict:
    """Return default geometry parameters for a shape type."""
    base = {
        "type": shape_type,
        "cx": 0.0,
        "cy": 0.0,
        "width": 100.0,
        "height": 100.0,
        "sidewall_angle_deg": 90.0,
        "vertices": [],
    }
    return base
