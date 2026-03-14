"""Tests for se_simulator.fitting.grid_builder."""
from __future__ import annotations

import numpy as np
import pytest

from se_simulator.config.recipe import FloatingParameter
from se_simulator.fitting.grid_builder import (
    build_bounds_from_floating_params,
    build_grid_from_floating_params,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fp(name: str, lo: float, hi: float, step: float, init: float | None = None) -> FloatingParameter:
    return FloatingParameter(
        name=name,
        target_field=f"forward_model.sample.inline.layers[0].{name}",
        min=lo,
        max=hi,
        initial=init if init is not None else lo,
        step=step,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_grid_values_match_arange():
    """Grid values should equal np.arange(min, max+step, step) for each param."""
    params = [
        _fp("thickness_nm", 50.0, 200.0, 10.0, 100.0),
        _fp("n_real",        1.0,   2.0,  0.1,  1.5),
    ]
    grid = build_grid_from_floating_params(params)

    assert set(grid.keys()) == {"thickness_nm", "n_real"}

    expected_t = np.arange(50.0, 200.0 + 10.0, 10.0)
    np.testing.assert_allclose(grid["thickness_nm"], expected_t)

    expected_n = np.arange(1.0, 2.0 + 0.1, 0.1)
    np.testing.assert_allclose(grid["n_real"], expected_n, atol=1e-10)


def test_build_bounds_order_and_values():
    """Bounds list should preserve parameter order and match (min, max)."""
    params = [
        _fp("thickness_nm",  80.0, 150.0, 5.0, 100.0),
        _fp("roughness_nm",   0.0,  10.0, 1.0,   2.0),
        _fp("cd_nm",        200.0, 400.0, 20.0, 300.0),
    ]
    bounds = build_bounds_from_floating_params(params)

    assert len(bounds) == 3
    assert bounds[0] == (80.0, 150.0)
    assert bounds[1] == (0.0, 10.0)
    assert bounds[2] == (200.0, 400.0)
