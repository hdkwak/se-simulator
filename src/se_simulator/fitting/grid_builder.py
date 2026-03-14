"""Grid and bounds builders for the TMM-direct fitting pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from se_simulator.config.recipe import FloatingParameter


def build_grid_from_floating_params(
    floating_params: list[FloatingParameter],
) -> dict[str, np.ndarray]:
    """Build a parameter grid for coarse search from floating parameters.

    Each axis is constructed with :func:`numpy.arange` using the parameter's
    ``min``, ``max``, and ``step`` fields.  The endpoint is included when it
    falls exactly on a step boundary (matches ``arange(min, max+step, step)``).

    Returns
    -------
    dict mapping parameter name → 1-D array of candidate values.
    """
    return {
        p.name: np.arange(p.min, p.max + p.step, p.step)
        for p in floating_params
    }


def build_bounds_from_floating_params(
    floating_params: list[FloatingParameter],
) -> list[tuple[float, float]]:
    """Build a bounds list for refinement from floating parameters.

    Returns
    -------
    List of ``(min, max)`` tuples in the same order as *floating_params*.
    """
    return [(p.min, p.max) for p in floating_params]
