"""Simulation runner: thin facade over RCWAEngine.

Accepts either a ``SimulationRecipe`` or raw ``SampleConfig`` +
``SimConditions`` objects and returns an ``RCWAResult``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from se_simulator.config.recipe import SimulationRecipe
    from se_simulator.config.schemas import SampleConfig, SimConditions
    from se_simulator.rcwa.results import RCWAResult

logger = logging.getLogger(__name__)


def run_simulation(
    recipe: SimulationRecipe | None = None,
    sample_config: SampleConfig | None = None,
    sim_conditions: SimConditions | None = None,
    recipe_path: str = "",
    wavelengths_nm: np.ndarray | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> RCWAResult:
    """Run a simulation from a recipe or from raw config objects.

    Exactly one of the following call signatures must be used:

    1. ``run_simulation(recipe=<SimulationRecipe>)`` — recipe drives everything.
    2. ``run_simulation(sample_config=..., sim_conditions=...)`` — raw objects.

    Parameters
    ----------
    recipe:
        Optional ``SimulationRecipe``; when provided, ``sample_config`` and
        ``sim_conditions`` are derived from it via ``RecipeManager``.
    sample_config:
        ``SampleConfig`` used when *recipe* is ``None``.
    sim_conditions:
        ``SimConditions`` used when *recipe* is ``None``.
    recipe_path:
        Optional file path to the recipe YAML; used to resolve relative
        ``ref:`` sample paths when *recipe* is provided.
    wavelengths_nm:
        Optional override for the wavelength array.  When ``None`` the
        wavelengths are derived from ``sim_conditions.wavelengths``.
    progress_callback:
        Optional ``(fraction: float) -> None`` callback (0.0 – 1.0).

    Returns
    -------
    ``RCWAResult`` dataclass.

    Raises
    ------
    ValueError
        If neither *recipe* nor both *sample_config* and *sim_conditions*
        are provided.
    """
    from pathlib import Path

    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.engine import RCWAEngine

    if recipe is not None:
        from se_simulator.recipe.manager import RecipeManager

        rpath = Path(recipe_path) if recipe_path else None
        mgr = RecipeManager()
        sample_config, sim_conditions = mgr.decompose_simulation(recipe, rpath)
        logger.info("[Runner] Decomposed SimulationRecipe.")
    elif sample_config is None or sim_conditions is None:
        raise ValueError(
            "Provide either a SimulationRecipe or both sample_config and sim_conditions."
        )

    db = MaterialDatabase()
    for spec in sample_config.materials.values():
        db.resolve(spec)

    engine = RCWAEngine(db)
    result = engine.run(
        sample_config,
        sim_conditions,
        wavelengths_nm=wavelengths_nm,
        progress_callback=progress_callback,
    )
    logger.info("[Runner] Simulation complete: %d wavelengths.", len(result.wavelengths_nm))
    return result
