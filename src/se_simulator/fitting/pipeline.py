"""Recipe-driven fitting pipeline entry point.

Orchestrates the full fitting workflow from a ``MeasurementRecipe``:

1. Decompose recipe into (SampleConfig, SimConditions, SystemConfig,
   floating_params, fitting_config) via ``RecipeManager``.
2. Select fitting mode via ``select_fitting_mode``.
3. Dispatch to either ``TmmDirectFitter`` (all-uniform stacks, no library)
   or the existing library-based ``FittingEngine``.
4. Optionally write results back to the recipe YAML file.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from se_simulator.config.recipe import FitResults, MeasurementRecipe

logger = logging.getLogger(__name__)


def run_fitting(
    recipe: MeasurementRecipe,
    target_spectrum: np.ndarray,
    recipe_path: str = "",
    progress_callback: Callable[[int, float], None] | None = None,
) -> FitResults:
    """Entry point for fitting via a ``MeasurementRecipe``.

    Parameters
    ----------
    recipe:
        Loaded and validated ``MeasurementRecipe`` object.
    target_spectrum:
        Measured spectrum to fit against.  Accepted shapes:

        - ``(N,)``   — psi or delta alone (based on ``fit_signals``)
        - ``(2*N,)`` — psi concatenated with delta
        - ``(N, 2)`` — psi in column 0, delta in column 1

    recipe_path:
        Optional path to the recipe YAML file.  When non-empty and
        ``recipe.output_options.save_recipe_with_results`` is ``True``,
        the fitted ``FitResults`` are appended back to the file.
    progress_callback:
        Optional ``(iteration: int, rmse: float) -> None`` callback.

    Returns
    -------
    ``FitResults`` Pydantic model.
    """
    from se_simulator.fitting.mode_selector import select_fitting_mode
    from se_simulator.recipe.manager import RecipeManager

    manager = RecipeManager()
    rpath = Path(recipe_path) if recipe_path else None

    sample_config, sim_conditions, system_config, floating_params, fitting_config = (
        manager.decompose_measurement(recipe, rpath)
    )

    mode = select_fitting_mode(recipe, sample_config)
    logger.info("[Pipeline] Fitting mode selected: %s", mode)

    if mode == "tmm_direct":
        fit_results = _run_tmm_direct(
            recipe,
            sample_config,
            sim_conditions,
            system_config,
            floating_params,
            fitting_config,
            target_spectrum,
            progress_callback,
        )
    else:
        fit_results = _run_library(
            recipe,
            sample_config,
            sim_conditions,
            system_config,
            floating_params,
            fitting_config,
            target_spectrum,
            progress_callback,
        )

    # Optionally persist results back into the recipe file
    if (
        recipe_path
        and recipe.output_options.save_recipe_with_results
        and rpath is not None
        and rpath.exists()
    ):
        try:
            manager.append_results(fit_results, rpath)
        except Exception:  # noqa: BLE001
            logger.warning("Could not write results back to %s", recipe_path, exc_info=True)

    return fit_results


# ---------------------------------------------------------------------------
# TMM-direct branch
# ---------------------------------------------------------------------------


def _run_tmm_direct(
    recipe: MeasurementRecipe,
    sample_config,
    sim_conditions,
    system_config,
    floating_params,
    fitting_config,
    target_spectrum: np.ndarray,
    progress_callback: Callable[[int, float], None] | None,
) -> FitResults:
    from se_simulator.fitting.tmm_direct_fitter import TmmDirectFitter

    # Prefer the inline Stack from the forward model when available
    fm = recipe.forward_model
    stack = fm.stack.inline if (fm.stack is not None and fm.stack.inline is not None) else None

    fitter = TmmDirectFitter(
        stack=stack,
        sample_config=sample_config if stack is None else None,
        sim_conditions=sim_conditions,
        system_config=system_config,
        floating_params=floating_params,
        fitting_config=fitting_config,
        progress_callback=progress_callback,
    )
    tmm_result = fitter.fit(target_spectrum)
    return fitter.to_fit_results(tmm_result)


# ---------------------------------------------------------------------------
# Library branch
# ---------------------------------------------------------------------------


def _run_library(
    recipe: MeasurementRecipe,
    sample_config,
    sim_conditions,
    system_config,
    floating_params,
    fitting_config,
    target_spectrum: np.ndarray,
    progress_callback: Callable[[int, float], None] | None,
) -> FitResults:
    """Delegate to the existing library-based ``FittingEngine``.

    This path requires a pre-built HDF5 library.  If no library is
    available, a ``FileNotFoundError`` is raised.
    """
    from datetime import UTC, datetime

    from se_simulator.config.recipe import FitResults
    from se_simulator.fitting.engine import FittingEngine
    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.engine import RCWAEngine

    library_path = recipe.library_reference.library_file.strip()
    if not library_path:
        raise FileNotFoundError(
            "Fitting mode is 'library' but no library_file is specified in the recipe."
        )

    db = MaterialDatabase()
    for spec in sample_config.materials.values():
        db.resolve(spec)

    engine = RCWAEngine(db)
    fitting_engine = FittingEngine(
        library_path=Path(library_path),
        rcwa_engine=engine,
        system=system_config,
        sim=sim_conditions,
    )

    # Build EllipsometryResult-like target from the flat target_spectrum array
    target_ell = _target_spectrum_to_ellipsometry_result(
        target_spectrum, sim_conditions, fitting_config
    )

    fc = _fitting_config_to_fitting_conditions(fitting_config, sim_conditions)

    def _lib_cb(stage: str, progress: float) -> None:
        if progress_callback is not None:
            # Map library stages to rough iteration counts
            iter_approx = int(progress * 100)
            progress_callback(iter_approx, 0.0)

    fit_result = fitting_engine.fit(target_ell, fitting_config=fc, progress_callback=_lib_cb)

    # Convert FitResult → FitResults Pydantic model
    fitted_params = {
        name: float(fit_result.final_params[i])
        for i, name in enumerate(fit_result.parameter_names)
    }
    return FitResults(
        fitted_parameters=fitted_params,
        fit_quality={
            "chi2": float(fit_result.final_chi2),
            "stages": ",".join(fit_result.pipeline_stages_run),
        },
        engine_used="library",
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


def _target_spectrum_to_ellipsometry_result(
    target_spectrum: np.ndarray,
    sim_conditions,
    fitting_config,
):
    """Wrap a flat target_spectrum array in an EllipsometryResult for library search."""
    from se_simulator.config.manager import ConfigManager
    from se_simulator.ellipsometer.signals import EllipsometryResult

    wls = ConfigManager().get_wavelengths(sim_conditions.wavelengths)
    n = len(wls)
    arr = np.asarray(target_spectrum, dtype=float)

    if arr.ndim == 2 and arr.shape[1] == 2:
        psi = arr[:, 0]
        delta = arr[:, 1]
    elif len(arr) == 2 * n:
        psi = arr[:n]
        delta = arr[n:]
    elif len(arr) == n:
        signals = fitting_config.fit_signals if fitting_config.fit_signals else ["psi"]
        if signals[0] == "delta":
            psi = np.zeros(n)
            delta = arr
        else:
            psi = arr
            delta = np.zeros(n)
    else:
        psi = arr[:n] if len(arr) >= n else np.zeros(n)
        delta = arr[n : 2 * n] if len(arr) >= 2 * n else np.zeros(n)

    return EllipsometryResult(
        wavelengths_nm=wls,
        psi_deg=psi,
        delta_deg=delta,
        alpha=np.zeros(n),
        beta=np.zeros(n),
        chi=np.zeros(n),
        xi=np.zeros(n),
        jones_reflection=np.zeros((n, 2, 2), dtype=complex),
        energy_conservation=np.ones(n),
    )


def _fitting_config_to_fitting_conditions(fitting_config, sim_conditions):
    """Convert FittingConfiguration (recipe) to FittingConditions (schema)."""

    fc = sim_conditions.fitting.model_copy(
        update={
            "fit_signals": fitting_config.fit_signals,
            "max_iterations": fitting_config.max_iterations,
            "convergence_tol": fitting_config.convergence_tolerance,
        }
    )
    return fc
