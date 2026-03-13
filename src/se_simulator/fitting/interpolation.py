"""Stage 2: library interpolation to refine search result below grid spacing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from se_simulator.config.schemas import FittingConditions
from se_simulator.ellipsometer.signals import EllipsometryResult
from se_simulator.fitting.library import LibraryStore
from se_simulator.fitting.search import SearchResult


@dataclass
class InterpolationResult:
    """Output from Stage 2 interpolation."""

    refined_params: np.ndarray
    refined_chi2: float
    parameter_names: list[str]
    chi2_map_data: dict | None


class LibraryInterpolator:
    """Stage 2: radial-basis-function interpolation of the χ² landscape."""

    def __init__(self, store: LibraryStore, search_result: SearchResult) -> None:
        self.store = store
        self.search_result = search_result

    def interpolate(
        self,
        target: EllipsometryResult,
        fitting_config: FittingConditions,
    ) -> InterpolationResult:
        """Refine the search result by minimising an RBF interpolant of χ²."""
        from scipy.interpolate import RBFInterpolator
        from scipy.optimize import minimize

        spec = self.store.get_spec()
        p_specs = spec.parameters
        n_params = len(p_specs)
        top_params = self.search_result.top_k_params  # (K, N_params)
        top_chi2 = self.search_result.top_k_chi2  # (K,)

        # Need at least 2 unique points to fit an RBF
        if n_params == 0 or len(top_params) < 2:
            return InterpolationResult(
                refined_params=self.search_result.best_fit_params.copy(),
                refined_chi2=self.search_result.best_fit_chi2,
                parameter_names=self.search_result.parameter_names,
                chi2_map_data=None,
            )

        p_mins = np.array([p.min_value for p in p_specs])
        p_maxs = np.array([p.max_value for p in p_specs])
        p_range = p_maxs - p_mins
        p_range[p_range == 0.0] = 1.0

        top_params_norm = (top_params - p_mins) / p_range  # [0, 1] normalised

        try:
            rbf = RBFInterpolator(
                top_params_norm, top_chi2, kernel="thin_plate_spline"
            )
        except Exception:
            return InterpolationResult(
                refined_params=self.search_result.best_fit_params.copy(),
                refined_chi2=self.search_result.best_fit_chi2,
                parameter_names=self.search_result.parameter_names,
                chi2_map_data=None,
            )

        x0_norm = (self.search_result.best_fit_params - p_mins) / p_range
        x0_norm = np.clip(x0_norm, 0.0, 1.0)

        def _objective(x_norm: np.ndarray) -> float:
            return float(rbf(x_norm[np.newaxis, :])[0])

        opt = minimize(
            _objective,
            x0_norm,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-7, "fatol": 1e-7},
        )

        refined_norm = np.clip(opt.x, 0.0, 1.0)
        refined_params = refined_norm * p_range + p_mins
        refined_chi2 = float(opt.fun)

        # If the optimizer made things worse, fall back to search best
        if refined_chi2 > self.search_result.best_fit_chi2:
            refined_params = self.search_result.best_fit_params.copy()
            refined_chi2 = self.search_result.best_fit_chi2

        return InterpolationResult(
            refined_params=refined_params,
            refined_chi2=refined_chi2,
            parameter_names=self.search_result.parameter_names,
            chi2_map_data=None,
        )

    def chi2_map(
        self,
        param_idx_x: int,
        param_idx_y: int,
        n_points: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a 2-D χ² map over two parameter axes via RBF interpolation.

        Returns ``(x_values, y_values, chi2_grid)`` where ``chi2_grid`` has
        shape ``(n_points, n_points)``.
        """
        from scipy.interpolate import RBFInterpolator

        spec = self.store.get_spec()
        p_specs = spec.parameters
        top_params = self.search_result.top_k_params
        top_chi2 = self.search_result.top_k_chi2

        p_mins = np.array([p.min_value for p in p_specs])
        p_maxs = np.array([p.max_value for p in p_specs])
        p_range = p_maxs - p_mins
        p_range[p_range == 0.0] = 1.0

        top_norm = (top_params - p_mins) / p_range
        rbf = RBFInterpolator(top_norm, top_chi2, kernel="thin_plate_spline")

        x_vals = np.linspace(
            p_specs[param_idx_x].min_value, p_specs[param_idx_x].max_value, n_points
        )
        y_vals = np.linspace(
            p_specs[param_idx_y].min_value, p_specs[param_idx_y].max_value, n_points
        )
        xx, yy = np.meshgrid(x_vals, y_vals)

        n_grid = n_points * n_points
        best = self.search_result.best_fit_params
        query = np.tile(best, (n_grid, 1))
        query[:, param_idx_x] = xx.ravel()
        query[:, param_idx_y] = yy.ravel()
        query_norm = (query - p_mins) / p_range

        chi2_flat = rbf(query_norm)
        chi2_grid = chi2_flat.reshape(n_points, n_points)

        return x_vals, y_vals, chi2_grid
