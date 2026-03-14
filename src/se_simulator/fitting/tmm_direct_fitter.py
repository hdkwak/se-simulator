"""TmmDirectFitter: TMM-based fitting without a pre-computed library.

For each forward evaluation the fitter:
  1. Applies the current parameter vector to a deep copy of the stack dict.
  2. Calls ``compute_tmm`` directly (bypassing the full RCWA engine).
  3. Derives psi/delta via ``compute_psi_delta``.
  4. Returns the concatenated residual vector.

Supported optimizers:
  - ``levenberg_marquardt`` — scipy ``least_squares`` with method ``'trf'``
  - ``nelder_mead``         — scipy ``minimize`` with method ``'Nelder-Mead'``
  - ``differential_evolution`` — scipy ``differential_evolution``
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from se_simulator.config.recipe import FitResults, FittingConfiguration, FloatingParameter
from se_simulator.config.schemas import SampleConfig, SimConditions, Stack, SystemConfig
from se_simulator.recipe.dotpath import resolve_set

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Legacy prefix constants — kept for backward compatibility (used by tests via _strip_prefix)
_FM_PREFIX = "forward_model."
_FM_SAMPLE_INLINE_PREFIX = "forward_model.sample.inline."

# All legacy prefixes that may appear in target_field paths
_LEGACY_PREFIXES = [
    "forward_model.sample.inline.",
    "sample.inline.",
    "forward_model.",
]


@dataclass
class TmmFitResult:
    """Result from TmmDirectFitter.fit()."""

    fitted_parameters: dict[str, float]
    fit_quality: dict[str, float]   # rmse, chi2, n_iter, converged
    engine_used: str
    timestamp: str
    best_fit_psi: np.ndarray
    best_fit_delta: np.ndarray
    wavelengths_nm: np.ndarray


def _strip_prefix(path: str) -> str:
    """Strip the forward_model.sample.inline. prefix from a target_field path.

    Falls back to stripping only ``forward_model.`` if the longer prefix is
    not present.  If neither prefix is present the path is returned unchanged
    so it can be applied directly to the *sample* dict.

    .. deprecated::
        Use :func:`se_simulator.recipe.manager._migrate_target_field` instead.
        Kept for backward compatibility with existing tests and external callers.
    """
    if path.startswith(_FM_SAMPLE_INLINE_PREFIX):
        return path[len(_FM_SAMPLE_INLINE_PREFIX):]
    if path.startswith(_FM_PREFIX):
        return path[len(_FM_PREFIX):]
    return path


def _local_path_from_target(path: str) -> str:
    """Strip any legacy prefix from a target_field, returning the Stack-rooted path."""
    for prefix in _LEGACY_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


class TmmDirectFitter:
    """Fit ellipsometric spectra directly using the TMM forward model.

    Parameters
    ----------
    stack:
        Base stack.  A deep copy is mutated for each forward call.
        Either *stack* or *sample_config* must be provided; *stack* takes
        precedence when both are given.
    sample_config:
        Deprecated — provide *stack* instead.  Accepted for backward
        compatibility with existing callers; internally converted to a Stack
        via ``Stack.from_sample_config()`` (not yet implemented) or kept as-is
        for the ``SampleConfig``-based forward path.
    sim_conditions:
        Wavelength range, AOI, azimuth — passed directly to ``compute_tmm``.
    system_config:
        Ellipsometer geometry used to compute psi/delta from Jones matrices.
    floating_params:
        Ordered list of parameters to optimise.  ``target_field`` paths should
        reference fields within the Stack (e.g. ``layers[0].thickness_nm``).
        Paths beginning with ``forward_model.sample.inline.`` are automatically
        stripped to their relative form.
    fitting_config:
        Controls optimizer selection, tolerances, and fit_signals.
    progress_callback:
        Optional ``(iteration: int, rmse: float) -> None`` callable invoked
        after each forward evaluation.
    """

    def __init__(
        self,
        sim_conditions: SimConditions,
        system_config: SystemConfig,
        floating_params: list[FloatingParameter],
        fitting_config: FittingConfiguration,
        stack: Stack | None = None,
        sample_config: SampleConfig | None = None,
        progress_callback: Callable[[int, float], None] | None = None,
    ) -> None:
        if stack is None and sample_config is None:
            raise ValueError("Either 'stack' or 'sample_config' must be provided")

        self._stack = stack
        # Keep sample_config reference for legacy callers that inspect it directly
        self.sample_config = sample_config
        self.sim_conditions = sim_conditions
        self.system_config = system_config
        self.floating_params = floating_params
        self.fitting_config = fitting_config
        self.progress_callback = progress_callback

        # Resolve wavelengths once
        from se_simulator.config.manager import ConfigManager

        self._wavelengths_nm: np.ndarray = ConfigManager().get_wavelengths(
            sim_conditions.wavelengths
        )

        # Build local dot-path keys (stripped of legacy prefixes)
        self._local_paths: list[str] = [
            _local_path_from_target(fp.target_field) for fp in floating_params
        ]

        # Determine the base SampleConfig for material pre-loading
        import warnings

        base_sample: SampleConfig
        if self._stack is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                base_sample = self._stack.to_sample_config()
        elif isinstance(self.sample_config, Stack):
            # Defensive: caller passed a Stack as sample_config — normalise it.
            self._stack = self.sample_config  # type: ignore[assignment]
            self.sample_config = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                base_sample = self._stack.to_sample_config()
        else:
            base_sample = self.sample_config  # type: ignore[assignment]

        # Pre-load material database and register all materials.
        from se_simulator.materials.database import MaterialDatabase

        self._db = MaterialDatabase()
        for spec in base_sample.materials.values():
            if spec.name not in self._db._cache:
                self._db.resolve(spec)

        # Iteration counter (shared via mutable container)
        self._n_evals: list[int] = [0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, target_spectrum: np.ndarray) -> TmmFitResult:
        """Fit against *target_spectrum* and return a ``TmmFitResult``.

        Parameters
        ----------
        target_spectrum:
            Array of shape ``(N,)`` (psi or delta alone) or ``(2*N,)``
            (psi concatenated with delta) or ``(N, 2)`` (psi in col 0,
            delta in col 1).  The shape must be consistent with
            ``fitting_config.fit_signals``.
        """
        target_flat = self._prepare_target(target_spectrum)
        x0 = np.array([fp.initial for fp in self.floating_params])
        self._n_evals[0] = 0

        opt_name = self.fitting_config.optimizer
        x_opt, info = self._run_optimizer(opt_name, x0, target_flat)

        psi_fit, delta_fit = self._compute_psi_delta(x_opt)
        fitted = {fp.name: float(x_opt[i]) for i, fp in enumerate(self.floating_params)}

        # Compute final RMSE
        final_vec = self._build_signal_vector(psi_fit, delta_fit)
        residuals = final_vec - target_flat
        rmse = float(np.sqrt(np.mean(residuals**2)))
        chi2 = float(np.sum(residuals**2))

        return TmmFitResult(
            fitted_parameters=fitted,
            fit_quality={
                "rmse": rmse,
                "chi2": chi2,
                "n_iter": int(self._n_evals[0]),
                "converged": float(info.get("converged", 0.0)),
            },
            engine_used="tmm_direct",
            timestamp=datetime.now(tz=UTC).isoformat(),
            best_fit_psi=psi_fit,
            best_fit_delta=delta_fit,
            wavelengths_nm=self._wavelengths_nm.copy(),
        )

    def to_fit_results(self, result: TmmFitResult) -> FitResults:
        """Convert a ``TmmFitResult`` to the Pydantic ``FitResults`` schema."""
        return FitResults(
            fitted_parameters=result.fitted_parameters,
            fit_quality=result.fit_quality,
            engine_used=result.engine_used,
            timestamp=result.timestamp,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Run TMM forward model for parameter vector *x*.

        Returns a flat 1-D array of the requested fit signals.
        """
        psi, delta = self._compute_psi_delta(x)
        return self._build_signal_vector(psi, delta)

    def _compute_psi_delta(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply *x*, run TMM, return (psi_deg, delta_deg) arrays."""
        from se_simulator.ellipsometer.prcsa import compute_psi_delta
        from se_simulator.rcwa.tmm import compute_tmm

        import warnings

        if self._stack is not None:
            # New Stack-based path
            stack_dict = self._stack.model_dump()
            for i, path in enumerate(self._local_paths):
                resolve_set(stack_dict, path, float(x[i]))
            stack_mod = Stack.model_validate(stack_dict)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                sample_mod = stack_mod.to_sample_config()
        else:
            # Legacy SampleConfig-based path
            sample_dict = self.sample_config.model_dump()  # type: ignore[union-attr]
            for i, path in enumerate(self._local_paths):
                resolve_set(sample_dict, path, float(x[i]))
            sample_mod = SampleConfig.model_validate(sample_dict)

        # Register any materials not yet in the cache.
        for spec in sample_mod.materials.values():
            if spec.name not in self._db._cache:
                self._db.resolve(spec)

        jones_r, _jones_t = compute_tmm(
            sample_mod,
            self._db,
            self._wavelengths_nm,
            self.sim_conditions.aoi_deg,
            self.sim_conditions.azimuth_deg,
        )

        n_wl = len(self._wavelengths_nm)
        psi = np.empty(n_wl)
        delta = np.empty(n_wl)
        for i in range(n_wl):
            psi[i], delta[i] = compute_psi_delta(jones_r[i])

        self._n_evals[0] += 1
        if self.progress_callback is not None:
            flat = self._build_signal_vector(psi, delta)
            # RMSE against a dummy zero target — caller gets iteration + current RMSE
            self.progress_callback(self._n_evals[0], float(np.sqrt(np.mean(flat**2))))

        return psi, delta

    def _build_signal_vector(self, psi: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Concatenate signals according to ``fit_signals``."""
        parts: list[np.ndarray] = []
        for sig in self.fitting_config.fit_signals:
            if sig == "psi":
                parts.append(psi)
            elif sig == "delta":
                parts.append(delta)
        if not parts:
            parts = [psi, delta]
        return np.concatenate(parts)

    def _prepare_target(self, target_spectrum: np.ndarray) -> np.ndarray:
        """Normalise *target_spectrum* to a 1-D flat array."""
        arr = np.asarray(target_spectrum, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            # (N, 2) — col 0 = psi, col 1 = delta; flatten in signal order
            parts: list[np.ndarray] = []
            for sig in self.fitting_config.fit_signals:
                if sig == "psi":
                    parts.append(arr[:, 0])
                elif sig == "delta":
                    parts.append(arr[:, 1])
            return np.concatenate(parts) if parts else arr.ravel()
        return arr.ravel()

    def _run_optimizer(
        self,
        optimizer: str,
        x0: np.ndarray,
        target_flat: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Dispatch to the selected scipy optimizer.

        Returns ``(x_optimal, info_dict)``.
        """
        bounds = [(fp.min, fp.max) for fp in self.floating_params]
        max_iter = self.fitting_config.max_iterations
        tol = self.fitting_config.convergence_tolerance

        if optimizer == "levenberg_marquardt":
            return self._run_lm(x0, target_flat, bounds, max_iter, tol)
        if optimizer == "nelder_mead":
            return self._run_nelder_mead(x0, target_flat, bounds, max_iter, tol)
        if optimizer == "differential_evolution":
            return self._run_de(target_flat, bounds, max_iter, tol)

        logger.warning("Unknown optimizer %r; falling back to levenberg_marquardt.", optimizer)
        return self._run_lm(x0, target_flat, bounds, max_iter, tol)

    # ---- individual optimizer backends ----

    def _run_lm(
        self,
        x0: np.ndarray,
        target_flat: np.ndarray,
        bounds: list[tuple[float, float]],
        max_iter: int,
        tol: float,
    ) -> tuple[np.ndarray, dict]:
        from scipy.optimize import least_squares

        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

        def _residuals(x: np.ndarray) -> np.ndarray:
            return self._forward(x) - target_flat

        result = least_squares(
            _residuals,
            x0,
            method="trf",
            bounds=(lb, ub),
            max_nfev=max_iter,
            ftol=tol,
            xtol=tol,
            gtol=tol,
        )
        return result.x, {"converged": float(result.success)}

    def _run_nelder_mead(
        self,
        x0: np.ndarray,
        target_flat: np.ndarray,
        bounds: list[tuple[float, float]],
        max_iter: int,
        tol: float,
    ) -> tuple[np.ndarray, dict]:
        from scipy.optimize import minimize

        def _objective(x: np.ndarray) -> float:
            residuals = self._forward(x) - target_flat
            return float(np.sum(residuals**2))

        result = minimize(
            _objective,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": max_iter, "fatol": tol, "xatol": tol},
        )
        return result.x, {"converged": float(result.success)}

    def _run_de(
        self,
        target_flat: np.ndarray,
        bounds: list[tuple[float, float]],
        max_iter: int,
        tol: float,
    ) -> tuple[np.ndarray, dict]:
        from scipy.optimize import differential_evolution

        def _objective(x: np.ndarray) -> float:
            residuals = self._forward(x) - target_flat
            return float(np.sum(residuals**2))

        result = differential_evolution(
            _objective,
            bounds,
            maxiter=max_iter,
            tol=tol,
            seed=42,
        )
        return result.x, {"converged": float(result.success)}
