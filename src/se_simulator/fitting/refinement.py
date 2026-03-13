"""Stage 3: gradient-based local optimisation starting from the interpolation result."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from se_simulator.config.schemas import (
    FittingConditions,
    SampleConfig,
    SimConditions,
    SystemConfig,
)
from se_simulator.ellipsometer.signals import EllipsometryResult
from se_simulator.fitting.library import ParameterSpec, apply_params
from se_simulator.fitting.search import chi_squared


@dataclass
class RefinementResult:
    """Output from Stage 3 gradient refinement."""

    final_params: np.ndarray
    final_chi2: float
    parameter_names: list[str]
    sigma_params: np.ndarray
    covariance_matrix: np.ndarray
    n_iterations: int
    converged: bool
    convergence_message: str
    best_fit_spectrum: EllipsometryResult


class GradientRefinement:
    """Stage 3: local optimisation using scipy.optimize.minimize."""

    def __init__(
        self,
        rcwa_engine: object,
        sample_base: SampleConfig,
        system: SystemConfig,
        sim: SimConditions,
        param_specs: list[ParameterSpec],
        fitting_config: FittingConditions,
    ) -> None:
        self.rcwa_engine = rcwa_engine
        self.sample_base = sample_base
        self.system = system
        self.sim = sim
        self.param_specs = param_specs
        self.fitting_config = fitting_config
        self._n_evals = 0

    def refine(
        self,
        target: EllipsometryResult,
        initial_params: np.ndarray,
        progress_callback: Callable[[int, float], None] | None = None,
    ) -> RefinementResult:
        """Run local optimisation from ``initial_params``.

        The objective is χ²(target, simulate(p)).  After convergence the
        parameter covariance is estimated from a finite-difference Hessian.
        """
        from scipy.optimize import minimize

        bounds = [(p.min_value, p.max_value) for p in self.param_specs]

        sig_map = {
            "psi": target.psi_deg,
            "delta": target.delta_deg,
            "alpha": target.alpha,
            "beta": target.beta,
            "chi": target.chi,
        }
        sigma_defaults: dict[str, float] = {
            "psi": self.fitting_config.sigma_psi,
            "delta": self.fitting_config.sigma_delta,
            "alpha": 0.01,
            "beta": 0.01,
            "chi": 0.01,
        }
        target_dict = {
            s: sig_map[s] for s in self.fitting_config.fit_signals if s in sig_map
        }
        sigma_dict: dict[str, float | np.ndarray] = {
            s: sigma_defaults.get(s, 1.0) for s in target_dict
        }

        self._n_evals = 0
        best_chi2_container: list[float] = [float("inf")]
        best_spectrum_container: list[EllipsometryResult] = []

        def _objective(params: np.ndarray) -> float:
            ell = self._forward_simulate(params)
            lib_dict: dict[str, np.ndarray] = {
                "psi": ell.psi_deg[np.newaxis, :],
                "delta": ell.delta_deg[np.newaxis, :],
                "alpha": ell.alpha[np.newaxis, :],
                "beta": ell.beta[np.newaxis, :],
                "chi": ell.chi[np.newaxis, :],
            }
            val = float(chi_squared(target_dict, lib_dict, sigma_dict)[0])
            self._n_evals += 1
            if val < best_chi2_container[0]:
                best_chi2_container[0] = val
                if best_spectrum_container:
                    best_spectrum_container[0] = ell
                else:
                    best_spectrum_container.append(ell)
            if progress_callback is not None:
                progress_callback(self._n_evals, val)
            return val

        method = self.fitting_config.refinement_algo
        options: dict = {"maxiter": self.fitting_config.max_iterations}
        if method == "L-BFGS-B":
            options["ftol"] = self.fitting_config.convergence_tol
        else:
            options["fatol"] = self.fitting_config.convergence_tol

        opt = minimize(
            _objective,
            initial_params,
            method=method,
            bounds=bounds if method == "L-BFGS-B" else None,
            options=options,
        )

        final_params = opt.x
        final_chi2 = float(opt.fun)
        n_params = len(final_params)
        n_data = len(target.psi_deg) * len(target_dict)

        # Finite-difference Hessian for covariance estimation
        h = 1e-4 * np.maximum(np.abs(final_params), 1.0)
        hessian = np.zeros((n_params, n_params))
        converged = bool(opt.success)
        try:
            f0 = final_chi2
            f_plus = np.array([
                _objective(final_params + h * np.eye(1, n_params, i).ravel())
                for i in range(n_params)
            ])
            for i in range(n_params):
                for j in range(i, n_params):
                    ei = np.zeros(n_params)
                    ej = np.zeros(n_params)
                    ei[i] = h[i]
                    ej[j] = h[j]
                    f_ij = _objective(final_params + ei + ej)
                    h_ij = (f_ij - f_plus[i] - f_plus[j] + f0) / (h[i] * h[j])
                    hessian[i, j] = h_ij
                    hessian[j, i] = h_ij

            dof = max(1, n_data - n_params)
            try:
                cov = np.linalg.inv(hessian) * final_chi2 / dof
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(hessian) * final_chi2 / dof
                converged = False

            if np.any(np.linalg.eigvalsh(cov) < 0):
                cov = np.linalg.pinv(hessian) * final_chi2 / dof

            sigma_params = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except Exception:
            cov = np.full((n_params, n_params), np.nan)
            sigma_params = np.full(n_params, np.nan)
            converged = False

        fit_spectrum = (
            best_spectrum_container[0]
            if best_spectrum_container
            else self._forward_simulate(final_params)
        )

        return RefinementResult(
            final_params=final_params,
            final_chi2=final_chi2,
            parameter_names=[p.name for p in self.param_specs],
            sigma_params=sigma_params,
            covariance_matrix=cov,
            n_iterations=int(opt.nit),
            converged=converged,
            convergence_message=str(opt.message),
            best_fit_spectrum=fit_spectrum,
        )

    def _forward_simulate(self, params: np.ndarray) -> EllipsometryResult:
        """Apply ``params`` to the base sample and run RCWA + ellipsometer."""
        from se_simulator.ellipsometer.prcsa import compute_spectrum

        sample_mod = apply_params(self.sample_base, self.param_specs, tuple(params))
        rcwa_result = self.rcwa_engine.run(sample_mod, self.sim)  # type: ignore[attr-defined]
        return compute_spectrum(rcwa_result, self.system)
