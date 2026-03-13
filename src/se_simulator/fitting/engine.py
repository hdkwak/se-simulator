"""FittingEngine: three-stage fitting pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from se_simulator.config.schemas import FittingConditions, SimConditions, SystemConfig
from se_simulator.ellipsometer.signals import EllipsometryResult
from se_simulator.fitting.interpolation import InterpolationResult, LibraryInterpolator
from se_simulator.fitting.library import LibraryStore
from se_simulator.fitting.refinement import RefinementResult
from se_simulator.fitting.search import NearestNeighborSearch, SearchResult


@dataclass
class FitResult:
    """Unified result from the full fitting pipeline."""

    stage1_search: SearchResult
    stage2_interpolation: InterpolationResult | None
    stage3_refinement: RefinementResult | None
    final_params: np.ndarray
    final_chi2: float
    parameter_names: list[str]
    sigma_params: np.ndarray | None
    best_fit_spectrum: EllipsometryResult
    pipeline_stages_run: list[str]

    def to_csv(self, path: Path) -> None:
        """Write fit results to CSV with comment-line header."""
        path = Path(path)
        now = datetime.now(UTC).isoformat()
        try:
            n_top_k = int(self.stage1_search.top_k_indices.size)
        except Exception:
            n_top_k = 0

        lines: list[str] = [
            f"# chi2_final={self.final_chi2:.6f}\n",
            f"# stages_run={','.join(self.pipeline_stages_run)}\n",
            f"# timestamp={now}\n",
            f"# n_library_entries_top_k={n_top_k}\n",
            "parameter_name,fitted_value,sigma_1\n",
        ]
        for i, name in enumerate(self.parameter_names):
            val = float(self.final_params[i]) if i < len(self.final_params) else float("nan")
            if self.sigma_params is not None and i < len(self.sigma_params):
                sig = float(self.sigma_params[i])
            else:
                sig = float("nan")
            lines.append(f"{name},{val:.6f},{sig:.6f}\n")

        path.write_text("".join(lines))

    def summary(self) -> str:
        """Return a human-readable multi-line summary string."""
        lines = [
            "FitResult Summary",
            f"  Stages run: {', '.join(self.pipeline_stages_run)}",
            f"  Final chi2: {self.final_chi2:.4f}",
            "  Parameters:",
        ]
        for i, name in enumerate(self.parameter_names):
            val = float(self.final_params[i]) if i < len(self.final_params) else float("nan")
            if self.sigma_params is not None and i < len(self.sigma_params):
                sigma_str = f" ± {float(self.sigma_params[i]):.4f}"
            else:
                sigma_str = ""
            lines.append(f"    {name}: {val:.4f}{sigma_str}")
        return "\n".join(lines)


class FittingEngine:
    """Orchestrates the three-stage library-based fitting pipeline."""

    def __init__(
        self,
        library_path: Path,
        rcwa_engine: object,
        system: SystemConfig,
        sim: SimConditions,
    ) -> None:
        self.library_path = Path(library_path)
        self.rcwa_engine = rcwa_engine
        self.system = system
        self.sim = sim
        self._store: LibraryStore | None = None

    def load_library(self) -> None:
        """Load the library store (called automatically on first ``fit()``)."""
        self._store = LibraryStore(self.library_path)

    def fit(
        self,
        target: EllipsometryResult,
        fitting_config: FittingConditions | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> FitResult:
        """Run the full three-stage fitting pipeline.

        Stage 1 (search) is always executed.
        Stage 2 (interpolation) runs when ``fitting_config.use_interpolation=True``.
        Stage 3 (refinement) runs when ``fitting_config.use_refinement=True``.
        """
        if self._store is None:
            self.load_library()
        assert self._store is not None

        fc = fitting_config if fitting_config is not None else self.sim.fitting

        # ----- Stage 1: nearest-neighbour search -----
        if progress_callback is not None:
            progress_callback("search", 0.0)

        searcher = NearestNeighborSearch(self._store, fc)
        search_result = searcher.search(target)
        stages_run = ["search"]

        if progress_callback is not None:
            progress_callback("search", 1.0)

        current_params = search_result.best_fit_params.copy()
        current_chi2 = search_result.best_fit_chi2
        current_spectrum = search_result.best_fit_spectrum

        interp_result: InterpolationResult | None = None
        refine_result: RefinementResult | None = None

        # ----- Stage 2: library interpolation -----
        if fc.use_interpolation:
            if progress_callback is not None:
                progress_callback("interpolation", 0.0)

            interpolator = LibraryInterpolator(self._store, search_result)
            interp_result = interpolator.interpolate(target, fc)
            current_params = interp_result.refined_params
            current_chi2 = interp_result.refined_chi2
            stages_run.append("interpolation")

            if progress_callback is not None:
                progress_callback("interpolation", 1.0)

        # ----- Stage 3: gradient refinement -----
        if fc.use_refinement:
            if progress_callback is not None:
                progress_callback("refinement", 0.0)

            from se_simulator.config.schemas import SampleConfig
            from se_simulator.fitting.refinement import GradientRefinement

            spec = self._store.get_spec()
            sample_base = SampleConfig.model_validate(spec.sample_config_snapshot)

            def _refine_cb(n_iter: int, chi2_val: float) -> None:
                if progress_callback is not None:
                    frac = min(1.0, n_iter / max(1, fc.max_iterations))
                    progress_callback("refinement", frac)

            refiner = GradientRefinement(
                self.rcwa_engine,
                sample_base,
                self.system,
                self.sim,
                spec.parameters,
                fc,
            )
            refine_result = refiner.refine(target, current_params, _refine_cb)
            current_params = refine_result.final_params
            current_chi2 = refine_result.final_chi2
            current_spectrum = refine_result.best_fit_spectrum
            stages_run.append("refinement")

            if progress_callback is not None:
                progress_callback("refinement", 1.0)

        sigma = refine_result.sigma_params if refine_result is not None else None

        return FitResult(
            stage1_search=search_result,
            stage2_interpolation=interp_result,
            stage3_refinement=refine_result,
            final_params=current_params,
            final_chi2=current_chi2,
            parameter_names=search_result.parameter_names,
            sigma_params=sigma,
            best_fit_spectrum=current_spectrum,
            pipeline_stages_run=stages_run,
        )
