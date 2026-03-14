"""FittingWorker: QThread for non-blocking library-based fitting."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from se_simulator.config.schemas import FittingConditions, SimConditions, SystemConfig
from se_simulator.ellipsometer.signals import EllipsometryResult


class FittingWorker(QThread):
    """Run the three-stage fitting pipeline in a background thread.

    Signals
    -------
    stage_progress(str, float):
        Emits (stage_name, fraction) as each fitting stage proceeds.
    result_ready(object):
        Emits a :class:`~se_simulator.fitting.engine.FitResult` on success.
    error_occurred(str):
        Emits the error message if any exception is raised.
    finished():
        Always emitted when the thread exits.
    """

    stage_progress = Signal(str, float)
    result_ready = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        target: EllipsometryResult,
        library_path: Path,
        system: SystemConfig,
        sim: SimConditions,
        fitting_config: FittingConditions | None = None,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._target = target
        self._library_path = Path(library_path)
        self._system = system
        self._sim = sim
        self._fitting_config = fitting_config

    def run(self) -> None:
        """Execute the fitting pipeline. Called automatically by QThread.start()."""
        try:
            from se_simulator.fitting.engine import FittingEngine
            from se_simulator.materials.database import MaterialDatabase
            from se_simulator.rcwa.engine import RCWAEngine

            db = MaterialDatabase()
            engine = RCWAEngine(db)

            fitting_engine = FittingEngine(
                library_path=self._library_path,
                rcwa_engine=engine,
                system=self._system,
                sim=self._sim,
            )

            def _progress_cb(stage: str, frac: float) -> None:
                if self.isInterruptionRequested():
                    return
                self.stage_progress.emit(stage, frac)

            result = fitting_engine.fit(
                self._target,
                fitting_config=self._fitting_config,
                progress_callback=_progress_cb,
            )

            self.result_ready.emit(result)

        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()


class RecipeFittingWorker(QThread):
    """Run the recipe-based fitting pipeline (TMM Direct or Library) in a background thread.

    Signals
    -------
    result_ready(object):
        Emits a :class:`~se_simulator.config.recipe.FitResults` on success.
    error_occurred(str):
        Emits the error message if any exception is raised.
    finished():
        Always emitted when the thread exits.
    """

    result_ready = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        recipe: object,
        target_spectrum_path: str,
        recipe_path: str = "",
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._recipe = recipe
        self._target_spectrum_path = target_spectrum_path
        self._recipe_path = recipe_path

    def run(self) -> None:
        """Execute the recipe-based fitting pipeline."""
        try:
            import numpy as np

            from se_simulator.fitting.mode_selector import select_fitting_mode
            from se_simulator.recipe.manager import RecipeManager

            # Load target spectrum CSV → numpy array (psi + delta concatenated)
            data = np.genfromtxt(
                self._target_spectrum_path,
                delimiter=",",
                names=True,
                dtype=float,
            )
            if "psi_deg" in data.dtype.names and "delta_deg" in data.dtype.names:
                target_spectrum = np.concatenate([data["psi_deg"], data["delta_deg"]])
            elif data.ndim == 2:
                target_spectrum = data[:, 1]
                wavelengths_from_csv = None
            else:
                target_spectrum = np.asarray(data.tolist(), dtype=float)
                wavelengths_from_csv = None

            # Decompose recipe
            manager = RecipeManager()
            rpath = Path(self._recipe_path) if self._recipe_path else None
            sample_config, sim_conditions, system_config, floating_params, fitting_config = (
                manager.decompose_measurement(self._recipe, rpath)
            )

            mode = select_fitting_mode(self._recipe, sample_config)

            if mode == "tmm_direct":
                from se_simulator.fitting.tmm_direct_fitter import TmmDirectFitter

                fitter = TmmDirectFitter(
                    sample_config=sample_config,
                    sim_conditions=sim_conditions,
                    system_config=system_config,
                    floating_params=floating_params,
                    fitting_config=fitting_config,
                )
                tmm_result = fitter.fit(target_spectrum)

                # Save best-fit spectrum to fit.csv
                wls = tmm_result.wavelengths_nm
                psi = tmm_result.best_fit_psi
                delta = tmm_result.best_fit_delta
                _save_fit_csv(wls, psi, delta)

                result = fitter.to_fit_results(tmm_result)
            else:
                # Library path — use full pipeline
                from se_simulator.fitting.pipeline import run_fitting
                result = run_fitting(
                    recipe=self._recipe,
                    target_spectrum=target_spectrum,
                    recipe_path=self._recipe_path,
                )

            # Optionally persist results back into the recipe file
            if (
                self._recipe_path
                and self._recipe.output_options.save_recipe_with_results
                and rpath is not None
                and rpath.exists()
            ):
                try:
                    manager.append_results(result, rpath)
                except Exception:  # noqa: BLE001
                    pass

            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()


def _save_fit_csv(wavelengths_nm: np.ndarray, psi_deg: np.ndarray, delta_deg: np.ndarray) -> None:
    """Write the best-fit model spectrum to fit.csv in the current working directory."""
    import numpy as np

    out = np.column_stack([wavelengths_nm, psi_deg, delta_deg])
    header = "wavelength_nm,psi_deg,delta_deg"
    np.savetxt("fit.csv", out, delimiter=",", header=header, comments="")
