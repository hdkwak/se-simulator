"""SimulationWorker: QThread for non-blocking RCWA + ellipsometer computation."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from se_simulator.config.schemas import SampleConfig, SimConditions, SystemConfig


class SimulationWorker(QThread):
    """Run RCWA simulation and ellipsometer spectrum computation in a background thread.

    Signals
    -------
    progress(float):
        Emits values from 0.0 to 1.0 as computation proceeds.
    result_ready(object):
        Emits an :class:`~se_simulator.ellipsometer.signals.EllipsometryResult`
        when computation completes successfully.
    error_occurred(str):
        Emits the error message string if any exception is raised.
    finished():
        Always emitted when the thread is about to exit (success or failure).
    """

    progress = Signal(float)
    result_ready = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        sample: SampleConfig,
        sim: SimConditions,
        system: SystemConfig,
        include_calibration_errors: bool = False,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample = sample
        self._sim = sim
        self._system = system
        self._include_calibration_errors = include_calibration_errors

    def run(self) -> None:
        """Execute the simulation pipeline. Called automatically by QThread.start()."""
        try:
            from se_simulator.ellipsometer.prcsa import compute_spectrum
            from se_simulator.materials.database import MaterialDatabase
            from se_simulator.rcwa.engine import RCWAEngine

            db = MaterialDatabase()
            engine = RCWAEngine(db)

            def _progress_cb(frac: float) -> None:
                if self.isInterruptionRequested():
                    return
                self.progress.emit(frac * 0.9)  # reserve last 10% for ellipsometer

            rcwa_result = engine.run(
                self._sample,
                self._sim,
                progress_callback=_progress_cb,
            )

            if self.isInterruptionRequested():
                return

            ell_result = compute_spectrum(
                rcwa_result,
                self._system,
                include_calibration_errors=self._include_calibration_errors,
            )

            self.progress.emit(1.0)
            self.result_ready.emit(ell_result)

        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()
