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
