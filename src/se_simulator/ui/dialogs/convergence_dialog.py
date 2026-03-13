"""ConvergenceDialog: run and display RCWA convergence test results."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import SampleConfig, SimConditions


class _ConvergenceWorker(QThread):
    """Worker for running convergence test in background."""

    result_ready = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        sample: SampleConfig,
        sim: SimConditions,
        wavelength_nm: float,
        n_min: int,
        n_max: int,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample = sample
        self._sim = sim
        self._wavelength_nm = wavelength_nm
        self._n_min = n_min
        self._n_max = n_max

    def run(self) -> None:
        try:
            from se_simulator.materials.database import MaterialDatabase
            from se_simulator.rcwa.engine import RCWAEngine

            db = MaterialDatabase()
            engine = RCWAEngine(db)
            result = engine.convergence_test(
                self._sample,
                self._sim,
                self._wavelength_nm,
                n_range=range(self._n_min, self._n_max + 1),
            )
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()


class ConvergenceDialog(QDialog):
    """Dialog for running and visualizing RCWA convergence tests.

    Shows how Ψ and Δ converge as the number of harmonics N increases.
    """

    def __init__(
        self,
        sample: SampleConfig | None = None,
        sim: SimConditions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample = sample
        self._sim = sim
        self._worker: _ConvergenceWorker | None = None
        self.setWindowTitle("Convergence Test")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Parameters
        params_group = QGroupBox("Test Parameters")
        params_form = QFormLayout(params_group)

        self._wl_spin = QDoubleSpinBox()
        self._wl_spin.setRange(100.0, 10000.0)
        self._wl_spin.setDecimals(1)
        self._wl_spin.setSuffix(" nm")
        self._wl_spin.setValue(633.0)
        params_form.addRow("Wavelength:", self._wl_spin)

        self._n_min_spin = QSpinBox()
        self._n_min_spin.setRange(1, 20)
        self._n_min_spin.setValue(1)
        params_form.addRow("N min:", self._n_min_spin)

        self._n_max_spin = QSpinBox()
        self._n_max_spin.setRange(1, 20)
        self._n_max_spin.setValue(8)
        params_form.addRow("N max:", self._n_max_spin)

        layout.addWidget(params_group)

        # Run button + progress
        self._btn_run = QPushButton("Run Convergence Test")
        self._btn_run.clicked.connect(self._run_test)
        layout.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Plot area
        from se_simulator.ui.plots.spectrum_plot import SpectrumPlot

        self._psi_plot = SpectrumPlot()
        self._psi_plot.set_xlabel("N harmonics")
        self._psi_plot.set_ylabel("Ψ (°)")
        self._psi_plot.set_title("Psi Convergence")
        layout.addWidget(self._psi_plot)

        self._delta_plot = SpectrumPlot()
        self._delta_plot.set_xlabel("N harmonics")
        self._delta_plot.set_ylabel("Δ (°)")
        self._delta_plot.set_title("Delta Convergence")
        layout.addWidget(self._delta_plot)

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_configs(self, sample: SampleConfig, sim: SimConditions) -> None:
        """Update configurations for the convergence test."""
        self._sample = sample
        self._sim = sim

    def _run_test(self) -> None:
        if self._sample is None or self._sim is None:
            return

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._psi_plot.clear()
        self._delta_plot.clear()

        self._worker = _ConvergenceWorker(
            self._sample,
            self._sim,
            self._wl_spin.value(),
            self._n_min_spin.value(),
            self._n_max_spin.value(),
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_result(self, result: dict) -> None:

        ns = result["N"].astype(float)
        self._psi_plot.add_dataset("Ψ", ns, result["psi"], color="#1f77b4")
        self._delta_plot.add_dataset("Δ", ns, result["delta"], color="#ff7f0e")

    def _on_error(self, msg: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(self, "Convergence Test Error", msg)

    def _on_done(self) -> None:
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
