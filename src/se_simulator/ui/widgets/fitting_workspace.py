"""FittingWorkspace: widget for loading a library and running the fitting pipeline."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import SimConditions, SystemConfig
from se_simulator.ellipsometer.signals import EllipsometryResult


class FittingWorkspace(QWidget):
    """Workspace for running the three-stage library fitting pipeline.

    Signals
    -------
    fit_completed(object):
        Emitted when fitting finishes; carries a FitResult.
    """

    fit_completed = Signal(object)

    def __init__(
        self,
        system: SystemConfig | None = None,
        sim: SimConditions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._system = system
        self._sim = sim
        self._target: EllipsometryResult | None = None
        self._library_path: Path | None = None
        self._worker: object | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Library selection
        lib_group = QGroupBox("Library")
        lib_layout = QHBoxLayout(lib_group)
        self._lib_path_edit = QLineEdit()
        self._lib_path_edit.setPlaceholderText("library.h5")
        self._btn_browse_lib = QPushButton("Browse…")
        lib_layout.addWidget(QLabel("Path:"))
        lib_layout.addWidget(self._lib_path_edit)
        lib_layout.addWidget(self._btn_browse_lib)
        layout.addWidget(lib_group)

        # Target data
        target_group = QGroupBox("Target Spectrum")
        target_layout = QHBoxLayout(target_group)
        self._target_path_edit = QLineEdit()
        self._target_path_edit.setPlaceholderText("measured.csv")
        self._btn_browse_target = QPushButton("Browse…")
        target_layout.addWidget(QLabel("Path:"))
        target_layout.addWidget(self._target_path_edit)
        target_layout.addWidget(self._btn_browse_target)
        layout.addWidget(target_group)

        # Run controls
        run_group = QGroupBox("Run Fitting")
        run_layout = QVBoxLayout(run_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._lbl_stage = QLabel("")
        run_layout.addWidget(self._lbl_stage)
        run_layout.addWidget(self._progress_bar)

        btn_row = QHBoxLayout()
        self._btn_fit = QPushButton("RUN Fit")
        self._btn_fit.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; }"
        )
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._btn_fit)
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch()
        run_layout.addLayout(btn_row)
        layout.addWidget(run_group)

        # Results summary
        summary_group = QGroupBox("Results Summary")
        summary_layout = QVBoxLayout(summary_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(150)
        summary_layout.addWidget(self._result_text)
        layout.addWidget(summary_group)
        layout.addStretch()

        # Connections
        self._btn_browse_lib.clicked.connect(self._browse_library)
        self._btn_browse_target.clicked.connect(self._browse_target)
        self._btn_fit.clicked.connect(self._start_fitting)
        self._btn_stop.clicked.connect(self._stop_fitting)

    def set_configs(self, system: SystemConfig, sim: SimConditions) -> None:
        """Update the configs used for fitting."""
        self._system = system
        self._sim = sim

    def set_target(self, target: EllipsometryResult) -> None:
        """Set the target spectrum to fit against."""
        self._target = target

    def _browse_library(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Library", "", "HDF5 Files (*.h5);;All Files (*)"
        )
        if path:
            self._lib_path_edit.setText(path)

    def _browse_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Measured Spectrum", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._target_path_edit.setText(path)
            import contextlib

            with contextlib.suppress(Exception):
                self._target = EllipsometryResult.from_csv(Path(path))

    def _start_fitting(self) -> None:
        lib_path_str = self._lib_path_edit.text().strip()
        if not lib_path_str or self._target is None:
            self._result_text.setPlainText("Error: load library and target spectrum first")
            return
        if self._system is None or self._sim is None:
            self._result_text.setPlainText("Error: no system/sim config loaded")
            return

        from se_simulator.ui.workers.fitting_worker import FittingWorker

        self._worker = FittingWorker(
            target=self._target,
            library_path=Path(lib_path_str),
            system=self._system,
            sim=self._sim,
        )
        self._worker.stage_progress.connect(self._on_stage_progress)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_done)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._btn_fit.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._worker.start()

    def _stop_fitting(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    def _on_stage_progress(self, stage: str, frac: float) -> None:
        self._lbl_stage.setText(f"Stage: {stage}")
        self._progress_bar.setValue(int(frac * 100))

    def _on_result(self, result: object) -> None:
        self._result_text.setPlainText(result.summary())
        self.fit_completed.emit(result)

    def _on_error(self, msg: str) -> None:
        self._result_text.setPlainText(f"Error: {msg}")

    def _on_worker_done(self) -> None:
        self._progress_bar.setVisible(False)
        self._btn_fit.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
