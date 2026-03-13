"""LibraryManager: widget for configuring and generating spectral libraries."""

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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import SampleConfig, SimConditions, SystemConfig
from se_simulator.fitting.library import LibrarySpec, ParameterSpec


class LibraryManager(QWidget):
    """Widget for defining library parameters and triggering generation.

    Signals
    -------
    generation_started():
        Emitted when the user starts library generation.
    generation_finished(Path):
        Emitted when generation completes; carries the output HDF5 path.
    """

    generation_started = Signal()
    generation_finished = Signal(object)

    def __init__(
        self,
        sample: SampleConfig | None = None,
        system: SystemConfig | None = None,
        sim: SimConditions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample = sample
        self._system = system
        self._sim = sim
        self._worker: object | None = None
        self._output_path: Path | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Output file
        out_group = QGroupBox("Output File")
        out_layout = QHBoxLayout(out_group)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("library.h5")
        self._btn_browse = QPushButton("Browse…")
        out_layout.addWidget(QLabel("Path:"))
        out_layout.addWidget(self._path_edit)
        out_layout.addWidget(self._btn_browse)
        layout.addWidget(out_group)

        # Parameter grid
        param_group = QGroupBox("Parameter Grid")
        param_layout = QVBoxLayout(param_group)

        param_toolbar = QHBoxLayout()
        self._btn_add_param = QPushButton("Add Parameter")
        self._btn_remove_param = QPushButton("Remove")
        param_toolbar.addWidget(self._btn_add_param)
        param_toolbar.addWidget(self._btn_remove_param)
        param_toolbar.addStretch()
        param_layout.addLayout(param_toolbar)

        self._param_table = QTableWidget()
        self._param_table.setColumnCount(4)
        self._param_table.setHorizontalHeaderLabels(["Path", "Min", "Max", "N Points"])
        param_layout.addWidget(self._param_table)

        layout.addWidget(param_group)

        # Generation controls
        gen_group = QGroupBox("Generate")
        gen_layout = QVBoxLayout(gen_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        gen_layout.addWidget(self._progress_bar)

        self._lbl_status = QLabel("Ready")
        gen_layout.addWidget(self._lbl_status)

        btn_row = QHBoxLayout()
        self._btn_generate = QPushButton("Generate Library")
        self._btn_generate.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; }"
        )
        self._btn_stop_gen = QPushButton("Stop")
        self._btn_stop_gen.setEnabled(False)
        btn_row.addWidget(self._btn_generate)
        btn_row.addWidget(self._btn_stop_gen)
        btn_row.addStretch()
        gen_layout.addLayout(btn_row)
        layout.addWidget(gen_group)
        layout.addStretch()

        # Connections
        self._btn_browse.clicked.connect(self._browse_output)
        self._btn_add_param.clicked.connect(self._add_param_row)
        self._btn_remove_param.clicked.connect(self._remove_param_row)
        self._btn_generate.clicked.connect(self._start_generation)
        self._btn_stop_gen.clicked.connect(self._stop_generation)

    def set_configs(
        self,
        sample: SampleConfig,
        system: SystemConfig,
        sim: SimConditions,
    ) -> None:
        """Update the configs used for generation."""
        self._sample = sample
        self._system = system
        self._sim = sim

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Library Output", "", "HDF5 Files (*.h5);;All Files (*)"
        )
        if path:
            self._path_edit.setText(path)

    def _add_param_row(self) -> None:
        row = self._param_table.rowCount()
        self._param_table.insertRow(row)
        self._param_table.setItem(row, 0, QTableWidgetItem("layers[0].thickness_nm"))
        self._param_table.setItem(row, 1, QTableWidgetItem("50"))
        self._param_table.setItem(row, 2, QTableWidgetItem("200"))
        self._param_table.setItem(row, 3, QTableWidgetItem("10"))

    def _remove_param_row(self) -> None:
        row = self._param_table.currentRow()
        if row >= 0:
            self._param_table.removeRow(row)

    def _build_spec(self) -> LibrarySpec | None:
        if self._sample is None or self._system is None or self._sim is None:
            return None

        params: list[ParameterSpec] = []
        for row in range(self._param_table.rowCount()):
            try:
                path = (self._param_table.item(row, 0) or QTableWidgetItem("")).text().strip()
                min_v = float((self._param_table.item(row, 1) or QTableWidgetItem("0")).text())
                max_v = float((self._param_table.item(row, 2) or QTableWidgetItem("1")).text())
                n_pts = int((self._param_table.item(row, 3) or QTableWidgetItem("5")).text())
                params.append(ParameterSpec(name=path, min_value=min_v, max_value=max_v, n_points=n_pts))
            except (ValueError, AttributeError):
                continue

        if not params:
            return None

        from se_simulator.fitting.library import build_library_spec

        return build_library_spec(params, self._sample, self._system, self._sim)

    def _start_generation(self) -> None:
        if self._sample is None or self._system is None or self._sim is None:
            self._lbl_status.setText("Error: no configuration loaded")
            return

        path_str = self._path_edit.text().strip()
        if not path_str:
            self._lbl_status.setText("Error: specify output path")
            return

        spec = self._build_spec()
        if spec is None:
            self._lbl_status.setText("Error: define at least one parameter")
            return

        self._output_path = Path(path_str)

        from se_simulator.ui.workers.library_worker import LibraryWorker

        self._worker = LibraryWorker(spec, self._output_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_done)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._btn_generate.setEnabled(False)
        self._btn_stop_gen.setEnabled(True)
        self._lbl_status.setText("Generating…")
        self.generation_started.emit()
        self._worker.start()

    def _stop_generation(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setValue(int(100 * done / total))
        self._lbl_status.setText(f"{done} / {total} entries")

    def _on_finished_ok(self) -> None:
        if self._output_path is not None:
            self.generation_finished.emit(self._output_path)

    def _on_error(self, msg: str) -> None:
        self._lbl_status.setText(f"Error: {msg}")

    def _on_worker_done(self) -> None:
        self._progress_bar.setVisible(False)
        self._btn_generate.setEnabled(True)
        self._btn_stop_gen.setEnabled(False)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
