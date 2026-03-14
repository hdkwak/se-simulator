"""FittingWorkspace: widget for loading a library and running the fitting pipeline."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import SimConditions, SystemConfig
from se_simulator.ellipsometer.signals import EllipsometryResult

# Fitting mode display → internal string
_FITTING_MODE_MAP = {
    "Auto": "auto",
    "Library": "library",
    "TMM Direct": "tmm_direct",
}
_FITTING_MODE_REVERSE = {v: k for k, v in _FITTING_MODE_MAP.items()}

# Floating params table columns
_COL_NAME = 0
_COL_FIELD = 1
_COL_MIN = 2
_COL_MAX = 3
_COL_INITIAL = 4
_COL_STEP = 5
_COL_UNITS = 6


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
        self._recipe: object | None = None
        self._recipe_path: Path | None = None
        self._fit_result: object | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Recipe section
        recipe_group = QGroupBox("Recipe")
        recipe_layout = QVBoxLayout(recipe_group)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("File:"))
        self._recipe_path_edit = QLineEdit()
        self._recipe_path_edit.setReadOnly(True)
        self._recipe_path_edit.setPlaceholderText("None")
        path_row.addWidget(self._recipe_path_edit)
        recipe_layout.addLayout(path_row)

        recipe_btn_row = QHBoxLayout()
        self._btn_load_recipe = QPushButton("Load")
        self._btn_clear_recipe = QPushButton("Clear")
        recipe_btn_row.addWidget(self._btn_load_recipe)
        recipe_btn_row.addWidget(self._btn_clear_recipe)
        recipe_btn_row.addStretch()
        recipe_layout.addLayout(recipe_btn_row)
        layout.addWidget(recipe_group)

        # Floating Parameters table
        params_group = QGroupBox("Floating Parameters")
        params_layout = QVBoxLayout(params_group)
        self._params_table = QTableWidget(0, 7)
        self._params_table.setHorizontalHeaderLabels(
            ["Name", "Field", "Min", "Max", "Initial", "Step", "Units"]
        )
        self._params_table.horizontalHeader().setSectionResizeMode(
            _COL_FIELD, QHeaderView.ResizeMode.Stretch
        )
        params_layout.addWidget(self._params_table)

        params_btn_row = QHBoxLayout()
        self._btn_add_param = QPushButton("+")
        self._btn_remove_param = QPushButton("−")
        self._btn_edit_in_editor = QPushButton("Edit in Recipe Editor")
        params_btn_row.addWidget(self._btn_add_param)
        params_btn_row.addWidget(self._btn_remove_param)
        params_btn_row.addStretch()
        params_btn_row.addWidget(self._btn_edit_in_editor)
        params_layout.addLayout(params_btn_row)
        layout.addWidget(params_group)

        # Fitting Mode
        mode_group = QGroupBox("Fitting Mode")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.addWidget(QLabel("Mode:"))
        self._fitting_mode_combo = QComboBox()
        self._fitting_mode_combo.addItems(list(_FITTING_MODE_MAP.keys()))
        mode_layout.addWidget(self._fitting_mode_combo)
        self._tmm_info_label = QLabel("(Library lookup disabled for TMM Direct)")
        self._tmm_info_label.setVisible(False)
        mode_layout.addWidget(self._tmm_info_label)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

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
        target_layout = QVBoxLayout(target_group)
        target_path_row = QHBoxLayout()
        self._target_path_edit = QLineEdit()
        self._target_path_edit.setPlaceholderText("measured.csv")
        self._btn_browse_target = QPushButton("Browse…")
        target_path_row.addWidget(QLabel("Path:"))
        target_path_row.addWidget(self._target_path_edit)
        target_path_row.addWidget(self._btn_browse_target)
        target_layout.addLayout(target_path_row)

        self._wl_warning_label = QLabel()
        self._wl_warning_label.setVisible(False)
        target_layout.addWidget(self._wl_warning_label)
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
        self._btn_save_results = QPushButton("Save Results to Recipe")
        self._btn_save_results.setEnabled(False)
        btn_row.addWidget(self._btn_fit)
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_save_results)
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
        self._btn_load_recipe.clicked.connect(self._on_load_recipe_clicked)
        self._btn_clear_recipe.clicked.connect(self._on_clear_recipe)
        self._btn_add_param.clicked.connect(self._add_param_row)
        self._btn_remove_param.clicked.connect(self._remove_param_row)
        self._btn_edit_in_editor.clicked.connect(self._open_recipe_editor)
        self._btn_save_results.clicked.connect(self._save_results_to_recipe)
        self._fitting_mode_combo.currentTextChanged.connect(self._on_fitting_mode_changed)

    # ------------------------------------------------------------------
    # Recipe support
    # ------------------------------------------------------------------

    def load_measurement_recipe(self, recipe: object, path: Path) -> None:
        """Populate UI from a MeasurementRecipe."""
        from se_simulator.recipe.manager import RecipeManager

        manager = RecipeManager()
        try:
            _sample, sim, system, floating_params, fitting_config = (
                manager.decompose_measurement(recipe, recipe_path=path)
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Recipe Error", str(exc))
            return

        self._recipe = recipe
        self._recipe_path = path
        self._fit_result = None
        self._btn_save_results.setEnabled(False)

        self._recipe_path_edit.setText(str(path))
        self.set_configs(system, sim)

        # Populate fitting mode
        fitting_mode = getattr(fitting_config, "fitting_mode", "auto")
        display = _FITTING_MODE_REVERSE.get(fitting_mode, "Auto")
        idx = self._fitting_mode_combo.findText(display)
        if idx >= 0:
            self._fitting_mode_combo.setCurrentIndex(idx)

        # Populate library path
        lib_file = ""
        if hasattr(recipe, "library_reference"):
            lib_file = getattr(recipe.library_reference, "library_file", "") or ""
        self._lib_path_edit.setText(lib_file)

        # Populate floating parameters
        self._populate_params_table(floating_params)

    def _populate_params_table(self, params: list) -> None:
        from PySide6.QtCore import Qt

        self._params_table.setRowCount(0)
        for fp in params:
            row = self._params_table.rowCount()
            self._params_table.insertRow(row)
            self._params_table.setItem(row, _COL_NAME, QTableWidgetItem(fp.name))
            field_item = QTableWidgetItem(fp.target_field)
            # Make field column read-only by removing ItemIsEditable flag
            flags = field_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            field_item.setFlags(flags)
            self._params_table.setItem(row, _COL_FIELD, field_item)
            self._params_table.setItem(row, _COL_MIN, QTableWidgetItem(str(fp.min)))
            self._params_table.setItem(row, _COL_MAX, QTableWidgetItem(str(fp.max)))
            self._params_table.setItem(row, _COL_INITIAL, QTableWidgetItem(str(fp.initial)))
            self._params_table.setItem(row, _COL_STEP, QTableWidgetItem(str(fp.step)))
            self._params_table.setItem(row, _COL_UNITS, QTableWidgetItem(fp.units))

    def _on_load_recipe_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Measurement Recipe", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if not path:
            return
        from ruamel.yaml import YAML as _YAML

        from se_simulator.recipe.manager import RecipeManager, RecipeValidationError

        # Check recipe_type before attempting Pydantic validation so we can
        # show a clear, actionable message instead of a raw Pydantic error.
        try:
            _yaml = _YAML()
            with open(path) as _fh:
                _raw = _yaml.load(_fh) or {}
            detected_type = (_raw.get("metadata") or {}).get("recipe_type", "")
        except Exception:  # noqa: BLE001
            detected_type = ""

        if detected_type == "simulation":
            QMessageBox.critical(
                self,
                "Wrong Recipe Type",
                "This file is a Simulation Recipe, not a Measurement Recipe.\n\n"
                "Use 'Load Simulation Recipe' instead, or open it in the Recipe Editor "
                "to convert it.",
            )
            return

        manager = RecipeManager()
        try:
            recipe = manager.load_measurement_recipe(Path(path))
        except RecipeValidationError as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self.load_measurement_recipe(recipe, Path(path))

    def _on_clear_recipe(self) -> None:
        self._recipe = None
        self._recipe_path = None
        self._fit_result = None
        self._recipe_path_edit.setText("")
        self._params_table.setRowCount(0)
        self._btn_save_results.setEnabled(False)

    def _add_param_row(self) -> None:
        row = self._params_table.rowCount()
        self._params_table.insertRow(row)
        for col in range(7):
            self._params_table.setItem(row, col, QTableWidgetItem(""))

    def _remove_param_row(self) -> None:
        current = self._params_table.currentRow()
        if current >= 0:
            self._params_table.removeRow(current)

    def _open_recipe_editor(self) -> None:
        from se_simulator.ui.recipe_editor import RecipeEditorDialog

        dlg = RecipeEditorDialog(recipe=self._recipe, path=self._recipe_path, parent=self)
        dlg.exec()

    def _on_fitting_mode_changed(self, text: str) -> None:
        is_tmm = _FITTING_MODE_MAP.get(text) == "tmm_direct"
        self._lib_path_edit.setEnabled(not is_tmm)
        self._btn_browse_lib.setEnabled(not is_tmm)
        self._tmm_info_label.setVisible(is_tmm)

    def _save_results_to_recipe(self) -> None:
        if self._fit_result is None or self._recipe_path is None:
            return
        from se_simulator.recipe.manager import RecipeManager

        manager = RecipeManager()
        try:
            manager.append_results(self._fit_result, self._recipe_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._btn_save_results.setEnabled(False)

    def _validate_target_wavelengths(self, path: Path) -> None:
        """Show yellow warning or green OK label based on wavelength coverage."""
        self._wl_warning_label.setVisible(False)
        try:
            import csv

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows or "wavelength_nm" not in rows[0]:
                return
            wls = [float(r["wavelength_nm"]) for r in rows if r.get("wavelength_nm")]
            if not wls:
                return
            wl_min, wl_max = min(wls), max(wls)

            if self._sim is not None and self._sim.wavelengths.range is not None:
                start, stop, _step = self._sim.wavelengths.range
                if wl_min > start or wl_max < stop:
                    self._wl_warning_label.setText(
                        f"Warning: target wavelengths [{wl_min:.0f}, {wl_max:.0f}] nm "
                        f"do not cover sim range [{start:.0f}, {stop:.0f}] nm"
                    )
                    self._wl_warning_label.setStyleSheet("color: #b8860b; background: #fffacd; "
                                                         "padding: 2px;")
                else:
                    self._wl_warning_label.setText(
                        f"Wavelength coverage OK: [{wl_min:.0f}, {wl_max:.0f}] nm"
                    )
                    self._wl_warning_label.setStyleSheet("color: #006400; background: #e8f5e9; "
                                                         "padding: 2px;")
                self._wl_warning_label.setVisible(True)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Existing API
    # ------------------------------------------------------------------

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
            self._validate_target_wavelengths(Path(path))

    def _start_fitting(self) -> None:
        target_path_str = self._target_path_edit.text().strip()

        # --- Recipe-based path (Step 2 pipeline) ---
        if self._recipe is not None:
            if not target_path_str:
                self._result_text.setPlainText("Error: load target spectrum first")
                return

            fitting_mode = _FITTING_MODE_MAP.get(self._fitting_mode_combo.currentText(), "auto")
            lib_path_str = self._lib_path_edit.text().strip()

            # Library mode requires a library file
            if fitting_mode == "library" and not lib_path_str:
                self._result_text.setPlainText("Error: Library fitting mode requires a library file")
                return

            from se_simulator.ui.workers.fitting_worker import RecipeFittingWorker

            self._worker = RecipeFittingWorker(
                recipe=self._recipe,
                target_spectrum_path=target_path_str,
                recipe_path=str(self._recipe_path) if self._recipe_path else "",
            )
            self._worker.result_ready.connect(self._on_result)
            self._worker.error_occurred.connect(self._on_error)
            self._worker.finished.connect(self._on_worker_done)
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(0)
            self._btn_fit.setEnabled(False)
            self._btn_stop.setEnabled(True)
            self._btn_save_results.setEnabled(False)
            self._worker.start()
            return

        # --- Legacy path (no recipe loaded) ---
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
        self._btn_save_results.setEnabled(False)
        self._worker.start()

    def _stop_fitting(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    def _on_stage_progress(self, stage: str, frac: float) -> None:
        self._lbl_stage.setText(f"Stage: {stage}")
        self._progress_bar.setValue(int(frac * 100))

    def _on_result(self, result: object) -> None:
        # Handle both legacy FitResult (has .summary()) and FitResults Pydantic model
        if hasattr(result, "summary") and callable(result.summary):
            text = result.summary()
        elif hasattr(result, "fitted_parameters"):
            # FitResults Pydantic model from recipe pipeline
            lines = ["=== Fit Results ===", f"Engine: {result.engine_used}"]
            lines.append("\nFitted Parameters:")
            for name, val in result.fitted_parameters.items():
                lines.append(f"  {name}: {val:.4g}")
            lines.append("\nFit Quality:")
            for key, val in result.fit_quality.items():
                lines.append(f"  {key}: {val:.4g}")
            lines.append(f"\nTimestamp: {result.timestamp}")
            lines.append("\nBest-fit spectrum saved to: fit.csv")
            text = "\n".join(lines)
        else:
            text = str(result)
        self._result_text.setPlainText(text)
        self._fit_result = result
        if self._recipe_path is not None:
            self._btn_save_results.setEnabled(True)
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
