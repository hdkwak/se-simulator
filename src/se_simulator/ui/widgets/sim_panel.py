"""SimulationPanel: editor for SimConditions and Run/Stop controls."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import SimConditions

try:
    from se_simulator.rcwa.dispatcher import select_engine as _select_engine

    _HAS_DISPATCHER = True
except ImportError:
    _HAS_DISPATCHER = False


class SimulationPanel(QWidget):
    """Panel for configuring and launching RCWA simulations.

    Signals
    -------
    run_requested():
        Emitted when the user clicks Run.
    stop_requested():
        Emitted when the user clicks Stop.
    settings_changed(SimConditions):
        Emitted when any setting is changed by the user.
    recipe_loaded(object, Path):
        Emitted when a simulation recipe is successfully loaded.
    """

    run_requested = Signal()
    stop_requested = Signal()
    settings_changed = Signal(object)
    recipe_loaded = Signal(object, object)  # (SimulationRecipe, Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sim: SimConditions | None = None
        self._recipe: object | None = None
        self._recipe_path: Path | None = None
        self._recipe_modified: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Recipe section (collapsible)
        recipe_group = QGroupBox("Recipe")
        recipe_group.setCheckable(True)
        recipe_group.setChecked(False)
        recipe_layout = QVBoxLayout(recipe_group)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("File:"))
        self._recipe_path_edit = QLineEdit()
        self._recipe_path_edit.setReadOnly(True)
        self._recipe_path_edit.setPlaceholderText("None")
        path_row.addWidget(self._recipe_path_edit)
        recipe_layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        self._btn_load_recipe = QPushButton("Load")
        self._btn_clear_recipe = QPushButton("Clear")
        self._btn_save_back = QPushButton("Save back to Recipe")
        self._btn_save_back.setEnabled(False)
        btn_row.addWidget(self._btn_load_recipe)
        btn_row.addWidget(self._btn_clear_recipe)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_save_back)
        recipe_layout.addLayout(btn_row)
        layout.addWidget(recipe_group)

        # Run controls
        run_group = QGroupBox("Run Controls")
        run_layout = QHBoxLayout(run_group)
        self._btn_run = QPushButton("RUN Simulation")
        self._btn_run.setObjectName("run_button")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #2ecc71; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #27ae60; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._engine_indicator = QLabel("Engine: \u2014 ")
        run_layout.addWidget(self._engine_indicator)
        run_layout.addStretch()
        run_layout.addWidget(self._btn_run)
        run_layout.addWidget(self._btn_stop)
        layout.addWidget(run_group)

        # Simulation conditions
        cond_group = QGroupBox("Simulation Conditions")
        form = QFormLayout(cond_group)

        self._aoi_spin = QDoubleSpinBox()
        self._aoi_spin.setRange(0.0, 89.9)
        self._aoi_spin.setDecimals(2)
        self._aoi_spin.setSuffix(" °")
        self._aoi_spin.setValue(65.0)
        form.addRow("Angle of Incidence:", self._aoi_spin)

        self._az_spin = QDoubleSpinBox()
        self._az_spin.setRange(0.0, 360.0)
        self._az_spin.setDecimals(2)
        self._az_spin.setSuffix(" °")
        self._az_spin.setValue(0.0)
        form.addRow("Azimuth Angle:", self._az_spin)

        self._nx_spin = QSpinBox()
        self._nx_spin.setRange(1, 20)
        self._nx_spin.setValue(5)
        form.addRow("Harmonics X:", self._nx_spin)

        self._ny_spin = QSpinBox()
        self._ny_spin.setRange(1, 20)
        self._ny_spin.setValue(5)
        form.addRow("Harmonics Y:", self._ny_spin)

        layout.addWidget(cond_group)

        # Wavelength range
        wl_group = QGroupBox("Wavelength Range")
        wl_form = QFormLayout(wl_group)

        self._wl_start = QDoubleSpinBox()
        self._wl_start.setRange(100.0, 10000.0)
        self._wl_start.setDecimals(1)
        self._wl_start.setSuffix(" nm")
        self._wl_start.setValue(300.0)
        wl_form.addRow("Start:", self._wl_start)

        self._wl_stop = QDoubleSpinBox()
        self._wl_stop.setRange(100.0, 10000.0)
        self._wl_stop.setDecimals(1)
        self._wl_stop.setSuffix(" nm")
        self._wl_stop.setValue(800.0)
        wl_form.addRow("Stop:", self._wl_stop)

        self._wl_step = QDoubleSpinBox()
        self._wl_step.setRange(0.1, 100.0)
        self._wl_step.setDecimals(1)
        self._wl_step.setSuffix(" nm")
        self._wl_step.setValue(10.0)
        wl_form.addRow("Step:", self._wl_step)

        layout.addWidget(wl_group)

        # Advanced options (collapsed by default)
        adv_group = QGroupBox("Advanced")
        adv_group.setCheckable(True)
        adv_group.setChecked(False)
        adv_form = QFormLayout(adv_group)

        self._chk_parallel = QCheckBox("Parallel wavelengths")
        self._chk_parallel.setChecked(True)
        adv_form.addRow(self._chk_parallel)

        self._chk_li = QCheckBox("Li factorization")
        self._chk_li.setChecked(True)
        adv_form.addRow(self._chk_li)

        layout.addWidget(adv_group)
        layout.addStretch()

        # Connect signals
        self._btn_run.clicked.connect(self.run_requested)
        self._btn_stop.clicked.connect(self.stop_requested)
        self._btn_load_recipe.clicked.connect(self._on_load_recipe_clicked)
        self._btn_clear_recipe.clicked.connect(self._on_clear_recipe)
        self._btn_save_back.clicked.connect(self._on_save_back)

        # Track modifications after recipe load
        self._aoi_spin.valueChanged.connect(self._mark_modified)
        self._az_spin.valueChanged.connect(self._mark_modified)
        self._nx_spin.valueChanged.connect(self._mark_modified)
        self._ny_spin.valueChanged.connect(self._mark_modified)
        self._wl_start.valueChanged.connect(self._mark_modified)
        self._wl_stop.valueChanged.connect(self._mark_modified)
        self._wl_step.valueChanged.connect(self._mark_modified)

    # ------------------------------------------------------------------
    # Recipe support
    # ------------------------------------------------------------------

    def load_recipe(self, recipe: object, path: Path) -> None:
        """Populate UI from a SimulationRecipe and record the path."""
        from se_simulator.recipe.manager import RecipeManager

        manager = RecipeManager()
        try:
            _sample, sim = manager.decompose_simulation(recipe, recipe_path=path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Recipe Error", str(exc))
            return

        self._recipe = recipe
        self._recipe_path = path
        self._recipe_modified = False

        # Block signals while populating to avoid marking as modified
        self._set_blocking(True)
        try:
            self.load_sim(sim)
        finally:
            self._set_blocking(False)

        self._recipe_path_edit.setText(str(path))
        self._btn_save_back.setEnabled(True)
        self.recipe_loaded.emit(recipe, path)

    def _set_blocking(self, block: bool) -> None:
        for widget in (
            self._aoi_spin,
            self._az_spin,
            self._nx_spin,
            self._ny_spin,
            self._wl_start,
            self._wl_stop,
            self._wl_step,
        ):
            widget.blockSignals(block)

    def _mark_modified(self) -> None:
        if self._recipe_path is not None and not self._recipe_modified:
            self._recipe_modified = True
            current = self._recipe_path_edit.text()
            if not current.endswith("*"):
                self._recipe_path_edit.setText(current + "*")

    def _on_load_recipe_clicked(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Simulation Recipe", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if not path:
            return
        from se_simulator.recipe.manager import RecipeManager, RecipeValidationError

        manager = RecipeManager()
        try:
            recipe = manager.load_simulation_recipe(Path(path))
        except RecipeValidationError as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self.load_recipe(recipe, Path(path))

    def _on_clear_recipe(self) -> None:
        self._recipe = None
        self._recipe_path = None
        self._recipe_modified = False
        self._recipe_path_edit.setText("")
        self._btn_save_back.setEnabled(False)

    def _on_save_back(self) -> None:
        if self._recipe is None or self._recipe_path is None:
            return
        from se_simulator.config.schemas import DataCollectionConfig
        from se_simulator.recipe.manager import RecipeManager

        sim = self.build_sim()
        wl = sim.wavelengths.range
        if wl is None:
            return

        # Get existing data_collection as base (for angle fields we don't control here)
        existing_dc = getattr(self._recipe, "data_collection", DataCollectionConfig())
        new_dc = existing_dc.model_copy(update={
            "wavelength_start_nm": wl[0],
            "wavelength_end_nm": wl[1],
            "wavelength_step_nm": wl[2],
            "aoi_deg": sim.aoi_deg,
            "azimuth_deg": sim.azimuth_deg,
        })

        recipe = self._recipe
        if hasattr(recipe, "model_copy"):
            recipe = recipe.model_copy(update={"data_collection": new_dc})

        manager = RecipeManager()
        try:
            manager.save_simulation_recipe(recipe, self._recipe_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", str(exc))
            return

        self._recipe = recipe
        self._recipe_modified = False
        self._recipe_path_edit.setText(str(self._recipe_path))

    # ------------------------------------------------------------------
    # Existing API
    # ------------------------------------------------------------------

    def load_sim(self, sim: SimConditions) -> None:
        """Populate controls from a SimConditions object."""
        self._sim = sim
        self._aoi_spin.setValue(sim.aoi_deg)
        self._az_spin.setValue(sim.azimuth_deg)
        self._nx_spin.setValue(sim.n_harmonics_x)
        self._ny_spin.setValue(sim.n_harmonics_y)
        if sim.wavelengths.range is not None:
            start, stop, step = sim.wavelengths.range
            self._wl_start.setValue(start)
            self._wl_stop.setValue(stop)
            self._wl_step.setValue(step)
        self._chk_parallel.setChecked(sim.parallel_wavelengths)
        self._chk_li.setChecked(sim.li_factorization)

    def build_sim(self) -> SimConditions:
        """Build a SimConditions from current UI values."""
        from se_simulator.config.schemas import WavelengthSpec

        wl_spec = WavelengthSpec(
            range=(
                self._wl_start.value(),
                self._wl_stop.value(),
                self._wl_step.value(),
            )
        )
        base = self._sim or SimConditions(
            aoi_deg=65.0,
            azimuth_deg=0.0,
            wavelengths=wl_spec,
        )
        return base.model_copy(
            update={
                "aoi_deg": self._aoi_spin.value(),
                "azimuth_deg": self._az_spin.value(),
                "n_harmonics_x": self._nx_spin.value(),
                "n_harmonics_y": self._ny_spin.value(),
                "wavelengths": wl_spec,
                "parallel_wavelengths": self._chk_parallel.isChecked(),
                "li_factorization": self._chk_li.isChecked(),
            }
        )

    def set_running(self, running: bool) -> None:
        """Toggle the run/stop button states."""
        self._btn_run.setEnabled(not running)
        self._btn_stop.setEnabled(running)

    def update_engine_indicator(
        self,
        sample_config: object,
        sim_conditions: object | None = None,
    ) -> None:
        """Update the engine indicator label based on sample and sim conditions."""
        override = getattr(sim_conditions, "engine_override", "auto")
        if _HAS_DISPATCHER:
            engine = _select_engine(sample_config, override)
        else:
            has_shapes = any(
                getattr(layer, "shapes", [])
                for layer in getattr(sample_config, "layers", [])
            )
            engine = "rcwa" if has_shapes else "tmm"
        qualifier = "(override)" if override != "auto" else "(auto)"
        self._engine_indicator.setText(f"Engine: {engine.upper()} {qualifier}")
