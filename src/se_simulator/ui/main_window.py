"""MainWindow: three-panel main application window for SE-RCWA Simulator."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from se_simulator.config.manager import ConfigManager
from se_simulator.config.schemas import SampleConfig, SimConditions, SystemConfig

try:
    from se_simulator.rcwa.dispatcher import select_engine as _select_engine

    _HAS_DISPATCHER = True
except ImportError:
    _HAS_DISPATCHER = False

# ------------------------------------------------------------------
# Tree node indices (must match order they are added)
# ------------------------------------------------------------------
_NODE_SYSTEM = 0
_NODE_SAMPLE = 1
_NODE_SIM = 2
_NODE_LIBRARY = 3
_NODE_RESULTS = 4
_NODE_FITTING = 5

# Central stack page indices
_PAGE_SYSTEM = 0
_PAGE_SAMPLE = 1
_PAGE_SIM = 2
_PAGE_LIBRARY = 3
_PAGE_RESULTS = 4
_PAGE_FITTING = 5


class MainWindow(QMainWindow):
    """
    Three-panel main window:

    ┌─────────────────────────────────────────────────────────┐
    │  Menu Bar                                               │
    ├───────────┬─────────────────────────────┬───────────────┤
    │           │                             │               │
    │  Project  │     Center Editor Area      │  Properties   │
    │   Tree    │   (QStackedWidget)          │  Inspector    │
    │  (240px)  │                             │   (280px)     │
    ├───────────┴─────────────────────────────┴───────────────┤
    │  Status Bar             [Progress Bar]  [Cancel Button] │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        project_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config_manager = config_manager or ConfigManager()
        self._project_dir = project_dir

        # Current config state
        self._system_config: SystemConfig | None = None
        self._sample_config: SampleConfig | None = None
        self._sim_conditions: SimConditions | None = None

        # Active simulation worker
        self._sim_worker: object | None = None
        self._last_result: object | None = None

        self.setWindowTitle("SE-RCWA Simulator v1.0")
        self.resize(1400, 900)

        self._build_menu()
        self._build_central_widget()
        self._build_status_bar()

        # Load configs if project dir provided
        if self._project_dir is not None:
            self._load_project_configs()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("New Project", self._new_project)
        file_menu.addAction("Open Project…", self._open_project)
        file_menu.addAction("Save Project", self._save_project, QKeySequence("Ctrl+S"))
        file_menu.addAction("Save As…", self._save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Import Sample Config…", self._import_sample)
        file_menu.addAction("Export Sample Config…", self._export_sample)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

        # Simulation menu
        sim_menu = menu_bar.addMenu("&Simulation")
        run_action = sim_menu.addAction("Run Simulation", self._run_simulation)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        sim_menu.addAction("Stop", self._stop_simulation)
        sim_menu.addSeparator()
        sim_menu.addAction("Run Parametric Sweep…", self._run_parametric_sweep)
        sim_menu.addSeparator()
        sim_menu.addAction("Generate Library…", self._open_library_manager)
        sim_menu.addSeparator()
        sim_menu.addAction("Preferences…", self._open_preferences)

        # Fitting menu
        fit_menu = menu_bar.addMenu("F&itting")
        fit_menu.addAction("Load Library…", self._load_library)
        fit_action = fit_menu.addAction("Run Fit", self._run_fit)
        fit_action.setShortcut(QKeySequence("Ctrl+F"))
        fit_menu.addSeparator()
        fit_menu.addAction("Fitting Settings…", self._open_fitting_settings)

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction("Convergence Test…", self._open_convergence)
        tools_menu.addAction("Calibration Dashboard", self._open_calibration)
        tools_menu.addSeparator()
        tools_menu.addAction("Material Editor", self._open_material_editor)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction("Documentation", self._open_docs)
        help_menu.addSeparator()
        help_menu.addAction("About", self._show_about)

    # ------------------------------------------------------------------
    # Central widget (splitter with 3 panels)
    # ------------------------------------------------------------------

    def _build_central_widget(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # --- Left panel: project tree ---
        self._project_tree = QTreeWidget()
        self._project_tree.setHeaderLabel("Project")
        self._project_tree.setMinimumWidth(200)
        self._project_tree.setMaximumWidth(300)

        # Add six top-level nodes
        node_labels = [
            "System Config",
            "Sample Structure",
            "Simulation Conditions",
            "Library",
            "Results",
            "Fitting",
        ]
        self._tree_nodes: list[QTreeWidgetItem] = []
        for label in node_labels:
            item = QTreeWidgetItem(self._project_tree, [label])
            self._tree_nodes.append(item)

        self._project_tree.expandAll()
        self._project_tree.itemClicked.connect(self._on_tree_item_clicked)

        splitter.addWidget(self._project_tree)

        # --- Center panel: stacked editor ---
        self._center_stack = QStackedWidget()

        # Page 0: SystemConfigEditor
        from se_simulator.ui.widgets.system_config_editor import SystemConfigEditor

        self._system_editor = SystemConfigEditor()
        self._system_editor.config_changed.connect(self._on_system_config_changed)
        self._center_stack.addWidget(self._system_editor)  # index 0

        # Page 1: StructureEditor
        from se_simulator.ui.widgets.structure_editor import StructureEditor

        self._structure_editor = StructureEditor()
        self._structure_editor.sample_changed.connect(self._on_sample_changed)
        self._center_stack.addWidget(self._structure_editor)  # index 1

        # Page 2: SimulationPanel
        from se_simulator.ui.widgets.sim_panel import SimulationPanel

        self._sim_panel = SimulationPanel()
        self._sim_panel.run_requested.connect(self._run_simulation)
        self._sim_panel.stop_requested.connect(self._stop_simulation)
        self._center_stack.addWidget(self._sim_panel)  # index 2

        # Page 3: LibraryManager
        from se_simulator.ui.widgets.library_manager import LibraryManager

        self._library_manager = LibraryManager()
        self._library_manager.generation_finished.connect(self._on_library_generated)
        self._center_stack.addWidget(self._library_manager)  # index 3

        # Page 4: ResultsViewer
        from se_simulator.ui.widgets.results_viewer import ResultsViewer

        self._results_viewer = ResultsViewer()
        self._results_viewer.export_requested.connect(self._on_export_csv)
        self._center_stack.addWidget(self._results_viewer)  # index 4

        # Page 5: FittingWorkspace
        from se_simulator.ui.widgets.fitting_workspace import FittingWorkspace

        self._fitting_workspace = FittingWorkspace(
            system=self._system_config,
            sim=self._sim_conditions,
        )
        self._fitting_workspace.fit_completed.connect(self._on_fit_completed)
        self._center_stack.addWidget(self._fitting_workspace)  # index 5

        splitter.addWidget(self._center_stack)

        # --- Right panel: properties inspector ---
        props_panel = QFrame()
        props_panel.setFrameShape(QFrame.Shape.StyledPanel)
        props_panel.setMinimumWidth(200)
        props_panel.setMaximumWidth(320)

        props_layout = QHBoxLayout(props_panel)
        props_layout.setContentsMargins(4, 4, 4, 4)

        # Stacked: page 0 = placeholder label, page 1 = layer inspector
        from se_simulator.ui.widgets.structure_editor import PropertiesInspectorWidget

        self._props_stack = QStackedWidget()

        self._props_label = QLabel("Select an item in the project tree.")
        self._props_label.setWordWrap(True)
        self._props_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._props_stack.addWidget(self._props_label)  # page 0

        self._layer_inspector = PropertiesInspectorWidget()
        self._props_stack.addWidget(self._layer_inspector)  # page 1

        props_layout.addWidget(self._props_stack)
        splitter.addWidget(props_panel)

        # Track which layer/shape index is being edited
        self._selected_layer_idx: int = -1
        self._selected_shape_idx: int = -1

        # Wire inspector changes back into the structure editor
        self._layer_inspector.layer_changed.connect(self._on_inspector_layer_changed)
        self._layer_inspector.shape_changed.connect(self._on_inspector_shape_changed)

        # Wire structure editor layer/shape selection to inspector
        self._structure_editor.layer_selected.connect(self._on_layer_selected_in_editor)
        self._structure_editor.shape_selected.connect(self._on_shape_selected_in_editor)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 0)

        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        self._status_label = QLabel("Ready")
        status_bar.addWidget(self._status_label, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        status_bar.addPermanentWidget(self._progress_bar)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setVisible(False)
        self._btn_cancel.clicked.connect(self._stop_simulation)
        status_bar.addPermanentWidget(self._btn_cancel)

        self._engine_label = QLabel("")
        status_bar.addPermanentWidget(self._engine_label)

    # ------------------------------------------------------------------
    # Project tree interaction
    # ------------------------------------------------------------------

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        top_items = [self._project_tree.topLevelItem(i) for i in range(6)]
        if item in top_items:
            idx = top_items.index(item)
            self._center_stack.setCurrentIndex(idx)
            self._update_props_label(idx)
        else:
            # Sub-item under Results
            self._center_stack.setCurrentIndex(_PAGE_RESULTS)

    def _update_props_label(self, page_idx: int) -> None:
        descriptions = {
            _PAGE_SYSTEM: "Edit instrument parameters: optical angles, compensator retardance, and calibration errors.",
            _PAGE_SAMPLE: "Select a layer in the stack to edit its properties.",
            _PAGE_SIM: "Configure simulation conditions: wavelengths, harmonics, AOI, and run the simulation.",
            _PAGE_LIBRARY: "Generate a pre-computed spectral library for fast fitting.",
            _PAGE_RESULTS: "View simulation and fitting results.",
            _PAGE_FITTING: "Load a pre-computed library and fit the simulated or measured spectrum.",
        }
        self._props_label.setText(descriptions.get(page_idx, ""))
        # When switching away from the sample page, reset inspector
        if page_idx != _PAGE_SAMPLE:
            self._selected_layer_idx = -1
            self._layer_inspector.clear()
            self._props_stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_project_configs(self) -> None:
        """Load all three config files from the project directory."""
        if self._project_dir is None:
            return

        system_path = self._project_dir / "system_config.yaml"
        sample_path = self._project_dir / "sample_config.yaml"
        sim_path = self._project_dir / "sim_conditions.yaml"

        try:
            if system_path.exists():
                self._system_config = self._config_manager.load_system(system_path)
                self._system_editor.load_config(self._system_config)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Warning: could not load system_config: {exc}")

        try:
            if sample_path.exists():
                self._sample_config = self._config_manager.load_sample(sample_path)
                self._structure_editor.load_sample(self._sample_config)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Warning: could not load sample_config: {exc}")

        try:
            if sim_path.exists():
                self._sim_conditions = self._config_manager.load_sim_conditions(sim_path)
                self._sim_panel.load_sim(self._sim_conditions)
                self._library_manager.set_configs(
                    self._sample_config or _make_default_sample(),
                    self._system_config or _make_default_system(),
                    self._sim_conditions,
                )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Warning: could not load sim_conditions: {exc}")

        self._fitting_workspace.set_configs(
            self._system_config or _make_default_system(),
            self._sim_conditions or _make_default_sim(),
        )
        self._set_status("Project loaded.")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _run_simulation(self) -> None:
        if self._sim_worker is not None:
            return

        sample = self._sample_config or _make_default_sample()
        system = self._system_config or _make_default_system()
        sim = self._sim_panel.build_sim()

        from se_simulator.ui.workers.simulation_worker import SimulationWorker

        self._sim_worker = SimulationWorker(sample, sim, system)
        self._sim_worker.progress.connect(self._on_sim_progress)
        self._sim_worker.result_ready.connect(self._on_sim_result)
        self._sim_worker.error_occurred.connect(self._on_sim_error)
        self._sim_worker.finished.connect(self._on_sim_finished)

        # Determine engine and update status bar
        override = getattr(sim, "engine_override", "auto")
        if _HAS_DISPATCHER:
            _engine = _select_engine(sample, override)
        else:
            _engine = "rcwa" if any(getattr(lr, "shapes", []) for lr in sample.layers) else "tmm"
        self._set_engine_label(_engine)
        self._sim_panel.update_engine_indicator(sample, sim)

        self._sim_panel.set_running(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._btn_cancel.setVisible(True)
        self._set_status("Running simulation…")

        self._sim_worker.start()

    def _stop_simulation(self) -> None:
        if self._sim_worker is not None:
            self._sim_worker.requestInterruption()
            self._set_status("Cancelling…")

    def _on_sim_progress(self, frac: float) -> None:
        self._progress_bar.setValue(int(frac * 100))

    def _on_sim_result(self, result: object) -> None:
        self._last_result = result
        self._results_viewer.load_result(result, label="Simulated")
        self._fitting_workspace.set_target(result)

        # Switch to results page and add sub-item to tree
        self._center_stack.setCurrentIndex(_PAGE_RESULTS)
        results_node = self._tree_nodes[_NODE_RESULTS]

        run_label = f"Run {results_node.childCount() + 1}"
        QTreeWidgetItem(results_node, [run_label])
        results_node.setExpanded(True)

        self._set_status("Simulation complete.")

    def _on_sim_error(self, msg: str) -> None:
        self._set_status(f"Simulation error: {msg}")
        QMessageBox.critical(self, "Simulation Error", msg)

    def _on_sim_finished(self) -> None:
        self._sim_panel.set_running(False)
        self._progress_bar.setVisible(False)
        self._btn_cancel.setVisible(False)
        if self._sim_worker is not None:
            self._sim_worker.deleteLater()
            self._sim_worker = None

    def _on_fit_completed(self, result: object) -> None:
        self._set_status(f"Fit complete — χ²={result.final_chi2:.4f}")
        # Show fit result spectrum in ResultsViewer
        if result.best_fit_spectrum is not None:
            self._results_viewer.load_result(result.best_fit_spectrum, label="Best Fit")
            self._center_stack.setCurrentIndex(_PAGE_RESULTS)

    # ------------------------------------------------------------------
    # Config change handlers
    # ------------------------------------------------------------------

    def _on_layer_selected_in_editor(self, idx: int) -> None:
        """Called when the user selects a layer row in the LayerStackWidget."""
        if self._sample_config is None:
            return
        if 0 <= idx < len(self._sample_config.layers):
            self._selected_layer_idx = idx
            self._layer_inspector.load_layer(
                self._sample_config.layers[idx], self._sample_config
            )
            self._props_stack.setCurrentIndex(1)  # show inspector
        else:
            self._selected_layer_idx = -1
            self._layer_inspector.clear()
            self._props_stack.setCurrentIndex(0)

    def _on_shape_selected_in_editor(self, layer_idx: int, shape_idx: int) -> None:
        """Called when the user clicks a shape on the 2D canvas."""
        if self._sample_config is None:
            return
        self._selected_shape_idx = shape_idx
        if shape_idx < 0:
            # Background clicked — fall back to showing the layer inspector
            if 0 <= layer_idx < len(self._sample_config.layers):
                self._selected_layer_idx = layer_idx
                self._layer_inspector.load_layer(
                    self._sample_config.layers[layer_idx], self._sample_config
                )
                self._props_stack.setCurrentIndex(1)
            else:
                self._layer_inspector.clear()
                self._props_stack.setCurrentIndex(0)
            return

        if (
            0 <= layer_idx < len(self._sample_config.layers)
            and 0 <= shape_idx < len(self._sample_config.layers[layer_idx].shapes)
        ):
            self._selected_layer_idx = layer_idx
            layer = self._sample_config.layers[layer_idx]
            shape = layer.shapes[shape_idx]
            self._layer_inspector.load_shape(shape, self._sample_config, layer=layer)
            self._props_stack.setCurrentIndex(1)

    def _on_inspector_layer_changed(self, updated_layer: object) -> None:
        """Called when the Properties Inspector edits a layer field."""
        if self._selected_layer_idx < 0 or self._sample_config is None:
            return
        self._structure_editor.apply_layer_edit(self._selected_layer_idx, updated_layer)
        # Keep sample_config in sync (apply_layer_edit mutates layers in-place)
        self._sample_config = self._structure_editor._sample  # type: ignore[assignment]

    def _on_inspector_shape_changed(self, updated_shape: object) -> None:
        """Called when the Properties Inspector edits a shape field."""
        if (
            self._selected_layer_idx < 0
            or self._selected_shape_idx < 0
            or self._sample_config is None
        ):
            return
        self._structure_editor.apply_shape_edit(
            self._selected_layer_idx, self._selected_shape_idx, updated_shape
        )
        self._sample_config = self._structure_editor._sample  # type: ignore[assignment]

    def _on_system_config_changed(self, config: SystemConfig) -> None:
        self._system_config = config
        self._fitting_workspace.set_configs(config, self._sim_conditions)
        self._set_status("System config updated.")

    def _set_engine_label(self, engine: str) -> None:
        """Update the status bar engine label."""
        self._engine_label.setText(f"Engine: {engine.upper()}")

    def _on_sample_changed(self, sample: SampleConfig) -> None:
        self._sample_config = sample
        self._set_status("Sample structure updated.")
        self._sim_panel.update_engine_indicator(sample, self._sim_conditions)

    def _on_sim_conditions_changed(self, sim: SimConditions) -> None:
        self._sim_conditions = sim
        if self._sample_config is not None:
            self._sim_panel.update_engine_indicator(self._sample_config, sim)
        self._fitting_workspace.set_configs(self._system_config, sim)

    def _on_library_generated(self, path: Path) -> None:
        self._set_status(f"Library generated: {path}")

    def _on_export_csv(self, path: Path) -> None:
        self._set_status(f"Results exported to {path}")

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _new_project(self) -> None:
        self._set_status("New project (not yet implemented).")

    def _open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Project Directory", "")
        if path:
            self._project_dir = Path(path)
            self._load_project_configs()

    def _save_project(self) -> None:
        if self._project_dir is None:
            self._save_project_as()
            return
        self._do_save(self._project_dir)

    def _save_project_as(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Save Project To…", "")
        if path:
            self._project_dir = Path(path)
            self._do_save(self._project_dir)

    def _do_save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if self._system_config is not None:
                self._config_manager.save_system(
                    self._system_config, directory / "system_config.yaml"
                )
            if self._sample_config is not None:
                self._config_manager.save_sample(
                    self._sample_config, directory / "sample_config.yaml"
                )
            if self._sim_conditions is not None:
                self._config_manager.save_sim_conditions(
                    self._sim_conditions, directory / "sim_conditions.yaml"
                )
            self._set_status(f"Project saved to {directory}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", str(exc))

    def _import_sample(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Sample Config", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            try:
                self._sample_config = self._config_manager.load_sample(Path(path))
                self._structure_editor.load_sample(self._sample_config)
                self._set_status(f"Sample imported from {path}")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Import Error", str(exc))

    def _export_sample(self) -> None:
        if self._sample_config is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Sample Config", "", "YAML Files (*.yaml);;All Files (*)"
        )
        if path:
            self._config_manager.save_sample(self._sample_config, Path(path))
            self._set_status(f"Sample exported to {path}")

    def _run_fit(self) -> None:
        # Pass latest result as fitting target if available
        if self._last_result is not None:
            self._fitting_workspace.set_target(self._last_result)
        self._center_stack.setCurrentIndex(_PAGE_FITTING)
        self._set_status("Fitting workspace ready.")

    def _load_library(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Library", "", "HDF5 Files (*.h5);;All Files (*)"
        )
        if path:
            self._set_status(f"Library loaded: {path}")

    def _open_fitting_settings(self) -> None:
        self._set_status("Fitting settings (not yet implemented).")

    def _run_parametric_sweep(self) -> None:
        self._set_status("Parametric sweep (not yet implemented).")

    def _open_library_manager(self) -> None:
        self._center_stack.setCurrentIndex(_PAGE_LIBRARY)

    def _open_preferences(self) -> None:
        self._set_status("Preferences (not yet implemented).")

    def _open_convergence(self) -> None:
        from se_simulator.ui.dialogs.convergence_dialog import ConvergenceDialog

        dlg = ConvergenceDialog(self._sample_config, self._sim_conditions, self)
        dlg.exec()

    def _open_calibration(self) -> None:
        from PySide6.QtWidgets import QDialog, QVBoxLayout

        from se_simulator.ui.widgets.calibration_dashboard import CalibrationDashboard

        dlg = QDialog(self)
        dlg.setWindowTitle("Calibration Dashboard")
        dlg.setMinimumWidth(400)
        dlg_layout = QVBoxLayout(dlg)
        dashboard = CalibrationDashboard()
        if self._system_config is not None:
            dashboard.load_config(self._system_config)
        dlg_layout.addWidget(dashboard)
        dlg.exec()

    def _open_material_editor(self) -> None:
        from PySide6.QtWidgets import QDialog, QVBoxLayout

        from se_simulator.ui.widgets.material_editor import MaterialEditor

        dlg = QDialog(self)
        dlg.setWindowTitle("Material Editor")
        dlg.setMinimumWidth(500)
        dlg.setMinimumHeight(600)
        dlg_layout = QVBoxLayout(dlg)
        editor = MaterialEditor()
        dlg_layout.addWidget(editor)
        dlg.exec()

    def _open_docs(self) -> None:
        import webbrowser

        webbrowser.open("https://github.com/se-rcwa-simulator")

    def _show_about(self) -> None:
        from se_simulator.ui.dialogs.about_dialog import AboutDialog

        dlg = AboutDialog(self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        """Update the status bar text."""
        self._status_label.setText(message)


# ------------------------------------------------------------------
# Default config factories (used when no project is loaded)
# ------------------------------------------------------------------


def _make_default_sample() -> SampleConfig:
    """Return a minimal default SampleConfig."""
    from se_simulator.config.schemas import GratingLayer, MaterialSpec

    return SampleConfig(
        Lx_nm=500.0,
        Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Si",
        layers=[
            GratingLayer(
                name="Layer1",
                type="uniform",
                thickness_nm=100.0,
                Lx_nm=500.0,
                Ly_nm=500.0,
                background_material="Air",
            )
        ],
        materials={
            "Air": MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0),
            "Si": MaterialSpec(name="Si", source="library", library_name="Si"),
        },
    )


def _make_default_system() -> SystemConfig:
    """Return a default SystemConfig."""
    from se_simulator.config.schemas import CompensatorRetardanceModel

    return SystemConfig(
        instrument_name="Default Instrument",
        polarizer_angle_deg=45.0,
        analyzer_angle_deg=45.0,
        compensator_angle_deg=0.0,
        compensator_retardance=CompensatorRetardanceModel(type="constant", value=90.0),
    )


def _make_default_sim() -> SimConditions:
    """Return a default SimConditions."""
    from se_simulator.config.schemas import WavelengthSpec

    return SimConditions(
        aoi_deg=65.0,
        azimuth_deg=0.0,
        wavelengths=WavelengthSpec(range=(400.0, 700.0, 50.0)),
        n_harmonics_x=3,
        n_harmonics_y=3,
        parallel_wavelengths=False,
    )
