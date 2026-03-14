"""RecipeEditorDialog: dual-pane recipe editor with live YAML preview."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


def _recipe_to_yaml_str(recipe: object) -> str:
    """Serialise a recipe object to a YAML string."""
    try:
        from io import StringIO

        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.default_flow_style = False
        buf = StringIO()
        data = recipe.model_dump() if hasattr(recipe, "model_dump") else {}
        yaml.dump(data, buf)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return ""


def _load_yaml_raw(path: Path) -> dict[str, Any]:
    from ruamel.yaml import YAML

    yaml = YAML()
    with open(path) as fh:
        data = yaml.load(fh)
    if data is None:
        return {}

    def _to_plain(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_plain(v) for v in obj]
        return obj

    return _to_plain(data)


# Column indices for the layer table
_LAYER_COL_TYPE = 0
_LAYER_COL_THICKNESS = 1
_LAYER_COL_MATERIAL = 2
_LAYER_COL_NAME = 3

# Column indices for the floating parameters table
_FP_COL_NAME = 0
_FP_COL_FIELD = 1
_FP_COL_MIN = 2
_FP_COL_MAX = 3
_FP_COL_INITIAL = 4
_FP_COL_STEP = 5
_FP_COL_UNITS = 6


class RecipeEditorDialog(QDialog):
    """Dual-pane recipe editor dialog.

    Left pane (60%): scrollable form editor.
    Right pane (40%): live YAML preview (read-only), updates with 300ms debounce.
    """

    def __init__(
        self,
        recipe: object | None = None,
        path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recipe Editor")
        self.resize(1100, 700)

        self._recipe: object | None = recipe
        self._path: Path | None = path
        self._last_saved_yaml: str = ""
        self._show_diff: bool = False
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._update_preview)

        self._setup_ui()

        if recipe is not None:
            self._populate_from_recipe(recipe)
        if path is not None:
            self._last_saved_yaml = _recipe_to_yaml_str(recipe) if recipe else ""

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QToolBar()
        self._btn_load = QPushButton("Load")
        self._btn_save = QPushButton("Save")
        self._btn_save_as = QPushButton("Save As")
        self._btn_validate = QPushButton("Validate")
        self._btn_export_sim = QPushButton("Export as Simulation Recipe")
        self._btn_toggle_diff = QPushButton("Show Diff")
        self._btn_toggle_diff.setCheckable(True)
        for btn in (
            self._btn_load,
            self._btn_save,
            self._btn_save_as,
            self._btn_validate,
            self._btn_export_sim,
            self._btn_toggle_diff,
        ):
            toolbar.addWidget(btn)
        main_layout.addWidget(toolbar)

        # Validation banner
        self._validation_label = QLabel()
        self._validation_label.setVisible(False)
        self._validation_label.setWordWrap(True)
        main_layout.addWidget(self._validation_label)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: form editor (scrollable)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_container = QWidget()
        self._form_layout = QVBoxLayout(form_container)
        self._form_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._build_form()

        scroll.setWidget(form_container)
        left_layout.addWidget(scroll)

        # Recent recipes (collapsible)
        recent_group = QGroupBox("Recent Recipes")
        recent_group.setCheckable(True)
        recent_group.setChecked(False)
        recent_layout = QVBoxLayout(recent_group)
        self._recent_list = QListWidget()
        self._recent_list.setMaximumHeight(120)
        recent_layout.addWidget(self._recent_list)
        left_layout.addWidget(recent_group)
        self._load_recent_list()

        splitter.addWidget(left_widget)
        splitter.setStretchFactor(0, 3)

        # Right: YAML preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("YAML Preview:"))
        self._yaml_preview = QPlainTextEdit()
        self._yaml_preview.setReadOnly(True)
        self._yaml_preview.setObjectName("yaml_preview")
        right_layout.addWidget(self._yaml_preview)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        # Wire signals
        self._btn_load.clicked.connect(self._on_load)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_save_as.clicked.connect(self._on_save_as)
        self._btn_validate.clicked.connect(self._on_validate)
        self._btn_export_sim.clicked.connect(self._on_export_sim)
        self._btn_toggle_diff.toggled.connect(self._on_toggle_diff)
        self._recent_list.itemDoubleClicked.connect(self._on_recent_double_clicked)

    def _build_form(self) -> None:
        """Build the form sections."""
        # Metadata section
        meta_group = QGroupBox("Metadata")
        meta_layout = QVBoxLayout(meta_group)

        self._meta_type_label = QLabel("Type: —")
        meta_layout.addWidget(self._meta_type_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("Author:"))
        self._meta_author = QLineEdit()
        self._meta_author.textChanged.connect(self._schedule_preview_update)
        row.addWidget(self._meta_author)
        meta_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Description:"))
        self._meta_description = QTextEdit()
        self._meta_description.setMaximumHeight(60)
        self._meta_description.textChanged.connect(self._schedule_preview_update)
        row2.addWidget(self._meta_description)
        meta_layout.addLayout(row2)

        self._form_layout.addWidget(meta_group)

        # Simulation Conditions section
        cond_group = QGroupBox("Simulation Conditions")
        cond_layout = QVBoxLayout(cond_group)

        for label, attr in (
            ("WL Start (nm):", "_cond_wl_start"),
            ("WL End (nm):", "_cond_wl_end"),
            ("WL Step (nm):", "_cond_wl_step"),
            ("AOI (deg):", "_cond_aoi"),
            ("Azimuth (deg):", "_cond_azimuth"),
            ("Polarizer (deg):", "_cond_polarizer"),
            ("Analyzer (deg):", "_cond_analyzer"),
        ):
            row_w = QHBoxLayout()
            row_w.addWidget(QLabel(label))
            edit = QLineEdit()
            edit.textChanged.connect(self._schedule_preview_update)
            setattr(self, attr, edit)
            row_w.addWidget(edit)
            cond_layout.addLayout(row_w)

        self._form_layout.addWidget(cond_group)

        # Sample section — editable layer stack
        self._build_sample_section()

        # Output options section
        out_group = QGroupBox("Output Options")
        out_layout = QVBoxLayout(out_group)
        out_layout.addWidget(QLabel("(Controlled by recipe YAML)"))
        self._form_layout.addWidget(out_group)

        # MeasurementRecipe-only sections
        self._build_floating_params_section()
        self._build_fitting_config_section()
        self._build_library_ref_section()

        # Initially hide measurement-only sections
        self._floating_params_group.setVisible(False)
        self._fitting_config_group.setVisible(False)
        self._library_ref_group.setVisible(False)

    def _build_sample_section(self) -> None:
        """Build the editable sample stack section."""
        sample_group = QGroupBox("Sample")
        sample_layout = QVBoxLayout(sample_group)

        # Superstrate row
        supra_row = QHBoxLayout()
        supra_row.addWidget(QLabel("Superstrate:"))
        self._sample_superstrate = QLineEdit()
        self._sample_superstrate.setPlaceholderText("Air")
        self._sample_superstrate.textChanged.connect(self._schedule_preview_update)
        supra_row.addWidget(self._sample_superstrate)
        sample_layout.addLayout(supra_row)

        # Layer table
        sample_layout.addWidget(QLabel("Layers (top to bottom):"))
        self._layer_table = QTableWidget(0, 4)
        self._layer_table.setHorizontalHeaderLabels(["Type", "Thickness (nm)", "Material", "Name"])
        self._layer_table.horizontalHeader().setSectionResizeMode(
            _LAYER_COL_MATERIAL, QHeaderView.ResizeMode.Stretch
        )
        self._layer_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._layer_table.setMaximumHeight(180)
        self._layer_table.itemChanged.connect(self._schedule_preview_update)
        sample_layout.addWidget(self._layer_table)

        # Layer buttons
        layer_btn_row = QHBoxLayout()
        self._btn_add_layer = QPushButton("+ Add Layer")
        self._btn_remove_layer = QPushButton("- Remove Layer")
        layer_btn_row.addWidget(self._btn_add_layer)
        layer_btn_row.addWidget(self._btn_remove_layer)
        layer_btn_row.addStretch()
        sample_layout.addLayout(layer_btn_row)

        # Substrate row
        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("Substrate:"))
        self._sample_substrate = QLineEdit()
        self._sample_substrate.setPlaceholderText("Si")
        self._sample_substrate.textChanged.connect(self._schedule_preview_update)
        sub_row.addWidget(self._sample_substrate)
        sample_layout.addLayout(sub_row)

        self._form_layout.addWidget(sample_group)

        # Connect buttons
        self._btn_add_layer.clicked.connect(self._add_layer_row)
        self._btn_remove_layer.clicked.connect(self._remove_layer_row)

    def _build_floating_params_section(self) -> None:
        """Build the Floating Parameters section (MeasurementRecipe only)."""
        self._floating_params_group = QGroupBox("Floating Parameters")
        fp_layout = QVBoxLayout(self._floating_params_group)

        self._fp_table = QTableWidget(0, 7)
        self._fp_table.setHorizontalHeaderLabels(
            ["Name", "Target Field", "Min", "Max", "Initial", "Step", "Units"]
        )
        self._fp_table.horizontalHeader().setSectionResizeMode(
            _FP_COL_FIELD, QHeaderView.ResizeMode.Stretch
        )
        self._fp_table.setMinimumHeight(120)
        self._fp_table.itemChanged.connect(self._schedule_preview_update)
        fp_layout.addWidget(self._fp_table)

        fp_btn_row = QHBoxLayout()
        self._btn_add_fp = QPushButton("+ Add")
        self._btn_remove_fp = QPushButton("- Remove")
        fp_btn_row.addWidget(self._btn_add_fp)
        fp_btn_row.addWidget(self._btn_remove_fp)
        fp_btn_row.addStretch()
        fp_layout.addLayout(fp_btn_row)

        self._form_layout.addWidget(self._floating_params_group)

        self._btn_add_fp.clicked.connect(self._add_fp_row)
        self._btn_remove_fp.clicked.connect(self._remove_fp_row)

    def _build_fitting_config_section(self) -> None:
        """Build the Fitting Configuration section (MeasurementRecipe only)."""
        self._fitting_config_group = QGroupBox("Fitting Configuration")
        fc_layout = QVBoxLayout(self._fitting_config_group)

        # Fitting Mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Fitting Mode:"))
        self._fc_fitting_mode = QComboBox()
        self._fc_fitting_mode.addItems(["auto", "library", "tmm_direct"])
        self._fc_fitting_mode.currentTextChanged.connect(self._on_fitting_mode_changed)
        self._fc_fitting_mode.currentTextChanged.connect(self._schedule_preview_update)
        mode_row.addWidget(self._fc_fitting_mode)
        mode_row.addStretch()
        fc_layout.addLayout(mode_row)

        # Fit Signals checkboxes
        signals_row = QHBoxLayout()
        signals_row.addWidget(QLabel("Fit Signals:"))
        self._fc_chk_psi = QCheckBox("psi")
        self._fc_chk_delta = QCheckBox("delta")
        self._fc_chk_mueller = QCheckBox("mueller_elements")
        for chk in (self._fc_chk_psi, self._fc_chk_delta, self._fc_chk_mueller):
            chk.stateChanged.connect(self._schedule_preview_update)
            signals_row.addWidget(chk)
        signals_row.addStretch()
        fc_layout.addLayout(signals_row)

        # Weights
        weights_row = QHBoxLayout()
        weights_row.addWidget(QLabel("Weights:"))
        self._fc_weights = QComboBox()
        self._fc_weights.addItems(["uniform", "snr", "custom"])
        self._fc_weights.currentTextChanged.connect(self._schedule_preview_update)
        weights_row.addWidget(self._fc_weights)
        weights_row.addStretch()
        fc_layout.addLayout(weights_row)

        # Optimizer
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Optimizer:"))
        self._fc_optimizer = QComboBox()
        self._fc_optimizer.addItems(
            ["levenberg_marquardt", "nelder_mead", "differential_evolution"]
        )
        self._fc_optimizer.currentTextChanged.connect(self._schedule_preview_update)
        opt_row.addWidget(self._fc_optimizer)
        opt_row.addStretch()
        fc_layout.addLayout(opt_row)

        # Max Iterations
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max Iterations:"))
        self._fc_max_iter = QSpinBox()
        self._fc_max_iter.setRange(1, 10000)
        self._fc_max_iter.setValue(200)
        self._fc_max_iter.valueChanged.connect(self._schedule_preview_update)
        iter_row.addWidget(self._fc_max_iter)
        iter_row.addStretch()
        fc_layout.addLayout(iter_row)

        # Convergence Tolerance
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Convergence Tolerance:"))
        self._fc_conv_tol = QLineEdit("1e-6")
        self._fc_conv_tol.textChanged.connect(self._schedule_preview_update)
        tol_row.addWidget(self._fc_conv_tol)
        fc_layout.addLayout(tol_row)

        # Gradient Step
        grad_row = QHBoxLayout()
        grad_row.addWidget(QLabel("Gradient Step:"))
        self._fc_grad_step = QLineEdit("1e-4")
        self._fc_grad_step.textChanged.connect(self._schedule_preview_update)
        grad_row.addWidget(self._fc_grad_step)
        fc_layout.addLayout(grad_row)

        self._form_layout.addWidget(self._fitting_config_group)

    def _build_library_ref_section(self) -> None:
        """Build the Library Reference section (MeasurementRecipe only)."""
        self._library_ref_group = QGroupBox("Library Reference")
        lib_layout = QHBoxLayout(self._library_ref_group)

        lib_layout.addWidget(QLabel("Library file:"))
        self._lib_file_edit = QLineEdit()
        self._lib_file_edit.setReadOnly(True)
        self._lib_file_edit.setPlaceholderText("path/to/library.h5")
        self._lib_file_edit.textChanged.connect(self._schedule_preview_update)
        lib_layout.addWidget(self._lib_file_edit)

        self._btn_browse_lib = QPushButton("Browse…")
        self._btn_browse_lib.clicked.connect(self._browse_library_file)
        lib_layout.addWidget(self._btn_browse_lib)

        self._form_layout.addWidget(self._library_ref_group)

    # ------------------------------------------------------------------
    # Layer table helpers
    # ------------------------------------------------------------------

    def _add_layer_row(self) -> None:
        self._layer_table.blockSignals(True)
        row = self._layer_table.rowCount()
        self._layer_table.insertRow(row)
        self._layer_table.setItem(row, _LAYER_COL_TYPE, QTableWidgetItem("uniform"))
        self._layer_table.setItem(row, _LAYER_COL_THICKNESS, QTableWidgetItem("100.0"))
        self._layer_table.setItem(row, _LAYER_COL_MATERIAL, QTableWidgetItem("Air"))
        self._layer_table.setItem(row, _LAYER_COL_NAME, QTableWidgetItem(f"Layer {row + 1}"))
        self._layer_table.blockSignals(False)
        self._schedule_preview_update()

    def _remove_layer_row(self) -> None:
        row = self._layer_table.currentRow()
        if row >= 0:
            self._layer_table.removeRow(row)
            self._schedule_preview_update()

    def _populate_layer_table(self, layers: list[dict[str, Any]]) -> None:
        """Populate the layer table from a list of inline layer dicts.

        Inline layer format uses ``material: {library_name: X}`` for the
        material name.  Legacy dicts that use ``background_material: X`` are
        also accepted as a fallback.
        """
        self._layer_table.blockSignals(True)
        self._layer_table.setRowCount(0)
        for layer in layers:
            row = self._layer_table.rowCount()
            self._layer_table.insertRow(row)
            layer_type = layer.get("type", "uniform")
            thickness = layer.get("thickness_nm", 100.0)
            # Prefer the canonical inline format; fall back to legacy GratingLayer field
            mat_dict = layer.get("material") or {}
            material = mat_dict.get("library_name") or layer.get("background_material", "Air")
            name = layer.get("name", f"Layer {row + 1}")
            self._layer_table.setItem(row, _LAYER_COL_TYPE, QTableWidgetItem(str(layer_type)))
            self._layer_table.setItem(
                row, _LAYER_COL_THICKNESS, QTableWidgetItem(str(thickness))
            )
            self._layer_table.setItem(row, _LAYER_COL_MATERIAL, QTableWidgetItem(str(material)))
            self._layer_table.setItem(row, _LAYER_COL_NAME, QTableWidgetItem(str(name)))
        self._layer_table.blockSignals(False)

    def _populate_layer_table_from_stack(self, stack: Any) -> None:
        """Populate the layer table from a Stack object."""
        self._layer_table.blockSignals(True)
        self._layer_table.setRowCount(0)
        for layer in stack.layers:
            row = self._layer_table.rowCount()
            self._layer_table.insertRow(row)
            mat_name = (
                layer.material.library_name or layer.material.name or "Air"
            )
            self._layer_table.setItem(
                row, _LAYER_COL_TYPE, QTableWidgetItem(str(layer.type))
            )
            self._layer_table.setItem(
                row, _LAYER_COL_THICKNESS, QTableWidgetItem(str(layer.thickness_nm))
            )
            self._layer_table.setItem(
                row, _LAYER_COL_MATERIAL, QTableWidgetItem(mat_name)
            )
            self._layer_table.setItem(
                row, _LAYER_COL_NAME, QTableWidgetItem(str(layer.name))
            )
        self._layer_table.blockSignals(False)

    def _collect_layers_from_table(self) -> list[Any]:
        """Build a list of StackLayer objects from the layer table."""
        from se_simulator.config.schemas import MaterialSpec, StackLayer

        def _cell(tbl: QTableWidget, row: int, col: int) -> str:
            item = tbl.item(row, col)
            return item.text().strip() if item else ""

        layers = []
        for row in range(self._layer_table.rowCount()):
            layer_type = _cell(self._layer_table, row, _LAYER_COL_TYPE) or "uniform"
            thickness_str = _cell(self._layer_table, row, _LAYER_COL_THICKNESS)
            try:
                thickness = float(thickness_str)
            except ValueError:
                thickness = 100.0
            mat_name = _cell(self._layer_table, row, _LAYER_COL_MATERIAL) or "Air"
            name = _cell(self._layer_table, row, _LAYER_COL_NAME) or f"Layer {row + 1}"
            mat_spec = MaterialSpec(library_name=mat_name, source="library", name=mat_name)
            layers.append(
                StackLayer(
                    name=name,
                    type=layer_type,  # type: ignore[arg-type]
                    thickness_nm=thickness,
                    material=mat_spec,
                )
            )
        return layers

    def _build_stack_ref(self) -> Any:
        """Build a StackRef from current form fields."""
        from se_simulator.config.recipe import StackRef
        from se_simulator.config.schemas import MaterialSpec, Stack

        sup_name = self._sample_superstrate.text().strip() or "Air"
        sub_name = self._sample_substrate.text().strip() or "Si"
        sup_spec = MaterialSpec(library_name=sup_name, source="library", name=sup_name)
        sub_spec = MaterialSpec(library_name=sub_name, source="library", name=sub_name)
        layers = self._collect_layers_from_table()
        stack = Stack(superstrate=sup_spec, substrate=sub_spec, layers=layers)
        return StackRef(inline=stack)

    # ------------------------------------------------------------------
    # Floating parameters table helpers
    # ------------------------------------------------------------------

    def _add_fp_row(self) -> None:
        self._fp_table.blockSignals(True)
        row = self._fp_table.rowCount()
        self._fp_table.insertRow(row)
        for col in range(7):
            self._fp_table.setItem(row, col, QTableWidgetItem(""))
        self._fp_table.blockSignals(False)
        self._schedule_preview_update()

    def _remove_fp_row(self) -> None:
        row = self._fp_table.currentRow()
        if row >= 0:
            self._fp_table.removeRow(row)
            self._schedule_preview_update()

    def _populate_fp_table(self, floating_params: list[Any]) -> None:
        """Populate the floating parameters table from a list of FloatingParameter objects."""
        self._fp_table.blockSignals(True)
        self._fp_table.setRowCount(0)
        for fp in floating_params:
            row = self._fp_table.rowCount()
            self._fp_table.insertRow(row)
            self._fp_table.setItem(row, _FP_COL_NAME, QTableWidgetItem(str(fp.name)))
            self._fp_table.setItem(
                row, _FP_COL_FIELD, QTableWidgetItem(str(fp.target_field))
            )
            self._fp_table.setItem(row, _FP_COL_MIN, QTableWidgetItem(str(fp.min)))
            self._fp_table.setItem(row, _FP_COL_MAX, QTableWidgetItem(str(fp.max)))
            self._fp_table.setItem(row, _FP_COL_INITIAL, QTableWidgetItem(str(fp.initial)))
            self._fp_table.setItem(row, _FP_COL_STEP, QTableWidgetItem(str(fp.step)))
            self._fp_table.setItem(row, _FP_COL_UNITS, QTableWidgetItem(str(fp.units)))
        self._fp_table.blockSignals(False)

    def _collect_fp_from_table(self) -> list[dict[str, Any]]:
        """Build a list of FloatingParameter dicts from the FP table."""

        def _cell(tbl: QTableWidget, row: int, col: int) -> str:
            item = tbl.item(row, col)
            return item.text().strip() if item else ""

        def _float_cell(tbl: QTableWidget, row: int, col: int, fallback: float = 0.0) -> float:
            try:
                return float(_cell(tbl, row, col))
            except ValueError:
                return fallback

        params = []
        for row in range(self._fp_table.rowCount()):
            name = _cell(self._fp_table, row, _FP_COL_NAME)
            if not name:
                continue
            params.append(
                {
                    "name": name,
                    "target_field": _cell(self._fp_table, row, _FP_COL_FIELD),
                    "min": _float_cell(self._fp_table, row, _FP_COL_MIN),
                    "max": _float_cell(self._fp_table, row, _FP_COL_MAX),
                    "initial": _float_cell(self._fp_table, row, _FP_COL_INITIAL),
                    "step": _float_cell(self._fp_table, row, _FP_COL_STEP),
                    "units": _cell(self._fp_table, row, _FP_COL_UNITS),
                }
            )
        return params

    # ------------------------------------------------------------------
    # Fitting mode callback
    # ------------------------------------------------------------------

    def _on_fitting_mode_changed(self, mode: str) -> None:
        is_tmm = mode == "tmm_direct"
        self._library_ref_group.setEnabled(not is_tmm)

    # ------------------------------------------------------------------
    # Library file browse
    # ------------------------------------------------------------------

    def _browse_library_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Library File", "", "HDF5 Files (*.h5);;All Files (*)"
        )
        if path:
            self._lib_file_edit.setReadOnly(False)
            self._lib_file_edit.setText(path)
            self._lib_file_edit.setReadOnly(True)

    # ------------------------------------------------------------------
    # Population from recipe
    # ------------------------------------------------------------------

    def _populate_from_recipe(self, recipe: object) -> None:
        from se_simulator.config.recipe import MeasurementRecipe

        is_measurement = isinstance(recipe, MeasurementRecipe)

        # Show/hide measurement-only groups
        self._floating_params_group.setVisible(is_measurement)
        self._fitting_config_group.setVisible(is_measurement)
        self._library_ref_group.setVisible(is_measurement)

        meta = getattr(recipe, "metadata", None)
        if meta is not None:
            self._meta_type_label.setText(f"Type: {getattr(meta, 'recipe_type', '—')}")
            self._meta_author.setText(getattr(meta, "author", "") or "")
            self._meta_description.setPlainText(getattr(meta, "description", "") or "")

        # Simulation conditions
        sim_conds = None
        if hasattr(recipe, "simulation_conditions"):
            sim_conds = recipe.simulation_conditions
        elif hasattr(recipe, "forward_model") and hasattr(
            recipe.forward_model, "simulation_conditions"
        ):
            sim_conds = recipe.forward_model.simulation_conditions

        if sim_conds is not None:
            self._cond_wl_start.setText(str(getattr(sim_conds, "wavelength_start_nm", "")))
            self._cond_wl_end.setText(str(getattr(sim_conds, "wavelength_end_nm", "")))
            self._cond_wl_step.setText(str(getattr(sim_conds, "wavelength_step_nm", "")))
            self._cond_aoi.setText(str(getattr(sim_conds, "aoi_degrees", "")))
            self._cond_azimuth.setText(str(getattr(sim_conds, "azimuth_degrees", "")))
            self._cond_polarizer.setText(str(getattr(sim_conds, "polarizer_degrees", "")))
            self._cond_analyzer.setText(str(getattr(sim_conds, "analyzer_degrees", "")))

        # Sample — prefer the new stack field, fall back to legacy sample field
        stack_ref = None
        if hasattr(recipe, "stack") and recipe.stack is not None:
            stack_ref = recipe.stack
        elif hasattr(recipe, "forward_model") and hasattr(recipe.forward_model, "stack"):
            stack_ref = recipe.forward_model.stack

        # Legacy fallback: sample field (old SampleRef with inline dict)
        if stack_ref is None:
            sample_ref = None
            if hasattr(recipe, "sample"):
                sample_ref = recipe.sample
            elif hasattr(recipe, "forward_model") and hasattr(recipe.forward_model, "sample"):
                sample_ref = recipe.forward_model.sample
            if sample_ref is not None:
                from se_simulator.config.recipe import _sampleref_to_stackref
                stack_ref = _sampleref_to_stackref(sample_ref)

        if stack_ref is not None:
            inline_stack = getattr(stack_ref, "inline", None)
            if inline_stack is not None:
                # inline_stack is a Stack object
                sup_name = (
                    inline_stack.superstrate.library_name
                    or inline_stack.superstrate.name
                    or "Air"
                )
                sub_name = (
                    inline_stack.substrate.library_name
                    or inline_stack.substrate.name
                    or "Si"
                )
                self._sample_superstrate.setText(sup_name)
                self._sample_substrate.setText(sub_name)
                self._populate_layer_table_from_stack(inline_stack)
            else:
                # ref-based stack — clear the table
                self._layer_table.setRowCount(0)
                self._sample_superstrate.setText("")
                self._sample_substrate.setText("")

        # MeasurementRecipe-only fields
        if is_measurement:
            # Floating parameters
            self._populate_fp_table(getattr(recipe, "floating_parameters", []))

            # Fitting configuration
            fc = getattr(recipe, "fitting_configuration", None)
            if fc is not None:
                mode = getattr(fc, "fitting_mode", "auto")
                idx = self._fc_fitting_mode.findText(mode)
                if idx >= 0:
                    self._fc_fitting_mode.setCurrentIndex(idx)

                fit_signals = getattr(fc, "fit_signals", [])
                self._fc_chk_psi.setChecked("psi" in fit_signals)
                self._fc_chk_delta.setChecked("delta" in fit_signals)
                self._fc_chk_mueller.setChecked("mueller_elements" in fit_signals)

                weights = getattr(fc, "weights", "uniform")
                widx = self._fc_weights.findText(weights)
                if widx >= 0:
                    self._fc_weights.setCurrentIndex(widx)

                optimizer = getattr(fc, "optimizer", "levenberg_marquardt")
                oidx = self._fc_optimizer.findText(optimizer)
                if oidx >= 0:
                    self._fc_optimizer.setCurrentIndex(oidx)

                self._fc_max_iter.setValue(int(getattr(fc, "max_iterations", 200)))
                self._fc_conv_tol.setText(str(getattr(fc, "convergence_tolerance", "1e-6")))
                self._fc_grad_step.setText(str(getattr(fc, "gradient_step", "1e-4")))

            # Library reference
            lib_ref = getattr(recipe, "library_reference", None)
            if lib_ref is not None:
                lib_file = getattr(lib_ref, "library_file", "") or ""
                self._lib_file_edit.setReadOnly(False)
                self._lib_file_edit.setText(lib_file)
                self._lib_file_edit.setReadOnly(True)

            # Trigger fitting mode callback to set library ref enabled state
            self._on_fitting_mode_changed(self._fc_fitting_mode.currentText())

        self._update_preview()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _schedule_preview_update(self) -> None:
        self._debounce_timer.start()

    def _collect_recipe_from_form(self) -> object | None:
        """Return an updated copy of ``self._recipe`` with form field values applied.

        Returns ``None`` if no base recipe is loaded.
        """
        if self._recipe is None:
            return None

        from se_simulator.config.recipe import (
            FittingConfiguration,
            FloatingParameter,
            LibraryReference,
            MeasurementRecipe,
            RecipeMetadata,
            SampleRef,
            SimulationConditionsEmbed,
            SimulationRecipe,
        )

        def _float_or(text: str, fallback: float) -> float:
            try:
                return float(text.strip())
            except ValueError:
                return fallback

        # --- helpers to build updated sub-objects ---

        def _updated_metadata(base: RecipeMetadata) -> RecipeMetadata:
            return base.model_copy(
                update={
                    "author": self._meta_author.text().strip(),
                    "description": self._meta_description.toPlainText().strip(),
                }
            )

        def _updated_conditions(base: SimulationConditionsEmbed) -> SimulationConditionsEmbed:
            return base.model_copy(
                update={
                    "wavelength_start_nm": _float_or(
                        self._cond_wl_start.text(), base.wavelength_start_nm
                    ),
                    "wavelength_end_nm": _float_or(
                        self._cond_wl_end.text(), base.wavelength_end_nm
                    ),
                    "wavelength_step_nm": _float_or(
                        self._cond_wl_step.text(), base.wavelength_step_nm
                    ),
                    "aoi_degrees": _float_or(self._cond_aoi.text(), base.aoi_degrees),
                    "azimuth_degrees": _float_or(
                        self._cond_azimuth.text(), base.azimuth_degrees
                    ),
                    "polarizer_degrees": _float_or(
                        self._cond_polarizer.text(), base.polarizer_degrees
                    ),
                    "analyzer_degrees": _float_or(
                        self._cond_analyzer.text(), base.analyzer_degrees
                    ),
                }
            )

        def _updated_sample_ref(base: SampleRef) -> SampleRef:
            """Build an updated SampleRef from the layer table, if inline."""
            if base.inline is None:
                # ref-based — leave unchanged
                return base
            updated_inline = dict(base.inline)
            supra = self._sample_superstrate.text().strip() or "Air"
            sub = self._sample_substrate.text().strip() or "Si"
            # Superstrate and substrate are MaterialSpec dicts: {library_name: X}
            updated_inline["superstrate"] = {"library_name": supra}
            updated_inline["substrate"] = {"library_name": sub}
            # Remove legacy flat keys that were written at the wrong level
            updated_inline.pop("superstrate_material", None)
            updated_inline.pop("substrate_material", None)
            updated_inline["layers"] = self._collect_layers_from_table()
            return base.model_copy(update={"inline": updated_inline})

        def _updated_fitting_config(base: FittingConfiguration) -> FittingConfiguration:
            fit_signals = []
            if self._fc_chk_psi.isChecked():
                fit_signals.append("psi")
            if self._fc_chk_delta.isChecked():
                fit_signals.append("delta")
            if self._fc_chk_mueller.isChecked():
                fit_signals.append("mueller_elements")
            return base.model_copy(
                update={
                    "fitting_mode": self._fc_fitting_mode.currentText(),
                    "fit_signals": fit_signals,
                    "weights": self._fc_weights.currentText(),
                    "optimizer": self._fc_optimizer.currentText(),
                    "max_iterations": self._fc_max_iter.value(),
                    "convergence_tolerance": _float_or(
                        self._fc_conv_tol.text(), base.convergence_tolerance
                    ),
                    "gradient_step": _float_or(
                        self._fc_grad_step.text(), base.gradient_step
                    ),
                }
            )

        def _updated_library_ref(base: LibraryReference) -> LibraryReference:
            lib_file = self._lib_file_edit.text().strip()
            return base.model_copy(update={"library_file": lib_file})

        def _updated_floating_params() -> list[FloatingParameter]:
            import contextlib

            raw = self._collect_fp_from_table()
            result = []
            for item in raw:
                with contextlib.suppress(Exception):
                    result.append(FloatingParameter.model_validate(item))
            return result

        if isinstance(self._recipe, SimulationRecipe):
            return self._recipe.model_copy(
                update={
                    "metadata": _updated_metadata(self._recipe.metadata),
                    "simulation_conditions": _updated_conditions(
                        self._recipe.simulation_conditions
                    ),
                    "stack": self._build_stack_ref(),
                }
            )

        if isinstance(self._recipe, MeasurementRecipe):
            fm = self._recipe.forward_model
            updated_fm = fm.model_copy(
                update={
                    "simulation_conditions": _updated_conditions(fm.simulation_conditions),
                    "stack": self._build_stack_ref(),
                }
            )
            return self._recipe.model_copy(
                update={
                    "metadata": _updated_metadata(self._recipe.metadata),
                    "forward_model": updated_fm,
                    "floating_parameters": _updated_floating_params(),
                    "fitting_configuration": _updated_fitting_config(
                        self._recipe.fitting_configuration
                    ),
                    "library_reference": _updated_library_ref(
                        self._recipe.library_reference
                    ),
                }
            )

        # Unknown recipe type — return as-is
        return self._recipe

    def _update_preview(self) -> None:
        if self._recipe is None:
            self._yaml_preview.setPlainText("")
            return

        current_recipe = self._collect_recipe_from_form()
        yaml_text = _recipe_to_yaml_str(current_recipe)

        if self._show_diff:
            diff_lines = list(
                difflib.unified_diff(
                    self._last_saved_yaml.splitlines(keepends=True),
                    yaml_text.splitlines(keepends=True),
                    fromfile="saved",
                    tofile="current",
                )
            )
            self._yaml_preview.setPlainText("".join(diff_lines))
        else:
            self._yaml_preview.setPlainText(yaml_text)

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Recipe", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if not path:
            return
        self._load_from_path(Path(path))

    def _load_from_path(self, path: Path) -> None:
        from se_simulator.recipe.manager import RecipeManager, RecipeValidationError

        manager = RecipeManager()
        raw = _load_yaml_raw(path)
        recipe_type = raw.get("metadata", {}).get("recipe_type")
        try:
            if recipe_type == "simulation":
                recipe = manager.load_simulation_recipe(path)
            elif recipe_type == "measurement":
                recipe = manager.load_measurement_recipe(path)
            else:
                QMessageBox.critical(self, "Load Error", "Unknown recipe type in file.")
                return
        except RecipeValidationError as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._recipe = recipe
        self._path = path
        self._last_saved_yaml = _recipe_to_yaml_str(recipe)
        self._populate_from_recipe(recipe)
        self._load_recent_list()

    def _on_save(self) -> None:
        if self._path is None:
            self._on_save_as()
            return
        self._do_save(self._path)

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recipe As", "", "YAML Files (*.yaml);;All Files (*)"
        )
        if path:
            self._path = Path(path)
            self._do_save(self._path)

    def _do_save(self, path: Path) -> None:
        recipe = self._collect_recipe_from_form()
        if recipe is None:
            QMessageBox.critical(
                self,
                "Save Error",
                "No recipe is loaded. Use Load or create a new recipe before saving.",
            )
            return
        from se_simulator.recipe.manager import RecipeManager

        manager = RecipeManager()
        try:
            from se_simulator.config.recipe import MeasurementRecipe, SimulationRecipe

            if isinstance(recipe, MeasurementRecipe):
                manager.save_measurement_recipe(recipe, path)
            elif isinstance(recipe, SimulationRecipe):
                manager.save_simulation_recipe(recipe, path)
            else:
                QMessageBox.critical(self, "Save Error", "Unknown recipe type; cannot save.")
                return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", f"Failed to write file:\n{exc}")
            return
        # Keep in-memory recipe in sync with what was written
        self._recipe = recipe
        self._last_saved_yaml = _recipe_to_yaml_str(recipe)
        self._update_preview()
        QMessageBox.information(self, "Saved", f"Recipe saved to:\n{path}")

    def _on_validate(self) -> None:
        if self._path is None or not self._path.exists():
            self._show_validation_banner(["No file path set — save first."], valid=False)
            return
        from se_simulator.recipe.manager import RecipeManager

        manager = RecipeManager()
        errors = manager.validate(self._path)
        if errors:
            self._show_validation_banner(errors, valid=False)
        else:
            self._show_validation_banner([], valid=True)

    def _show_validation_banner(self, errors: list[str], valid: bool) -> None:
        self._validation_label.setVisible(True)
        if valid:
            self._validation_label.setText("Validation OK")
            self._validation_label.setStyleSheet(
                "background: #e8f5e9; color: #2e7d32; padding: 4px; border-radius: 3px;"
            )
        else:
            msg = "\n".join(f"• {e}" for e in errors) if errors else "Unknown error"
            self._validation_label.setText(f"Validation errors:\n{msg}")
            self._validation_label.setStyleSheet(
                "background: #ffebee; color: #c62828; padding: 4px; border-radius: 3px;"
            )

    def _on_export_sim(self) -> None:
        if self._recipe is None:
            return
        from se_simulator.config.recipe import MeasurementRecipe

        if not isinstance(self._recipe, MeasurementRecipe):
            QMessageBox.information(
                self, "Export", "Only MeasurementRecipe can be exported as SimulationRecipe."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export as Simulation Recipe", "", "YAML Files (*.yaml);;All Files (*)"
        )
        if not path:
            return

        from se_simulator.recipe.manager import RecipeManager

        manager = RecipeManager()
        try:
            sim_recipe = manager.export_as_simulation(self._recipe)
            manager.save_simulation_recipe(sim_recipe, Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        QMessageBox.information(self, "Export", f"Saved simulation recipe to:\n{path}")

    def _on_toggle_diff(self, checked: bool) -> None:
        self._show_diff = checked
        self._btn_toggle_diff.setText("Hide Diff" if checked else "Show Diff")
        self._update_preview()

    # ------------------------------------------------------------------
    # Recent recipes
    # ------------------------------------------------------------------

    def _load_recent_list(self) -> None:
        self._recent_list.clear()
        try:
            from se_simulator.recipe.manager import RecipeManager

            manager = RecipeManager()
            recent = manager.get_recent(10)
            for rpath, rtype in recent:
                self._recent_list.addItem(f"[{rtype}] {rpath}")
        except Exception:  # noqa: BLE001
            pass

    def _on_recent_double_clicked(self, item: object) -> None:
        text = item.text()
        # Format: "[type] /path/to/file"
        parts = text.split("] ", 1)
        if len(parts) == 2:
            path = Path(parts[1].strip())
            if path.exists():
                self._load_from_path(path)
