"""LayerStackWidget: editable table of grating layers."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import GratingLayer, MaterialSpec, SampleConfig, Stack, StackLayer
from se_simulator.materials.database import MaterialDatabase


class LayerStackWidget(QWidget):
    """Widget displaying the ordered stack of grating layers.

    Users can add, remove, and reorder layers. Each row shows the layer's
    name, type, and thickness.

    Signals
    -------
    layers_changed():
        Emitted whenever the layer list is modified.
    layer_selected(int):
        Emitted when a row is selected; passes the layer index.
    """

    layers_changed = Signal()
    layer_selected = Signal(int)

    _HEADERS = ["Name", "Type", "Thickness (nm)", "Period Lx (nm)", "Period Ly (nm)"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sample: SampleConfig | None = None
        self._stack: Stack | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_add = QPushButton("Add Layer")
        self._btn_remove = QPushButton("Remove Layer")
        self._btn_up = QPushButton("Move Up")
        self._btn_down = QPushButton("Move Down")
        for btn in (self._btn_add, self._btn_remove, self._btn_up, self._btn_down):
            toolbar.addWidget(btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(len(self._HEADERS))
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        # Substrate selector
        substrate_row = QHBoxLayout()
        substrate_row.addWidget(QLabel("Substrate:"))
        self._substrate_combo = QComboBox()
        substrate_row.addWidget(self._substrate_combo, stretch=1)
        layout.addLayout(substrate_row)

        # Connect signals
        self._btn_add.clicked.connect(self._add_layer)
        self._btn_remove.clicked.connect(self._remove_layer)
        self._btn_up.clicked.connect(self._move_up)
        self._btn_down.clicked.connect(self._move_down)
        self._table.currentItemChanged.connect(
            lambda cur, _prev: self._on_row_changed(self._table.row(cur) if cur else -1)
        )
        self._substrate_combo.currentTextChanged.connect(self._on_substrate_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_stack(self, stack: Stack) -> None:
        """Load layers from a Stack object (primary API)."""
        self._stack = stack
        # Sync _sample so existing helpers still work
        self._sample = stack.to_sample_config()
        self._refresh_table()
        self._refresh_substrate_combo()

    def load_sample(self, sample: SampleConfig) -> None:
        """Load layers from a SampleConfig (backward-compat shim)."""
        self._sample = sample
        self._stack = None
        self._refresh_table()
        self._refresh_substrate_combo()

    def get_stack(self) -> Stack | None:
        """Read current UI state and return a Stack, or None if nothing is loaded."""
        if self._sample is None:
            return None
        # Build Stack from the current _sample state
        def _mat_spec(name: str) -> MaterialSpec:
            return MaterialSpec(library_name=name, source="library", name=name)

        sup_name = self._sample.superstrate_material
        sub_name = self._sample.substrate_material
        stack_layers = []
        for gl in self._sample.layers:
            mat_name = gl.background_material
            sl = StackLayer(
                name=gl.name,
                type=gl.type,
                thickness_nm=gl.thickness_nm,
                Lx_nm=gl.Lx_nm,
                Ly_nm=gl.Ly_nm,
                material=_mat_spec(mat_name),
                shapes=gl.shapes,
                incoherent=gl.incoherent,
            )
            stack_layers.append(sl)
        return Stack(
            superstrate=_mat_spec(sup_name),
            substrate=_mat_spec(sub_name),
            layers=stack_layers,
        )

    def get_sample_config(self) -> SampleConfig | None:
        """Return current state as SampleConfig (backward-compat shim)."""
        stack = self.get_stack()
        return stack.to_sample_config() if stack is not None else None

    def get_layers(self) -> list[GratingLayer]:
        """Return the current list of layers (order matches table rows)."""
        if self._sample is None:
            return []
        return list(self._sample.layers)

    @property
    def row_count(self) -> int:
        """Current number of rows in the layer table."""
        return self._table.rowCount()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        layers = self.get_layers()
        self._table.setRowCount(len(layers))
        for row, layer in enumerate(layers):
            self._table.setItem(row, 0, QTableWidgetItem(layer.name))
            self._table.setItem(row, 1, QTableWidgetItem(layer.type))
            self._table.setItem(row, 2, QTableWidgetItem(f"{layer.thickness_nm:.1f}"))
            self._table.setItem(row, 3, QTableWidgetItem(f"{layer.Lx_nm:.1f}"))
            self._table.setItem(row, 4, QTableWidgetItem(f"{layer.Ly_nm:.1f}"))

    def _add_layer(self) -> None:
        """Add a default uniform layer."""
        if self._sample is None:
            return
        new_layer = GratingLayer(
            name=f"Layer {len(self._sample.layers) + 1}",
            type="uniform",
            thickness_nm=100.0,
            Lx_nm=self._sample.Lx_nm,
            Ly_nm=self._sample.Ly_nm,
            background_material=self._sample.superstrate_material,
        )
        self._sample.layers.append(new_layer)
        self._refresh_table()
        self.layers_changed.emit()

    def _remove_layer(self) -> None:
        """Remove the currently selected layer."""
        if self._sample is None:
            return
        row = self._table.currentRow()
        if row < 0 or row >= len(self._sample.layers):
            return
        self._sample.layers.pop(row)
        self._refresh_table()
        self.layers_changed.emit()

    def _move_up(self) -> None:
        """Move the selected layer up by one position."""
        if self._sample is None:
            return
        row = self._table.currentRow()
        if row <= 0:
            return
        layers = self._sample.layers
        layers[row], layers[row - 1] = layers[row - 1], layers[row]
        self._refresh_table()
        self._table.selectRow(row - 1)
        self.layers_changed.emit()

    def _move_down(self) -> None:
        """Move the selected layer down by one position."""
        if self._sample is None:
            return
        row = self._table.currentRow()
        layers = self._sample.layers
        if row < 0 or row >= len(layers) - 1:
            return
        layers[row], layers[row + 1] = layers[row + 1], layers[row]
        self._refresh_table()
        self._table.selectRow(row + 1)
        self.layers_changed.emit()

    def _refresh_substrate_combo(self) -> None:
        """Repopulate the substrate combo from user-defined and built-in library materials."""
        if self._sample is None:
            return
        db = MaterialDatabase()
        builtin = set(db.list_library_materials())
        user_defined = set(self._sample.materials.keys())
        names = sorted(builtin | user_defined)
        self._substrate_combo.blockSignals(True)
        self._substrate_combo.clear()
        for name in names:
            self._substrate_combo.addItem(name)
        self._substrate_combo.setCurrentText(self._sample.substrate_material)
        self._substrate_combo.blockSignals(False)

    def _on_substrate_changed(self, material: str) -> None:
        """Update the sample substrate and notify listeners."""
        if self._sample is None or not material:
            return
        if material not in self._sample.materials:
            self._sample.materials[material] = MaterialSpec(
                name=material, source="library", library_name=material
            )
        self._sample.substrate_material = material
        self.layers_changed.emit()

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.layer_selected.emit(row)
