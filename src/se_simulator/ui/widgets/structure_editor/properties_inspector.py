"""PropertiesInspectorWidget: editable property panel for a selected layer or shape."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QStackedWidget,
    QWidget,
)

from se_simulator.config.schemas import GratingLayer, MaterialSpec, SampleConfig, ShapeRegion
from se_simulator.materials.database import MaterialDatabase


def _all_material_names(sample: SampleConfig) -> list[str]:
    """Return sorted union of user-defined and built-in library materials."""
    db = MaterialDatabase()
    builtin = set(db.list_library_materials())
    user_defined = set(sample.materials.keys())
    return sorted(builtin | user_defined)

_GRATING_1D = "grating_1d"

_LAYER_TYPES = ["uniform", "grating_1d", "grating_2d"]


class PropertiesInspectorWidget(QWidget):
    """Right-panel inspector showing editable properties for a selected layer or shape.

    Two modes are supported:

    * **Layer mode** — shows layer name, type, thickness, periods, and background
      material.  Activated via :meth:`load_layer`.
    * **Shape mode** — shows shape type (read-only), cx/cy, width/height, sidewall
      angle, and assigned material.  Activated via :meth:`load_shape`.

    When any field is edited the corresponding signal is emitted:

    Signals
    -------
    layer_changed(GratingLayer):
        Emitted after any layer-field edit; carries the mutated layer object.
    shape_changed(ShapeRegion):
        Emitted after any shape-field edit; carries the mutated ShapeRegion object.
    """

    layer_changed = Signal(object)  # GratingLayer
    shape_changed = Signal(object)  # ShapeRegion

    # Stack page indices
    _PAGE_PLACEHOLDER = 0
    _PAGE_LAYER = 1
    _PAGE_SHAPE = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer: GratingLayer | None = None
        self._shape: ShapeRegion | None = None
        self._sample: SampleConfig | None = None
        self._parent_layer: GratingLayer | None = None  # parent layer when in shape mode
        self._blocking = False  # guard against recursive signal loops
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        from PySide6.QtWidgets import QVBoxLayout

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._stack = QStackedWidget()
        root.addWidget(self._stack)

        # --- Page 0: placeholder ---
        placeholder_page = QWidget()
        ph_layout = QFormLayout(placeholder_page)
        ph_layout.setContentsMargins(8, 8, 8, 8)
        self._placeholder = QLabel("Select a layer or shape to edit its properties.")
        self._placeholder.setWordWrap(True)
        ph_layout.addRow(self._placeholder)
        self._stack.addWidget(placeholder_page)  # index 0

        # --- Page 1: layer properties ---
        layer_page = QWidget()
        layer_form = QFormLayout(layer_page)
        layer_form.setContentsMargins(8, 8, 8, 8)
        layer_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Layer name")
        layer_form.addRow("Name:", self._name_edit)

        self._type_combo = QComboBox()
        for t in _LAYER_TYPES:
            self._type_combo.addItem(t)
        layer_form.addRow("Type:", self._type_combo)

        self._thickness_spin = QDoubleSpinBox()
        self._thickness_spin.setRange(0.1, 10_000.0)
        self._thickness_spin.setSuffix(" nm")
        self._thickness_spin.setDecimals(1)
        self._thickness_spin.setSingleStep(10.0)
        layer_form.addRow("Thickness:", self._thickness_spin)

        self._lx_spin = QDoubleSpinBox()
        self._lx_spin.setRange(0.1, 100_000.0)
        self._lx_spin.setSuffix(" nm")
        self._lx_spin.setDecimals(1)
        self._lx_spin.setSingleStep(10.0)
        layer_form.addRow("Period Lx:", self._lx_spin)

        self._ly_spin = QDoubleSpinBox()
        self._ly_spin.setRange(0.1, 100_000.0)
        self._ly_spin.setSuffix(" nm")
        self._ly_spin.setDecimals(1)
        self._ly_spin.setSingleStep(10.0)
        layer_form.addRow("Period Ly:", self._ly_spin)

        self._material_combo = QComboBox()
        layer_form.addRow("Background material:", self._material_combo)
        # Keep a reference to the label so it can be updated in load_layer
        self._material_label = layer_form.labelForField(self._material_combo)

        self._stack.addWidget(layer_page)  # index 1

        # --- Page 2: shape properties ---
        shape_page = QWidget()
        shape_form = QFormLayout(shape_page)
        shape_form.setContentsMargins(8, 8, 8, 8)
        shape_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self._shape_type_label = QLabel("")
        shape_form.addRow("Shape type:", self._shape_type_label)

        self._cx_spin = QDoubleSpinBox()
        self._cx_spin.setRange(-100_000.0, 100_000.0)
        self._cx_spin.setSuffix(" nm")
        self._cx_spin.setDecimals(1)
        self._cx_spin.setSingleStep(1.0)
        shape_form.addRow("Center X:", self._cx_spin)

        self._cy_spin = QDoubleSpinBox()
        self._cy_spin.setRange(-100_000.0, 100_000.0)
        self._cy_spin.setSuffix(" nm")
        self._cy_spin.setDecimals(1)
        self._cy_spin.setSingleStep(1.0)
        shape_form.addRow("Center Y:", self._cy_spin)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(0.1, 100_000.0)
        self._width_spin.setSuffix(" nm")
        self._width_spin.setDecimals(1)
        self._width_spin.setSingleStep(1.0)
        shape_form.addRow("Width:", self._width_spin)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(0.1, 100_000.0)
        self._height_spin.setSuffix(" nm")
        self._height_spin.setDecimals(1)
        self._height_spin.setSingleStep(1.0)
        shape_form.addRow("Height:", self._height_spin)

        self._sidewall_spin = QDoubleSpinBox()
        self._sidewall_spin.setRange(60.0, 90.0)
        self._sidewall_spin.setSuffix(" deg")
        self._sidewall_spin.setDecimals(1)
        self._sidewall_spin.setSingleStep(0.5)
        shape_form.addRow("Sidewall angle:", self._sidewall_spin)

        self._shape_material_combo = QComboBox()
        shape_form.addRow("Material:", self._shape_material_combo)

        self._stack.addWidget(shape_page)  # index 2

        # Connect layer-page signals
        self._name_edit.editingFinished.connect(self._on_name_changed)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        self._thickness_spin.valueChanged.connect(self._on_thickness_changed)
        self._lx_spin.valueChanged.connect(self._on_lx_changed)
        self._ly_spin.valueChanged.connect(self._on_ly_changed)
        self._material_combo.currentTextChanged.connect(self._on_material_changed)

        # Connect shape-page signals
        self._cx_spin.valueChanged.connect(self._on_cx_changed)
        self._cy_spin.valueChanged.connect(self._on_cy_changed)
        self._width_spin.valueChanged.connect(self._on_width_changed)
        self._height_spin.valueChanged.connect(self._on_height_changed)
        self._sidewall_spin.valueChanged.connect(self._on_sidewall_changed)
        self._shape_material_combo.currentTextChanged.connect(self._on_shape_material_changed)

        self._stack.setCurrentIndex(self._PAGE_PLACEHOLDER)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_layer(self, layer: GratingLayer, sample: SampleConfig) -> None:
        """Populate the layer form from *layer* and use *sample* for the material list."""
        self._layer = layer
        self._shape = None
        self._sample = sample
        self._blocking = True
        try:
            self._name_edit.setText(layer.name)

            type_idx = _LAYER_TYPES.index(layer.type) if layer.type in _LAYER_TYPES else 0
            self._type_combo.setCurrentIndex(type_idx)

            self._thickness_spin.setValue(layer.thickness_nm)
            self._lx_spin.setValue(layer.Lx_nm)
            self._ly_spin.setValue(layer.Ly_nm)

            self._material_combo.clear()
            for mat_name in _all_material_names(sample):
                self._material_combo.addItem(mat_name)
            mat_idx = self._material_combo.findText(layer.background_material)
            if mat_idx >= 0:
                self._material_combo.setCurrentIndex(mat_idx)

            if layer.type == "uniform":
                self._material_label.setText("Layer material:")
            else:
                self._material_label.setText("Background material:")

            self._stack.setCurrentIndex(self._PAGE_LAYER)
        finally:
            self._blocking = False

    def load_shape(
        self,
        shape: ShapeRegion,
        sample: SampleConfig,
        layer: GratingLayer | None = None,
    ) -> None:
        """Populate the shape form from *shape* and use *sample* for the material list.

        Parameters
        ----------
        shape:
            The :class:`~se_simulator.config.schemas.ShapeRegion` to display.
        sample:
            The parent :class:`~se_simulator.config.schemas.SampleConfig` (used
            to populate the material combo).
        layer:
            Optional parent :class:`~se_simulator.config.schemas.GratingLayer`.
            When provided and the layer type is ``grating_1d``, the height spin
            is locked to ``layer.Ly_nm`` so the user cannot violate the 1D
            grating constraint.
        """
        self._shape = shape
        self._layer = None
        self._parent_layer = layer
        self._sample = sample
        self._blocking = True
        try:
            geom = shape.geometry
            self._shape_type_label.setText(geom.type)
            self._cx_spin.setValue(geom.cx)
            self._cy_spin.setValue(geom.cy)
            self._width_spin.setValue(geom.width)

            is_grating_1d = layer is not None and layer.type == _GRATING_1D
            if is_grating_1d:
                locked_height = layer.Ly_nm  # type: ignore[union-attr]
                self._height_spin.setValue(locked_height)
                self._height_spin.setEnabled(False)
                self._height_spin.setToolTip("Height is locked to Ly for 1D gratings")
            else:
                self._height_spin.setValue(geom.height)
                self._height_spin.setEnabled(True)
                self._height_spin.setToolTip("")

            self._sidewall_spin.setValue(
                max(60.0, min(90.0, geom.sidewall_angle_deg))
            )

            self._shape_material_combo.clear()
            for mat_name in _all_material_names(sample):
                self._shape_material_combo.addItem(mat_name)
            mat_idx = self._shape_material_combo.findText(shape.material)
            if mat_idx >= 0:
                self._shape_material_combo.setCurrentIndex(mat_idx)

            self._stack.setCurrentIndex(self._PAGE_SHAPE)
        finally:
            self._blocking = False

    def clear(self) -> None:
        """Reset the inspector to the empty/placeholder state."""
        self._layer = None
        self._shape = None
        self._sample = None
        self._parent_layer = None
        self._stack.setCurrentIndex(self._PAGE_PLACEHOLDER)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_layer_if_ready(self) -> None:
        if not self._blocking and self._layer is not None:
            self.layer_changed.emit(self._layer)

    def _emit_shape_if_ready(self) -> None:
        if not self._blocking and self._shape is not None:
            self.shape_changed.emit(self._shape)

    # ------------------------------------------------------------------
    # Layer field change handlers
    # ------------------------------------------------------------------

    def _on_name_changed(self) -> None:
        if self._blocking or self._layer is None:
            return
        self._layer = self._layer.model_copy(update={"name": self._name_edit.text()})
        self._emit_layer_if_ready()

    def _on_type_changed(self, value: str) -> None:
        if self._blocking or self._layer is None:
            return
        updated = self._layer.model_copy(update={"type": value})  # type: ignore[arg-type]
        if value == "uniform":
            # Clear any shapes when reverting to a uniform layer
            updated = updated.model_copy(update={"shapes": []})
        elif value in ("grating_1d", "grating_2d") and not updated.shapes:
            from se_simulator.config.schemas import ShapeGeometry, ShapeRegion

            # Pick the first non-background material available, fall back to "Si"
            bg = updated.background_material
            available = []
            if self._sample is not None:
                available = [m for m in sorted(self._sample.materials.keys()) if m != bg]
            shape_material = available[0] if available else "Si"

            lx = updated.Lx_nm or 500.0
            ly = updated.Ly_nm or 500.0
            # For grating_1d the shape must span the full y-period
            shape_height = ly if value == _GRATING_1D else ly * 0.3
            shape_cy = ly / 2
            default_shape = ShapeRegion(
                geometry=ShapeGeometry(
                    type="rectangle",
                    cx=lx / 2,
                    cy=shape_cy,
                    width=lx * 0.3,
                    height=shape_height,
                ),
                material=shape_material,
            )
            updated = updated.model_copy(update={"shapes": [default_shape]})
        self._layer = updated
        self._emit_layer_if_ready()

    def _on_thickness_changed(self, value: float) -> None:
        if self._blocking or self._layer is None:
            return
        self._layer = self._layer.model_copy(update={"thickness_nm": value})
        self._emit_layer_if_ready()

    def _on_lx_changed(self, value: float) -> None:
        if self._blocking or self._layer is None:
            return
        self._layer = self._layer.model_copy(update={"Lx_nm": value})
        self._emit_layer_if_ready()

    def _on_ly_changed(self, value: float) -> None:
        if self._blocking or self._layer is None:
            return
        self._layer = self._layer.model_copy(update={"Ly_nm": value})
        self._emit_layer_if_ready()

    def _on_material_changed(self, value: str) -> None:
        if self._blocking or self._layer is None or not value:
            return
        if self._sample is not None and value not in self._sample.materials:
            self._sample.materials[value] = MaterialSpec(
                name=value, source="library", library_name=value
            )
        self._layer = self._layer.model_copy(update={"background_material": value})
        self._emit_layer_if_ready()

    # ------------------------------------------------------------------
    # Shape field change handlers
    # ------------------------------------------------------------------

    def _on_cx_changed(self, value: float) -> None:
        if self._blocking or self._shape is None:
            return
        new_geom = self._shape.geometry.model_copy(update={"cx": value})
        self._shape = self._shape.model_copy(update={"geometry": new_geom})
        self._emit_shape_if_ready()

    def _on_cy_changed(self, value: float) -> None:
        if self._blocking or self._shape is None:
            return
        new_geom = self._shape.geometry.model_copy(update={"cy": value})
        self._shape = self._shape.model_copy(update={"geometry": new_geom})
        self._emit_shape_if_ready()

    def _on_width_changed(self, value: float) -> None:
        if self._blocking or self._shape is None:
            return
        new_geom = self._shape.geometry.model_copy(update={"width": value})
        self._shape = self._shape.model_copy(update={"geometry": new_geom})
        self._emit_shape_if_ready()

    def _on_height_changed(self, value: float) -> None:
        if self._blocking or self._shape is None:
            return
        # Enforce the 1D grating constraint: height must equal Ly_nm
        if self._parent_layer is not None and self._parent_layer.type == _GRATING_1D:
            value = self._parent_layer.Ly_nm
        new_geom = self._shape.geometry.model_copy(update={"height": value})
        self._shape = self._shape.model_copy(update={"geometry": new_geom})
        self._emit_shape_if_ready()

    def _on_sidewall_changed(self, value: float) -> None:
        if self._blocking or self._shape is None:
            return
        new_geom = self._shape.geometry.model_copy(update={"sidewall_angle_deg": value})
        self._shape = self._shape.model_copy(update={"geometry": new_geom})
        self._emit_shape_if_ready()

    def _on_shape_material_changed(self, value: str) -> None:
        if self._blocking or self._shape is None or not value:
            return
        if self._sample is not None and value not in self._sample.materials:
            self._sample.materials[value] = MaterialSpec(
                name=value, source="library", library_name=value
            )
        self._shape = self._shape.model_copy(update={"material": value})
        self._emit_shape_if_ready()
