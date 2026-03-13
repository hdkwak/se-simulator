"""StructureEditor: main composite editor for SampleConfig."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QSplitter, QTabWidget, QVBoxLayout, QWidget

from se_simulator.config.schemas import SampleConfig
from se_simulator.ui.widgets.structure_editor.canvas_2d import Canvas2DWidget
from se_simulator.ui.widgets.structure_editor.layer_stack import LayerStackWidget
from se_simulator.ui.widgets.structure_editor.preview_3d import Preview3DWidget


class StructureEditor(QWidget):
    """Full structure editor: layer stack + 2D canvas + 3D preview.

    Signals
    -------
    sample_changed(SampleConfig):
        Emitted when the sample is modified.
    layer_selected(int):
        Emitted when a layer row is selected; passes the layer index.
    shape_selected(int, int):
        Emitted when a shape is clicked on the 2D canvas.  Carries
        ``(layer_idx, shape_idx)``.  ``shape_idx == -1`` means background
        (deselect).
    """

    sample_changed = Signal(object)
    layer_selected = Signal(int)
    shape_selected = Signal(int, int)  # (layer_idx, shape_idx)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sample: SampleConfig | None = None
        self._selected_layer_idx: int = -1
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter()

        # Left: layer stack
        self.layer_stack = LayerStackWidget()
        splitter.addWidget(self.layer_stack)

        # Right: tabs for 2D canvas + 3D preview
        tabs = QTabWidget()
        self._canvas_2d = Canvas2DWidget()
        tabs.addTab(self._canvas_2d, "2D Top View")
        self._preview_3d = Preview3DWidget()
        tabs.addTab(self._preview_3d, "3D Preview")
        splitter.addWidget(tabs)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # Connect
        self.layer_stack.layer_selected.connect(self._on_layer_selected)
        self.layer_stack.layers_changed.connect(self._on_layers_changed)
        self._canvas_2d.shape_selected.connect(self._on_canvas_shape_selected)

    def load_sample(self, sample: SampleConfig) -> None:
        """Load a SampleConfig into all sub-editors."""
        self._sample = sample
        self.layer_stack.load_sample(sample)
        if sample.layers:
            self._canvas_2d.set_layer(sample.layers[0])
        self._preview_3d.set_layers(sample.layers)

    def apply_layer_edit(self, idx: int, updated_layer: object) -> None:
        """Apply an edited layer from the Properties Inspector back into the sample.

        Parameters
        ----------
        idx:
            Index of the layer within ``_sample.layers`` to replace.
        updated_layer:
            The updated ``GratingLayer`` produced by the inspector.
        """
        if self._sample is None:
            return
        if 0 <= idx < len(self._sample.layers):
            from se_simulator.config.schemas import GratingLayer

            if isinstance(updated_layer, GratingLayer):
                # When a layer becomes grating_1d, clamp all shape heights to Ly_nm
                if updated_layer.type == "grating_1d" and updated_layer.shapes:
                    clamped_shapes = []
                    for shape in updated_layer.shapes:
                        clamped_geom = shape.geometry.model_copy(
                            update={"height": updated_layer.Ly_nm}
                        )
                        clamped_shapes.append(
                            shape.model_copy(update={"geometry": clamped_geom})
                        )
                    updated_layer = updated_layer.model_copy(
                        update={"shapes": clamped_shapes}
                    )
                self._sample.layers[idx] = updated_layer
                self.layer_stack.load_sample(self._sample)
                self._canvas_2d.set_layer(self._sample.layers[idx])
                self._preview_3d.set_layers(self._sample.layers)
                self.sample_changed.emit(self._sample)

    def apply_shape_edit(
        self, layer_idx: int, shape_idx: int, updated_shape: object
    ) -> None:
        """Apply an edited shape from the Properties Inspector back into the sample.

        Parameters
        ----------
        layer_idx:
            Index of the parent layer within ``_sample.layers``.
        shape_idx:
            Index of the shape within the layer's ``shapes`` list.
        updated_shape:
            The updated ``ShapeRegion`` produced by the inspector.
        """
        if self._sample is None:
            return
        from se_simulator.config.schemas import ShapeRegion

        if (
            isinstance(updated_shape, ShapeRegion)
            and 0 <= layer_idx < len(self._sample.layers)
        ):
            layer = self._sample.layers[layer_idx]
            if 0 <= shape_idx < len(layer.shapes):
                # Enforce 1D grating constraint: height must equal Ly_nm
                if layer.type == "grating_1d":
                    clamped_geom = updated_shape.geometry.model_copy(
                        update={"height": layer.Ly_nm}
                    )
                    updated_shape = updated_shape.model_copy(
                        update={"geometry": clamped_geom}
                    )
                layer.shapes[shape_idx] = updated_shape
                self._canvas_2d.set_layer(layer)
                self._preview_3d.set_layers(self._sample.layers)
                self.sample_changed.emit(self._sample)

    def _on_layer_selected(self, idx: int) -> None:
        if self._sample is None:
            return
        self._selected_layer_idx = idx
        if 0 <= idx < len(self._sample.layers):
            self._canvas_2d.set_layer(self._sample.layers[idx])
        self.layer_selected.emit(idx)

    def _on_canvas_shape_selected(self, shape_idx: int) -> None:
        """Forward canvas shape-click as (layer_idx, shape_idx)."""
        self.shape_selected.emit(self._selected_layer_idx, shape_idx)

    def _on_layers_changed(self) -> None:
        if self._sample is not None:
            self._preview_3d.set_layers(self._sample.layers)
            self.sample_changed.emit(self._sample)
