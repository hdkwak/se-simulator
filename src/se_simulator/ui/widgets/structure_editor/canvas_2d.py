"""Canvas2DWidget: 2D top-down view of a grating layer unit cell."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from se_simulator.config.schemas import GratingLayer

# Palette of distinct fill colors (RGBA) cycled per unique material name
_MATERIAL_PALETTE: list[QColor] = [
    QColor(100, 180, 255, 180),
    QColor(120, 210, 120, 180),
    QColor(255, 180, 80, 180),
    QColor(210, 120, 210, 180),
    QColor(80, 210, 210, 180),
    QColor(255, 120, 120, 180),
]


def _material_color(material: str, material_index_map: dict[str, int]) -> QColor:
    """Return a stable fill colour for *material* based on insertion order."""
    if material not in material_index_map:
        material_index_map[material] = len(material_index_map)
    return _MATERIAL_PALETTE[material_index_map[material] % len(_MATERIAL_PALETTE)]


class Canvas2DWidget(QWidget):
    """Simple 2D canvas rendering the shapes in a grating layer unit cell.

    Signals
    -------
    shape_selected(int):
        Emitted when the user clicks a shape; carries the shape index within
        the current layer's ``shapes`` list.  Emitted with ``-1`` when the
        click lands on the background (deselect).
    """

    shape_selected = Signal(int)

    _SHAPE_SELECTED_COLOR = QColor(255, 160, 40, 210)
    _BG_COLOR = QColor(245, 245, 245)
    _BORDER_COLOR = QColor(60, 60, 60)
    _SELECTED_BORDER_COLOR = QColor(200, 80, 0)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer: GratingLayer | None = None
        self._selected_shape_idx: int = -1
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_layer(self, layer: GratingLayer) -> None:
        """Set the layer to render and trigger a repaint."""
        self._layer = layer
        self._selected_shape_idx = -1
        self.update()

    def clear_selection(self) -> None:
        """Deselect any selected shape without emitting a signal."""
        self._selected_shape_idx = -1
        self.update()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: object) -> None:  # noqa: N802
        """Select the topmost shape under the cursor, or deselect."""
        if self._layer is None:
            return

        from PySide6.QtCore import QPoint

        pos: QPoint = event.pos()  # type: ignore[union-attr]
        mx, my = pos.x(), pos.y()

        margin = 20
        w = self.width()
        h = self.height()
        draw_w = w - 2 * margin
        draw_h = h - 2 * margin
        lx = self._layer.Lx_nm or 1.0
        ly = self._layer.Ly_nm or 1.0
        scale_x = draw_w / lx
        scale_y = draw_h / ly

        hit = -1
        # Iterate in reverse so the topmost-drawn shape is picked first
        for idx in reversed(range(len(self._layer.shapes))):
            geom = self._layer.shapes[idx].geometry
            shape_type = geom.type
            if shape_type in ("rectangle", "trapezoid"):
                x0 = geom.cx - geom.width / 2
                y0 = geom.cy - geom.height / 2
                px = int(margin + x0 * scale_x)
                py = int(margin + y0 * scale_y)
                pw = int(geom.width * scale_x)
                ph = int(geom.height * scale_y)
                if px <= mx <= px + pw and py <= my <= py + ph:
                    hit = idx
                    break
            elif shape_type == "ellipse":
                cx_px = margin + geom.cx * scale_x
                cy_px = margin + geom.cy * scale_y
                rx = geom.width / 2 * scale_x
                ry = geom.height / 2 * scale_y
                if rx > 0 and ry > 0:
                    dx = (mx - cx_px) / rx
                    dy = (my - cy_px) / ry
                    if dx * dx + dy * dy <= 1.0:
                        hit = idx
                        break

        self._selected_shape_idx = hit
        self.shape_selected.emit(hit)
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: object) -> None:  # noqa: ARG002, N802
        if self._layer is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 20

        draw_w = w - 2 * margin
        draw_h = h - 2 * margin

        lx = self._layer.Lx_nm or 1.0
        ly = self._layer.Ly_nm or 1.0

        scale_x = draw_w / lx
        scale_y = draw_h / ly

        def to_px(x: float, y: float) -> tuple[int, int]:
            return (int(margin + x * scale_x), int(margin + y * scale_y))

        # Draw background (unit cell)
        painter.fillRect(margin, margin, draw_w, draw_h, QBrush(self._BG_COLOR))
        painter.setPen(QPen(self._BORDER_COLOR, 2))
        painter.drawRect(margin, margin, draw_w, draw_h)

        # Build a stable material → colour mapping for this layer
        material_index_map: dict[str, int] = {}

        # Draw shapes
        for idx, shape_region in enumerate(self._layer.shapes):
            selected = idx == self._selected_shape_idx
            fill_color = (
                self._SHAPE_SELECTED_COLOR
                if selected
                else _material_color(shape_region.material, material_index_map)
            )
            border_color = self._SELECTED_BORDER_COLOR if selected else self._BORDER_COLOR
            border_width = 2 if selected else 1

            painter.setBrush(QBrush(fill_color))
            painter.setPen(QPen(border_color, border_width))

            geom = shape_region.geometry
            shape_type = geom.type

            if shape_type in ("rectangle", "trapezoid"):
                x0 = geom.cx - geom.width / 2
                y0 = geom.cy - geom.height / 2
                px, py = to_px(x0, y0)
                pw = int(geom.width * scale_x)
                ph = int(geom.height * scale_y)
                painter.drawRect(px, py, pw, ph)

            elif shape_type == "ellipse":
                x0 = geom.cx - geom.width / 2
                y0 = geom.cy - geom.height / 2
                px, py = to_px(x0, y0)
                pw = int(geom.width * scale_x)
                ph = int(geom.height * scale_y)
                painter.drawEllipse(px, py, pw, ph)

        painter.end()
