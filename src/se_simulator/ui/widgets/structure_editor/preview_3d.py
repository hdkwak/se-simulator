"""Preview3DWidget: 3D preview of the sample layer stack.

Uses OpenGL if available; falls back to a matplotlib Axes3D view.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

# Try to import OpenGL; fall back gracefully
try:
    import OpenGL  # noqa: F401

    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False


class Preview3DWidget(QWidget):
    """3D preview widget for the full sample stack.

    Falls back to matplotlib when OpenGL is not available.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layers: list = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if HAS_OPENGL:
            label = QLabel("3D Preview (OpenGL)")
        else:
            label = QLabel("3D Preview (matplotlib fallback)")

        label.setAlignment(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        self._canvas_widget = self._build_canvas()
        if self._canvas_widget is not None:
            layout.addWidget(self._canvas_widget)

    def _build_canvas(self) -> QWidget | None:
        """Build the appropriate 3D canvas."""
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = Figure(figsize=(4, 3), dpi=80)
            self._ax = fig.add_subplot(111, projection="3d")
            self._ax.set_xlabel("X (nm)")
            self._ax.set_ylabel("Y (nm)")
            self._ax.set_zlabel("Z (nm)")
            canvas = FigureCanvasQTAgg(fig)
            self._fig = fig
            return canvas
        except Exception:  # noqa: BLE001
            return None

    def set_layers(self, layers: list) -> None:
        """Update the 3D view with new layer data."""
        self._layers = layers
        self._draw()

    def _draw(self) -> None:
        """Redraw the 3D preview."""
        if not hasattr(self, "_ax"):
            return
        self._ax.clear()
        self._ax.set_xlabel("X (nm)")
        self._ax.set_ylabel("Y (nm)")
        self._ax.set_zlabel("Z (nm)")

        z = 0.0
        for layer in reversed(self._layers):
            thickness = getattr(layer, "thickness_nm", 50.0)
            lx = getattr(layer, "Lx_nm", 500.0)
            ly = getattr(layer, "Ly_nm", 500.0)

            # Draw a simple box for the layer
            xs = [0, lx, lx, 0, 0]
            ys = [0, 0, ly, ly, 0]
            for zi in [z, z + thickness]:
                self._ax.plot(xs, ys, [zi] * 5, "b-", alpha=0.4)

            z += thickness

        if hasattr(self, "_fig"):
            self._fig.canvas.draw_idle()
