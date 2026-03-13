"""NKPlot: n(λ) and k(λ) dual-axis plot widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget


class NKPlot(QWidget):
    """Plot n(λ) and k(λ) optical constants for one or more materials.

    Uses pyqtgraph with two y-axes (left: n, right: k).
    Import of pyqtgraph is deferred until after QApplication creation.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg
        from PySide6.QtWidgets import QVBoxLayout

        self._pg = pg
        self._n_curves: dict[str, object] = {}
        self._k_curves: dict[str, object] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        # Top plot: n(λ)
        self._p_n = self._glw.addPlot(row=0, col=0)
        self._p_n.setLabel("left", "n")
        self._p_n.setLabel("bottom", "Wavelength (nm)")
        self._p_n.showGrid(x=True, y=True, alpha=0.3)
        self._p_n.addLegend()

        # Bottom plot: k(λ)
        self._p_k = self._glw.addPlot(row=1, col=0)
        self._p_k.setLabel("left", "k")
        self._p_k.setLabel("bottom", "Wavelength (nm)")
        self._p_k.showGrid(x=True, y=True, alpha=0.3)
        self._p_k.addLegend()

        self._p_n.setXLink(self._p_k)

    def add_material(
        self,
        name: str,
        wavelengths_nm: np.ndarray,
        n: np.ndarray,
        k: np.ndarray,
        color: str | None = None,
    ) -> None:
        """Add or replace n/k curves for a named material."""
        self.remove_material(name)
        pen_n = self._pg.mkPen(color=color or "b", width=2)
        pen_k = self._pg.mkPen(color=color or "r", width=2, style=self._pg.QtCore.Qt.PenStyle.DashLine)
        self._n_curves[name] = self._p_n.plot(wavelengths_nm, n, pen=pen_n, name=name)
        self._k_curves[name] = self._p_k.plot(wavelengths_nm, k, pen=pen_k, name=name)

    def remove_material(self, name: str) -> None:
        """Remove a material's curves."""
        if name in self._n_curves:
            self._p_n.removeItem(self._n_curves.pop(name))
        if name in self._k_curves:
            self._p_k.removeItem(self._k_curves.pop(name))

    def clear(self) -> None:
        """Remove all material curves."""
        for name in list(self._n_curves.keys()):
            self.remove_material(name)
