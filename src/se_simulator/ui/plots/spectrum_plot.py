"""SpectrumPlot: pyqtgraph-based spectrum visualization widget."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    pass

# Color palette for datasets
_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

_STYLE_MAP = {
    "solid": Qt.PenStyle.SolidLine,
    "dash": Qt.PenStyle.DashLine,
    "dot": Qt.PenStyle.DotLine,
    "dashdot": Qt.PenStyle.DashDotLine,
}


class SpectrumPlot(QWidget):
    """Multi-dataset spectrum plot with export capabilities.

    Wraps a pyqtgraph GraphicsLayoutWidget. Import is deferred so that
    pg is never imported at module load time (requires QApplication to exist).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg
        from PySide6.QtWidgets import QVBoxLayout

        self._pg = pg
        self._datasets: dict[str, object] = {}  # name -> PlotDataItem
        self._color_index = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.getPlotItem().addLegend()
        layout.addWidget(self._plot_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_dataset(
        self,
        name: str,
        wavelengths_nm: np.ndarray,
        y_values: np.ndarray,
        color: str | None = None,
        style: str = "solid",
    ) -> None:
        """Add or replace a named dataset on the plot."""
        if name in self._datasets:
            self.remove_dataset(name)

        if color is None:
            color = _COLORS[self._color_index % len(_COLORS)]
            self._color_index += 1

        pen_style = _STYLE_MAP.get(style, Qt.PenStyle.SolidLine)
        pen = self._pg.mkPen(color=color, width=2, style=pen_style)
        curve = self._plot_widget.plot(
            wavelengths_nm,
            y_values,
            pen=pen,
            name=name,
        )
        self._datasets[name] = curve

    def remove_dataset(self, name: str) -> None:
        """Remove a dataset by name."""
        if name in self._datasets:
            self._plot_widget.removeItem(self._datasets.pop(name))

    def clear(self) -> None:
        """Remove all datasets."""
        for name in list(self._datasets.keys()):
            self.remove_dataset(name)
        self._color_index = 0

    def set_ylabel(self, label: str) -> None:
        """Set the y-axis label."""
        self._plot_widget.setLabel("left", label)

    def set_xlabel(self, label: str) -> None:
        """Set the x-axis label."""
        self._plot_widget.setLabel("bottom", label)

    def set_title(self, title: str) -> None:
        """Set the plot title."""
        self._plot_widget.setTitle(title)

    def export_png(self, path: str | Path, dpi: int = 300) -> None:
        """Export the plot as a PNG image."""
        from PySide6.QtGui import QImage, QPainter

        path = Path(path)
        size = self._plot_widget.size()
        scale = dpi / 96
        img = QImage(
            int(size.width() * scale),
            int(size.height() * scale),
            QImage.Format.Format_RGB32,
        )
        img.fill(QColor("white"))
        painter = QPainter(img)
        painter.scale(scale, scale)
        self._plot_widget.render(painter)
        painter.end()
        img.save(str(path))

    def export_svg(self, path: str | Path) -> None:
        """Export the plot as an SVG file."""
        from PySide6.QtSvg import QSvgGenerator

        path = Path(path)
        generator = QSvgGenerator()
        generator.setFileName(str(path))
        size = self._plot_widget.size()
        generator.setSize(size)

        from PySide6.QtGui import QPainter

        painter = QPainter(generator)
        self._plot_widget.render(painter)
        painter.end()

    @property
    def dataset_names(self) -> list[str]:
        """Return the names of all currently plotted datasets."""
        return list(self._datasets.keys())
