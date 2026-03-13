"""Chi2MapWidget: 2D heatmap for chi-squared landscape visualization."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget


class Chi2MapWidget(QWidget):
    """2D heatmap of chi-squared values over a 2-parameter grid.

    Uses pyqtgraph ImageItem with log-scale option and minimum marker.
    Import of pyqtgraph is deferred until after QApplication creation.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg
        from PySide6.QtWidgets import QVBoxLayout

        self._pg = pg
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        self._plot = self._glw.addPlot()
        self._plot.setLabel("bottom", "Parameter 1")
        self._plot.setLabel("left", "Parameter 2")

        self._img_item = pg.ImageItem()
        self._plot.addItem(self._img_item)

        self._colorbar = pg.ColorBarItem(
            colorMap=pg.colormap.get("inferno"),
            interactive=False,
        )
        self._colorbar.setImageItem(self._img_item, insert_in=self._plot)

        self._min_marker: object | None = None

    def set_data(
        self,
        chi2_grid: np.ndarray,
        x_values: np.ndarray,
        y_values: np.ndarray,
        log_scale: bool = True,
        x_label: str = "Parameter 1",
        y_label: str = "Parameter 2",
    ) -> None:
        """Set the chi-squared grid data.

        Parameters
        ----------
        chi2_grid:
            2D array of shape (N_x, N_y) with chi-squared values.
        x_values:
            1D array of parameter 1 values (length N_x).
        y_values:
            1D array of parameter 2 values (length N_y).
        log_scale:
            If True, display log10(chi2) for better dynamic range.
        x_label:
            X-axis label text.
        y_label:
            Y-axis label text.
        """
        self._plot.setLabel("bottom", x_label)
        self._plot.setLabel("left", y_label)

        display_data = np.log10(np.maximum(chi2_grid, 1e-30)) if log_scale else chi2_grid.copy()

        # Set image transform to map pixels to parameter values
        dx = (x_values[-1] - x_values[0]) / max(len(x_values) - 1, 1)
        dy = (y_values[-1] - y_values[0]) / max(len(y_values) - 1, 1)
        transform = self._pg.QtGui.QTransform()
        transform.translate(x_values[0], y_values[0])
        transform.scale(dx, dy)
        self._img_item.setTransform(transform)
        self._img_item.setImage(display_data)

        self._colorbar.setLevels((float(display_data.min()), float(display_data.max())))

    def mark_minimum(self, x: float, y: float) -> None:
        """Place a scatter marker at the minimum chi-squared location."""
        if self._min_marker is not None:
            self._plot.removeItem(self._min_marker)
        self._min_marker = self._pg.ScatterPlotItem(
            [x], [y], symbol="x", size=12, pen=self._pg.mkPen("r", width=2)
        )
        self._plot.addItem(self._min_marker)

    def clear(self) -> None:
        """Clear the heatmap."""
        self._img_item.clear()
        if self._min_marker is not None:
            self._plot.removeItem(self._min_marker)
            self._min_marker = None
