"""ResultsViewer: tabbed display of ellipsometric simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from se_simulator.ellipsometer.signals import EllipsometryResult


class ResultsViewer(QWidget):
    """Display Ψ/Δ and Fourier coefficient spectra after a simulation.

    Signals
    -------
    export_requested(Path):
        Emitted when the user triggers CSV export; carries the chosen file path.
    """

    export_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: EllipsometryResult | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._lbl_status = QLabel("No results loaded")
        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_png = QPushButton("Export PNG")
        self._btn_export_png.setEnabled(False)
        toolbar.addWidget(self._lbl_status)
        toolbar.addStretch()
        toolbar.addWidget(self._btn_export_csv)
        toolbar.addWidget(self._btn_export_png)
        layout.addLayout(toolbar)

        # Tabs
        self._tabs = QTabWidget()

        # Ψ/Δ tab
        self._psi_delta_tab = QWidget()
        pd_layout = QVBoxLayout(self._psi_delta_tab)

        from se_simulator.ui.plots.spectrum_plot import SpectrumPlot

        self._psi_plot = SpectrumPlot()
        self._psi_plot.set_xlabel("Wavelength (nm)")
        self._psi_plot.set_ylabel("Ψ (°)")
        self._psi_plot.set_title("Psi")
        pd_layout.addWidget(self._psi_plot)

        self._delta_plot = SpectrumPlot()
        self._delta_plot.set_xlabel("Wavelength (nm)")
        self._delta_plot.set_ylabel("Δ (°)")
        self._delta_plot.set_title("Delta")
        pd_layout.addWidget(self._delta_plot)

        self._tabs.addTab(self._psi_delta_tab, "Ψ / Δ")

        # Fourier tab
        self._fourier_tab = QWidget()
        f_layout = QVBoxLayout(self._fourier_tab)

        self._alpha_beta_plot = SpectrumPlot()
        self._alpha_beta_plot.set_xlabel("Wavelength (nm)")
        self._alpha_beta_plot.set_ylabel("Coefficient")
        self._alpha_beta_plot.set_title("Alpha / Beta")
        f_layout.addWidget(self._alpha_beta_plot)

        self._chi_xi_plot = SpectrumPlot()
        self._chi_xi_plot.set_xlabel("Wavelength (nm)")
        self._chi_xi_plot.set_ylabel("Coefficient")
        self._chi_xi_plot.set_title("Chi / Xi")
        f_layout.addWidget(self._chi_xi_plot)

        self._tabs.addTab(self._fourier_tab, "Fourier Coefficients")

        # Energy conservation tab
        self._energy_tab = QWidget()
        e_layout = QVBoxLayout(self._energy_tab)
        self._energy_plot = SpectrumPlot()
        self._energy_plot.set_xlabel("Wavelength (nm)")
        self._energy_plot.set_ylabel("R + T")
        self._energy_plot.set_title("Energy Conservation")
        e_layout.addWidget(self._energy_plot)
        self._tabs.addTab(self._energy_tab, "Energy Conservation")

        # Mueller Matrix tab
        self._setup_mueller_tab()

        layout.addWidget(self._tabs)

        # Connect signals
        self._btn_export_csv.clicked.connect(self._on_export_csv)
        self._btn_export_png.clicked.connect(self._on_export_png)

    def _setup_mueller_tab(self) -> None:
        """Create the Mueller Matrix tab content."""
        from se_simulator.ui.plots.spectrum_plot import SpectrumPlot

        self._mueller_tab = QWidget()
        tab_layout = QVBoxLayout(self._mueller_tab)
        tab_layout.setContentsMargins(4, 4, 4, 4)

        # Header row with export button
        header = QHBoxLayout()
        self._btn_export_mueller = QPushButton("Export Mueller CSV")
        self._btn_export_mueller.setEnabled(False)
        self._btn_export_mueller.clicked.connect(self._on_export_mueller_csv)
        header.addWidget(self._btn_export_mueller)
        header.addStretch()
        tab_layout.addLayout(header)

        # 4×4 grid of buttons
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(4)

        self._mueller_buttons: list[list[QPushButton]] = []
        for i in range(4):
            row: list[QPushButton] = []
            for j in range(4):
                btn = QPushButton(f"m{i + 1}{j + 1}")
                if i == j:
                    btn.setStyleSheet("font-weight: bold;")
                if i == 0 and j == 0:
                    btn.setText("m11\n1.000")
                    btn.setEnabled(False)
                else:
                    # Use default args to capture i, j by value
                    btn.clicked.connect(
                        lambda checked=False, _i=i, _j=j: self._on_mueller_cell_clicked(_i, _j)
                    )
                grid_layout.addWidget(btn, i, j)
                row.append(btn)
            self._mueller_buttons.append(row)

        tab_layout.addWidget(grid_widget)

        # Detail plot for the selected element
        self._mueller_detail_plot = SpectrumPlot()
        self._mueller_detail_plot.set_xlabel("Wavelength (nm)")
        self._mueller_detail_plot.set_ylabel("mij")
        self._mueller_detail_plot.set_title("Mueller Element (click a cell above)")
        tab_layout.addWidget(self._mueller_detail_plot)

        self._tabs.addTab(self._mueller_tab, "Mueller Matrix")

    def load_result(self, result: EllipsometryResult, label: str = "Simulated") -> None:
        """Display a new EllipsometryResult."""
        self._result = result
        wl = result.wavelengths_nm

        # Ψ/Δ plots
        self._psi_plot.clear()
        self._psi_plot.add_dataset(label, wl, result.psi_deg, color="#1f77b4")

        self._delta_plot.clear()
        self._delta_plot.add_dataset(label, wl, result.delta_deg, color="#ff7f0e")

        # Fourier coefficients
        self._alpha_beta_plot.clear()
        self._alpha_beta_plot.add_dataset(f"{label} α", wl, result.alpha, color="#2ca02c")
        self._alpha_beta_plot.add_dataset(f"{label} β", wl, result.beta, color="#d62728", style="dash")

        self._chi_xi_plot.clear()
        self._chi_xi_plot.add_dataset(f"{label} χ", wl, result.chi, color="#9467bd")
        self._chi_xi_plot.add_dataset(f"{label} ξ", wl, result.xi, color="#8c564b", style="dash")

        # Energy conservation
        self._energy_plot.clear()
        self._energy_plot.add_dataset(label, wl, result.energy_conservation, color="#7f7f7f")

        # Update Mueller tab
        self._btn_export_mueller.setEnabled(result.mueller_matrix is not None)
        if result.mueller_matrix is not None:
            self._update_mueller_grid(result.mueller_matrix)

        n_wl = len(wl)
        self._lbl_status.setText(f"{label} — {n_wl} wavelengths")
        self._btn_export_csv.setEnabled(True)
        self._btn_export_png.setEnabled(True)

    def add_measured_data(
        self,
        wavelengths_nm: object,
        psi_deg: object,
        delta_deg: object,
    ) -> None:
        """Overlay measured data on the Ψ/Δ plots."""
        wl = np.asarray(wavelengths_nm)
        self._psi_plot.add_dataset(
            "Measured", wl, np.asarray(psi_deg), color="#e377c2", style="dot"
        )
        self._delta_plot.add_dataset(
            "Measured", wl, np.asarray(delta_deg), color="#17becf", style="dot"
        )

    @property
    def psi_plot(self) -> object:
        """Access the Ψ plot widget (for tests)."""
        return self._psi_plot

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_export_csv(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            out = Path(path)
            self._result.to_csv(out)
            self.export_requested.emit(out)

    def _on_export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "", "PNG Images (*.png);;All Files (*)"
        )
        if path:
            self._psi_plot.export_png(path)

    def _on_export_mueller_csv(self) -> None:
        """Handle Export Mueller CSV button click."""
        if self._result is None or self._result.mueller_matrix is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mueller CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self._result.export_mueller_csv(Path(path))

    def _on_mueller_cell_clicked(self, i: int, j: int) -> None:
        """Update the detail plot when a Mueller matrix cell is clicked."""
        if self._result is None or self._result.mueller_matrix is None:
            return
        label = f"m{i + 1}{j + 1}"
        self._mueller_detail_plot.clear()
        self._mueller_detail_plot.add_dataset(
            label,
            self._result.wavelengths_nm,
            self._result.mueller_matrix[:, i, j],
        )
        self._mueller_detail_plot.set_ylabel(label)
        self._mueller_detail_plot.set_xlabel("Wavelength (nm)")
        self._mueller_detail_plot.set_title(f"Mueller Element {label}(λ)")
        self._mueller_detail_plot._plot_widget.setYRange(-1.1, 1.1)

    def _update_mueller_grid(self, mueller_matrix: np.ndarray) -> None:
        """Update Mueller grid buttons with mean values across wavelengths."""
        for i in range(4):
            for j in range(4):
                btn = self._mueller_buttons[i][j]
                mean_val = float(np.mean(mueller_matrix[:, i, j]))
                if i == 0 and j == 0:
                    btn.setText("m11\n1.000")
                else:
                    btn.setText(f"m{i + 1}{j + 1}\n{mean_val:+.3f}")
