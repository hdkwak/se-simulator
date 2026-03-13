"""MaterialEditor: widget for viewing and editing material optical constants."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import MaterialSpec
from se_simulator.materials.database import MaterialDatabase


class MaterialEditor(QWidget):
    """Editor for a single MaterialSpec with live n/k preview.

    Signals
    -------
    material_changed(MaterialSpec):
        Emitted when the user modifies material parameters.
    """

    material_changed = Signal(object)

    _SOURCES = ["constant_nk", "tabulated_file", "library", "dispersion_model"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = MaterialDatabase()
        self._spec: MaterialSpec | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Source selection
        src_group = QGroupBox("Material Source")
        src_form = QFormLayout(src_group)

        self._name_edit = QLineEdit()
        src_form.addRow("Name:", self._name_edit)

        self._source_combo = QComboBox()
        self._source_combo.addItems(self._SOURCES)
        src_form.addRow("Source:", self._source_combo)

        layout.addWidget(src_group)

        # Stacked parameter panels per source type
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        self._stack = QStackedWidget()

        # constant_nk panel
        cnk_panel = QWidget()
        cnk_form = QFormLayout(cnk_panel)
        self._n_spin = QDoubleSpinBox()
        self._n_spin.setRange(0.0, 100.0)
        self._n_spin.setDecimals(4)
        self._n_spin.setValue(1.5)
        self._k_spin = QDoubleSpinBox()
        self._k_spin.setRange(0.0, 100.0)
        self._k_spin.setDecimals(4)
        self._k_spin.setValue(0.0)
        cnk_form.addRow("n:", self._n_spin)
        cnk_form.addRow("k:", self._k_spin)
        self._stack.addWidget(cnk_panel)  # index 0

        # tabulated_file panel
        tab_panel = QWidget()
        tab_form = QFormLayout(tab_panel)
        self._file_path_edit = QLineEdit()
        self._btn_browse_file = QPushButton("Browse…")
        file_row = QHBoxLayout()
        file_row.addWidget(self._file_path_edit)
        file_row.addWidget(self._btn_browse_file)
        tab_form.addRow("File Path:", file_row)
        self._stack.addWidget(tab_panel)  # index 1

        # library panel
        lib_panel = QWidget()
        lib_form = QFormLayout(lib_panel)
        self._lib_name_combo = QComboBox()
        lib_mats = self._db.list_library_materials()
        self._lib_name_combo.addItems(lib_mats)
        lib_form.addRow("Library Material:", self._lib_name_combo)
        self._stack.addWidget(lib_panel)  # index 2

        # dispersion_model panel
        disp_panel = QWidget()
        disp_form = QFormLayout(disp_panel)
        self._disp_model_combo = QComboBox()
        self._disp_model_combo.addItems(["cauchy", "sellmeier", "drude", "tauc_lorentz"])
        self._disp_params_edit = QLineEdit()
        self._disp_params_edit.setPlaceholderText("e.g. 1.5, 0.003")
        disp_form.addRow("Model:", self._disp_model_combo)
        disp_form.addRow("Parameters:", self._disp_params_edit)
        self._stack.addWidget(disp_panel)  # index 3

        params_layout.addWidget(self._stack)
        layout.addWidget(params_group)

        # n/k preview plot
        preview_group = QGroupBox("n/k Preview")
        preview_layout = QVBoxLayout(preview_group)

        from se_simulator.ui.plots.nk_plot import NKPlot

        self._nk_plot = NKPlot()
        preview_layout.addWidget(self._nk_plot)

        self._btn_preview = QPushButton("Refresh Preview")
        preview_layout.addWidget(self._btn_preview)
        layout.addWidget(preview_group)

        # Connect signals
        self._source_combo.currentIndexChanged.connect(self._stack.setCurrentIndex)
        self._btn_preview.clicked.connect(self._refresh_preview)
        self._btn_browse_file.clicked.connect(self._browse_file)

    def load_spec(self, spec: MaterialSpec) -> None:
        """Populate the editor from a MaterialSpec."""
        self._spec = spec
        self._name_edit.setText(spec.name)
        idx = self._SOURCES.index(spec.source) if spec.source in self._SOURCES else 0
        self._source_combo.setCurrentIndex(idx)

        if spec.source == "constant_nk":
            self._n_spin.setValue(spec.n or 1.5)
            self._k_spin.setValue(spec.k or 0.0)
        elif spec.source == "tabulated_file":
            self._file_path_edit.setText(spec.file_path or "")
        elif spec.source == "library":
            lib_name = spec.library_name or spec.name
            combo_idx = self._lib_name_combo.findText(lib_name)
            if combo_idx >= 0:
                self._lib_name_combo.setCurrentIndex(combo_idx)
        elif spec.source == "dispersion_model":
            model_idx = self._disp_model_combo.findText(spec.dispersion_model or "cauchy")
            if model_idx >= 0:
                self._disp_model_combo.setCurrentIndex(model_idx)
            self._disp_params_edit.setText(", ".join(str(p) for p in spec.dispersion_params))

    def _refresh_preview(self) -> None:
        """Rebuild the spec and refresh the n/k plot preview."""
        import numpy as np

        spec = self._build_spec()
        if spec is None:
            return

        try:
            self._db.resolve(spec)
            wl = np.linspace(200, 1000, 200)
            n, k = self._db.get_nk(spec.name, wl)
            self._nk_plot.clear()
            self._nk_plot.add_material(spec.name, wl, n, k, color="#1f77b4")
            self.material_changed.emit(spec)
        except Exception:  # noqa: BLE001
            pass

    def _build_spec(self) -> MaterialSpec | None:
        """Construct a MaterialSpec from current UI values."""
        try:
            src = self._SOURCES[self._source_combo.currentIndex()]
            name = self._name_edit.text().strip() or "Material"

            if src == "constant_nk":
                return MaterialSpec(
                    name=name,
                    source="constant_nk",
                    n=self._n_spin.value(),
                    k=self._k_spin.value(),
                )
            elif src == "tabulated_file":
                return MaterialSpec(
                    name=name,
                    source="tabulated_file",
                    file_path=self._file_path_edit.text().strip(),
                )
            elif src == "library":
                return MaterialSpec(
                    name=name,
                    source="library",
                    library_name=self._lib_name_combo.currentText(),
                )
            else:
                params_str = self._disp_params_edit.text().strip()
                params = [float(p.strip()) for p in params_str.split(",") if p.strip()]
                return MaterialSpec(
                    name=name,
                    source="dispersion_model",
                    dispersion_model=self._disp_model_combo.currentText(),
                    dispersion_params=params,
                )
        except Exception:  # noqa: BLE001
            return None

    def _browse_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Open n,k File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._file_path_edit.setText(path)
