"""SystemConfigEditor: widget for editing SystemConfig parameters."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import SystemConfig


class SystemConfigEditor(QWidget):
    """Editor widget for the SystemConfig (instrument parameters).

    Signals
    -------
    config_changed(SystemConfig):
        Emitted when the user modifies and applies configuration.
    """

    config_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config: SystemConfig | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Instrument identity
        ident_group = QGroupBox("Instrument Identity")
        ident_form = QFormLayout(ident_group)
        self._name_edit = QLineEdit()
        self._serial_edit = QLineEdit()
        ident_form.addRow("Instrument Name:", self._name_edit)
        ident_form.addRow("Serial Number:", self._serial_edit)
        layout.addWidget(ident_group)

        # Optical angles
        angle_group = QGroupBox("Optical Angles")
        angle_form = QFormLayout(angle_group)
        self._polarizer_spin = QDoubleSpinBox()
        self._polarizer_spin.setRange(-360.0, 360.0)
        self._polarizer_spin.setDecimals(2)
        self._polarizer_spin.setSuffix(" °")

        self._analyzer_spin = QDoubleSpinBox()
        self._analyzer_spin.setRange(-360.0, 360.0)
        self._analyzer_spin.setDecimals(2)
        self._analyzer_spin.setSuffix(" °")

        self._compensator_spin = QDoubleSpinBox()
        self._compensator_spin.setRange(-360.0, 360.0)
        self._compensator_spin.setDecimals(2)
        self._compensator_spin.setSuffix(" °")

        angle_form.addRow("Polarizer P:", self._polarizer_spin)
        angle_form.addRow("Analyzer A:", self._analyzer_spin)
        angle_form.addRow("Compensator C:", self._compensator_spin)
        layout.addWidget(angle_group)

        # Compensator retardance
        ret_group = QGroupBox("Compensator Retardance")
        ret_form = QFormLayout(ret_group)
        self._retardance_spin = QDoubleSpinBox()
        self._retardance_spin.setRange(0.0, 360.0)
        self._retardance_spin.setDecimals(2)
        self._retardance_spin.setSuffix(" °")
        self._retardance_spin.setValue(90.0)
        ret_form.addRow("Retardance (constant):", self._retardance_spin)
        layout.addWidget(ret_group)

        # Measurement parameters
        meas_group = QGroupBox("Measurement")
        meas_form = QFormLayout(meas_group)
        self._n_rev_spin = QSpinBox()
        self._n_rev_spin.setRange(1, 1000)
        self._n_rev_spin.setValue(20)
        self._n_pts_spin = QSpinBox()
        self._n_pts_spin.setRange(10, 10000)
        self._n_pts_spin.setValue(50)
        meas_form.addRow("Revolutions:", self._n_rev_spin)
        meas_form.addRow("Points/Revolution:", self._n_pts_spin)
        layout.addWidget(meas_group)

        # Apply button
        self._btn_apply = QPushButton("Apply Changes")
        self._btn_apply.clicked.connect(self._on_apply)
        layout.addWidget(self._btn_apply)
        layout.addStretch()

    def load_config(self, config: SystemConfig) -> None:
        """Populate the editor from a SystemConfig."""
        self._config = config
        self._name_edit.setText(config.instrument_name)
        self._serial_edit.setText(config.serial_number)
        self._polarizer_spin.setValue(config.polarizer_angle_deg)
        self._analyzer_spin.setValue(config.analyzer_angle_deg)
        self._compensator_spin.setValue(config.compensator_angle_deg)
        if config.compensator_retardance.type == "constant":
            val = config.compensator_retardance.value
            self._retardance_spin.setValue(val if val is not None else 90.0)
        self._n_rev_spin.setValue(config.n_revolutions)
        self._n_pts_spin.setValue(config.n_points_per_revolution)

    def build_config(self) -> SystemConfig | None:
        """Construct a SystemConfig from current UI values."""
        if self._config is None:
            return None
        from se_simulator.config.schemas import CompensatorRetardanceModel

        retardance = CompensatorRetardanceModel(
            type="constant",
            value=self._retardance_spin.value(),
        )
        return self._config.model_copy(
            update={
                "instrument_name": self._name_edit.text().strip(),
                "serial_number": self._serial_edit.text().strip(),
                "polarizer_angle_deg": self._polarizer_spin.value(),
                "analyzer_angle_deg": self._analyzer_spin.value(),
                "compensator_angle_deg": self._compensator_spin.value(),
                "compensator_retardance": retardance,
                "n_revolutions": self._n_rev_spin.value(),
                "n_points_per_revolution": self._n_pts_spin.value(),
            }
        )

    def _on_apply(self) -> None:
        config = self.build_config()
        if config is not None:
            self._config = config
            self.config_changed.emit(config)
