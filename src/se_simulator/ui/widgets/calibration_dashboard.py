"""CalibrationDashboard: widget for viewing and editing calibration error parameters."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import CalibrationErrors, SystemConfig


class CalibrationDashboard(QWidget):
    """Dashboard for calibration error parameters of the ellipsometer.

    Signals
    -------
    calibration_changed(CalibrationErrors):
        Emitted when the user applies changes.
    """

    calibration_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config: SystemConfig | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        errors_group = QGroupBox("Calibration Errors")
        errors_form = QFormLayout(errors_group)

        self._dp_spin = QDoubleSpinBox()
        self._dp_spin.setRange(-10.0, 10.0)
        self._dp_spin.setDecimals(4)
        self._dp_spin.setSuffix(" °")
        errors_form.addRow("ΔP (Polarizer offset):", self._dp_spin)

        self._da_spin = QDoubleSpinBox()
        self._da_spin.setRange(-10.0, 10.0)
        self._da_spin.setDecimals(4)
        self._da_spin.setSuffix(" °")
        errors_form.addRow("ΔA (Analyzer offset):", self._da_spin)

        self._dc_spin = QDoubleSpinBox()
        self._dc_spin.setRange(-10.0, 10.0)
        self._dc_spin.setDecimals(4)
        self._dc_spin.setSuffix(" °")
        errors_form.addRow("ΔC (Compensator offset):", self._dc_spin)

        layout.addWidget(errors_group)

        info_label = QLabel(
            "These parameters model systematic alignment errors in the "
            "polarizer, analyzer, and compensator positions."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self._btn_apply = QPushButton("Apply Calibration")
        self._btn_apply.clicked.connect(self._on_apply)
        layout.addWidget(self._btn_apply)

        self._btn_reset = QPushButton("Reset to Zero")
        self._btn_reset.clicked.connect(self._on_reset)
        layout.addWidget(self._btn_reset)

        layout.addStretch()

    def load_config(self, config: SystemConfig) -> None:
        """Populate from a SystemConfig."""
        self._config = config
        errors = config.calibration_errors
        self._dp_spin.setValue(errors.delta_P_deg)
        self._da_spin.setValue(errors.delta_A_deg)
        self._dc_spin.setValue(errors.delta_C_deg)

    def build_errors(self) -> CalibrationErrors:
        """Build CalibrationErrors from current UI values."""
        return CalibrationErrors(
            delta_P_deg=self._dp_spin.value(),
            delta_A_deg=self._da_spin.value(),
            delta_C_deg=self._dc_spin.value(),
        )

    def _on_apply(self) -> None:
        errors = self.build_errors()
        if self._config is not None:
            self._config = self._config.model_copy(
                update={"calibration_errors": errors}
            )
        self.calibration_changed.emit(errors)

    def _on_reset(self) -> None:
        self._dp_spin.setValue(0.0)
        self._da_spin.setValue(0.0)
        self._dc_spin.setValue(0.0)
