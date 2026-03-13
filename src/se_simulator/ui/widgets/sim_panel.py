"""SimulationPanel: editor for SimConditions and Run/Stop controls."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from se_simulator.config.schemas import SimConditions

try:
    from se_simulator.rcwa.dispatcher import select_engine as _select_engine

    _HAS_DISPATCHER = True
except ImportError:
    _HAS_DISPATCHER = False


class SimulationPanel(QWidget):
    """Panel for configuring and launching RCWA simulations.

    Signals
    -------
    run_requested():
        Emitted when the user clicks Run.
    stop_requested():
        Emitted when the user clicks Stop.
    settings_changed(SimConditions):
        Emitted when any setting is changed by the user.
    """

    run_requested = Signal()
    stop_requested = Signal()
    settings_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sim: SimConditions | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Run controls
        run_group = QGroupBox("Run Controls")
        run_layout = QHBoxLayout(run_group)
        self._btn_run = QPushButton("RUN Simulation")
        self._btn_run.setObjectName("run_button")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #2ecc71; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #27ae60; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._engine_indicator = QLabel("Engine: \u2014 ")
        run_layout.addWidget(self._engine_indicator)
        run_layout.addStretch()
        run_layout.addWidget(self._btn_run)
        run_layout.addWidget(self._btn_stop)
        layout.addWidget(run_group)

        # Simulation conditions
        cond_group = QGroupBox("Simulation Conditions")
        form = QFormLayout(cond_group)

        self._aoi_spin = QDoubleSpinBox()
        self._aoi_spin.setRange(0.0, 89.9)
        self._aoi_spin.setDecimals(2)
        self._aoi_spin.setSuffix(" °")
        self._aoi_spin.setValue(65.0)
        form.addRow("Angle of Incidence:", self._aoi_spin)

        self._az_spin = QDoubleSpinBox()
        self._az_spin.setRange(0.0, 360.0)
        self._az_spin.setDecimals(2)
        self._az_spin.setSuffix(" °")
        self._az_spin.setValue(0.0)
        form.addRow("Azimuth Angle:", self._az_spin)

        self._nx_spin = QSpinBox()
        self._nx_spin.setRange(1, 20)
        self._nx_spin.setValue(5)
        form.addRow("Harmonics X:", self._nx_spin)

        self._ny_spin = QSpinBox()
        self._ny_spin.setRange(1, 20)
        self._ny_spin.setValue(5)
        form.addRow("Harmonics Y:", self._ny_spin)

        layout.addWidget(cond_group)

        # Wavelength range
        wl_group = QGroupBox("Wavelength Range")
        wl_form = QFormLayout(wl_group)

        self._wl_start = QDoubleSpinBox()
        self._wl_start.setRange(100.0, 10000.0)
        self._wl_start.setDecimals(1)
        self._wl_start.setSuffix(" nm")
        self._wl_start.setValue(300.0)
        wl_form.addRow("Start:", self._wl_start)

        self._wl_stop = QDoubleSpinBox()
        self._wl_stop.setRange(100.0, 10000.0)
        self._wl_stop.setDecimals(1)
        self._wl_stop.setSuffix(" nm")
        self._wl_stop.setValue(800.0)
        wl_form.addRow("Stop:", self._wl_stop)

        self._wl_step = QDoubleSpinBox()
        self._wl_step.setRange(0.1, 100.0)
        self._wl_step.setDecimals(1)
        self._wl_step.setSuffix(" nm")
        self._wl_step.setValue(10.0)
        wl_form.addRow("Step:", self._wl_step)

        layout.addWidget(wl_group)

        # Advanced options (collapsed by default)
        adv_group = QGroupBox("Advanced")
        adv_group.setCheckable(True)
        adv_group.setChecked(False)
        adv_form = QFormLayout(adv_group)

        self._chk_parallel = QCheckBox("Parallel wavelengths")
        self._chk_parallel.setChecked(True)
        adv_form.addRow(self._chk_parallel)

        self._chk_li = QCheckBox("Li factorization")
        self._chk_li.setChecked(True)
        adv_form.addRow(self._chk_li)

        layout.addWidget(adv_group)
        layout.addStretch()

        # Connect signals
        self._btn_run.clicked.connect(self.run_requested)
        self._btn_stop.clicked.connect(self.stop_requested)

    def load_sim(self, sim: SimConditions) -> None:
        """Populate controls from a SimConditions object."""
        self._sim = sim
        self._aoi_spin.setValue(sim.aoi_deg)
        self._az_spin.setValue(sim.azimuth_deg)
        self._nx_spin.setValue(sim.n_harmonics_x)
        self._ny_spin.setValue(sim.n_harmonics_y)
        if sim.wavelengths.range is not None:
            start, stop, step = sim.wavelengths.range
            self._wl_start.setValue(start)
            self._wl_stop.setValue(stop)
            self._wl_step.setValue(step)
        self._chk_parallel.setChecked(sim.parallel_wavelengths)
        self._chk_li.setChecked(sim.li_factorization)

    def build_sim(self) -> SimConditions:
        """Build a SimConditions from current UI values."""
        from se_simulator.config.schemas import WavelengthSpec

        wl_spec = WavelengthSpec(
            range=(
                self._wl_start.value(),
                self._wl_stop.value(),
                self._wl_step.value(),
            )
        )
        base = self._sim or SimConditions(
            aoi_deg=65.0,
            azimuth_deg=0.0,
            wavelengths=wl_spec,
        )
        return base.model_copy(
            update={
                "aoi_deg": self._aoi_spin.value(),
                "azimuth_deg": self._az_spin.value(),
                "n_harmonics_x": self._nx_spin.value(),
                "n_harmonics_y": self._ny_spin.value(),
                "wavelengths": wl_spec,
                "parallel_wavelengths": self._chk_parallel.isChecked(),
                "li_factorization": self._chk_li.isChecked(),
            }
        )

    def set_running(self, running: bool) -> None:
        """Toggle the run/stop button states."""
        self._btn_run.setEnabled(not running)
        self._btn_stop.setEnabled(running)

    def update_engine_indicator(
        self,
        sample_config: object,
        sim_conditions: object | None = None,
    ) -> None:
        """Update the engine indicator label based on sample and sim conditions."""
        override = getattr(sim_conditions, "engine_override", "auto")
        if _HAS_DISPATCHER:
            engine = _select_engine(sample_config, override)
        else:
            has_shapes = any(
                getattr(layer, "shapes", [])
                for layer in getattr(sample_config, "layers", [])
            )
            engine = "rcwa" if has_shapes else "tmm"
        qualifier = "(override)" if override != "auto" else "(auto)"
        self._engine_indicator.setText(f"Engine: {engine.upper()} {qualifier}")
