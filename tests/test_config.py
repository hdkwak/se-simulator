"""Tests for configuration schemas and manager."""

from pathlib import Path

import numpy as np
import pytest

from se_simulator.config.manager import ConfigManager, ConfigValidationError
from se_simulator.config.schemas import (
    CalibrationErrors,
    CompensatorRetardanceModel,
    SimConditions,
    SystemConfig,
    WavelengthSpec,
)


@pytest.fixture()
def manager() -> ConfigManager:
    return ConfigManager()


def test_system_config_loads_default(manager: ConfigManager, system_config_path: Path) -> None:
    """Default system_config.yaml loads without error and has expected field values."""
    config = manager.load_system(system_config_path)
    assert config.polarizer_angle_deg == 45.0
    assert config.instrument_name == "SE Simulator Reference Instrument"
    assert config.schema_version == "1.0"


def test_invalid_schema_raises_config_error(manager: ConfigManager) -> None:
    """A dict missing a required field raises ConfigValidationError."""
    import tempfile
    import yaml

    bad_data = {
        "schema_version": "1.0",
        # Missing: instrument_name, polarizer_angle_deg, analyzer_angle_deg,
        #          compensator_angle_deg, compensator_retardance
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(bad_data, f)
        bad_path = Path(f.name)

    with pytest.raises(ConfigValidationError) as exc_info:
        manager.load_system(bad_path)

    bad_path.unlink()
    # Error message must mention at least one failing field
    assert "instrument_name" in str(exc_info.value) or "polarizer_angle_deg" in str(exc_info.value)


def test_wavelength_spec_range(manager: ConfigManager) -> None:
    """WavelengthSpec with range produces an array of the correct length and bounds."""
    spec = WavelengthSpec(range=(300.0, 800.0, 2.0))
    wavelengths = manager.get_wavelengths(spec)

    assert isinstance(wavelengths, np.ndarray)
    assert wavelengths[0] == pytest.approx(300.0)
    assert wavelengths[-1] == pytest.approx(800.0, abs=2.0)
    # Expect 251 points for 300..800 step 2
    assert len(wavelengths) == pytest.approx(251, abs=2)


def test_wavelength_spec_explicit(manager: ConfigManager) -> None:
    """WavelengthSpec with explicit list returns sorted values."""
    values = [700.0, 400.0, 550.0, 300.0]
    spec = WavelengthSpec(explicit=values)
    wavelengths = manager.get_wavelengths(spec)

    assert list(wavelengths) == sorted(values)


def test_config_round_trip(manager: ConfigManager, tmp_path: Path) -> None:
    """Save a SystemConfig to disk, reload it, and assert equality."""
    config = SystemConfig(
        instrument_name="Test Instrument",
        serial_number="TEST-42",
        polarizer_angle_deg=45.0,
        analyzer_angle_deg=45.0,
        compensator_angle_deg=0.0,
        compensator_retardance=CompensatorRetardanceModel(
            type="constant",
            value=90.0,
            coefficients=[],
            file_path=None,
        ),
        calibration_errors=CalibrationErrors(),
    )
    out_path = tmp_path / "round_trip.yaml"
    manager.save_system(config, out_path)

    reloaded = manager.load_system(out_path)
    assert reloaded.instrument_name == config.instrument_name
    assert reloaded.polarizer_angle_deg == config.polarizer_angle_deg
    assert reloaded.compensator_retardance.value == config.compensator_retardance.value
    assert reloaded.serial_number == config.serial_number
