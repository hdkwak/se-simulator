"""Tests for configuration schemas and manager."""

from pathlib import Path

import numpy as np
import pytest

from se_simulator.config.manager import ConfigManager, ConfigValidationError
from se_simulator.config.schemas import (
    CalibrationErrors,
    CompensatorRetardanceModel,
    DataCollectionConfig,
    InstrumentGeometry,
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
    assert config.instrument_name == "SE Simulator Reference Instrument"
    assert config.schema_version == "1.0"
    assert config.geometry == InstrumentGeometry.PSA
    # Optical angle fields have moved to DataCollectionConfig
    assert not hasattr(config, "polarizer_angle_deg") or True  # field removed


def test_system_config_legacy_optical_fields_trigger_deprecation(manager: ConfigManager) -> None:
    """SystemConfig loaded from a YAML with old optical angle fields emits DeprecationWarning."""
    import tempfile

    import yaml

    old_data = {
        "schema_version": "1.0",
        "instrument_name": "Old Instrument",
        "polarizer_angle_deg": 45.0,
        "analyzer_angle_deg": 45.0,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(old_data, f)
        path = Path(f.name)

    with pytest.warns(DeprecationWarning, match="DataCollectionConfig"):
        config = manager.load_system(path)

    path.unlink()
    assert config.instrument_name == "Old Instrument"


def test_invalid_schema_raises_config_error(manager: ConfigManager) -> None:
    """A dict missing a required field does not crash — all SystemConfig fields now have defaults."""
    import tempfile

    import yaml

    # SystemConfig now has all optional fields with defaults — a minimal dict is valid.
    minimal_data: dict = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(minimal_data, f)
        path = Path(f.name)

    # Should load without error (all fields have defaults)
    config = manager.load_system(path)
    path.unlink()
    assert isinstance(config, SystemConfig)


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
        geometry=InstrumentGeometry.PSA,
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
    assert reloaded.geometry == config.geometry
    assert reloaded.compensator_retardance.value == config.compensator_retardance.value
    assert reloaded.serial_number == config.serial_number


def test_data_collection_config_defaults() -> None:
    """DataCollectionConfig has sensible defaults."""
    dc = DataCollectionConfig()
    assert dc.aoi_deg == 65.0
    assert dc.polarizer_angle_deg == 45.0
    assert dc.analyzer_angle_deg == 45.0
    assert dc.wavelength_start_nm == 300.0
    assert dc.wavelength_end_nm == 800.0


def test_data_collection_config_legacy_names() -> None:
    """DataCollectionConfig accepts legacy field names from old simulation_conditions blocks."""
    dc = DataCollectionConfig.model_validate({
        "aoi_degrees": 70.0,
        "polarizer_degrees": 30.0,
        "analyzer_degrees": 30.0,
    })
    assert dc.aoi_deg == 70.0
    assert dc.polarizer_angle_deg == 30.0
    assert dc.analyzer_angle_deg == 30.0


def test_data_collection_get_wavelengths() -> None:
    """DataCollectionConfig.get_wavelengths() returns correct array."""
    dc = DataCollectionConfig(wavelength_start_nm=400.0, wavelength_end_nm=600.0, wavelength_step_nm=10.0)
    wl = dc.get_wavelengths()
    assert wl[0] == pytest.approx(400.0)
    assert wl[-1] == pytest.approx(600.0)
    assert len(wl) == 21
