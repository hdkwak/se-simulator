"""Shared fixtures for SE-RCWA Simulator tests."""

from pathlib import Path

import pytest

DEFAULTS_DIR = (
    Path(__file__).parent.parent
    / "src"
    / "se_simulator"
    / "config"
    / "defaults"
)


@pytest.fixture()
def defaults_dir() -> Path:
    """Return path to the default config directory."""
    return DEFAULTS_DIR


@pytest.fixture()
def system_config_path(defaults_dir: Path) -> Path:
    return defaults_dir / "system_config.yaml"


@pytest.fixture()
def sample_config_path(defaults_dir: Path) -> Path:
    return defaults_dir / "sample_config.yaml"


@pytest.fixture()
def sim_conditions_path(defaults_dir: Path) -> Path:
    return defaults_dir / "sim_conditions.yaml"
