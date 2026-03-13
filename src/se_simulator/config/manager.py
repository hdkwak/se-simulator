"""ConfigManager: load/save YAML configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from se_simulator.config.schemas import SampleConfig, SimConditions, SystemConfig, WavelengthSpec


class ConfigValidationError(ValueError):
    """Raised when a config file fails Pydantic validation."""


class ConfigManager:
    """Load and save YAML config files as validated Pydantic models."""

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def load_system(self, path: Path) -> SystemConfig:
        """Load and validate a SystemConfig YAML file."""
        try:
            return SystemConfig.model_validate(self._load_yaml(path))
        except ValidationError as exc:
            raise ConfigValidationError(str(exc)) from exc

    def load_sample(self, path: Path) -> SampleConfig:
        """Load and validate a SampleConfig YAML file."""
        try:
            return SampleConfig.model_validate(self._load_yaml(path))
        except ValidationError as exc:
            raise ConfigValidationError(str(exc)) from exc

    def load_sim_conditions(self, path: Path) -> SimConditions:
        """Load and validate a SimConditions YAML file."""
        try:
            return SimConditions.model_validate(self._load_yaml(path))
        except ValidationError as exc:
            raise ConfigValidationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_system(self, config: SystemConfig, path: Path) -> None:
        """Serialise SystemConfig to YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    def save_sample(self, config: SampleConfig, path: Path) -> None:
        """Serialise SampleConfig to YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    def save_sim_conditions(self, config: SimConditions, path: Path) -> None:
        """Serialise SimConditions to YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Wavelength helper
    # ------------------------------------------------------------------

    def get_wavelengths(self, spec: WavelengthSpec) -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
        """Return a sorted numpy array of wavelengths from a WavelengthSpec."""
        import numpy as np

        if spec.explicit is not None:
            return np.sort(np.asarray(spec.explicit, dtype=float))
        if spec.range is not None:
            start, stop, step = spec.range
            return np.arange(start, stop + step * 0.5, step, dtype=float)
        msg = "WavelengthSpec must have either 'explicit' or 'range' set."
        raise ValueError(msg)
