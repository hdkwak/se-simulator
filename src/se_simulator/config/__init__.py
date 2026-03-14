"""Configuration system for SE-RCWA Simulator."""

from se_simulator.config.manager import ConfigManager, ConfigValidationError
from se_simulator.config.schemas import (
    CalibrationErrors,
    CompensatorRetardanceModel,
    DepolarizationConfig,
    FittingConditions,
    GratingLayer,
    MaterialSpec,
    SampleConfig,
    ShapeGeometry,
    ShapeRegion,
    SimConditions,
    Stack,
    StackLayer,
    SystemConfig,
    WavelengthSpec,
)

__all__ = [
    "CalibrationErrors",
    "CompensatorRetardanceModel",
    "ConfigManager",
    "ConfigValidationError",
    "DepolarizationConfig",
    "FittingConditions",
    "GratingLayer",
    "MaterialSpec",
    "SampleConfig",
    "ShapeGeometry",
    "ShapeRegion",
    "SimConditions",
    "Stack",
    "StackLayer",
    "SystemConfig",
    "WavelengthSpec",
]
