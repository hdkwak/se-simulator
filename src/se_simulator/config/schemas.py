"""Pydantic v2 configuration schemas for SE-RCWA Simulator."""
from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Material spec
# ---------------------------------------------------------------------------

class MaterialSpec(BaseModel):
    """Optical material specification."""

    name: str
    source: Literal["library", "constant_nk", "cauchy", "sellmeier", "drude", "tauc_lorentz"]
    # library
    library_name: str | None = None
    # constant n/k
    n: float | None = None
    k: float | None = None
    # dispersion model coefficients
    coefficients: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Shape / layer geometry
# ---------------------------------------------------------------------------

class ShapeGeometry(BaseModel):
    """Geometry of a single shape region in a grating layer."""

    type: Literal["rectangle", "ellipse", "polygon"] = "rectangle"
    cx: float = 0.0
    cy: float = 0.0
    width: float = 100.0
    height: float = 100.0
    sidewall_angle_deg: float = 90.0
    vertices: list[list[float]] = Field(default_factory=list)  # for polygon


class ShapeRegion(BaseModel):
    """A filled region inside a grating layer."""

    geometry: ShapeGeometry = Field(default_factory=ShapeGeometry)
    material: str = "Si"


class GratingLayer(BaseModel):
    """A single layer in the sample stack."""

    name: str = "layer"
    type: Literal["uniform", "grating_1d", "grating_2d"] = "uniform"
    thickness_nm: float = 100.0
    Lx_nm: float = 500.0  # noqa: N815
    Ly_nm: float = 500.0  # noqa: N815
    background_material: str = "Air"
    shapes: list[ShapeRegion] = Field(default_factory=list)
    incoherent: bool = False


# ---------------------------------------------------------------------------
# Sample config
# ---------------------------------------------------------------------------

class SampleConfig(BaseModel):
    """Internal engine representation of a sample stack.

    .. deprecated::
        Use :class:`Stack` instead. ``SampleConfig`` is an internal type
        and will be removed from the public API in a future release.
        Pass ``Stack`` objects directly to ``RCWAEngine.run()`` and ``compute_tmm()``.
    """

    schema_version: str = "1.0"
    sample_id: str = "unnamed"
    Lx_nm: float = 500.0  # noqa: N815
    Ly_nm: float = 500.0  # noqa: N815
    superstrate_material: str = "Air"
    substrate_material: str = "Si"
    layers: list[GratingLayer] = Field(default_factory=list)
    materials: dict[str, MaterialSpec] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wavelength spec
# ---------------------------------------------------------------------------

class WavelengthSpec(BaseModel):
    """Either an explicit list or a (start, stop, step) range."""

    explicit: list[float] | None = None
    range: tuple[float, float, float] | None = None  # (start, stop, step)

    @model_validator(mode="after")
    def _check_one_set(self) -> WavelengthSpec:
        if self.explicit is None and self.range is None:
            msg = "WavelengthSpec: exactly one of 'explicit' or 'range' must be set."
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Instrument geometry enum
# ---------------------------------------------------------------------------


class InstrumentGeometry(str, Enum):
    """Optical element ordering of the ellipsometer.

    Determines which DataCollectionConfig fields are physically meaningful.
    """

    PSA = "PSA"    # Polarizer – Sample – Analyzer (no compensator)
    PSCA = "PSCA"  # Polarizer – Sample – Compensator – Analyzer (rotating compensator)
    PCSA = "PCSA"  # Polarizer – Compensator – Sample – Analyzer (rotating analyzer)


# ---------------------------------------------------------------------------
# Data collection config — optical + wavelength setup for a measurement
# ---------------------------------------------------------------------------


class DataCollectionConfig(BaseModel):
    """Per-measurement optical and wavelength settings.

    These fields belong to the measurement/experiment setup, not the simulation
    algorithm.  They are separated from SimConditions so that the same
    computational knobs (harmonics, Li factorisation, …) can be reused across
    measurements with different optical geometry.
    """

    model_config = ConfigDict(extra="forbid")

    aoi_deg: float = 65.0
    azimuth_deg: float = 0.0
    polarizer_angle_deg: float = 45.0
    analyzer_angle_deg: float = 45.0
    compensator_angle_deg: float = 0.0   # ignored when geometry == PSA
    wavelength_start_nm: float = 300.0
    wavelength_end_nm: float = 800.0
    wavelength_step_nm: float = 2.0

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_names(cls, data: Any) -> Any:
        """Accept old field names from legacy simulation_conditions blocks."""
        if not isinstance(data, dict):
            return data
        renames = {
            "aoi_degrees": "aoi_deg",
            "azimuth_degrees": "azimuth_deg",
            "polarizer_degrees": "polarizer_angle_deg",
            "analyzer_degrees": "analyzer_angle_deg",
        }
        for old, new in renames.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        return data

    def get_wavelengths(self) -> list[float]:
        """Return wavelength array as a plain Python list."""
        import numpy as np

        return list(
            np.arange(self.wavelength_start_nm, self.wavelength_end_nm + 1e-9, self.wavelength_step_nm)
        )


# ---------------------------------------------------------------------------
# Fitting conditions
# ---------------------------------------------------------------------------

class FittingConditions(BaseModel):
    """Parameters controlling the fitting pipeline."""

    fit_signals: list[str] = Field(default_factory=lambda: ["psi", "delta"])
    sigma_psi: float = 0.05
    sigma_delta: float = 0.1
    top_k_candidates: int = 10
    use_interpolation: bool = True
    use_refinement: bool = False
    refinement_algo: str = "Nelder-Mead"
    max_iterations: int = 200
    convergence_tol: float = 1e-5
    wavelength_mask: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Simulation conditions
# ---------------------------------------------------------------------------

class SimConditions(BaseModel):
    """Simulation run conditions."""

    schema_version: str = "1.0"
    aoi_deg: float = 65.0
    azimuth_deg: float = 0.0
    wavelengths: WavelengthSpec = Field(
        default_factory=lambda: WavelengthSpec(range=(300.0, 800.0, 2.0))
    )
    n_harmonics_x: int = 5
    n_harmonics_y: int = 5
    li_factorization: bool = True
    parallel_wavelengths: bool = False
    output_jones: bool = False
    output_orders: bool = False
    fitting: FittingConditions = Field(default_factory=FittingConditions)
    engine_override: Literal["auto", "tmm", "rcwa"] = "auto"


# ---------------------------------------------------------------------------
# Calibration errors
# ---------------------------------------------------------------------------

class CalibrationErrors(BaseModel):
    """Systematic offset errors in the optical components."""

    delta_P_deg: float = 0.0  # noqa: N815
    delta_A_deg: float = 0.0  # noqa: N815
    delta_C_deg: float = 0.0  # noqa: N815
    delta_retardance_deg: float = 0.0  # noqa: N815


# ---------------------------------------------------------------------------
# Compensator retardance model
# ---------------------------------------------------------------------------

class CompensatorRetardanceModel(BaseModel):
    """Model for wavelength-dependent compensator retardance."""

    type: Literal["constant", "polynomial", "tabulated"] = "constant"
    value: float = 90.0           # used when type=="constant"
    coefficients: list[float] = Field(default_factory=list)  # ascending order
    file_path: str | None = None  # for tabulated


# ---------------------------------------------------------------------------
# Depolarization config
# ---------------------------------------------------------------------------

class DepolarizationConfig(BaseModel):
    """Incoherent averaging parameters."""

    aoi_spread_deg: float = 0.0
    wavelength_bandwidth_nm: float = 0.0


# ---------------------------------------------------------------------------
# System config
# ---------------------------------------------------------------------------

class SystemConfig(BaseModel):
    """Instrument (ellipsometer) hardware configuration.

    Only fields that describe fixed properties of the instrument belong here.
    Per-measurement optical settings (AOI, polarizer/analyzer/compensator angles,
    wavelength range) now live in :class:`DataCollectionConfig`.
    """

    schema_version: str = "1.0"
    instrument_name: str = "SE Simulator Reference Instrument"
    serial_number: str = ""
    geometry: InstrumentGeometry = InstrumentGeometry.PSA
    compensator_retardance: CompensatorRetardanceModel = Field(
        default_factory=CompensatorRetardanceModel
    )
    n_revolutions: int = 20
    n_points_per_revolution: int = 50
    calibration_errors: CalibrationErrors = Field(default_factory=CalibrationErrors)
    depolarization: DepolarizationConfig = Field(default_factory=DepolarizationConfig)

    @model_validator(mode="before")
    @classmethod
    def _absorb_legacy_optical_fields(cls, data: Any) -> Any:
        """Drop per-measurement angle fields that have moved to DataCollectionConfig."""
        if not isinstance(data, dict):
            return data
        _MOVED = ("polarizer_angle_deg", "analyzer_angle_deg", "compensator_angle_deg")
        found = [k for k in _MOVED if k in data]
        if found:
            warnings.warn(
                f"SystemConfig: {found} have moved to DataCollectionConfig. "
                "Remove them from system_config.yaml and add a data_collection "
                "block to your recipe instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            for k in found:
                data.pop(k)
        return data

    @classmethod
    def default(cls) -> SystemConfig:
        """Return a zero-calibration-error SystemConfig for use when no file is provided."""
        return cls(instrument_name="SE Simulator (no system config)")


# ---------------------------------------------------------------------------
# Stack — first-class sample representation (Phase 1 addition)
# ---------------------------------------------------------------------------

class StackLayer(BaseModel):
    """A single layer in a Stack, owning its MaterialSpec directly."""

    model_config = ConfigDict(extra="forbid")

    name: str = "layer"
    type: Literal["uniform", "grating_1d", "grating_2d"] = "uniform"
    thickness_nm: float = 100.0
    Lx_nm: float = 500.0  # noqa: N815
    Ly_nm: float = 500.0  # noqa: N815
    material: MaterialSpec
    shapes: list[ShapeRegion] = Field(default_factory=list)
    incoherent: bool = False


class Stack(BaseModel):
    """First-class sample representation that bridges to SampleConfig for engine use."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    sample_id: str = "unnamed"
    superstrate: MaterialSpec
    substrate: MaterialSpec
    layers: list[StackLayer] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_sample_config(self) -> SampleConfig:
        """Convert to SampleConfig for consumption by the RCWA engine.

        .. deprecated::
            Pass ``Stack`` directly to ``RCWAEngine.run()`` or ``compute_tmm()``.
            ``SampleConfig`` will be removed from the public API in a future release.
        """
        import warnings

        warnings.warn(
            "Stack.to_sample_config() is deprecated. Pass Stack directly to "
            "RCWAEngine.run() or compute_tmm(). SampleConfig will be removed "
            "from the public API in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

        def _mat_key(mat: MaterialSpec) -> str:
            return mat.library_name if mat.library_name is not None else mat.name

        materials: dict[str, MaterialSpec] = {}
        materials[_mat_key(self.superstrate)] = self.superstrate
        materials[_mat_key(self.substrate)] = self.substrate

        layers_out: list[GratingLayer] = []
        for sl in self.layers:
            mat_key = _mat_key(sl.material)
            materials[mat_key] = sl.material
            gl = GratingLayer(
                name=sl.name,
                type=sl.type,
                thickness_nm=sl.thickness_nm,
                Lx_nm=sl.Lx_nm,
                Ly_nm=sl.Ly_nm,
                background_material=mat_key,
                shapes=sl.shapes,
                incoherent=sl.incoherent,
            )
            layers_out.append(gl)

        return SampleConfig(
            schema_version=self.schema_version,
            sample_id=self.sample_id,
            superstrate_material=_mat_key(self.superstrate),
            substrate_material=_mat_key(self.substrate),
            layers=layers_out,
            materials=materials,
            metadata=self.metadata,
        )
