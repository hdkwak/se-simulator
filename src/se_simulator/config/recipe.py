"""Pydantic v2 models for Simulation and Measurement Recipes."""
from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from se_simulator.config.schemas import DataCollectionConfig, MaterialSpec, Stack, StackLayer, WavelengthSpec


class RecipeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe_type: Literal["simulation", "measurement"]
    version: str = "1.0"
    created: str | None = None
    author: str = ""
    description: str = ""
    material_db_version: str = ""


class SampleRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inline: dict[str, Any] | None = None
    ref: str | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> SampleRef:
        if (self.inline is None) == (self.ref is None):
            raise ValueError("Exactly one of 'inline' or 'ref' must be provided")
        return self


class StackRef(BaseModel):
    """Thin envelope referencing a Stack either inline or by file path."""

    model_config = ConfigDict(extra="forbid")

    inline: Stack | None = None
    ref: str | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> StackRef:
        if (self.inline is None) == (self.ref is None):
            raise ValueError("Exactly one of 'inline' or 'ref' must be provided")
        return self


# ---------------------------------------------------------------------------
# Helper: convert a legacy SampleRef to a StackRef
# ---------------------------------------------------------------------------


def _inline_dict_to_stack(inline: dict[str, Any]) -> Stack:
    """Convert a raw inline dict (old SampleRef.inline format) to a Stack."""
    # Parse superstrate
    sup = inline.get("superstrate") or {}
    sup_name = sup.get("library_name", "Air")
    sup_spec = MaterialSpec(library_name=sup_name, source="library", name=sup_name)

    # Parse substrate
    sub = inline.get("substrate") or {}
    sub_name = sub.get("library_name", "Si")
    sub_spec = MaterialSpec(library_name=sub_name, source="library", name=sub_name)

    # Parse layers
    layers: list[StackLayer] = []
    for ld in inline.get("layers", []):
        mat_dict = ld.get("material") or {}
        mat_name = mat_dict.get("library_name") or ld.get("background_material", "Air")
        mat_spec = MaterialSpec(library_name=mat_name, source="library", name=mat_name)
        sl = StackLayer(
            name=ld.get("name", "Layer"),
            type=ld.get("type", "uniform"),
            thickness_nm=float(ld.get("thickness_nm", 100.0)),
            Lx_nm=float(ld.get("Lx_nm", 500.0)),
            Ly_nm=float(ld.get("Ly_nm", 500.0)),
            material=mat_spec,
            shapes=ld.get("shapes", []),
            incoherent=bool(ld.get("incoherent", False)),
        )
        layers.append(sl)

    return Stack(superstrate=sup_spec, substrate=sub_spec, layers=layers)


def _sampleref_to_stackref(sample_ref: SampleRef) -> StackRef:
    """Convert a legacy SampleRef to a StackRef."""
    if sample_ref.ref is not None:
        return StackRef(ref=sample_ref.ref)
    # Convert inline dict to Stack
    inline = sample_ref.inline or {}
    stack = _inline_dict_to_stack(inline)
    return StackRef(inline=stack)


# ---------------------------------------------------------------------------
# Simulation conditions embed
# ---------------------------------------------------------------------------


class SimulationConditionsEmbed(BaseModel):
    """Algorithm-only simulation knobs embedded in a recipe.

    Optical and wavelength settings (AOI, polarizer/analyzer angles, wavelength
    range) now live in :class:`DataCollectionConfig`.  Legacy YAMLs that still
    carry those fields under ``simulation_conditions`` are silently promoted to
    ``data_collection`` by the parent model validator.
    """

    model_config = ConfigDict(extra="ignore")

    # Algorithm parameters
    n_harmonics_x: int = 5
    n_harmonics_y: int = 5
    li_factorization: bool = True
    parallel_wavelengths: bool = False
    output_jones: bool = False
    output_orders: bool = False
    engine_override: Literal["auto", "rcwa", "tmm"] = "auto"


class SimulationRecipeOutputOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_psi_delta: bool = True
    save_jones: bool = False
    save_mueller: bool = False


def _extract_legacy_data_collection(data: dict) -> dict:
    """Pull optical/wavelength fields out of simulation_conditions into data_collection.

    Mutates *data* in-place and returns it.  Emits DeprecationWarning when
    any migration happens.
    """
    sc = data.get("simulation_conditions")
    if not isinstance(sc, dict):
        return data

    _OPTICAL_KEYS = {
        "wavelength_start_nm", "wavelength_end_nm", "wavelength_step_nm",
        "aoi_degrees", "azimuth_degrees", "polarizer_degrees", "analyzer_degrees",
    }
    found = {k: sc.pop(k) for k in list(sc) if k in _OPTICAL_KEYS}
    if not found:
        return data

    warnings.warn(
        "Recipe simulation_conditions contains deprecated optical/wavelength fields "
        f"({list(found)}). Move them to a data_collection block instead.",
        DeprecationWarning,
        stacklevel=5,
    )

    if "data_collection" not in data:
        data["data_collection"] = {}
    dc = data["data_collection"]

    # Renames: legacy → DataCollectionConfig field names
    _RENAME = {
        "aoi_degrees": "aoi_deg",
        "azimuth_degrees": "azimuth_deg",
        "polarizer_degrees": "polarizer_angle_deg",
        "analyzer_degrees": "analyzer_angle_deg",
    }
    for old, val in found.items():
        new_key = _RENAME.get(old, old)
        if new_key not in dc:
            dc[new_key] = val

    return data


class SimulationRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: RecipeMetadata
    # Legacy field — kept for backward compatibility; new field is `stack`
    sample: SampleRef | None = None
    # New first-class field
    stack: StackRef | None = None
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    simulation_conditions: SimulationConditionsEmbed = Field(
        default_factory=SimulationConditionsEmbed
    )
    output_options: SimulationRecipeOutputOptions = Field(
        default_factory=SimulationRecipeOutputOptions
    )

    @model_validator(mode="before")
    @classmethod
    def _promote_legacy_optical_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _extract_legacy_data_collection(data)
        return data

    @model_validator(mode="after")
    def _migrate_sample_to_stack(self) -> SimulationRecipe:
        """Auto-migrate legacy `sample` key to `stack`."""
        if self.stack is None and self.sample is not None:
            self.stack = _sampleref_to_stackref(self.sample)
        elif self.stack is None and self.sample is None:
            raise ValueError("Either 'stack' or 'sample' must be provided")
        return self


class FloatingParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    target_field: str
    """Dot-path rooted at the Stack object when using the new stack-based API.

    Use short paths like 'layers[0].thickness_nm'. The legacy prefix
    'forward_model.sample.inline.' is auto-stripped at load time for
    backward compatibility.
    """
    min: float
    max: float
    initial: float
    step: float
    units: str = ""


class FittingConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fitting_mode: Literal["auto", "library", "tmm_direct"] = "auto"
    fit_signals: list[str] = Field(default_factory=lambda: ["psi", "delta"])
    weights: Literal["uniform", "snr", "custom"] = "uniform"
    optimizer: Literal[
        "levenberg_marquardt", "nelder_mead", "differential_evolution"
    ] = "levenberg_marquardt"
    max_iterations: int = 200
    convergence_tolerance: float = 1e-6
    gradient_step: float = 1e-4


class LibraryReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    library_file: str = ""


class MeasurementRecipeOutputOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_recipe_with_results: bool = True
    save_fit_report: bool = True
    save_fitted_spectrum: bool = True


class ForwardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Legacy field — kept for backward compatibility; new field is `stack`
    sample: SampleRef | None = None
    # New first-class field
    stack: StackRef | None = None
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    simulation_conditions: SimulationConditionsEmbed = Field(
        default_factory=SimulationConditionsEmbed
    )
    system_config_ref: str = ""

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_layout(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # 1. Absorb old `system` dict → extract system_config_ref
        if "system" in data:
            sys_dict = data.pop("system") or {}
            if isinstance(sys_dict, dict):
                ref = sys_dict.get("system_config_ref", "")
                if ref and not data.get("system_config_ref"):
                    warnings.warn(
                        "ForwardModel.system is deprecated; use system_config_ref (a file path) instead.",
                        DeprecationWarning,
                        stacklevel=5,
                    )
                    data["system_config_ref"] = ref
        # 2. Promote optical fields from simulation_conditions → data_collection
        _extract_legacy_data_collection(data)
        return data

    @model_validator(mode="after")
    def _migrate_sample_to_stack(self) -> ForwardModel:
        """Auto-migrate legacy `sample` key to `stack`."""
        if self.stack is None and self.sample is not None:
            self.stack = _sampleref_to_stackref(self.sample)
        elif self.stack is None and self.sample is None:
            raise ValueError("Either 'stack' or 'sample' must be provided in forward_model")
        return self


class FitResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fitted_parameters: dict[str, float]
    fit_quality: dict[str, float]
    engine_used: str
    timestamp: str


class MeasurementRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: RecipeMetadata
    forward_model: ForwardModel
    floating_parameters: list[FloatingParameter]
    fitting_configuration: FittingConfiguration = Field(
        default_factory=FittingConfiguration
    )
    library_reference: LibraryReference = Field(default_factory=LibraryReference)
    output_options: MeasurementRecipeOutputOptions = Field(
        default_factory=MeasurementRecipeOutputOptions
    )
    results: FitResults | None = None
