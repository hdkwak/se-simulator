"""Integration tests for se_simulator.fitting.pipeline.run_fitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from se_simulator.config.recipe import (
    FitResults,
    FittingConfiguration,
    FloatingParameter,
    ForwardModel,
    LibraryReference,
    MeasurementRecipe,
    MeasurementRecipeOutputOptions,
    RecipeMetadata,
    SampleRef,
    SimulationConditionsEmbed,
)
from se_simulator.config.schemas import (
    DataCollectionConfig,
    GratingLayer,
    MaterialSpec,
    SampleConfig,
    SimConditions,
    SystemConfig,
    WavelengthSpec,
)
from se_simulator.fitting.pipeline import run_fitting

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WL_START = 400.0
_WL_END = 700.0
_WL_STEP = 25.0


def _data_collection() -> DataCollectionConfig:
    return DataCollectionConfig(
        aoi_deg=65.0,
        wavelength_start_nm=_WL_START,
        wavelength_end_nm=_WL_END,
        wavelength_step_nm=_WL_STEP,
    )


def _sim_cond_embed() -> SimulationConditionsEmbed:
    return SimulationConditionsEmbed()


# ---------------------------------------------------------------------------
# Sample helpers — use constant_nk / cauchy so no CSV is needed
# ---------------------------------------------------------------------------


def _make_sample_config(thickness_nm: float = 100.0) -> SampleConfig:
    """SiO2/Si stack with constant_nk/cauchy materials (no library CSV required)."""
    mat_air = MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)
    mat_sio2 = MaterialSpec(
        name="SiO2", source="cauchy", coefficients=[1.45, 3560.0, 0.0]
    )
    mat_si = MaterialSpec(name="Si", source="constant_nk", n=3.88, k=0.02)
    layer = GratingLayer(
        name="SiO2",
        type="uniform",
        thickness_nm=thickness_nm,
        background_material="SiO2",
    )
    return SampleConfig(
        superstrate_material="Air",
        substrate_material="Si",
        layers=[layer],
        materials={"Air": mat_air, "SiO2": mat_sio2, "Si": mat_si},
    )


def _write_sample_yaml(sample: SampleConfig, path: Path) -> None:
    """Serialise a SampleConfig to a YAML file for use with ``ref:``."""
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.default_flow_style = False
    with open(path, "w") as fh:
        yaml.dump(sample.model_dump(), fh)


def _make_measurement_recipe_ref(
    sample_yaml_path: Path,
    thickness_initial: float = 100.0,
    fitting_mode: str = "auto",
    library_file: str = "",
) -> MeasurementRecipe:
    """Recipe that references an external sample YAML (avoids inline library_name issue)."""
    return MeasurementRecipe(
        metadata=RecipeMetadata(recipe_type="measurement"),
        forward_model=ForwardModel(
            sample=SampleRef(ref=str(sample_yaml_path)),
            data_collection=_data_collection(),
            simulation_conditions=_sim_cond_embed(),
        ),
        floating_parameters=[
            FloatingParameter(
                name="thickness_nm",
                # target_field must be resolvable against the recipe dict;
                # for ref: samples the field resolves within forward_model.sample.ref
                # but TmmDirectFitter uses the decomposed SampleConfig directly
                target_field="layers[0].thickness_nm",
                min=50.0,
                max=250.0,
                initial=thickness_initial,
                step=5.0,
            )
        ],
        fitting_configuration=FittingConfiguration(
            fitting_mode=fitting_mode,
            fit_signals=["psi", "delta"],
            optimizer="levenberg_marquardt",
            max_iterations=200,
            convergence_tolerance=1e-6,
        ),
        library_reference=LibraryReference(library_file=library_file),
        output_options=MeasurementRecipeOutputOptions(
            save_recipe_with_results=False,
        ),
    )


def _generate_synthetic_target(true_thickness: float = 97.3) -> np.ndarray:
    """Generate a synthetic psi+delta spectrum via TMM.

    Uses constant_nk/cauchy material models to avoid the library CSV
    header-comment issue.
    """
    from se_simulator.config.manager import ConfigManager
    from se_simulator.ellipsometer.prcsa import compute_psi_delta
    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.tmm import compute_tmm

    sample = _make_sample_config(true_thickness)
    sim = SimConditions(
        aoi_deg=65.0,
        wavelengths=WavelengthSpec(range=(_WL_START, _WL_END, _WL_STEP)),
    )
    wls = ConfigManager().get_wavelengths(sim.wavelengths)
    db = MaterialDatabase()
    for spec in sample.materials.values():
        db.resolve(spec)
    jones_r, _ = compute_tmm(sample, db, wls, sim.aoi_deg, sim.azimuth_deg)
    n = len(wls)
    psi = np.empty(n)
    delta = np.empty(n)
    for i in range(n):
        psi[i], delta[i] = compute_psi_delta(jones_r[i])
    return np.concatenate([psi, delta])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_yaml(tmp_path: Path) -> Path:
    """Write a sample YAML with dispersion-correct materials to a temp file."""
    p = tmp_path / "sample.yaml"
    _write_sample_yaml(_make_sample_config(100.0), p)
    return p


# ---------------------------------------------------------------------------
# Test 1: TMM-direct end-to-end
# ---------------------------------------------------------------------------


def test_tmm_direct_end_to_end_returns_fit_results(sample_yaml: Path):
    """run_fitting with auto mode + uniform sample + no library → FitResults."""
    true_thickness = 97.3
    recipe = _make_measurement_recipe_ref(
        sample_yaml_path=sample_yaml,
        thickness_initial=100.0,
        fitting_mode="auto",
    )
    target = _generate_synthetic_target(true_thickness)

    result = run_fitting(recipe, target)

    assert isinstance(result, FitResults)
    assert "thickness_nm" in result.fitted_parameters
    fitted = result.fitted_parameters["thickness_nm"]
    assert abs(fitted - true_thickness) < 2.0, (
        f"Pipeline fitted {fitted:.3f} nm; expected near {true_thickness} nm"
    )


# ---------------------------------------------------------------------------
# Test 2: returns FitResults with correct structure
# ---------------------------------------------------------------------------


def test_run_fitting_result_has_required_fields(sample_yaml: Path):
    """FitResults must contain fitted_parameters, fit_quality, engine_used, timestamp."""
    recipe = _make_measurement_recipe_ref(sample_yaml)
    target = _generate_synthetic_target()
    result = run_fitting(recipe, target)

    assert isinstance(result.fitted_parameters, dict)
    assert isinstance(result.fit_quality, dict)
    assert isinstance(result.engine_used, str)
    assert isinstance(result.timestamp, str)
    assert result.engine_used == "tmm_direct"


# ---------------------------------------------------------------------------
# Test 3: library mode without library_file raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_library_mode_no_file_raises(sample_yaml: Path):
    """Explicit library mode with no library_file → FileNotFoundError."""
    recipe = _make_measurement_recipe_ref(
        sample_yaml, fitting_mode="library", library_file=""
    )
    target = _generate_synthetic_target()
    with pytest.raises(FileNotFoundError, match="library_file"):
        run_fitting(recipe, target)


# ---------------------------------------------------------------------------
# Test 4: target array with (N, 2) shape accepted
# ---------------------------------------------------------------------------


def test_target_2d_shape_accepted(sample_yaml: Path):
    """run_fitting should accept (N, 2) targets (psi col 0, delta col 1)."""
    true_thickness = 97.3
    recipe = _make_measurement_recipe_ref(sample_yaml, fitting_mode="auto")
    flat = _generate_synthetic_target(true_thickness)
    n_wl = len(flat) // 2
    target_2d = np.column_stack([flat[:n_wl], flat[n_wl:]])  # (N, 2)

    result = run_fitting(recipe, target_2d)

    assert isinstance(result, FitResults)
    fitted = result.fitted_parameters["thickness_nm"]
    assert abs(fitted - true_thickness) < 2.0
