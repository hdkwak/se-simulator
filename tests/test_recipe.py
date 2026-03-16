"""Tests for the Recipe data layer (Step 1).

19+ test cases covering:
  - SimulationRecipe and MeasurementRecipe Pydantic validation
  - DataCollectionConfig (new)
  - dotpath get/set utilities
  - RecipeManager load/save round-trips
  - RecipeManager.decompose_simulation / decompose_measurement
  - RecipeManager.validate (6 checks)
  - RecipeManager.export_as_simulation
  - RecipeManager.append_results
  - RecipeManager.get_recent
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from se_simulator.config.recipe import (
    FitResults,
    FittingConfiguration,
    FloatingParameter,
    ForwardModel,
    MeasurementRecipe,
    RecipeMetadata,
    SampleRef,
    SimulationConditionsEmbed,
    SimulationRecipe,
    SimulationRecipeOutputOptions,
)
from se_simulator.config.schemas import (
    DataCollectionConfig,
    SampleConfig,
    SimConditions,
    Stack,
    SystemConfig,
)
from se_simulator.recipe.dotpath import resolve_get, resolve_set
from se_simulator.recipe.manager import RecipeManager, RecipeValidationError


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

INLINE_SAMPLE: dict[str, Any] = {
    "superstrate": {"library_name": "Air"},
    "substrate": {"library_name": "Si"},
    "layers": [
        {"type": "uniform", "thickness_nm": 100.0, "material": {"library_name": "SiO2"}}
    ],
}

# Legacy-style sim conditions (optical fields in simulation_conditions) for
# backward-compat testing.
LEGACY_SIM_CONDITIONS: dict[str, Any] = {
    "wavelength_start_nm": 300.0,
    "wavelength_end_nm": 900.0,
    "wavelength_step_nm": 2.0,
    "aoi_degrees": 65.0,
    "azimuth_degrees": 0.0,
    "polarizer_degrees": 45.0,
    "analyzer_degrees": 45.0,
}

DATA_COLLECTION: dict[str, Any] = {
    "aoi_deg": 65.0,
    "azimuth_deg": 0.0,
    "polarizer_angle_deg": 45.0,
    "analyzer_angle_deg": 45.0,
    "wavelength_start_nm": 300.0,
    "wavelength_end_nm": 900.0,
    "wavelength_step_nm": 2.0,
}


def _make_sim_recipe() -> SimulationRecipe:
    return SimulationRecipe(
        metadata=RecipeMetadata(recipe_type="simulation"),
        sample=SampleRef(inline=INLINE_SAMPLE),
        data_collection=DataCollectionConfig(**DATA_COLLECTION),
        simulation_conditions=SimulationConditionsEmbed(),
    )


def _make_meas_recipe() -> MeasurementRecipe:
    fm = ForwardModel(
        sample=SampleRef(inline=INLINE_SAMPLE),
        data_collection=DataCollectionConfig(**DATA_COLLECTION),
        simulation_conditions=SimulationConditionsEmbed(),
        system_config_ref="",
    )
    fp = FloatingParameter(
        name="sio2_thickness",
        target_field="layers[0].thickness_nm",
        min=0.0,
        max=500.0,
        initial=100.0,
        step=1.0,
    )
    return MeasurementRecipe(
        metadata=RecipeMetadata(recipe_type="measurement"),
        forward_model=fm,
        floating_parameters=[fp],
    )


@pytest.fixture()
def manager() -> RecipeManager:
    return RecipeManager()


# ---------------------------------------------------------------------------
# 1. SimulationRecipe validation — happy path
# ---------------------------------------------------------------------------

def test_simulation_recipe_valid() -> None:
    recipe = _make_sim_recipe()
    assert recipe.metadata.recipe_type == "simulation"
    assert recipe.simulation_conditions.engine_override == "auto"
    assert recipe.output_options.save_psi_delta is True
    assert recipe.data_collection.aoi_deg == pytest.approx(65.0)


# ---------------------------------------------------------------------------
# 2. SimulationRecipe validation — missing stack/sample raises
# ---------------------------------------------------------------------------

def test_simulation_recipe_missing_field() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # Both stack and sample absent → validator raises
        SimulationRecipe.model_validate(
            {
                "metadata": {"recipe_type": "simulation"},
                # no stack, no sample
            }
        )


# ---------------------------------------------------------------------------
# 3. SampleRef — must have exactly one of inline/ref
# ---------------------------------------------------------------------------

def test_sample_ref_exclusive() -> None:
    from pydantic import ValidationError

    # Neither provided
    with pytest.raises(ValidationError):
        SampleRef()

    # Both provided — also invalid
    with pytest.raises(ValidationError):
        SampleRef(inline=INLINE_SAMPLE, ref="some/path.yaml")

    # Exactly one is fine
    assert SampleRef(inline=INLINE_SAMPLE).inline is not None
    assert SampleRef(ref="some/path.yaml").ref == "some/path.yaml"


# ---------------------------------------------------------------------------
# 4. MeasurementRecipe validation — happy path
# ---------------------------------------------------------------------------

def test_measurement_recipe_valid() -> None:
    recipe = _make_meas_recipe()
    assert recipe.metadata.recipe_type == "measurement"
    assert len(recipe.floating_parameters) == 1
    assert recipe.floating_parameters[0].name == "sio2_thickness"


# ---------------------------------------------------------------------------
# 5. FittingConfiguration defaults
# ---------------------------------------------------------------------------

def test_fitting_configuration_defaults() -> None:
    fc = FittingConfiguration()
    assert fc.optimizer == "levenberg_marquardt"
    assert fc.max_iterations == 200
    assert fc.fit_signals == ["psi", "delta"]


# ---------------------------------------------------------------------------
# 6. dotpath resolve_get — simple key
# ---------------------------------------------------------------------------

def test_dotpath_resolve_get_simple() -> None:
    data = {"a": {"b": {"c": 42}}}
    assert resolve_get(data, "a.b.c") == 42


# ---------------------------------------------------------------------------
# 7. dotpath resolve_get — list index
# ---------------------------------------------------------------------------

def test_dotpath_resolve_get_list() -> None:
    data = {"layers": [{"thickness_nm": 100.0}, {"thickness_nm": 200.0}]}
    assert resolve_get(data, "layers[0].thickness_nm") == 100.0
    assert resolve_get(data, "layers[1].thickness_nm") == 200.0


# ---------------------------------------------------------------------------
# 8. dotpath resolve_set — simple key
# ---------------------------------------------------------------------------

def test_dotpath_resolve_set_simple() -> None:
    data: dict[str, Any] = {"a": {"b": 1}}
    resolve_set(data, "a.b", 99)
    assert data["a"]["b"] == 99


# ---------------------------------------------------------------------------
# 9. dotpath resolve_set — list index
# ---------------------------------------------------------------------------

def test_dotpath_resolve_set_list() -> None:
    data: dict[str, Any] = {
        "layers": [{"thickness_nm": 100.0}, {"thickness_nm": 200.0}]
    }
    resolve_set(data, "layers[0].thickness_nm", 150.0)
    assert data["layers"][0]["thickness_nm"] == 150.0


# ---------------------------------------------------------------------------
# 10. RecipeManager.save_simulation_recipe + load_simulation_recipe round-trip
# ---------------------------------------------------------------------------

def test_sim_recipe_round_trip(
    manager: RecipeManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import se_simulator.recipe.manager as rm_module

    monkeypatch.setattr(rm_module, "_RECENT_FILE", tmp_path / "recent.json")

    recipe = _make_sim_recipe()
    out = tmp_path / "sim.yaml"
    manager.save_simulation_recipe(recipe, out)
    assert out.exists()

    loaded = manager.load_simulation_recipe(out)
    assert loaded.metadata.recipe_type == "simulation"
    assert loaded.data_collection.aoi_deg == pytest.approx(65.0)
    assert loaded.data_collection.wavelength_start_nm == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# 11. RecipeManager.save_measurement_recipe + load_measurement_recipe round-trip
# ---------------------------------------------------------------------------

def test_meas_recipe_round_trip(
    manager: RecipeManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import se_simulator.recipe.manager as rm_module

    monkeypatch.setattr(rm_module, "_RECENT_FILE", tmp_path / "recent.json")

    recipe = _make_meas_recipe()
    out = tmp_path / "meas.yaml"
    manager.save_measurement_recipe(recipe, out)
    assert out.exists()

    loaded = manager.load_measurement_recipe(out)
    assert loaded.metadata.recipe_type == "measurement"
    assert loaded.floating_parameters[0].initial == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 12. RecipeManager.decompose_simulation returns correct types and values
# ---------------------------------------------------------------------------

def test_decompose_simulation(manager: RecipeManager) -> None:
    recipe = _make_sim_recipe()
    sample, sim_cond = manager.decompose_simulation(recipe)

    assert isinstance(sample, Stack)
    assert isinstance(sim_cond, SimConditions)

    assert sample.superstrate.library_name == "Air" or sample.superstrate.name == "Air"
    assert sample.substrate.library_name == "Si" or sample.substrate.name == "Si"
    assert sim_cond.aoi_deg == pytest.approx(65.0)
    assert sim_cond.wavelengths.range is not None
    assert sim_cond.wavelengths.range[0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# 13. RecipeManager.decompose_measurement returns five objects of correct types
# ---------------------------------------------------------------------------

def test_decompose_measurement(manager: RecipeManager) -> None:
    recipe = _make_meas_recipe()
    sample, sim_cond, dc, sys_cfg, floats, fit_cfg = manager.decompose_measurement(recipe)

    assert isinstance(sample, Stack)
    assert isinstance(sim_cond, SimConditions)
    assert isinstance(dc, DataCollectionConfig)
    assert isinstance(sys_cfg, SystemConfig)
    assert len(floats) == 1
    assert isinstance(fit_cfg, FittingConfiguration)
    # Optical angles now live in DataCollectionConfig
    assert dc.polarizer_angle_deg == pytest.approx(45.0)
    assert dc.aoi_deg == pytest.approx(65.0)
    assert sim_cond.aoi_deg == pytest.approx(65.0)


# ---------------------------------------------------------------------------
# 14. RecipeManager.validate — valid simulation file returns empty list
# ---------------------------------------------------------------------------

def test_validate_valid_simulation(manager: RecipeManager, tmp_path: Path) -> None:
    recipe = _make_sim_recipe()
    out = tmp_path / "valid_sim.yaml"
    manager.save_simulation_recipe(recipe, out)

    errors = manager.validate(out)
    assert errors == []


# ---------------------------------------------------------------------------
# 15. RecipeManager.validate — missing file returns error
# ---------------------------------------------------------------------------

def test_validate_missing_file(manager: RecipeManager, tmp_path: Path) -> None:
    errors = manager.validate(tmp_path / "nonexistent.yaml")
    assert len(errors) == 1
    assert "not found" in errors[0].lower() or "File not found" in errors[0]


# ---------------------------------------------------------------------------
# 16. RecipeManager.validate — bad YAML returns error
# ---------------------------------------------------------------------------

def test_validate_bad_yaml(manager: RecipeManager, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(": invalid: yaml: [[[")
    errors = manager.validate(bad)
    assert len(errors) >= 1


# ---------------------------------------------------------------------------
# 17. RecipeManager.validate — floating parameter out of bounds flagged
# ---------------------------------------------------------------------------

def test_validate_floating_param_out_of_bounds(manager: RecipeManager, tmp_path: Path) -> None:
    recipe = _make_meas_recipe()
    recipe.floating_parameters[0].initial = 9999.0  # max is 500
    out = tmp_path / "bad_param.yaml"
    manager.save_measurement_recipe(recipe, out)

    errors = manager.validate(out)
    assert any("initial" in e or "not in" in e for e in errors)


# ---------------------------------------------------------------------------
# 18. RecipeManager.export_as_simulation pins floated params to initial values
# ---------------------------------------------------------------------------

def test_export_as_simulation(manager: RecipeManager) -> None:
    recipe = _make_meas_recipe()
    recipe.floating_parameters[0].initial = 250.0

    sim_recipe = manager.export_as_simulation(recipe)

    assert isinstance(sim_recipe, SimulationRecipe)
    assert sim_recipe.metadata.recipe_type == "simulation"
    assert sim_recipe.stack is not None
    assert sim_recipe.stack.inline is not None
    assert sim_recipe.stack.inline.layers[0].thickness_nm == pytest.approx(250.0)
    # DataCollectionConfig should be preserved
    assert sim_recipe.data_collection.aoi_deg == pytest.approx(65.0)


# ---------------------------------------------------------------------------
# 19. RecipeManager.append_results adds results block; get_recent tracks files
# ---------------------------------------------------------------------------

def test_append_results_and_get_recent(
    manager: RecipeManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import se_simulator.recipe.manager as rm_module

    fake_recent = tmp_path / "recent_recipes.json"
    monkeypatch.setattr(rm_module, "_RECENT_FILE", fake_recent)

    recipe = _make_meas_recipe()
    out = tmp_path / "meas_with_results.yaml"
    manager.save_measurement_recipe(recipe, out)
    manager.load_measurement_recipe(out)

    results = FitResults(
        fitted_parameters={"sio2_thickness": 123.4},
        fit_quality={"mse": 0.001},
        engine_used="tmm",
        timestamp="2026-03-14T00:00:00+00:00",
    )
    manager.append_results(results, out)

    reloaded = manager.load_measurement_recipe(out)
    assert reloaded.results is not None
    assert reloaded.results.fitted_parameters["sio2_thickness"] == pytest.approx(123.4)

    recent = manager.get_recent(5)
    assert len(recent) >= 1
    paths = [p for p, _ in recent]
    assert out in paths


# ---------------------------------------------------------------------------
# 20. Backward-compat: loading legacy YAML with `sample:` key migrates to `stack`
# ---------------------------------------------------------------------------

def test_legacy_sample_key_migrates_to_stack(manager: RecipeManager, tmp_path: Path) -> None:
    """A YAML file using the old `sample:` key must load and auto-migrate to `stack`."""
    from se_simulator.config.recipe import StackRef

    legacy_yaml = tmp_path / "legacy_sim.yaml"
    legacy_yaml.write_text(
        "metadata:\n"
        "  recipe_type: simulation\n"
        "sample:\n"
        "  inline:\n"
        "    superstrate:\n"
        "      library_name: Air\n"
        "    substrate:\n"
        "      library_name: Si\n"
        "    layers:\n"
        "      - name: SiO2 Layer\n"
        "        type: uniform\n"
        "        thickness_nm: 150.0\n"
        "        material:\n"
        "          library_name: SiO2\n"
        "simulation_conditions:\n"
        "  wavelength_start_nm: 300.0\n"
        "  wavelength_end_nm: 800.0\n"
        "  wavelength_step_nm: 10.0\n"
        "  aoi_degrees: 65.0\n"
    )

    recipe = manager.load_simulation_recipe(legacy_yaml)

    assert isinstance(recipe.stack, StackRef)
    assert recipe.stack.inline is not None
    assert recipe.stack.inline.layers[0].thickness_nm == pytest.approx(150.0)
    assert recipe.stack.inline.superstrate.library_name == "Air"
    assert recipe.stack.inline.substrate.library_name == "Si"
    # Legacy optical fields should be promoted to data_collection
    assert recipe.data_collection.aoi_deg == pytest.approx(65.0)
    assert recipe.data_collection.wavelength_start_nm == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# 21. StackRef inline — SimulationRecipe created directly with stack= field
# ---------------------------------------------------------------------------

def test_simulation_recipe_with_stack_field() -> None:
    from se_simulator.config.recipe import StackRef
    from se_simulator.config.schemas import MaterialSpec, Stack, StackLayer

    sup = MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)
    sub = MaterialSpec(name="Si", source="constant_nk", n=3.88, k=0.02)
    layer = StackLayer(
        name="SiO2",
        type="uniform",
        thickness_nm=200.0,
        material=MaterialSpec(name="SiO2", source="constant_nk", n=1.46, k=0.0),
    )
    stack = Stack(superstrate=sup, substrate=sub, layers=[layer])

    recipe = SimulationRecipe(
        metadata=RecipeMetadata(recipe_type="simulation"),
        stack=StackRef(inline=stack),
        data_collection=DataCollectionConfig(**DATA_COLLECTION),
        simulation_conditions=SimulationConditionsEmbed(),
    )

    assert recipe.stack is not None
    assert recipe.stack.inline is not None
    assert recipe.stack.inline.layers[0].thickness_nm == pytest.approx(200.0)
    assert recipe.sample is None


# ---------------------------------------------------------------------------
# 22. Backward-compat: legacy simulation_conditions optical fields promoted to data_collection
# ---------------------------------------------------------------------------

def test_legacy_sim_conditions_promoted_to_data_collection() -> None:
    """Old recipes with optical fields in simulation_conditions get promoted automatically."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        recipe = SimulationRecipe(
            metadata=RecipeMetadata(recipe_type="simulation"),
            sample=SampleRef(inline=INLINE_SAMPLE),
            simulation_conditions=SimulationConditionsEmbed(**LEGACY_SIM_CONDITIONS),
        )

    assert recipe.data_collection.aoi_deg == pytest.approx(65.0)
    assert recipe.data_collection.polarizer_angle_deg == pytest.approx(45.0)
    assert recipe.data_collection.wavelength_start_nm == pytest.approx(300.0)
