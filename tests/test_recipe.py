"""Tests for the Recipe data layer (Step 1).

19 test cases covering:
  - SimulationRecipe and MeasurementRecipe Pydantic validation
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
from se_simulator.config.schemas import SampleConfig, SimConditions, Stack, SystemConfig
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

SIM_CONDITIONS: dict[str, Any] = {
    "wavelength_start_nm": 300.0,
    "wavelength_end_nm": 900.0,
    "wavelength_step_nm": 2.0,
    "aoi_degrees": 65.0,
    "azimuth_degrees": 0.0,
    "polarizer_degrees": 45.0,
    "analyzer_degrees": 45.0,
}


def _make_sim_recipe() -> SimulationRecipe:
    return SimulationRecipe(
        metadata=RecipeMetadata(recipe_type="simulation"),
        sample=SampleRef(inline=INLINE_SAMPLE),
        simulation_conditions=SimulationConditionsEmbed(**SIM_CONDITIONS),
    )


def _make_meas_recipe() -> MeasurementRecipe:
    fm = ForwardModel(
        sample=SampleRef(inline=INLINE_SAMPLE),
        simulation_conditions=SimulationConditionsEmbed(**SIM_CONDITIONS),
        system={"system_config_ref": ""},
    )
    fp = FloatingParameter(
        name="sio2_thickness",
        target_field="forward_model.sample.inline.layers[0].thickness_nm",
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
    assert recipe.engine_override == "auto"
    assert recipe.output_options.save_psi_delta is True


# ---------------------------------------------------------------------------
# 2. SimulationRecipe validation — missing required field
# ---------------------------------------------------------------------------

def test_simulation_recipe_missing_field() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # simulation_conditions is required
        SimulationRecipe.model_validate(
            {
                "metadata": {"recipe_type": "simulation"},
                "sample": {"inline": INLINE_SAMPLE},
                # missing simulation_conditions
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
    assert loaded.simulation_conditions.aoi_degrees == pytest.approx(65.0)
    assert loaded.simulation_conditions.wavelength_start_nm == pytest.approx(300.0)


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

    # decompose_simulation now returns Stack (preferred API)
    assert isinstance(sample, Stack)
    assert isinstance(sim_cond, SimConditions)

    # Stack fields: superstrate/substrate are MaterialSpec objects
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
    sample, sim_cond, sys_cfg, floats, fit_cfg = manager.decompose_measurement(recipe)

    # decompose_measurement now returns Stack (preferred API)
    assert isinstance(sample, Stack)
    assert isinstance(sim_cond, SimConditions)
    assert isinstance(sys_cfg, SystemConfig)
    assert len(floats) == 1
    assert isinstance(fit_cfg, FittingConfiguration)
    assert sys_cfg.polarizer_angle_deg == pytest.approx(45.0)


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
    # Should have a YAML parse error OR a structural error
    assert len(errors) >= 1


# ---------------------------------------------------------------------------
# 17. RecipeManager.validate — floating parameter out of bounds flagged
# ---------------------------------------------------------------------------

def test_validate_floating_param_out_of_bounds(manager: RecipeManager, tmp_path: Path) -> None:
    recipe = _make_meas_recipe()
    # Mutate so initial is out of range
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
    # Set initial to something distinct from the template value
    recipe.floating_parameters[0].initial = 250.0

    sim_recipe = manager.export_as_simulation(recipe)

    assert isinstance(sim_recipe, SimulationRecipe)
    assert sim_recipe.metadata.recipe_type == "simulation"
    # The layer thickness should have been pinned to the initial value.
    # New Stack-based access path:
    assert sim_recipe.stack is not None
    assert sim_recipe.stack.inline is not None
    assert sim_recipe.stack.inline.layers[0].thickness_nm == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# 19. RecipeManager.append_results adds results block; get_recent tracks files
# ---------------------------------------------------------------------------

def test_append_results_and_get_recent(
    manager: RecipeManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Redirect recent file to tmp_path so we don't pollute the real one
    import se_simulator.recipe.manager as rm_module

    fake_recent = tmp_path / "recent_recipes.json"
    monkeypatch.setattr(rm_module, "_RECENT_FILE", fake_recent)

    recipe = _make_meas_recipe()
    out = tmp_path / "meas_with_results.yaml"
    manager.save_measurement_recipe(recipe, out)
    manager.load_measurement_recipe(out)  # populates recent list

    results = FitResults(
        fitted_parameters={"sio2_thickness": 123.4},
        fit_quality={"mse": 0.001},
        engine_used="tmm",
        timestamp="2026-03-14T00:00:00+00:00",
    )
    manager.append_results(results, out)

    # Reload and check results block is present
    reloaded = manager.load_measurement_recipe(out)
    assert reloaded.results is not None
    assert reloaded.results.fitted_parameters["sio2_thickness"] == pytest.approx(123.4)

    # Check get_recent
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

    # Write a minimal legacy simulation recipe YAML by hand
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

    # stack must be populated from the legacy sample key
    assert isinstance(recipe.stack, StackRef)
    assert recipe.stack.inline is not None
    assert recipe.stack.inline.layers[0].thickness_nm == pytest.approx(150.0)
    assert recipe.stack.inline.superstrate.library_name == "Air"
    assert recipe.stack.inline.substrate.library_name == "Si"


# ---------------------------------------------------------------------------
# 21. StackRef inline — SimulationRecipe created directly with stack= field
# ---------------------------------------------------------------------------

def test_simulation_recipe_with_stack_field() -> None:
    """SimulationRecipe constructed with `stack=` (no `sample=`) validates correctly."""
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
        simulation_conditions=SimulationConditionsEmbed(**SIM_CONDITIONS),
    )

    assert recipe.stack is not None
    assert recipe.stack.inline is not None
    assert recipe.stack.inline.layers[0].thickness_nm == pytest.approx(200.0)
    # stack-based recipe has no legacy sample field
    assert recipe.sample is None
