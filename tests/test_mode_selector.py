"""Tests for se_simulator.fitting.mode_selector.select_fitting_mode."""
from __future__ import annotations

import pytest

from se_simulator.config.recipe import (
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
    GratingLayer,
    MaterialSpec,
    SampleConfig,
    ShapeGeometry,
    ShapeRegion,
)
from se_simulator.fitting.mode_selector import select_fitting_mode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIM_COND = SimulationConditionsEmbed(
    wavelength_start_nm=400.0,
    wavelength_end_nm=800.0,
    wavelength_step_nm=10.0,
    aoi_degrees=65.0,
)

_FM_SYSTEM: dict = {}


def _make_recipe(
    fitting_mode: str = "auto",
    library_file: str = "",
) -> MeasurementRecipe:
    """Build a minimal MeasurementRecipe with configurable fitting_mode/library."""
    return MeasurementRecipe(
        metadata=RecipeMetadata(recipe_type="measurement"),
        forward_model=ForwardModel(
            sample=SampleRef(inline={
                "superstrate": {"library_name": "Air"},
                "substrate": {"library_name": "Si"},
                "layers": [
                    {"type": "uniform", "thickness_nm": 100.0,
                     "material": {"library_name": "SiO2"}}
                ],
            }),
            simulation_conditions=_SIM_COND,
            system={},
        ),
        floating_parameters=[
            FloatingParameter(
                name="thickness",
                target_field="forward_model.sample.inline.layers[0].thickness_nm",
                min=50.0,
                max=200.0,
                initial=100.0,
                step=5.0,
            )
        ],
        fitting_configuration=FittingConfiguration(fitting_mode=fitting_mode),
        library_reference=LibraryReference(library_file=library_file),
    )


def _make_uniform_sample() -> SampleConfig:
    """SiO2 / Si stack — no shapes in any layer."""
    mat_air = MaterialSpec(name="Air", source="library", library_name="Air")
    mat_sio2 = MaterialSpec(name="SiO2", source="library", library_name="SiO2")
    mat_si = MaterialSpec(name="Si", source="library", library_name="Si")
    layer = GratingLayer(name="SiO2", type="uniform", thickness_nm=100.0)
    return SampleConfig(
        superstrate_material="Air",
        substrate_material="Si",
        layers=[layer],
        materials={"Air": mat_air, "SiO2": mat_sio2, "Si": mat_si},
    )


def _make_patterned_sample() -> SampleConfig:
    """Sample with one layer that has a rectangular shape (patterned)."""
    mat_air = MaterialSpec(name="Air", source="library", library_name="Air")
    mat_si = MaterialSpec(name="Si", source="library", library_name="Si")
    shape = ShapeRegion(geometry=ShapeGeometry(type="rectangle"), material="Si")
    layer = GratingLayer(
        name="grating",
        type="grating_1d",
        thickness_nm=100.0,
        shapes=[shape],
    )
    return SampleConfig(
        superstrate_material="Air",
        substrate_material="Si",
        layers=[layer],
        materials={"Air": mat_air, "Si": mat_si},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_auto_uniform_no_library_returns_tmm_direct():
    """auto + all-uniform + no library → 'tmm_direct'."""
    recipe = _make_recipe(fitting_mode="auto", library_file="")
    sample = _make_uniform_sample()
    assert select_fitting_mode(recipe, sample) == "tmm_direct"


def test_auto_uniform_with_library_returns_library():
    """auto + all-uniform + library present → 'library'."""
    recipe = _make_recipe(fitting_mode="auto", library_file="some_lib.h5")
    sample = _make_uniform_sample()
    assert select_fitting_mode(recipe, sample) == "library"


def test_auto_patterned_returns_library():
    """auto + patterned layer (shapes present) → 'library'."""
    recipe = _make_recipe(fitting_mode="auto", library_file="")
    sample = _make_patterned_sample()
    assert select_fitting_mode(recipe, sample) == "library"


def test_explicit_tmm_direct_uniform_returns_tmm_direct():
    """fitting_mode='tmm_direct' + uniform stack → 'tmm_direct'."""
    recipe = _make_recipe(fitting_mode="tmm_direct")
    sample = _make_uniform_sample()
    assert select_fitting_mode(recipe, sample) == "tmm_direct"


def test_explicit_tmm_direct_patterned_raises():
    """fitting_mode='tmm_direct' + patterned layer → ValueError."""
    recipe = _make_recipe(fitting_mode="tmm_direct")
    sample = _make_patterned_sample()
    with pytest.raises(ValueError, match="patterned layers"):
        select_fitting_mode(recipe, sample)


def test_explicit_library_uniform_returns_library():
    """fitting_mode='library' + uniform stack → 'library' (library always wins)."""
    recipe = _make_recipe(fitting_mode="library")
    sample = _make_uniform_sample()
    assert select_fitting_mode(recipe, sample) == "library"
