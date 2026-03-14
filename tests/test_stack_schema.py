"""Tests for StackLayer, Stack, and StackRef (Phase 1 schema additions)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from se_simulator.config.schemas import (
    MaterialSpec,
    SampleConfig,
    ShapeGeometry,
    ShapeRegion,
    Stack,
    StackLayer,
)
from se_simulator.config.recipe import StackRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _air() -> MaterialSpec:
    return MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)


def _si() -> MaterialSpec:
    return MaterialSpec(name="Si", source="library", library_name="Si")


def _sio2() -> MaterialSpec:
    return MaterialSpec(name="SiO2", source="library", library_name="SiO2")


def _sio2_layer(thickness_nm: float = 100.0) -> StackLayer:
    return StackLayer(
        name="sio2_layer",
        type="uniform",
        thickness_nm=thickness_nm,
        material=_sio2(),
    )


def _si_layer(thickness_nm: float = 50.0) -> StackLayer:
    return StackLayer(
        name="si_layer",
        type="uniform",
        thickness_nm=thickness_nm,
        material=_si(),
    )


def _basic_stack() -> Stack:
    return Stack(
        superstrate=_air(),
        substrate=_si(),
        layers=[_sio2_layer(), _si_layer()],
    )


# ---------------------------------------------------------------------------
# 1. StackLayer round-trip
# ---------------------------------------------------------------------------

def test_stack_layer_round_trip() -> None:
    layer = _sio2_layer()
    dumped = layer.model_dump()
    revalidated = StackLayer.model_validate(dumped)
    assert revalidated == layer
    assert revalidated.name == "sio2_layer"
    assert revalidated.thickness_nm == pytest.approx(100.0)
    assert revalidated.material.name == "SiO2"


# ---------------------------------------------------------------------------
# 2. Stack round-trip
# ---------------------------------------------------------------------------

def test_stack_round_trip() -> None:
    stack = _basic_stack()
    dumped = stack.model_dump()
    revalidated = Stack.model_validate(dumped)
    assert revalidated == stack
    assert revalidated.superstrate.name == "Air"
    assert revalidated.substrate.name == "Si"
    assert len(revalidated.layers) == 2


# ---------------------------------------------------------------------------
# 3. to_sample_config — material keys
# ---------------------------------------------------------------------------

def test_stack_to_sample_config_materials() -> None:
    stack = _basic_stack()
    sample = stack.to_sample_config()

    assert isinstance(sample, SampleConfig)
    # Air uses name (no library_name); Si and SiO2 use library_name
    assert "Air" in sample.materials
    assert "Si" in sample.materials
    assert "SiO2" in sample.materials


# ---------------------------------------------------------------------------
# 4. to_sample_config — layer background_material
# ---------------------------------------------------------------------------

def test_stack_to_sample_config_layers() -> None:
    stack = _basic_stack()
    sample = stack.to_sample_config()

    assert len(sample.layers) == 2
    # First layer is SiO2 (library_name = "SiO2")
    assert sample.layers[0].background_material == "SiO2"
    # Second layer is Si (library_name = "Si")
    assert sample.layers[1].background_material == "Si"


# ---------------------------------------------------------------------------
# 5. to_sample_config — superstrate field
# ---------------------------------------------------------------------------

def test_stack_to_sample_config_superstrate() -> None:
    stack = _basic_stack()
    sample = stack.to_sample_config()

    # Air has no library_name, so key falls back to .name
    assert sample.superstrate_material == "Air"
    assert sample.substrate_material == "Si"


# ---------------------------------------------------------------------------
# 6. shapes preserved through to_sample_config
# ---------------------------------------------------------------------------

def test_stack_shapes_preserved() -> None:
    geom = ShapeGeometry(type="rectangle", cx=250.0, cy=250.0, width=200.0, height=400.0)
    shape = ShapeRegion(geometry=geom, material="Si")
    layer = StackLayer(
        name="patterned",
        type="grating_1d",
        thickness_nm=80.0,
        material=_air(),
        shapes=[shape],
    )
    stack = Stack(superstrate=_air(), substrate=_si(), layers=[layer])
    sample = stack.to_sample_config()

    assert len(sample.layers[0].shapes) == 1
    s = sample.layers[0].shapes[0]
    assert s.geometry.width == pytest.approx(200.0)
    assert s.geometry.height == pytest.approx(400.0)
    assert s.material == "Si"


# ---------------------------------------------------------------------------
# 7. StackRef — inline branch
# ---------------------------------------------------------------------------

def test_stack_ref_inline() -> None:
    stack = _basic_stack()
    sr = StackRef(inline=stack)
    assert sr.inline is not None
    assert sr.ref is None
    assert sr.inline.substrate.name == "Si"


# ---------------------------------------------------------------------------
# 8. StackRef — external ref branch
# ---------------------------------------------------------------------------

def test_stack_ref_external() -> None:
    sr = StackRef(ref="./sample.yaml")
    assert sr.ref == "./sample.yaml"
    assert sr.inline is None


# ---------------------------------------------------------------------------
# 9. StackRef — both provided raises
# ---------------------------------------------------------------------------

def test_stack_ref_both_raises() -> None:
    stack = _basic_stack()
    with pytest.raises(ValidationError):
        StackRef(inline=stack, ref="./sample.yaml")


# ---------------------------------------------------------------------------
# 10. StackRef — neither provided raises
# ---------------------------------------------------------------------------

def test_stack_ref_neither_raises() -> None:
    with pytest.raises(ValidationError):
        StackRef()


# ---------------------------------------------------------------------------
# 11. to_sample_config result validates as SampleConfig and is engine-runnable
# ---------------------------------------------------------------------------

def test_stack_to_sample_config_engine_runnable() -> None:
    from se_simulator.config.schemas import SimConditions, WavelengthSpec
    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.engine import RCWAEngine

    stack = Stack(
        superstrate=_air(),
        substrate=_si(),
        layers=[_sio2_layer(100.0)],
    )
    sample = stack.to_sample_config()

    # Confirm it is a valid SampleConfig
    assert isinstance(sample, SampleConfig)
    # Re-validate through Pydantic to be sure
    SampleConfig.model_validate(sample.model_dump())

    # Build a minimal MaterialDatabase from the specs
    db = MaterialDatabase()
    for spec in sample.materials.values():
        db.resolve(spec)

    sim = SimConditions(
        aoi_deg=65.0,
        azimuth_deg=0.0,
        wavelengths=WavelengthSpec(explicit=[633.0]),
        n_harmonics_x=1,
        n_harmonics_y=1,
    )

    result = RCWAEngine(db).run(sample, sim)
    assert result.jones_reflection.shape == (1, 2, 2)
