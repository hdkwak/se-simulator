"""Tests for the RCWA engine (Step 2b): global S-matrix propagation and orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from se_simulator.config.schemas import (
    GratingLayer,
    MaterialSpec,
    SampleConfig,
    ShapeGeometry,
    ShapeRegion,
    SimConditions,
    Stack,
    StackLayer,
    WavelengthSpec,
)
from se_simulator.materials.database import MaterialDatabase
from se_simulator.rcwa.engine import RCWAEngine


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_db(*material_specs: MaterialSpec) -> MaterialDatabase:
    db = MaterialDatabase()
    for spec in material_specs:
        db.resolve(spec)
    return db


def _air_spec() -> MaterialSpec:
    return MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)


def _sio2_spec() -> MaterialSpec:
    return MaterialSpec(name="SiO2", source="library", library_name="SiO2")


def _si_spec() -> MaterialSpec:
    return MaterialSpec(name="Si", source="library", library_name="Si")


def _uniform_sio2_layer(thickness_nm: float = 100.0) -> GratingLayer:
    return GratingLayer(
        name="sio2_layer",
        type="uniform",
        thickness_nm=thickness_nm,
        Lx_nm=500.0,
        Ly_nm=500.0,
        background_material="SiO2",
        shapes=[],
    )


def _uniform_si_layer(thickness_nm: float = 100.0) -> GratingLayer:
    return GratingLayer(
        name="si_layer",
        type="uniform",
        thickness_nm=thickness_nm,
        Lx_nm=500.0,
        Ly_nm=500.0,
        background_material="Si",
        shapes=[],
    )


def _si_grating_layer() -> GratingLayer:
    geom = ShapeGeometry(
        type="rectangle",
        cx=250.0, cy=250.0,
        width=250.0, height=500.0,
    )
    shape = ShapeRegion(geometry=geom, material="Si")
    return GratingLayer(
        name="grating",
        type="grating_1d",
        thickness_nm=100.0,
        Lx_nm=500.0,
        Ly_nm=500.0,
        background_material="Air",
        shapes=[shape],
    )


def _sim(
    aoi_deg: float = 65.0,
    wavelengths: list[float] | None = None,
    n: int = 3,
    parallel: bool = False,
    output_orders: bool = False,
    li: bool = True,
) -> SimConditions:
    if wavelengths is None:
        wavelengths = [633.0]
    return SimConditions(
        aoi_deg=aoi_deg,
        azimuth_deg=0.0,
        wavelengths=WavelengthSpec(explicit=wavelengths),
        n_harmonics_x=n,
        n_harmonics_y=n,
        li_factorization=li,
        parallel_wavelengths=parallel,
        output_orders=output_orders,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_jones_matrix_shape():
    """jones_reflection must have shape (10, 2, 2) with complex dtype."""
    wls = list(np.linspace(500.0, 700.0, 10))
    db = _make_db(_air_spec(), _si_spec())
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_si_grating_layer()],
        materials={"Air": _air_spec(), "Si": _si_spec()},
    )
    result = RCWAEngine(db).run(sample, _sim(wavelengths=wls))

    assert result.jones_reflection.shape == (10, 2, 2)
    assert np.issubdtype(result.jones_reflection.dtype, np.complexfloating)


def test_energy_conservation_lossless():
    """Lossless SiO2 layer between Air and Air: R+T > 0.998."""
    db = _make_db(_air_spec(), _sio2_spec())
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_uniform_sio2_layer(100.0)],
        materials={"Air": _air_spec(), "SiO2": _sio2_spec()},
    )
    result = RCWAEngine(db).run(sample, _sim(aoi_deg=45.0, n=3))

    ec = result.energy_conservation[0]
    assert ec > 0.998, f"R+T = {ec:.6f} (expected > 0.998)"


def test_energy_conservation_lossy():
    """Lossy Si layer: R+T < 1.0 and > 0.0."""
    db = _make_db(_air_spec(), _si_spec())
    # Si is absorptive at 400 nm
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_uniform_si_layer(50.0)],
        materials={"Air": _air_spec(), "Si": _si_spec()},
    )
    result = RCWAEngine(db).run(sample, _sim(aoi_deg=45.0, wavelengths=[400.0], n=3))

    ec = result.energy_conservation[0]
    assert 0.0 < ec < 1.0, f"R+T = {ec:.6f}, expected in (0, 1)"


def test_uniform_layer_matches_fresnel():
    """RCWA |Rpp| for Air | SiO2 (thin) | SiO2 matches single-interface Fresnel r_p."""
    # Use SiO2 substrate so the thin SiO2 layer is invisible (same medium)
    sio2_substrate = MaterialSpec(name="SiO2sub", source="library", library_name="SiO2")
    sio2_layer_spec = MaterialSpec(name="SiO2", source="library", library_name="SiO2")
    db = MaterialDatabase()
    db.resolve(_air_spec())
    db.resolve(sio2_layer_spec)
    db.resolve(sio2_substrate)

    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="SiO2sub",
        layers=[_uniform_sio2_layer(0.1)],
        materials={"Air": _air_spec(), "SiO2": sio2_layer_spec, "SiO2sub": sio2_substrate},
    )
    result = RCWAEngine(db).run(sample, _sim(aoi_deg=65.0, wavelengths=[633.0], n=1))

    rpp_rcwa = abs(result.jones_reflection[0, 1, 1])

    # Analytic Fresnel r_p for Air → SiO2 at 633 nm
    n1, n2 = 1.0, 1.457
    theta1 = np.radians(65.0)
    sin_t2 = n1 / n2 * np.sin(theta1)
    theta2 = np.arcsin(sin_t2)
    r_p = (n2 * np.cos(theta1) - n1 * np.cos(theta2)) / (n2 * np.cos(theta1) + n1 * np.cos(theta2))
    rpp_fresnel = abs(r_p)

    assert abs(rpp_rcwa - rpp_fresnel) < 1e-2, (
        f"|Rpp|_RCWA={rpp_rcwa:.4f}, |r_p|_Fresnel={rpp_fresnel:.4f}, "
        f"diff={abs(rpp_rcwa-rpp_fresnel):.4f}"
    )


def test_rpp_not_nan():
    """Default Si grating sample: no NaN in jones_reflection across all wavelengths."""
    wls = list(np.linspace(400.0, 700.0, 20))
    db = _make_db(_air_spec(), _si_spec())
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_si_grating_layer()],
        materials={"Air": _air_spec(), "Si": _si_spec()},
    )
    result = RCWAEngine(db).run(sample, _sim(wavelengths=wls, n=3))

    assert not np.any(np.isnan(result.jones_reflection)), "NaN in jones_reflection"


def test_single_matches_batch():
    """run_single at 500 nm matches run() at the same wavelength (Rpp within 1e-8)."""
    db = _make_db(_air_spec(), _si_spec())
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_si_grating_layer()],
        materials={"Air": _air_spec(), "Si": _si_spec()},
    )
    engine = RCWAEngine(db)
    sim = _sim(wavelengths=[500.0], n=3)

    jr_single, _ = engine.run_single(sample, sim, 500.0)
    result_batch = engine.run(sample, sim)
    jr_batch = result_batch.jones_reflection[0]

    assert abs(jr_single[1, 1] - jr_batch[1, 1]) < 1e-8, (
        f"Rpp single={jr_single[1,1]}, batch={jr_batch[1,1]}"
    )


def test_output_orders_shape():
    """With output_orders=True, r_orders has shape (Nλ, n_orders)."""
    db = _make_db(_air_spec(), _si_spec())
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_si_grating_layer()],
        materials={"Air": _air_spec(), "Si": _si_spec()},
    )
    sim = _sim(wavelengths=[633.0], n=2, output_orders=True)
    result = RCWAEngine(db).run(sample, sim)

    n_orders = (2 * 2 + 1) ** 2  # = 25
    assert result.r_orders is not None
    assert result.r_orders.shape == (1, n_orders)


def test_energy_conservation_stored():
    """energy_conservation has shape (Nλ,) and values in [0, 1.05]."""
    db = _make_db(_air_spec(), _sio2_spec())
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_uniform_sio2_layer(100.0)],
        materials={"Air": _air_spec(), "SiO2": _sio2_spec()},
    )
    wls = [500.0, 600.0, 700.0]
    result = RCWAEngine(db).run(sample, _sim(wavelengths=wls, n=3))

    assert result.energy_conservation.shape == (3,)
    assert np.all(result.energy_conservation >= 0.0)
    assert np.all(result.energy_conservation <= 1.05)


def test_li_factorization_improves_convergence():
    """Li factorization at N=3 should be closer to N=7 reference than without Li."""
    pytest.importorskip("se_simulator.ellipsometer", reason="Step 3 not yet implemented")

    db = _make_db(_air_spec(), _si_spec())
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_si_grating_layer()],
        materials={"Air": _air_spec(), "Si": _si_spec()},
    )
    engine = RCWAEngine(db)
    sim_ref = _sim(wavelengths=[633.0], n=7, li=True)
    jr_ref, _ = engine.run_single(sample, sim_ref, 633.0)
    rpp_ref = abs(jr_ref[1, 1])

    jr_li, _ = engine.run_single(sample, _sim(wavelengths=[633.0], n=3, li=True), 633.0)
    jr_no_li, _ = engine.run_single(sample, _sim(wavelengths=[633.0], n=3, li=False), 633.0)

    err_li = abs(abs(jr_li[1, 1]) - rpp_ref)
    err_no_li = abs(abs(jr_no_li[1, 1]) - rpp_ref)
    assert err_li < err_no_li, (
        f"Li err={err_li:.4f} should be < no-Li err={err_no_li:.4f}"
    )


def test_parallel_matches_serial():
    """Parallel and serial runs must return identical jones_reflection (< 1e-8)."""
    db_s = _make_db(_air_spec(), _sio2_spec())
    db_p = _make_db(_air_spec(), _sio2_spec())
    sample = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_uniform_sio2_layer(100.0)],
        materials={"Air": _air_spec(), "SiO2": _sio2_spec()},
    )
    wls = [500.0, 550.0, 600.0, 650.0, 700.0]

    result_serial = RCWAEngine(db_s).run(sample, _sim(wavelengths=wls, n=3, parallel=False))
    result_parallel = RCWAEngine(db_p).run(sample, _sim(wavelengths=wls, n=3, parallel=True))

    max_diff = np.max(np.abs(result_serial.jones_reflection - result_parallel.jones_reflection))
    assert max_diff < 1e-8, f"Max diff between parallel and serial: {max_diff:.2e}"


# ---------------------------------------------------------------------------
# Phase 4: Stack-as-first-class-API tests
# ---------------------------------------------------------------------------


def _make_sio2_stack() -> Stack:
    """Minimal Stack: Air | 100 nm SiO2 | Air."""
    air = MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)
    sio2 = MaterialSpec(name="SiO2", source="library", library_name="SiO2")
    layer = StackLayer(
        name="sio2_layer",
        type="uniform",
        thickness_nm=100.0,
        material=sio2,
    )
    return Stack(superstrate=air, substrate=air, layers=[layer])


def test_engine_accepts_stack():
    """RCWAEngine.run() accepts a Stack directly without SampleConfig conversion by caller."""
    stack = _make_sio2_stack()
    db = _make_db(_air_spec(), _sio2_spec())
    sim = _sim(aoi_deg=45.0, wavelengths=[633.0], n=3)

    result = RCWAEngine(db).run(stack, sim)

    assert result.jones_reflection.shape == (1, 2, 2)
    assert result.energy_conservation[0] > 0.998


def test_engine_run_single_accepts_stack():
    """RCWAEngine.run_single() accepts a Stack directly."""
    stack = _make_sio2_stack()
    db = _make_db(_air_spec(), _sio2_spec())
    sim = _sim(aoi_deg=45.0, wavelengths=[633.0], n=3)

    jr, jt = RCWAEngine(db).run_single(stack, sim, 633.0)

    assert jr.shape == (2, 2)
    assert not np.any(np.isnan(jr))


def test_engine_stack_and_sampleconfig_match():
    """Stack and equivalent SampleConfig produce identical jones_reflection."""
    stack = _make_sio2_stack()
    # Equivalent SampleConfig (backward-compat path)
    sample_config = SampleConfig(
        Lx_nm=500.0, Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[_uniform_sio2_layer(100.0)],
        materials={"Air": _air_spec(), "SiO2": _sio2_spec()},
    )

    db1 = _make_db(_air_spec(), _sio2_spec())
    db2 = _make_db(_air_spec(), _sio2_spec())
    sim = _sim(aoi_deg=45.0, wavelengths=[500.0, 633.0, 700.0], n=3)

    result_stack = RCWAEngine(db1).run(stack, sim)
    result_cfg = RCWAEngine(db2).run(sample_config, sim)

    max_diff = np.max(np.abs(result_stack.jones_reflection - result_cfg.jones_reflection))
    assert max_diff < 1e-10, f"Stack vs SampleConfig max diff: {max_diff:.2e}"
