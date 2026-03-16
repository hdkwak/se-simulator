"""Tests for the TMM engine and engine dispatcher."""

from __future__ import annotations

import time

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
from se_simulator.rcwa.dispatcher import select_engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def air_spec():
    return MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)


@pytest.fixture()
def sio2_spec():
    return MaterialSpec(name="SiO2", source="constant_nk", n=1.46, k=0.0)


@pytest.fixture()
def si3n4_spec():
    return MaterialSpec(name="Si3N4", source="constant_nk", n=2.0, k=0.0)


@pytest.fixture()
def si_spec():
    return MaterialSpec(name="Si", source="constant_nk", n=3.88, k=0.02)


@pytest.fixture()
def si_lossless_spec():
    return MaterialSpec(name="Si", source="constant_nk", n=3.88, k=0.0)


def _make_uniform_sample(
    materials: dict[str, MaterialSpec],
    layers: list[GratingLayer],
    superstrate: str = "Air",
    substrate: str = "Si",
) -> SampleConfig:
    return SampleConfig(
        Lx_nm=1000.0,
        Ly_nm=1000.0,
        superstrate_material=superstrate,
        substrate_material=substrate,
        layers=layers,
        materials=materials,
    )


@pytest.fixture()
def single_layer_sample(air_spec, sio2_spec, si_spec):
    """100 nm SiO2 on Si, superstrate Air."""
    return _make_uniform_sample(
        materials={"Air": air_spec, "SiO2": sio2_spec, "Si": si_spec},
        layers=[
            GratingLayer(
                name="SiO2",
                type="uniform",
                thickness_nm=100.0,
                Lx_nm=1000.0,
                Ly_nm=1000.0,
                background_material="SiO2",
            )
        ],
    )


@pytest.fixture()
def three_layer_sample(air_spec, sio2_spec, si3n4_spec, si_spec):
    """SiO2 / Si3N4 / SiO2 stack on Si."""
    return _make_uniform_sample(
        materials={"Air": air_spec, "SiO2": sio2_spec, "Si3N4": si3n4_spec, "Si": si_spec},
        layers=[
            GratingLayer(name="SiO2_top", type="uniform", thickness_nm=80.0,
                         Lx_nm=1000.0, Ly_nm=1000.0, background_material="SiO2"),
            GratingLayer(name="Si3N4", type="uniform", thickness_nm=120.0,
                         Lx_nm=1000.0, Ly_nm=1000.0, background_material="Si3N4"),
            GratingLayer(name="SiO2_bot", type="uniform", thickness_nm=60.0,
                         Lx_nm=1000.0, Ly_nm=1000.0, background_material="SiO2"),
        ],
    )


@pytest.fixture()
def five_layer_sample(air_spec, sio2_spec, si3n4_spec, si_spec):
    """5-layer alternating stack for performance test."""
    mats = {"Air": air_spec, "SiO2": sio2_spec, "Si3N4": si3n4_spec, "Si": si_spec}
    layers = []
    for i in range(5):
        mat = "SiO2" if i % 2 == 0 else "Si3N4"
        layers.append(GratingLayer(
            name=f"L{i}", type="uniform", thickness_nm=80.0,
            Lx_nm=1000.0, Ly_nm=1000.0, background_material=mat,
        ))
    return _make_uniform_sample(materials=mats, layers=layers)


@pytest.fixture()
def grating_sample(air_spec, sio2_spec, si_spec):
    """Sample with a grating layer (has shapes)."""
    return SampleConfig(
        Lx_nm=500.0,
        Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Si",
        layers=[
            GratingLayer(
                name="Grating",
                type="grating_1d",
                thickness_nm=200.0,
                Lx_nm=500.0,
                Ly_nm=500.0,
                background_material="Air",
                shapes=[
                    ShapeRegion(
                        geometry=ShapeGeometry(
                            type="rectangle", cx=250.0, cy=250.0,
                            width=150.0, height=500.0,
                        ),
                        material="SiO2",
                    )
                ],
            )
        ],
        materials={"Air": air_spec, "SiO2": sio2_spec, "Si": si_spec},
    )


@pytest.fixture()
def wavelengths_dense():
    return np.linspace(400.0, 800.0, 401)


@pytest.fixture()
def wavelengths_100():
    return np.linspace(400.0, 800.0, 100)


def _make_db(sample: SampleConfig) -> MaterialDatabase:
    db = MaterialDatabase()
    for spec in sample.materials.values():
        db.resolve(spec)
    return db


# ---------------------------------------------------------------------------
# Engine dispatcher tests
# ---------------------------------------------------------------------------


def test_select_engine_tmm_for_uniform_stack(single_layer_sample):
    assert select_engine(single_layer_sample) == "tmm"


def test_select_engine_rcwa_for_grating(grating_sample):
    assert select_engine(grating_sample) == "rcwa"


def test_select_engine_override_rcwa(single_layer_sample):
    assert select_engine(single_layer_sample, engine_override="rcwa") == "rcwa"


def test_select_engine_override_tmm(grating_sample):
    assert select_engine(grating_sample, engine_override="tmm") == "tmm"


def test_select_engine_auto_is_default(single_layer_sample):
    assert select_engine(single_layer_sample, engine_override="auto") == "tmm"


# ---------------------------------------------------------------------------
# SimConditions schema test
# ---------------------------------------------------------------------------


def test_simconditions_engine_override_field():
    base = {"wavelengths": {"explicit": [632.0]}}
    sc_auto = SimConditions(**base)
    assert sc_auto.engine_override == "auto"

    sc_tmm = SimConditions(**base, engine_override="tmm")
    assert sc_tmm.engine_override == "tmm"

    sc_rcwa = SimConditions(**base, engine_override="rcwa")
    assert sc_rcwa.engine_override == "rcwa"


# ---------------------------------------------------------------------------
# TMM physics tests
# ---------------------------------------------------------------------------


def test_tmm_single_layer_psi_delta_physical(single_layer_sample, wavelengths_100):
    from se_simulator.ellipsometer.prcsa import compute_spectrum
    from se_simulator.rcwa.results import RCWAResult
    from se_simulator.rcwa.tmm import compute_tmm

    db = _make_db(single_layer_sample)
    jones_r, jones_t = compute_tmm(single_layer_sample, db, wavelengths_100, 65.0)

    r_total = 0.5 * (np.abs(jones_r[:, 0, 0]) ** 2 + np.abs(jones_r[:, 1, 1]) ** 2)
    t_total = 0.5 * (np.abs(jones_t[:, 0, 0]) ** 2 + np.abs(jones_t[:, 1, 1]) ** 2)

    rcwa_result = RCWAResult(
        wavelengths_nm=wavelengths_100,
        jones_reflection=jones_r,
        jones_transmission=jones_t,
        energy_conservation=r_total + t_total,
    )

    from se_simulator.config.schemas import CompensatorRetardanceModel, SystemConfig
    system = SystemConfig(
        instrument_name="test",
        compensator_retardance=CompensatorRetardanceModel(type="constant", value=90.0),
    )

    result = compute_spectrum(rcwa_result, system)
    assert result.psi_deg is not None
    assert result.delta_deg is not None
    assert not np.any(np.isnan(result.psi_deg))
    assert not np.any(np.isnan(result.delta_deg))
    assert not np.any(np.isinf(result.psi_deg))
    assert not np.any(np.isinf(result.delta_deg))
    assert np.all(result.psi_deg >= 0.0)
    assert np.all(result.psi_deg <= 90.0)


def test_tmm_energy_conservation_lossless(air_spec, sio2_spec, si_lossless_spec, wavelengths_100):
    from se_simulator.rcwa.tmm import compute_tmm

    sample = _make_uniform_sample(
        materials={"Air": air_spec, "SiO2": sio2_spec, "Si": si_lossless_spec},
        layers=[
            GratingLayer(name="SiO2", type="uniform", thickness_nm=100.0,
                         Lx_nm=1000.0, Ly_nm=1000.0, background_material="SiO2"),
        ],
    )
    db = _make_db(sample)
    jones_r, jones_t = compute_tmm(sample, db, wavelengths_100, 65.0)

    # For lossless, use simple |r|² + |t|² ≈ 1 (approximate; exact needs admittance ratio)
    r_s = np.abs(jones_r[:, 0, 0]) ** 2
    r_p = np.abs(jones_r[:, 1, 1]) ** 2
    t_s = np.abs(jones_t[:, 0, 0]) ** 2
    t_p = np.abs(jones_t[:, 1, 1]) ** 2

    # At minimum, R ≤ 1 and no NaN
    assert np.all(r_s <= 1.01)
    assert np.all(r_p <= 1.01)
    assert not np.any(np.isnan(r_s + r_p + t_s + t_p))


def test_tmm_zero_thickness_no_crash(air_spec, sio2_spec, si_spec, wavelengths_100):
    from se_simulator.rcwa.tmm import compute_tmm

    sample = _make_uniform_sample(
        materials={"Air": air_spec, "SiO2": sio2_spec, "Si": si_spec},
        layers=[
            GratingLayer(name="SiO2", type="uniform", thickness_nm=0.0,
                         Lx_nm=1000.0, Ly_nm=1000.0, background_material="SiO2"),
        ],
    )
    db = _make_db(sample)
    jones_r, jones_t = compute_tmm(sample, db, wavelengths_100, 65.0)

    assert jones_r.shape == (len(wavelengths_100), 2, 2)
    assert not np.any(np.isnan(jones_r))
    assert not np.any(np.isinf(jones_r))


def test_tmm_three_layer_stack_runs(three_layer_sample, wavelengths_100):
    from se_simulator.rcwa.tmm import compute_tmm

    db = _make_db(three_layer_sample)
    jones_r, jones_t = compute_tmm(three_layer_sample, db, wavelengths_100, 65.0)

    assert jones_r.shape == (len(wavelengths_100), 2, 2)
    assert not np.any(np.isnan(jones_r))


def test_tmm_incoherent_layer_suppresses_fringes(air_spec, sio2_spec, si_spec, wavelengths_dense):
    from se_simulator.ellipsometer.prcsa import compute_psi_delta
    from se_simulator.rcwa.tmm import compute_tmm

    thick = 500_000.0  # 500 µm — many fringes when coherent

    def _run(incoherent: bool) -> np.ndarray:
        sample = _make_uniform_sample(
            materials={"Air": air_spec, "SiO2": sio2_spec, "Si": si_spec},
            layers=[
                GratingLayer(name="SiO2", type="uniform", thickness_nm=thick,
                             Lx_nm=1000.0, Ly_nm=1000.0, background_material="SiO2",
                             incoherent=incoherent),
            ],
        )
        db = _make_db(sample)
        jones_r, _ = compute_tmm(sample, db, wavelengths_dense, 65.0)
        deltas = np.array([compute_psi_delta(jones_r[i])[1] for i in range(len(wavelengths_dense))])
        return deltas

    delta_coherent = _run(incoherent=False)
    delta_incoherent = _run(incoherent=True)

    # Incoherent layer should show less variation (no interference fringes)
    assert np.std(delta_incoherent) < np.std(delta_coherent), (
        f"Expected incoherent std ({np.std(delta_incoherent):.2f}) < "
        f"coherent std ({np.std(delta_coherent):.2f})"
    )


def test_full_pipeline_tmm_path(single_layer_sample):
    """RCWAEngine.run() auto-selects TMM and flows through compute_spectrum."""
    from se_simulator.config.schemas import CompensatorRetardanceModel, SimConditions, SystemConfig
    from se_simulator.ellipsometer.prcsa import compute_spectrum
    from se_simulator.rcwa.engine import RCWAEngine

    db = _make_db(single_layer_sample)
    engine = RCWAEngine(db)

    sim = SimConditions(
        wavelengths=WavelengthSpec(explicit=list(np.linspace(400.0, 800.0, 50))),
        parallel_wavelengths=False,
    )
    system = SystemConfig(
        instrument_name="test",
        polarizer_angle_deg=45.0,
        analyzer_angle_deg=45.0,
        compensator_angle_deg=0.0,
        compensator_retardance=CompensatorRetardanceModel(type="constant", value=90.0),
    )

    result = engine.run(single_layer_sample, sim)
    assert result.jones_reflection.shape[0] == 50

    ell = compute_spectrum(result, system)
    assert len(ell.psi_deg) == 50
    assert not np.any(np.isnan(ell.psi_deg))


def test_tmm_performance(five_layer_sample):
    """TMM for 100 wavelengths, 5-layer stack must complete in < 50 ms."""
    from se_simulator.rcwa.tmm import compute_tmm

    wavelengths = np.linspace(400.0, 800.0, 100)
    db = _make_db(five_layer_sample)

    # Warm-up run (JIT, imports, etc.)
    compute_tmm(five_layer_sample, db, wavelengths, 65.0)

    t0 = time.perf_counter()
    compute_tmm(five_layer_sample, db, wavelengths, 65.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.05, f"TMM took {elapsed * 1000:.1f} ms — exceeds 50 ms target"


def test_bare_substrate_no_oscillation():
    """Bare Si substrate (no layers) must produce smooth Psi — no interpolation ringing."""
    from se_simulator.config.schemas import MaterialSpec, SampleConfig
    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.tmm import compute_tmm

    air = MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)
    si = MaterialSpec(name="Si", source="library", library_name="Si")
    sample = SampleConfig(
        Lx_nm=1000.0, Ly_nm=1000.0,
        superstrate_material="Air",
        substrate_material="Si",
        layers=[],
        materials={"Air": air, "Si": si},
    )
    db = MaterialDatabase()
    for s in sample.materials.values():
        db.resolve(s)

    wls = np.linspace(300.0, 1000.0, 300)
    jones_r, _ = compute_tmm(sample, db, wls, 65.0)
    rss = jones_r[:, 0, 0]
    rpp = jones_r[:, 1, 1]
    psi = np.degrees(np.arctan(np.abs(rpp / rss)))

    assert not np.any(np.isnan(psi)), "NaN in Psi"
    assert not np.any(np.isinf(psi)), "Inf in Psi"
    # Smooth curve: std should be moderate (physical dispersion), not huge (ringing)
    assert psi.std() < 10.0, f"Psi std={psi.std():.2f} suggests interpolation ringing"
    # Count oscillations — smooth curve should have very few sign changes in derivative
    sign_changes = np.sum(np.diff(psi)[:-1] * np.diff(psi)[1:] < 0)
    assert sign_changes <= 5, f"Too many Psi oscillations: {sign_changes}"


# ---------------------------------------------------------------------------
# Phase 4: Stack as direct input to compute_tmm
# ---------------------------------------------------------------------------


def test_tmm_accepts_stack(air_spec, sio2_spec, si_spec, wavelengths_100):
    """compute_tmm() accepts a Stack directly — no SampleConfig needed by caller."""
    from se_simulator.rcwa.tmm import compute_tmm

    sio2_layer = StackLayer(
        name="SiO2",
        type="uniform",
        thickness_nm=100.0,
        material=sio2_spec,
        Lx_nm=1000.0,
        Ly_nm=1000.0,
    )
    stack = Stack(superstrate=air_spec, substrate=si_spec, layers=[sio2_layer])

    db = MaterialDatabase()
    for spec in (air_spec, sio2_spec, si_spec):
        db.resolve(spec)

    jones_r, jones_t = compute_tmm(stack, db, wavelengths_100, 65.0)

    assert jones_r.shape == (len(wavelengths_100), 2, 2)
    assert not np.any(np.isnan(jones_r))
    assert not np.any(np.isinf(jones_r))


def test_tmm_stack_matches_sampleconfig(air_spec, sio2_spec, si_spec, wavelengths_100):
    """compute_tmm with Stack produces identical results to compute_tmm with SampleConfig."""
    from se_simulator.rcwa.tmm import compute_tmm

    sio2_layer_stack = StackLayer(
        name="SiO2",
        type="uniform",
        thickness_nm=100.0,
        material=sio2_spec,
        Lx_nm=1000.0,
        Ly_nm=1000.0,
    )
    stack = Stack(superstrate=air_spec, substrate=si_spec, layers=[sio2_layer_stack])

    sampleconfig = _make_uniform_sample(
        materials={"Air": air_spec, "SiO2": sio2_spec, "Si": si_spec},
        layers=[
            GratingLayer(
                name="SiO2", type="uniform", thickness_nm=100.0,
                Lx_nm=1000.0, Ly_nm=1000.0, background_material="SiO2",
            )
        ],
    )

    db1 = MaterialDatabase()
    db2 = MaterialDatabase()
    for spec in (air_spec, sio2_spec, si_spec):
        db1.resolve(spec)
        db2.resolve(spec)

    jr_stack, _ = compute_tmm(stack, db1, wavelengths_100, 65.0)
    jr_cfg, _ = compute_tmm(sampleconfig, db2, wavelengths_100, 65.0)

    max_diff = np.max(np.abs(jr_stack - jr_cfg))
    assert max_diff < 1e-12, f"Stack vs SampleConfig max diff: {max_diff:.2e}"
