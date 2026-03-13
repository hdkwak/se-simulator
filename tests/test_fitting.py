"""Tests for the Step 4 fitting engine (library generation and search)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from se_simulator.config.schemas import (
    FittingConditions,
    SampleConfig,
    SimConditions,
    SystemConfig,
)
from se_simulator.ellipsometer.signals import EllipsometryResult
from se_simulator.fitting.engine import FitResult, FittingEngine
from se_simulator.fitting.interpolation import LibraryInterpolator
from se_simulator.fitting.library import (
    LibraryGenerator,
    LibrarySpec,
    LibraryStore,
    ParameterSpec,
    apply_params,
    build_library_spec,
)
from se_simulator.fitting.search import NearestNeighborSearch, SearchResult, chi_squared


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_result(n_wl: int = 10) -> EllipsometryResult:
    wl = np.linspace(400.0, 700.0, n_wl)
    return EllipsometryResult(
        wavelengths_nm=wl,
        psi_deg=np.full(n_wl, 30.0),
        delta_deg=np.full(n_wl, 90.0),
        alpha=np.zeros(n_wl),
        beta=np.zeros(n_wl),
        chi=np.zeros(n_wl),
        xi=np.zeros(n_wl),
        jones_reflection=np.zeros((n_wl, 2, 2), dtype=complex),
        energy_conservation=np.ones(n_wl),
    )


def _make_test_sample(thickness_nm: float = 100.0) -> SampleConfig:
    """A simple SiO2-like uniform film on Si-like substrate (constant nk)."""
    return SampleConfig.model_validate({
        "schema_version": "1.0",
        "sample_id": "test_film",
        "Lx_nm": 500.0,
        "Ly_nm": 500.0,
        "superstrate_material": "Air",
        "substrate_material": "Sub",
        "layers": [
            {
                "name": "film",
                "type": "uniform",
                "thickness_nm": thickness_nm,
                "Lx_nm": 500.0,
                "Ly_nm": 500.0,
                "background_material": "Film",
                "shapes": [],
            }
        ],
        "materials": {
            "Air": {"name": "Air", "source": "constant_nk", "n": 1.0, "k": 0.0},
            "Film": {"name": "Film", "source": "constant_nk", "n": 1.46, "k": 0.0},
            "Sub": {"name": "Sub", "source": "constant_nk", "n": 3.9, "k": 0.0},
        },
    })


def _make_test_system() -> SystemConfig:
    return SystemConfig.model_validate({
        "schema_version": "1.0",
        "instrument_name": "Test",
        "polarizer_angle_deg": 45.0,
        "analyzer_angle_deg": 45.0,
        "compensator_angle_deg": 0.0,
        "compensator_retardance": {
            "type": "constant",
            "value": 90.0,
            "coefficients": [],
            "file_path": None,
        },
        "n_revolutions": 20,
        "n_points_per_revolution": 50,
    })


def _make_test_sim(use_refinement: bool = False) -> SimConditions:
    """Few-wavelength, 1-harmonic SimConditions for fast tests."""
    return SimConditions.model_validate({
        "schema_version": "1.0",
        "aoi_deg": 65.0,
        "azimuth_deg": 0.0,
        "wavelengths": {"range": [400.0, 700.0, 100.0]},
        "n_harmonics_x": 1,
        "n_harmonics_y": 1,
        "li_factorization": True,
        "parallel_wavelengths": False,
        "output_jones": False,
        "output_orders": False,
        "fitting": {
            "fit_signals": ["psi", "delta"],
            "sigma_psi": 0.05,
            "sigma_delta": 0.1,
            "top_k_candidates": 10,
            "use_interpolation": True,
            "use_refinement": use_refinement,
            "refinement_algo": "Nelder-Mead",
            "max_iterations": 50,
            "convergence_tol": 1e-4,
        },
    })


def _simulate_target(thickness_nm: float) -> EllipsometryResult:
    """Run a real RCWA + ellipsometer simulation for a given thickness."""
    from se_simulator.ellipsometer.prcsa import compute_spectrum
    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.engine import RCWAEngine

    sample = _make_test_sample(thickness_nm)
    system = _make_test_system()
    sim = _make_test_sim()
    db = MaterialDatabase()
    engine = RCWAEngine(db)
    rcwa_result = engine.run(sample, sim)
    return compute_spectrum(rcwa_result, system)


def _build_small_library(
    tmp_path: Path,
    thickness_values: list[float],
) -> tuple[LibraryStore, LibrarySpec]:
    """Generate a small 1-parameter library for the given thickness values."""
    sample = _make_test_sample()
    system = _make_test_system()
    sim = _make_test_sim()

    n = len(thickness_values)
    param_spec = ParameterSpec(
        name="layers[0].thickness_nm",
        min_value=float(thickness_values[0]),
        max_value=float(thickness_values[-1]),
        n_points=n,
    )
    lib_spec = build_library_spec([param_spec], sample, system, sim, signals=["psi", "delta"])

    lib_path = tmp_path / "test_library.h5"
    store = LibraryStore(lib_path)
    store.create(lib_spec, overwrite=True)

    gen = LibraryGenerator(lib_spec, store, n_workers=1)
    gen.generate(resume=False)

    return store, lib_spec


# ---------------------------------------------------------------------------
# Unit tests — no simulation required
# ---------------------------------------------------------------------------


def test_chi_squared_zero_for_identical() -> None:
    """χ²(target, library) = 0 when target equals the library entry."""
    target = {"psi": np.array([30.0, 35.0, 40.0])}
    library = {"psi": np.array([[30.0, 35.0, 40.0], [31.0, 36.0, 41.0]])}
    sigma = {"psi": 1.0}
    chi2 = chi_squared(target, library, sigma)
    assert chi2.shape == (2,)
    assert float(chi2[0]) == pytest.approx(0.0, abs=1e-10)
    assert float(chi2[1]) > 0.0


def test_chi_squared_vectorized_shape() -> None:
    """chi_squared() with N=100 library entries returns shape (100,)."""
    n_lib = 100
    n_wl = 50
    rng = np.random.default_rng(42)
    target = {"psi": rng.random(n_wl), "delta": rng.random(n_wl)}
    library = {"psi": rng.random((n_lib, n_wl)), "delta": rng.random((n_lib, n_wl))}
    sigma = {"psi": 0.05, "delta": 0.1}
    chi2 = chi_squared(target, library, sigma)
    assert chi2.shape == (n_lib,)
    assert np.all(chi2 >= 0.0)


def test_chi_squared_increases_with_error() -> None:
    """χ² is strictly larger when the target deviates more from the library."""
    n_wl = 20
    base = np.zeros(n_wl)
    target_small = {"psi": base + 0.1}
    target_large = {"psi": base + 1.0}
    library = {"psi": np.zeros((1, n_wl))}
    sigma = {"psi": 1.0}

    chi2_small = float(chi_squared(target_small, library, sigma)[0])
    chi2_large = float(chi_squared(target_large, library, sigma)[0])
    assert chi2_large > chi2_small


# ---------------------------------------------------------------------------
# LibraryStore tests
# ---------------------------------------------------------------------------


def test_library_create_and_write(tmp_path: Path) -> None:
    """Create a 2-param × 3-point library; write 9 mock entries; check completeness."""
    n_wl = 8
    params = [
        ParameterSpec("layers[0].thickness_nm", 80.0, 120.0, 3),
        ParameterSpec("layers[0].Lx_nm", 400.0, 600.0, 3),
    ]
    spec = LibrarySpec(
        parameters=params,
        system_config_snapshot={},
        sample_config_snapshot={},
        sim_conditions_snapshot={},
        created_at="2026-01-01T00:00:00+00:00",
        rcwa_version="0.1.0",
        n_wavelengths=n_wl,
        wavelengths_nm=list(np.linspace(400.0, 700.0, n_wl)),
        signals=["psi", "delta"],
    )

    lib_path = tmp_path / "lib.h5"
    store = LibraryStore(lib_path)
    store.create(spec)

    assert store.n_entries == 9
    assert not store.is_complete

    mock = _make_mock_result(n_wl)
    for i in range(9):
        store.write_entry(i, np.array([100.0 + i, 500.0]), mock)

    assert store.is_complete


def test_library_read_all(tmp_path: Path) -> None:
    """Write 9 entries; read_all() returns correct shapes."""
    n_wl = 5
    params = [
        ParameterSpec("layers[0].thickness_nm", 80.0, 120.0, 3),
        ParameterSpec("layers[0].Lx_nm", 400.0, 600.0, 3),
    ]
    spec = LibrarySpec(
        parameters=params,
        system_config_snapshot={},
        sample_config_snapshot={},
        sim_conditions_snapshot={},
        created_at="2026-01-01T00:00:00+00:00",
        rcwa_version="0.1.0",
        n_wavelengths=n_wl,
        wavelengths_nm=list(np.linspace(400.0, 700.0, n_wl)),
        signals=["psi", "delta"],
    )

    lib_path = tmp_path / "lib.h5"
    store = LibraryStore(lib_path)
    store.create(spec)

    mock = _make_mock_result(n_wl)
    for i in range(9):
        store.write_entry(i, np.array([float(i), float(i) * 2]), mock)

    parameters, spectra = store.read_all()
    assert parameters.shape == (9, 2)
    assert spectra["psi"].shape == (9, n_wl)
    assert spectra["delta"].shape == (9, n_wl)


def test_parameter_path_apply() -> None:
    """_apply_params sets layers[0].thickness_nm on a deep copy; original unchanged."""
    base = _make_test_sample(100.0)
    param_specs = [ParameterSpec("layers[0].thickness_nm", 80.0, 200.0, 5)]

    modified = apply_params(base, param_specs, (150.0,))

    assert modified.layers[0].thickness_nm == pytest.approx(150.0)
    assert base.layers[0].thickness_nm == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Integration tests — require actual RCWA simulation
# ---------------------------------------------------------------------------


def test_search_finds_correct_entry(tmp_path: Path) -> None:
    """NearestNeighborSearch must rank the 100 nm entry first for a 100 nm target."""
    thickness_values = [80.0, 90.0, 100.0, 110.0, 120.0]
    store, lib_spec = _build_small_library(tmp_path, thickness_values)

    target = _simulate_target(100.0)
    sim = _make_test_sim()

    searcher = NearestNeighborSearch(store, sim.fitting)
    result = searcher.search(target)

    assert abs(float(result.best_fit_params[0]) - 100.0) < 1.0


def test_fit_result_csv_round_trip(tmp_path: Path) -> None:
    """FitResult.to_csv() writes a readable file; re-read values match."""
    n_wl = 5
    mock_spectrum = _make_mock_result(n_wl)

    # Build a minimal SearchResult
    search_result = SearchResult(
        top_k_indices=np.array([0, 1, 2]),
        top_k_chi2=np.array([0.01, 0.5, 1.2]),
        top_k_params=np.array([[100.0], [90.0], [110.0]]),
        parameter_names=["layers[0].thickness_nm"],
        best_fit_params=np.array([100.0]),
        best_fit_chi2=0.01,
        best_fit_spectrum=mock_spectrum,
    )

    fit = FitResult(
        stage1_search=search_result,
        stage2_interpolation=None,
        stage3_refinement=None,
        final_params=np.array([97.5]),
        final_chi2=0.003,
        parameter_names=["layers[0].thickness_nm"],
        sigma_params=np.array([2.1]),
        best_fit_spectrum=mock_spectrum,
        pipeline_stages_run=["search", "interpolation"],
    )

    csv_path = tmp_path / "result.csv"
    fit.to_csv(csv_path)

    # Parse the CSV back
    text = csv_path.read_text()
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert lines[0] == "parameter_name,fitted_value,sigma_1"
    parts = lines[1].split(",")
    assert parts[0] == "layers[0].thickness_nm"
    assert float(parts[1]) == pytest.approx(97.5, abs=1e-4)
    assert float(parts[2]) == pytest.approx(2.1, abs=1e-4)


def test_fitting_engine_run(tmp_path: Path) -> None:
    """Integration: 5-point library, target=95 nm, interpolation-only fit within 8 nm."""
    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.engine import RCWAEngine

    thickness_values = [80.0, 90.0, 100.0, 110.0, 120.0]
    store, lib_spec = _build_small_library(tmp_path, thickness_values)

    target = _simulate_target(95.0)
    sim = _make_test_sim(use_refinement=False)

    db = MaterialDatabase()
    engine = RCWAEngine(db)
    system = _make_test_system()

    fit_engine = FittingEngine(
        library_path=tmp_path / "test_library.h5",
        rcwa_engine=engine,
        system=system,
        sim=sim,
    )

    fit_config = FittingConditions(
        fit_signals=["psi", "delta"],
        sigma_psi=0.05,
        sigma_delta=0.1,
        top_k_candidates=5,
        use_interpolation=True,
        use_refinement=False,
    )

    result = fit_engine.fit(target, fitting_config=fit_config)

    assert "search" in result.pipeline_stages_run
    assert "interpolation" in result.pipeline_stages_run
    assert abs(float(result.final_params[0]) - 95.0) < 8.0


def test_chi2_map_shape(tmp_path: Path) -> None:
    """LibraryInterpolator.chi2_map() returns arrays of correct shape."""
    n_wl = 4
    # Build a 2-parameter library with mock data (no RCWA needed)
    params = [
        ParameterSpec("layers[0].thickness_nm", 80.0, 120.0, 3),
        ParameterSpec("layers[0].Lx_nm", 400.0, 600.0, 3),
    ]
    spec = LibrarySpec(
        parameters=params,
        system_config_snapshot={},
        sample_config_snapshot={},
        sim_conditions_snapshot={},
        created_at="2026-01-01T00:00:00+00:00",
        rcwa_version="0.1.0",
        n_wavelengths=n_wl,
        wavelengths_nm=list(np.linspace(400.0, 700.0, n_wl)),
        signals=["psi", "delta"],
    )

    lib_path = tmp_path / "lib2d.h5"
    store = LibraryStore(lib_path)
    store.create(spec)

    mock = _make_mock_result(n_wl)
    for i in range(9):
        store.write_entry(i, np.array([80.0 + (i % 3) * 20.0, 400.0 + (i // 3) * 100.0]), mock)

    mock_spectrum = _make_mock_result(n_wl)
    all_params = np.array([
        [80.0, 400.0], [100.0, 400.0], [120.0, 400.0],
        [80.0, 500.0], [100.0, 500.0], [120.0, 500.0],
        [80.0, 600.0], [100.0, 600.0], [120.0, 600.0],
    ])
    rng = np.random.default_rng(0)
    chi2_vals = rng.random(9)

    search_result = SearchResult(
        top_k_indices=np.arange(9),
        top_k_chi2=chi2_vals,
        top_k_params=all_params,
        parameter_names=["layers[0].thickness_nm", "layers[0].Lx_nm"],
        best_fit_params=np.array([100.0, 500.0]),
        best_fit_chi2=float(chi2_vals.min()),
        best_fit_spectrum=mock_spectrum,
    )

    interpolator = LibraryInterpolator(store, search_result)
    n_pts = 20
    x_vals, y_vals, chi2_grid = interpolator.chi2_map(0, 1, n_points=n_pts)

    assert x_vals.shape == (n_pts,)
    assert y_vals.shape == (n_pts,)
    assert chi2_grid.shape == (n_pts, n_pts)
