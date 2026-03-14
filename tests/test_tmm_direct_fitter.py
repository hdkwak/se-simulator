"""Tests for se_simulator.fitting.tmm_direct_fitter.TmmDirectFitter."""
from __future__ import annotations

import numpy as np
import pytest

from se_simulator.config.recipe import FittingConfiguration, FloatingParameter
from se_simulator.config.schemas import (
    GratingLayer,
    MaterialSpec,
    SampleConfig,
    SimConditions,
    SystemConfig,
    WavelengthSpec,
)
from se_simulator.fitting.tmm_direct_fitter import TmmDirectFitter, _strip_prefix

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


def _make_sio2_si_sample(thickness_nm: float = 97.3) -> SampleConfig:
    """SiO2 on Si sample — standard ellipsometry test stack.

    Uses constant_nk / cauchy models so the test does not depend on
    the library CSV files (which have a known header-comment issue).
    """
    mat_air = MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)
    # SiO2: realistic Cauchy dispersion (Malitson coefficients, simplified)
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


def _make_sim_conditions(wl_start: float = 400.0, wl_end: float = 800.0, step: float = 20.0) -> SimConditions:
    return SimConditions(
        aoi_deg=65.0,
        wavelengths=WavelengthSpec(range=(wl_start, wl_end, step)),
    )


def _make_system_config() -> SystemConfig:
    return SystemConfig(
        instrument_name="test",
        polarizer_angle_deg=45.0,
        analyzer_angle_deg=45.0,
    )


def _make_floating_params(lo: float = 50.0, hi: float = 200.0) -> list[FloatingParameter]:
    return [
        FloatingParameter(
            name="thickness",
            target_field="forward_model.sample.inline.layers[0].thickness_nm",
            min=lo,
            max=hi,
            initial=100.0,
            step=5.0,
        )
    ]


def _make_fitting_config(optimizer: str = "levenberg_marquardt") -> FittingConfiguration:
    return FittingConfiguration(
        fitting_mode="tmm_direct",
        fit_signals=["psi", "delta"],
        optimizer=optimizer,
        max_iterations=300,
        convergence_tolerance=1e-6,
    )


def _generate_synthetic_spectrum(
    sample: SampleConfig,
    sim: SimConditions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate psi/delta from the TMM engine for the given sample.

    Returns (psi_deg, delta_deg, wavelengths_nm).
    """
    from se_simulator.config.manager import ConfigManager
    from se_simulator.ellipsometer.prcsa import compute_psi_delta
    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.tmm import compute_tmm

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
    return psi, delta, wls


# ---------------------------------------------------------------------------
# Test 1: LM convergence on SiO2/Si
# ---------------------------------------------------------------------------


def test_lm_convergence_sio2_si():
    """LM fitter must converge to within 0.5 nm of the true thickness (97.3 nm)."""
    true_thickness = 97.3
    sample_true = _make_sio2_si_sample(thickness_nm=true_thickness)
    sim = _make_sim_conditions()
    system = _make_system_config()

    psi, delta, _ = _generate_synthetic_spectrum(sample_true, sim)
    target = np.concatenate([psi, delta])

    # Start from 100.0 nm (offset of ~2.7 nm)
    floating = _make_floating_params(lo=50.0, hi=200.0)
    floating[0] = FloatingParameter(
        name="thickness",
        target_field="forward_model.sample.inline.layers[0].thickness_nm",
        min=50.0,
        max=200.0,
        initial=100.0,
        step=5.0,
    )

    fitter = TmmDirectFitter(
        sample_config=_make_sio2_si_sample(thickness_nm=100.0),
        sim_conditions=sim,
        system_config=system,
        floating_params=floating,
        fitting_config=_make_fitting_config("levenberg_marquardt"),
    )
    result = fitter.fit(target)
    fitted_t = result.fitted_parameters["thickness"]
    assert abs(fitted_t - true_thickness) < 0.5, (
        f"Fitted thickness {fitted_t:.3f} nm is not within 0.5 nm of truth {true_thickness} nm"
    )


# ---------------------------------------------------------------------------
# Test 2: forward model is deterministic
# ---------------------------------------------------------------------------


def test_forward_deterministic():
    """Two calls with the same x should return identical results."""
    sample = _make_sio2_si_sample(100.0)
    sim = _make_sim_conditions()
    system = _make_system_config()
    floating = _make_floating_params()
    fitter = TmmDirectFitter(
        sample_config=sample,
        sim_conditions=sim,
        system_config=system,
        floating_params=floating,
        fitting_config=_make_fitting_config(),
    )
    x = np.array([120.0])
    out1 = fitter._forward(x)
    out2 = fitter._forward(x)
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Test 3: fit_signals='psi' only uses psi
# ---------------------------------------------------------------------------


def test_fit_signals_psi_only():
    """fit_signals=['psi'] should return a vector of length n_wavelengths."""
    sample = _make_sio2_si_sample(100.0)
    sim = _make_sim_conditions()
    system = _make_system_config()
    floating = _make_floating_params()

    fc = FittingConfiguration(
        fitting_mode="tmm_direct",
        fit_signals=["psi"],
        optimizer="levenberg_marquardt",
    )
    fitter = TmmDirectFitter(
        sample_config=sample,
        sim_conditions=sim,
        system_config=system,
        floating_params=floating,
        fitting_config=fc,
    )
    x = np.array([100.0])
    signal = fitter._forward(x)
    from se_simulator.config.manager import ConfigManager

    n_wl = len(ConfigManager().get_wavelengths(sim.wavelengths))
    assert signal.shape == (n_wl,)


# ---------------------------------------------------------------------------
# Test 4: Nelder-Mead optimizer converges
# ---------------------------------------------------------------------------


def test_nelder_mead_convergence():
    """Nelder-Mead fitter should converge to within 2 nm of the true thickness."""
    true_thickness = 97.3
    sample_true = _make_sio2_si_sample(thickness_nm=true_thickness)
    sim = _make_sim_conditions()
    system = _make_system_config()

    psi, delta, _ = _generate_synthetic_spectrum(sample_true, sim)
    target = np.concatenate([psi, delta])

    floating = _make_floating_params()
    fitter = TmmDirectFitter(
        sample_config=_make_sio2_si_sample(thickness_nm=100.0),
        sim_conditions=sim,
        system_config=system,
        floating_params=floating,
        fitting_config=_make_fitting_config("nelder_mead"),
    )
    result = fitter.fit(target)
    fitted_t = result.fitted_parameters["thickness"]
    assert abs(fitted_t - true_thickness) < 2.0, (
        f"Nelder-Mead fitted {fitted_t:.3f} nm; expected near {true_thickness} nm"
    )


# ---------------------------------------------------------------------------
# Test 5: to_fit_results returns correct Pydantic schema
# ---------------------------------------------------------------------------


def test_to_fit_results_returns_fit_results_schema():
    """to_fit_results() must return a FitResults Pydantic model with correct fields."""
    from se_simulator.config.recipe import FitResults

    sample = _make_sio2_si_sample(100.0)
    sim = _make_sim_conditions()
    system = _make_system_config()
    floating = _make_floating_params()

    fitter = TmmDirectFitter(
        sample_config=sample,
        sim_conditions=sim,
        system_config=system,
        floating_params=floating,
        fitting_config=_make_fitting_config(),
    )

    psi, delta, _ = _generate_synthetic_spectrum(sample, sim)
    target = np.concatenate([psi, delta])
    tmm_result = fitter.fit(target)
    fit_results = fitter.to_fit_results(tmm_result)

    assert isinstance(fit_results, FitResults)
    assert "thickness" in fit_results.fitted_parameters
    assert fit_results.engine_used == "tmm_direct"
    assert "rmse" in fit_results.fit_quality


# ---------------------------------------------------------------------------
# Test 6: progress_callback is invoked
# ---------------------------------------------------------------------------


def test_progress_callback_invoked():
    """Progress callback should be called at least once during fitting."""
    sample = _make_sio2_si_sample(100.0)
    sim = _make_sim_conditions()
    system = _make_system_config()
    floating = _make_floating_params()

    calls: list[tuple[int, float]] = []

    def _cb(iteration: int, rmse: float) -> None:
        calls.append((iteration, rmse))

    psi, delta, _ = _generate_synthetic_spectrum(sample, sim)
    target = np.concatenate([psi, delta])

    fitter = TmmDirectFitter(
        sample_config=sample,
        sim_conditions=sim,
        system_config=system,
        floating_params=floating,
        fitting_config=_make_fitting_config(),
        progress_callback=_cb,
    )
    fitter.fit(target)

    assert len(calls) > 0, "progress_callback was never called"
    # Iterations should be monotonically non-decreasing
    iterations = [c[0] for c in calls]
    assert iterations == sorted(iterations)


# ---------------------------------------------------------------------------
# Test 7: _strip_prefix handles all prefix cases
# ---------------------------------------------------------------------------


def test_strip_prefix_cases():
    """_strip_prefix correctly removes known prefixes."""
    full_inline = "forward_model.sample.inline.layers[0].thickness_nm"
    assert _strip_prefix(full_inline) == "layers[0].thickness_nm"

    fm_only = "forward_model.simulation_conditions.aoi_degrees"
    assert _strip_prefix(fm_only) == "simulation_conditions.aoi_degrees"

    bare = "layers[0].thickness_nm"
    assert _strip_prefix(bare) == "layers[0].thickness_nm"
