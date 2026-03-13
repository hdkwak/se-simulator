"""Tests for Step 3a: PRCSA ellipsometer core."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from se_simulator.config.schemas import (
    CompensatorRetardanceModel,
    SystemConfig,
)
from se_simulator.ellipsometer.jones import (
    linear_polarizer,
    rotating_compensator,
    wave_plate,
)
from se_simulator.ellipsometer.prcsa import (
    compute_fourier_coefficients,
    compute_psi_delta,
    compute_spectrum,
    resolve_retardance,
)
from se_simulator.ellipsometer.signals import EllipsometryResult
from se_simulator.rcwa.results import RCWAResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_system(retardance_deg: float = 90.0) -> SystemConfig:
    return SystemConfig(
        instrument_name="Test",
        polarizer_angle_deg=45.0,
        analyzer_angle_deg=45.0,
        compensator_angle_deg=0.0,
        compensator_retardance=CompensatorRetardanceModel(type="constant", value=retardance_deg),
    )


def _mock_rcwa_result(n_wl: int = 50) -> RCWAResult:
    wls = np.linspace(400.0, 700.0, n_wl)
    jones_r = np.zeros((n_wl, 2, 2), dtype=complex)
    # Set Rpp = 0.5, Rss = 1.0 (diagonal) for each wavelength
    jones_r[:, 1, 1] = 0.5
    jones_r[:, 0, 0] = 1.0
    return RCWAResult(
        wavelengths_nm=wls,
        jones_reflection=jones_r,
        jones_transmission=np.zeros((n_wl, 2, 2), dtype=complex),
        energy_conservation=np.ones(n_wl),
    )


# ---------------------------------------------------------------------------
# Jones matrix tests
# ---------------------------------------------------------------------------

def test_linear_polarizer_transmits_aligned():
    """Polarizer at 0° transmits x-polarized light: J·[1,0]^T = [1,0]^T."""
    j = linear_polarizer(0.0)
    e_in = np.array([1.0, 0.0], dtype=complex)
    e_out = j @ e_in
    assert abs(e_out[0] - 1.0) < 1e-12
    assert abs(e_out[1]) < 1e-12


def test_wave_plate_quarter_wave():
    """QWP at 45° converts linear (x) to circular: |Ex|=|Ey|, phase diff = 90°."""
    j = wave_plate(fast_axis_deg=45.0, retardance_deg=90.0)
    e_in = np.array([1.0, 0.0], dtype=complex)
    e_out = j @ e_in
    assert abs(abs(e_out[0]) - abs(e_out[1])) < 1e-10
    phase_diff = np.angle(e_out[1]) - np.angle(e_out[0])
    assert abs(abs(phase_diff) - np.pi / 2) < 1e-10


# ---------------------------------------------------------------------------
# psi/delta tests
# ---------------------------------------------------------------------------

def test_psi_delta_from_identity():
    """Identity Jones matrix: Ψ=45°, Δ=0°."""
    jr = np.eye(2, dtype=complex)
    # eye has [0,0]=1=Rss, [1,1]=1=Rpp → ρ=1 → Ψ=45°, Δ=0°
    psi, delta = compute_psi_delta(jr)
    assert abs(psi - 45.0) < 1e-10
    assert abs(delta) < 1e-10


def test_psi_delta_from_known_ratio():
    """Rpp = 0.5·exp(i·30°), Rss = 1.0 → Ψ≈26.565°, Δ=30°."""
    jr = np.zeros((2, 2), dtype=complex)
    jr[1, 1] = 0.5 * np.exp(1j * np.radians(30.0))  # Rpp
    jr[0, 0] = 1.0                                    # Rss
    psi, delta = compute_psi_delta(jr)
    expected_psi = np.degrees(np.arctan(0.5))
    assert abs(psi - expected_psi) < 0.01, f"Ψ={psi:.4f}, expected {expected_psi:.4f}"
    assert abs(delta - 30.0) < 0.01, f"Δ={delta:.4f}, expected 30.0"


# ---------------------------------------------------------------------------
# Fourier coefficient tests
# ---------------------------------------------------------------------------

def test_fourier_coefficients_normalization():
    """Fourier coefficients are real and finite for 10 random Jones matrices; I₀ > 0."""
    rng = np.random.default_rng(42)
    system = _default_system(90.0)
    for _ in range(10):
        jr = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        coeffs = compute_fourier_coefficients(jr, system, 90.0)
        assert coeffs["I0"] > 0.0, "I₀ must be positive"
        for key in ("alpha", "beta", "chi", "xi"):
            assert np.isfinite(coeffs[key]), f"{key} is not finite: {coeffs[key]}"


def test_alpha_beta_dc_component():
    """I₀ > 0 for a non-zero Jones matrix."""
    jr = np.eye(2, dtype=complex)
    system = _default_system(90.0)
    coeffs = compute_fourier_coefficients(jr, system, 90.0)
    assert coeffs["I0"] > 0.0


# ---------------------------------------------------------------------------
# compute_spectrum tests
# ---------------------------------------------------------------------------

def test_compute_spectrum_shapes():
    """compute_spectrum with Nλ=50 returns EllipsometryResult with shape (50,) arrays."""
    result = _mock_rcwa_result(50)
    system = _default_system()
    er = compute_spectrum(result, system)
    for arr_name in ("psi_deg", "delta_deg", "alpha", "beta", "chi", "xi",
                     "energy_conservation", "wavelengths_nm"):
        arr = getattr(er, arr_name)
        assert arr.shape == (50,), f"{arr_name} shape={arr.shape}"
    assert er.jones_reflection.shape == (50, 2, 2)


# ---------------------------------------------------------------------------
# CSV round-trip test
# ---------------------------------------------------------------------------

def test_csv_round_trip():
    """Save and reload EllipsometryResult; psi_deg matches to 6 decimal places."""
    result = _mock_rcwa_result(20)
    system = _default_system()
    er = compute_spectrum(result, system)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = Path(f.name)

    try:
        er.to_csv(path, metadata={"instrument": "TestSE", "sample": "Si"})
        er2 = EllipsometryResult.from_csv(path)
        np.testing.assert_array_almost_equal(er.psi_deg, er2.psi_deg, decimal=6)
        np.testing.assert_array_almost_equal(er.delta_deg, er2.delta_deg, decimal=6)
        np.testing.assert_array_almost_equal(er.alpha, er2.alpha, decimal=6)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Retardance resolver tests
# ---------------------------------------------------------------------------

def test_retardance_resolver_constant():
    """Constant retardance=90° → all values == 90.0."""
    system = _default_system(retardance_deg=90.0)
    wls = np.linspace(400.0, 700.0, 100)
    ret = resolve_retardance(system, wls)
    assert ret.shape == (100,)
    np.testing.assert_array_almost_equal(ret, 90.0)


def test_retardance_resolver_polynomial():
    """Polynomial [90.0, 0.0] (ascending: 90 + 0*λ) → all values ≈ 90.0."""
    from se_simulator.config.schemas import CompensatorRetardanceModel, SystemConfig

    system = SystemConfig(
        instrument_name="Test",
        polarizer_angle_deg=45.0,
        analyzer_angle_deg=45.0,
        compensator_angle_deg=0.0,
        compensator_retardance=CompensatorRetardanceModel(
            type="polynomial",
            coefficients=[90.0, 0.0],
        ),
    )
    wls = np.linspace(400.0, 700.0, 50)
    ret = resolve_retardance(system, wls)
    np.testing.assert_array_almost_equal(ret, 90.0, decimal=6)


# ---------------------------------------------------------------------------
# Step 3b: calibration error tests
# ---------------------------------------------------------------------------

def test_calibration_errors_no_op_when_zero():
    """apply_calibration_errors with all offsets == 0 returns jones_r unchanged (< 1e-12)."""
    from se_simulator.ellipsometer.calibration import apply_calibration_errors

    system = _default_system()
    jr = np.array([[0.7 + 0j, 0.02 + 0.01j], [0.01 - 0.02j, 0.5 * np.exp(1j * np.radians(-40))]])
    jr_eff = apply_calibration_errors(jr, system, 632.8)
    assert np.max(np.abs(jr_eff - jr)) < 1e-12


def test_calibration_errors_change_signal():
    """apply_calibration_errors with delta_P=1.0° produces a different Jones matrix (> 1e-6)."""
    from se_simulator.config.schemas import CalibrationErrors
    from se_simulator.ellipsometer.calibration import apply_calibration_errors

    jr = np.array([[0.7 + 0j, 0.0j], [0.0j, 0.5 * np.exp(1j * np.radians(-40))]])
    system_err = _default_system().model_copy(
        update={"calibration_errors": CalibrationErrors(delta_P_deg=1.0)}
    )
    jr_eff = apply_calibration_errors(jr, system_err, 632.8)
    assert np.max(np.abs(jr_eff - jr)) > 1e-6


def test_sensitivity_finite_difference():
    """sensitivity_spectrum for delta_P on SiO2-like Jones matrix: dΨ/dP < 1 deg/deg."""
    from se_simulator.ellipsometer.calibration import sensitivity_spectrum

    # SiO2-like diagonal Jones matrix at 65° AOI
    jones_r = np.zeros((1, 2, 2), dtype=complex)
    jones_r[0, 0, 0] = 0.7                                    # Rss
    jones_r[0, 1, 1] = 0.5 * np.exp(1j * np.radians(-40.0))  # Rpp

    rcwa_result = RCWAResult(
        wavelengths_nm=np.array([632.8]),
        jones_reflection=jones_r,
        jones_transmission=np.zeros((1, 2, 2), dtype=complex),
        energy_conservation=np.ones(1),
    )

    system = _default_system()
    sens = sensitivity_spectrum(rcwa_result, system, parameters=["delta_P"])
    assert abs(sens["delta_P_psi"][0]) < 1.0, f"dΨ/dP = {sens['delta_P_psi'][0]:.4f} ≥ 1"
